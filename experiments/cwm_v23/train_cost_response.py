"""Two calibrated origins x two response losses; original controller untouched."""
import argparse
import collections
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cost_data_audit import audit_data, read_dataset
from cost_origin_model import (LocalTwoHead, CalibratedOrigin, FrozenOriginResponse,
    public_cost_inputs, calibrated_motion_loss, evaluate_cost_model, normalization)
from deployed_cost_loss import (FrozenLocalScore, original_scores, cost_population_weights,
    contrast_cache, audit_deployed_score, cost_response_loss, contrast_metrics)
from cost_qualification import validate_cost_training, qualify_cost_models
sys.path.insert(0, str(ROOT.parent / 'cwm_v19'))
from train_frozen_response import (prepare_contexts, control_metrics, state_digest,
    frozen_configs, sha, BASELINE_SHA)
from local_shadow import rank_metrics
from support_audit import map_actual
sys.path.insert(0, str(ROOT.parent / 'cwm_v20'))
from response_diagnostics import decompose


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')


def cost_decision_records(calls, indices, outputs, contexts):
    result = {'geometry_union': [], 'actual_unique': []}
    for i, output in zip(indices, outputs):
        if i not in contexts:
            continue  # Same complete UNION support for BOTH populations.
        row, v = calls[i]['record'], calls[i]['values']
        actions = v['proposed'][:, :, row['agent']]
        count = len(actions)
        cv = v['reference'][None] + .1 * np.arange(1, 9)[:, None] * v['velocity'][None]
        paths = {'model': output['prediction'], 'own_motion_component': output['reference'],
            'original_gru': np.repeat(v['backbone'][None], count, 0),
            'constant_velocity': np.repeat(cv[None], count, 0), 'action_specific_truth': v['target']}
        costs = {k: original_scores(contexts[i], actions, path) for k, path in paths.items()}
        raw = [int(np.flatnonzero((actions == a[None]).all((1, 2)))[0]) for a in v['actual_candidates']]
        if not np.allclose(costs['original_gru'][raw], v['original_local_costs'], rtol=1e-10, atol=1e-8):
            raise ValueError('Original actual candidate cost reproduction differs')
        actual, duplicates = map_actual(v, row['agent'])
        for population, selected in (('geometry_union', np.arange(count)), ('actual_unique', actual)):
            subset = {k: value[selected].tolist() for k, value in costs.items()}
            measured = {k: rank_metrics(value, subset['action_specific_truth']) for k, value in subset.items()}
            if population == 'actual_unique':
                anchor = np.flatnonzero(selected == row['actual_choice_pool_index'])
                if len(anchor) != 1 or measured['original_gru']['choice'] != int(anchor[0]):
                    raise ValueError('Original actual solver choice reproduction differs')
            result[population].append({**{k: row[k] for k in ('episode_index', 'step', 'agent', 'group')},
                'population': population, 'indices': selected.tolist(), 'actual_duplicate_candidates': duplicates,
                'costs': subset, 'metrics': measured})
    return result


def make_cost_caches(calls, indices, outputs, contexts, scores, audit=False):
    caches, records = {}, []
    for i, output in zip(indices, outputs):
        if i not in contexts:
            continue
        cache = contrast_cache(scores[i], output['reference'], calls[i]['values'])
        caches[i] = cache
        if audit:
            records.append({'call_index': i, **{k: calls[i]['record'][k]
                for k in ('episode_index', 'step', 'agent', 'group', 'split')},
                **audit_deployed_score(contexts[i], calls[i]['values'], output['reference'], scores[i], cache)})
    if set(caches) != set(contexts):
        raise ValueError('Incomplete canonical deployed-reference cache')
    return caches, records


def cache_arrays(caches):
    return {f'call{i}_{k}': v.numpy() for i, cache in caches.items() for k, v in cache.items() if torch.is_tensor(v)}


def run_cost_epochs(model, optimizer, name, calls, indices, mean, scale, weights, counts, caches, config, protocol):
    motion_stage = name.endswith('_motion_only')
    epochs = config['motion_epochs' if motion_stage else 'response_epochs']
    sampler = np.random.default_rng(config['seed'])
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        order = sampler.permutation(indices)
        for start in range(0, len(order), config['batch_calls']):
            chosen = order[start:start + config['batch_calls']]
            if motion_stage:
                loss, terms = calibrated_motion_loss(model, calls, chosen, mean, scale, counts, config['residual_penalty'])
            else:
                inputs, slices = public_cost_inputs(calls, chosen, mean, scale)
                _, _, response = model(*inputs)
                cost_weight = config['cost_contrast_weight'] if protocol['response_configurations'][name]['auxiliary_cost_contrast'] else 0.
                loss, terms = cost_response_loss(response, calls, chosen, slices, weights, caches,
                                                  config['residual_penalty'], cost_weight)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite fixed-budget preregistered loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config['gradient_clip_norm'])
            optimizer.step()
            losses.append([float(loss.detach()), *[float(t.detach()) for t in terms]])
        history.append({'epoch': epoch, 'mean_batch_loss_terms': np.mean(losses, 0).tolist()})
        if epoch % 20 == 0:
            print(json.dumps({'model': name, 'seed': config['seed'], **history[-1]}), flush=True)
    return history, sampler.bit_generator.state


def run_cost_training(data, public, output):
    protocol = json.loads((ROOT / 'training_protocol.json').read_text(encoding='utf8'))
    validate_cost_training(protocol)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    output.mkdir(parents=True, exist_ok=False)
    capsule = ROOT.parent / 'cwm_v1/baseline/capsule.zip'
    if sha(capsule) != BASELINE_SHA:
        raise ValueError('Original baseline capsule changed')
    configs = frozen_configs(capsule, output / 'restored')
    read = read_dataset(data, public)
    calls, data_report = audit_data(read, configs, output / 'restored')
    indices = {split: [i for i, c in enumerate(calls) if c['record']['split'] == split]
               for split in ('train', 'development_validation')}
    weights, population = cost_population_weights(calls, indices['train'])
    save_json(output / 'population_weights.json', {'population': population, 'weights': weights})
    mean, scale = normalization(calls, indices['train'])
    np.savez_compressed(output / 'normalization.npz', mean=mean, scale=scale)
    all_indices = list(range(len(calls)))
    contexts = prepare_contexts(calls, all_indices, read, configs)
    scores = {i: FrozenLocalScore(context, calls[i]['values']['proposed'][:, :, calls[i]['record']['agent']])
              for i, context in contexts.items()}
    controls = control_metrics(calls, indices['development_validation'])
    diagnostic = json.loads((ROOT.parent / 'cwm_v20/diagnostic_protocol.json').read_text(encoding='utf8'))
    sources = sorted({Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, '__file__', None)
        and Path(m.__file__).resolve().is_relative_to(ROOT.parent)} |
        {Path(__file__).resolve(), ROOT / 'data_protocol.json', ROOT / 'training_protocol.json',
         ROOT.parent / 'cwm_v11/protocol.json', ROOT.parent / 'cwm_v20/diagnostic_protocol.json'})
    hashes = {p.relative_to(ROOT.parent).as_posix(): sha(p) for p in sources}
    for source in sources:
        dest = output / 'source' / source.relative_to(ROOT.parent)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    print(json.dumps({'status': 'fresh_v23_independent_public_original_cost_data_audit_passed',
                      'calls': {k: len(v) for k, v in indices.items()}, 'full_union_support': len(contexts)}), flush=True)
    # Audit both public initial references BEFORE ANY optimizer is instantiated.
    initial_audits = {}
    for origin in ('gru', 'cv'):
        initial_outputs = []
        for c in calls:
            v = c['values']
            path = v['backbone'] if origin == 'gru' else v['reference'][None] + .1 * np.arange(1, 9)[:, None] * v['velocity'][None]
            initial_outputs.append({'reference': np.repeat(path[None], len(v['proposed']), 0)})
        _, initial_audits[origin] = make_cost_caches(calls, all_indices, initial_outputs, contexts, scores, True)
        save_json(output / f'initial_{origin}_score_audit.json', initial_audits[origin])
        print(json.dumps({'status': 'initial_original_cost_and_gradient_audit_passed',
                          'origin': origin, 'complete_calls': len(initial_audits[origin])}), flush=True)
    config = protocol['training']
    counts = collections.Counter(calls[i]['record']['group'] for i in indices['train'])
    models, decisions, actual_decisions = [], {}, {}
    for seed in config['seeds']:
        common_by_origin, motion_training, caches_by_origin = {}, {}, {}
        # Matched independent pretraining: reset init/RNG and sampler for each origin.
        for origin in ('gru', 'cv'):
            torch.manual_seed(seed)
            core = LocalTwoHead('motion_only', config['motion_scale_m'], config['response_scale_m'])
            initial = state_digest(core)
            common = CalibratedOrigin(core, origin)
            optimizer = torch.optim.Adam([p for p in common.parameters() if p.requires_grad], lr=config['learning_rate'])
            history, rng = run_cost_epochs(common, optimizer, origin + '_motion_only', calls,
                indices['train'], mean, scale, weights, counts, {}, {**config, 'seed': seed}, protocol)
            trained = {'model_state': copy.deepcopy(common.state_dict()), 'optimizer_state': copy.deepcopy(optimizer.state_dict()),
                'initial_state_digest': initial, 'torch_rng_state': torch.get_rng_state(), 'sampler_rng_state': rng}
            # Preserve the completed pretraining stage BEFORE a later strict
            # score/cache audit can stop this run. This adds no optimization.
            torch.save({**trained, 'origin': origin, 'seed': seed, 'protocol': protocol,
                'source_hashes': hashes, 'online_promoted': False}, output / f'{origin}_seed{seed}_motion_stage.pt')
            save_json(output / f'{origin}_seed{seed}_motion_stage_history.json', history)
            common.eval().requires_grad_(False)
            for p in common.parameters():
                p.grad = None
            common_by_origin[origin] = common
            motion_training[origin] = (trained, history, initial)
            # Canonical frozen reference uses the same fixed batch32 evaluation
            # order as final forecasts, independent of response minibatch shuffles.
            canonical = {}
            for chosen in indices.values():
                _, split_outputs = evaluate_cost_model(common, calls, chosen, mean, scale, config['batch_calls'])
                canonical.update(zip(chosen, split_outputs))
            motion_outputs = [canonical[i] for i in all_indices]
            caches, audit = make_cost_caches(calls, all_indices, motion_outputs, contexts, scores, True)
            caches_by_origin[origin] = caches
            save_json(output / f'{origin}_seed{seed}_deployed_score_audit.json', audit)
            np.savez_compressed(output / f'{origin}_seed{seed}_deployed_cost_cache.npz', **cache_arrays(caches))
            print(json.dumps({'status': 'calibrated_original_cost_and_gradient_audit_passed',
                              'origin': origin, 'seed': seed, 'complete_calls': len(audit)}), flush=True)
        if motion_training['gru'][2] != motion_training['cv'][2]:
            raise ValueError('Matched origin initialization differs')
        response_initials = []
        for name in protocol['models']:
            origin = name.split('_')[0]
            common = common_by_origin[origin]
            common_digest = state_digest(common)
            caches = caches_by_origin[origin]
            if name.endswith('_motion_only'):
                model = common
                trained, used_history, initial = motion_training[origin]
            else:
                torch.manual_seed(seed)
                core = LocalTwoHead('plain', config['motion_scale_m'], config['response_scale_m'])
                initial = state_digest(core)
                response_initials.append(initial)
                model = FrozenOriginResponse(copy.deepcopy(common), core)
                optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config['learning_rate'])
                common_ids = {id(p) for p in model.common.parameters()}
                if any(id(p) in common_ids for g in optimizer.param_groups for p in g['params']):
                    raise ValueError('Frozen common motion entered response optimizer')
                used_history, rng = run_cost_epochs(model, optimizer, name, calls, indices['train'],
                    mean, scale, weights, counts, caches, {**config, 'seed': seed}, protocol)
                if state_digest(model.common) != common_digest or any(p.grad is not None or p.requires_grad for p in model.common.parameters()):
                    raise ValueError('Frozen origin motion mutated')
                trained = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                    'initial_state_digest': initial, 'torch_rng_state': torch.get_rng_state(), 'sampler_rng_state': rng}
            identifier = f'{name}_seed{seed}'
            split_metrics, split_diagnostics, split_decisions = {}, {}, {}
            for split, chosen in indices.items():
                measured, outputs = evaluate_cost_model(model, calls, chosen, mean, scale, config['batch_calls'])
                # Canonical per-split fixed batch32 evaluation must exactly match
                # cached M, even if skipped data makes split sizes irregular.
                if any(not np.array_equal(outputs[j]['reference'], caches[i]['reference'].numpy())
                       for j, i in enumerate(chosen) if i in caches):
                    raise ValueError('Final deployed reference differs from loss cache')
                measured.update(contrast_metrics(calls, chosen, outputs, caches))
                np.savez_compressed(output / f'{identifier}_{split}_predictions.npz',
                    **{f'call{j}_{k}': v for j, o in enumerate(outputs) for k, v in o.items()})
                split_metrics[split] = measured
                split_diagnostics[split] = decompose(calls, chosen, outputs, diagnostic)
                if not np.isclose(split_diagnostics[split]['overall']['group_equal_response_error_m'],
                                  measured['group_equal_paired_response_error_m'], rtol=0, atol=1e-12):
                    raise ValueError('Diagnostic response support differs from gate')
                split_decisions[split] = cost_decision_records(calls, chosen, outputs, contexts)
                if split == 'development_validation':
                    decisions[identifier] = split_decisions[split]['geometry_union']
                    actual_decisions[identifier] = split_decisions[split]['actual_unique']
            save_json(output / f'{identifier}_decisions.json', split_decisions)
            save_json(output / f'{identifier}_history.json', used_history)
            save_json(output / f'{identifier}_diagnostics.json', split_diagnostics)
            checkpoint = {**trained, 'configuration': name, 'seed': seed, 'origin': origin,
                'common_motion_state_digest': common_digest, 'normalizer_mean': torch.from_numpy(mean),
                'normalizer_scale': torch.from_numpy(scale), 'protocol': protocol, 'source_hashes': hashes,
                'data_summary_sha256': sha(data / 'summary.json'), 'public_summary_sha256': sha(public / 'summary.json'),
                'population_weights_sha256': sha(output / 'population_weights.json'),
                'deployed_cost_cache_sha256': sha(output / f'{origin}_seed{seed}_deployed_cost_cache.npz'),
                'online_promoted': False, 'baseline_weights_included': False, 'mediator_optimized': False,
                'common_motion_in_response_optimizer': False}
            torch.save(checkpoint, output / f'{identifier}.pt')
            row = {'configuration': name, 'seed': seed, 'origin': origin, 'initial_state_digest': initial,
                'common_motion_state_digest': common_digest, 'parameters': sum(p.numel() for p in model.parameters()),
                'checkpoint_sha256': sha(output / f'{identifier}.pt'),
                'train': split_metrics['train'], 'development': split_metrics['development_validation']}
            models.append(row)
            print(json.dumps({'status': 'final_model_measured', 'model': identifier,
                'dev_ade': row['development']['group_equal_ade_m'],
                'dev_response': row['development']['group_equal_paired_response_error_m'],
                'dev_cost_contrast': row['development']['group_equal_normalized_cost_contrast_l1']}), flush=True)
        if len(set(response_initials)) != 1 or len(response_initials) != 4:
            raise ValueError('Matched response-core initialization differs')
    selected, gates, factorial = qualify_cost_models(models, controls, decisions, protocol)
    # Actual-library comparisons are additionally disclosed, not gate substitutes.
    _, actual_gates, actual_factorial = qualify_cost_models(models, controls, actual_decisions, protocol)
    if sha(capsule) != BASELINE_SHA or any(sha(ROOT.parent / n) != digest for n, digest in hashes.items()):
        raise ValueError('Frozen original capsule or run-used sources changed')
    summary = {'status': 'offline_fresh_deployed_cost_contrast_finished_not_promoted',
        'protocol': protocol, 'source_hashes': hashes, 'models': models, 'controls': controls,
        'qualification': gates, 'factorial_comparisons': factorial,
        'actual_unique_diagnostic_qualification': actual_gates, 'actual_unique_factorial_diagnostics': actual_factorial,
        'actual_unique_is_promotion_evidence': False,
        'primary_research_eligible': gates[protocol['primary_configuration']]['research_eligible'],
        'selected_median_seeds': selected, 'split_calls': {k: len(v) for k, v in indices.items()},
        'split_groups': {k: sorted({calls[i]['record']['group'] for i in v}) for k, v in indices.items()},
        'full_cost_calls': {k: sum(i in contexts for i in v) for k, v in indices.items()},
        'data_summary_sha256': sha(data / 'summary.json'), 'public_summary_sha256': sha(public / 'summary.json'),
        'population_weights_sha256': sha(output / 'population_weights.json'),
        'enhanced_control_enabled': False, 'baseline_weights_included': False, 'holdout_used': False,
        'prior_gate_overridden': False, 'common_motion_in_response_optimizer': False, 'scope': protocol['later_required']}
    save_json(output / 'summary.json', summary)
    print(json.dumps({'status': summary['status'], 'qualification': gates, 'selected': selected}), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'public', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        run_cost_training(args.data, args.public, args.output)
    except Exception as error:
        if args.output.is_dir() and not (args.output / 'summary.json').exists():
            failure = args.output / 'implementation_failure.json'
            if not failure.exists():
                with failure.open('x', encoding='utf8') as file:
                    json.dump({'status': 'incomplete_attempt_no_qualification',
                        'exception_type': type(error).__name__, 'exception': str(error)[:300],
                        'enhanced_control_enabled': False, 'holdout_used': False}, file, indent=2)
        raise


if __name__ == '__main__':
    main()
