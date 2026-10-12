"""Full fixed V34 matched training; original controller never enters optimizer.

Must run only after COMPLETE native data audit and exact source publication.
All original score/derivative checks precede each appropriate optimizer stage.
No resume, budget override, holdout, active selector or best-epoch option.
"""
import argparse
import collections
import copy
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'cwm_v35'), str(HERE.parent/'cwm_v33'), str(HERE.parent/'cwm_v28')]
from contract import load_contract
from audited_stream import AuditedCorpus, population_weights, sha
from stream_objective import public_batch, batch_objective
from prefix_response_core import PrefixQueryResponse
from train_ranking_response import (LocalTwoHead, CalibratedOrigin, FrozenOriginResponse,
    calibrated_motion_loss, frozen_configs, save_json, state_digest, BASELINE_SHA, verify)
from ranking_loss import ranking_kl
from cost_runtime import (CanonicalCache, audit_public_origins, require_motion_permit, context_for)
from stream_metrics import PredictionMetrics
from decision_gate import build_decisions, qualify, library_contributions, DecisionRows
from collect_prefix_data import source_hashes, GitHubAPI, credential_token
from input_precision import audit_command_precision


def published_sources():
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if GitHubAPI(credential_token()).head() != commit:
        raise ValueError('Publish exact scientific source/contract HEAD before training')
    hashes = source_hashes()
    for path in (*HERE.glob('*.py'), HERE/'protocol.json', HERE.parent/'cwm_v11/protocol.json'):
        hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    for name, digest in sorted(hashes.items()):
        if subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT) != (ROOT/name).read_bytes():
            raise ValueError('Actual scientific source differs from exact published bytes: '+name)
    return {'commit': commit, 'scientific_sources': dict(sorted(hashes.items())),
        'exact_published_bytes_verified': True}


def verify_train_normalizer(corpus, mean, scale, count):
    """Independent fixed-block reconstruction; no development history access."""
    n = 0; average = np.zeros(252, dtype=np.float64); squares = np.zeros(252, dtype=np.float64)
    for index in corpus.indices['train']:
        block = np.asarray(corpus.load(index)['values']['history'], dtype=np.float64)
        local_average = block.sum(axis=0)/8
        delta = local_average-average; total = n+8
        local_squares = np.square(block-local_average).sum(axis=0)
        squares += local_squares+delta**2*(n*8/total)
        average += delta*(8/total); n = total
    expected_scale = np.maximum(np.sqrt(np.maximum(0., squares/n)), .01).reshape(1, 1, 252)
    if n != count or mean.tobytes() != average.reshape(1, 1, 252).tobytes() or scale.tobytes() != expected_scale.tobytes():
        raise ValueError('Complete TRAIN-only fixed block normalizer differs')


def checked_response_parameters(model):
    if not isinstance(model, FrozenOriginResponse) or not isinstance(model.response_core, PrefixQueryResponse):
        raise ValueError('Only matched V33 response with original frozen common allowed')
    if any(p.requires_grad or p.grad is not None for p in model.common.parameters()):
        raise ValueError('Common motion must be frozen and gradients cleared BEFORE optimizer')
    parameters = [p for p in model.parameters() if p.requires_grad]
    common_ids = {id(p) for p in model.common.parameters()}
    if (any(id(p) in common_ids for p in parameters) or
        sum(p.numel() for p in parameters) != 31432 or
        sum(p.numel() for p in model.response_core.parameters()) != 32992):
        raise ValueError('Fixed matched response-only parameter budget/ownership required')
    return parameters


def verify_used_sources(prereg, output, capsule, native_manifest):
    if sha(capsule) != BASELINE_SHA:
        raise ValueError('Original baseline capsule identity changed before optimizer')
    verify(output/'r', native_manifest)
    for name, digest in prereg['scientific_sources'].items():
        if sha(ROOT/name) != digest or sha(output/'source'/name) != digest:
            raise ValueError('Actual published scientific sources changed before optimizer')


def run_epochs(model, optimizer, name, corpus, mean, scale, counts, weights, cache, config):
    motion = name == 'cv_motion_only'; sampler = np.random.default_rng(config['seed']); history = []
    epochs = config['motion_epochs' if motion else 'response_epochs']
    if epochs != 80 or config['batch_calls'] != 16:
        raise ValueError('Full fixed80epoch/16call budget required; no smaller pilot')
    common_digest = None if motion else state_digest(model.common)
    for epoch in range(1, 81):
        model.train(); losses = []; order = sampler.permutation(corpus.indices['train'])
        for start in range(0, len(order), 16):
            ids = [int(i) for i in order[start:start+16]]; calls = corpus.batch(ids)
            if motion:
                loss, terms = calibrated_motion_loss(model, calls, list(range(len(calls))), mean, scale, counts, .001)
            else:
                inputs, slices = public_batch(calls, mean, scale); _, _, response = model(*inputs)
                loss, terms = batch_objective(response, calls, ids, slices, weights, cache.batch(ids, calls))
            if not torch.isfinite(loss): raise ValueError('Nonfinite full fixed-budget loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
            optimizer.step(); losses.append([float(loss.detach()), *[float(v.detach()) for v in terms]])
        if not losses: raise ValueError('No complete TRAIN population')
        if not motion and (state_digest(model.common) != common_digest or any(p.grad is not None or p.requires_grad for p in model.common.parameters())):
            raise ValueError('Original frozen common changed during response training')
        history.append({'epoch': epoch, 'mean_batch_loss_terms': np.mean(losses, axis=0).tolist()})
        if epoch % 20 == 0: print(json.dumps({'configuration': name, 'seed': config['seed'], **history[-1]}), flush=True)
    return history, sampler.bit_generator.state


def controls(corpus):
    result = {}
    for name in ('frozen_gru', 'constant_velocity'):
        measured = PredictionMetrics()
        for index in corpus.indices['development_validation']:
            call = corpus.load(index); v = call['values']; n = len(v['proposed'])
            backbone = np.repeat(v['backbone'][None], n, 0)
            prediction = backbone if name == 'frozen_gru' else np.repeat((v['reference'][None]+.1*np.arange(1, 9)[:, None]*v['velocity'][None])[None], n, 0)
            output = {'prediction': prediction, 'motion': prediction-backbone, 'response': np.zeros_like(backbone)}
            measured.add(call['record'], v, output, v['bounded_shared_indices'])
        result[name] = measured.result({corpus.records[i]['group'] for i in corpus.indices['development_validation']})
    return result


def evaluate(model, corpus, mean, scale, cache, output):
    """Same global canonical blocks; no entire prediction arrays in memory."""
    model.eval(); output.mkdir(parents=True, exist_ok=False)
    base = {s: PredictionMetrics() for s in corpus.indices}; augmented = {s: PredictionMetrics() for s in corpus.indices}
    axes = [variant['name'] for variant in corpus.protocol['variants']]
    axis_metrics = {s: {a: PredictionMetrics() for a in axes} for s in corpus.indices}
    populations = ('actual_unique', 'public_geometry', 'bounded_shared')
    decision_paths = {s: {p: output/f'{s}_{p}.jsonl' for p in populations} for s in corpus.indices}
    decision_counts = {s: {p: 0 for p in populations} for s in corpus.indices}; files = {}
    for start in range(0, len(corpus), 16):
        ids = list(range(start, min(start+16, len(corpus)))); calls = corpus.batch(ids)
        inputs, slices = public_batch(calls, mean, scale)
        with torch.no_grad():
            tensors = [v.numpy() for v in model(*inputs)]
            reference = model.common(*inputs)[0].numpy() if isinstance(model, FrozenOriginResponse) else tensors[0]
        for index, call, (a, b) in zip(ids, calls, slices):
            out = {key: value[a:b].copy() for key, value in zip(('prediction', 'motion', 'response', 'reference'), [*tensors, reference])}
            canonical = cache.arrays(index)['reference_augmented']
            if out['reference'].dtype != canonical.dtype or out['reference'].shape != canonical.shape or out['reference'].tobytes() != canonical.tobytes():
                raise ValueError('Final cloned common reference differs from canonical global blocks')
            row, v = call['record'], call['values']; split = row['split']; ranking = cache.load(index, call)
            with torch.no_grad(): kl = None if ranking is None else float(ranking_kl(ranking, torch.from_numpy(out['response'][:int(v['base_candidate_count'])])))
            base[split].add(row, v, out, v['bounded_shared_indices'], kl)
            augmented[split].add(row, v, out, np.arange(len(v['proposed'])))
            axis_metrics[split][row['variant']].add(row, v, out, v['bounded_shared_indices'], kl)
            if ranking is not None:
                contexts = context_for(corpus, call, cache.configs)
                records = build_decisions(call, out, contexts)
                for population, record in records.items():
                    # Each output directory is exclusive; append only our records.
                    with decision_paths[split][population].open('a', encoding='utf8') as file:
                        file.write(json.dumps(record, allow_nan=False)+'\n')
                    decision_counts[split][population] += 1
            path = output/f'{index}.npz'; np.savez_compressed(path, **out)
            files[str(index)] = {'path': path.name, 'sha256': sha(path)}
    summary = {}
    for split in corpus.indices:
        groups = {corpus.records[i]['group'] for i in corpus.indices[split]}
        summary[split] = base[split].result(groups, require_ranking=True)
        summary[split]['augmented_prediction_diagnostic_not_selection'] = augmented[split].result(groups)
        summary[split]['by_geometry_axis'] = {a: m.result(groups, require_ranking=True) for a, m in axis_metrics[split].items()}
    save_json(output/'manifest.json', files)
    decisions = {s: {p: DecisionRows(path, sha(path), decision_counts[s][p]) for p, path in rows.items()} for s, rows in decision_paths.items()}
    save_json(output/'decision_manifest.json', {s: {p: {'path': r.path.name, 'sha256': r.digest, 'calls': r.count} for p, r in rows.items()} for s, rows in decisions.items()})
    return summary, decisions


def train(data, audit, output):
    if output.exists(): raise ValueError('Exclusive NEW complete scientific run required')
    contract, plan = load_contract()
    # No output/model/normalizer/optimizer until complete data + publication.
    corpus = AuditedCorpus(data, audit); prereg = published_sources()
    precision = audit_command_precision(corpus)
    mean, scale, frames = corpus.normalization(); verify_train_normalizer(corpus, mean, scale, frames)
    weights, population = population_weights(corpus)
    output.mkdir(parents=True, exist_ok=False); save_json(output/'preregistration.json', prereg)
    save_json(output/'public_input_precision_audit.json', precision)
    capsule = HERE.parent/'cwm_v1/baseline/capsule.zip'
    if sha(capsule) != BASELINE_SHA: raise ValueError('Original baseline capsule changed')
    configs = frozen_configs(capsule, output/'r')
    with zipfile.ZipFile(capsule) as archive:
        native_manifest = json.loads(archive.read('CAPSULE_MANIFEST.json'))
    np.savez_compressed(output/'normalization.npz', mean=mean, scale=scale, train_frames=np.asarray(frames, dtype=np.int64))
    save_json(output/'population_weights.json', {'population': population, 'weights': weights})
    for name, digest in prereg['scientific_sources'].items():
        destination = output/'source'/name; destination.parent.mkdir(parents=True, exist_ok=True)
        if sha(ROOT/name) != digest: raise ValueError('Published source changed before scientific run')
        destination.write_bytes((ROOT/name).read_bytes())
    initial = audit_public_origins(corpus, configs, output)
    verify_used_sources(prereg, output, capsule, native_manifest)
    require_motion_permit(corpus, initial, output)
    counts = collections.Counter(corpus.records[i]['group'] for i in corpus.indices['train'])
    config = plan['training']; measured_controls = controls(corpus)
    rows, selected_decisions, all_decisions, contributions = [], {}, {}, {}
    for seed in config['seeds']:
        verify_used_sources(prereg, output, capsule, native_manifest)
        torch.manual_seed(seed)
        common = CalibratedOrigin(LocalTwoHead('motion_only', 2.5, 2.5), 'cv'); initial_common = state_digest(common)
        optimizer = torch.optim.Adam(common.parameters(), lr=.001)
        motion_history, motion_rng = run_epochs(common, optimizer, 'cv_motion_only', corpus, mean, scale, counts, weights, None, {**config, 'seed': seed})
        trained_motion = {'model_state': copy.deepcopy(common.state_dict()), 'optimizer_state': copy.deepcopy(optimizer.state_dict()),
            'torch_rng_state': torch.get_rng_state(), 'sampler_rng_state': motion_rng, 'initial_state_digest': initial_common}
        torch.save({**trained_motion, 'seed': seed, 'v34_plan': plan, 'v36_contract': contract,
            'source_hashes': prereg['scientific_sources'], 'online_promoted': False, 'baseline_weights_included': False}, output/f'cv_seed{seed}_motion_stage.pt')
        save_json(output/f'cv_seed{seed}_motion_stage_history.json', motion_history)
        common.eval().requires_grad_(False)
        for p in common.parameters(): p.grad = None
        common_digest = state_digest(common)
        cache = CanonicalCache.build(common, corpus, configs, mean, scale, output/f'cache{seed}')
        cache.require_response_permit(); response_initializations = []
        for name in plan['models']:
            if name == 'cv_motion_only':
                model, trained, history, initial_digest = common, trained_motion, motion_history, initial_common
            else:
                torch.manual_seed(seed); core = PrefixQueryResponse(name == 'cv_prefix_rank_l2')
                initial_digest = state_digest(core); response_initializations.append(initial_digest)
                model = FrozenOriginResponse(copy.deepcopy(common), core)
                verify_used_sources(prereg, output, capsule, native_manifest)
                parameters = checked_response_parameters(model)
                optimizer = torch.optim.Adam(parameters, lr=.001)
                history, rng = run_epochs(model, optimizer, name, corpus, mean, scale, counts, weights, cache, {**config, 'seed': seed})
                trained = {'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                    'torch_rng_state': torch.get_rng_state(), 'sampler_rng_state': rng, 'initial_state_digest': initial_digest}
                if state_digest(model.common) != common_digest: raise ValueError('Frozen common component changed')
            identifier = f'{name}_seed{seed}'
            measured, decisions = evaluate(model, corpus, mean, scale, cache, output/'predictions'/identifier)
            groups = {s: {corpus.records[i]['group'] for i in corpus.indices[s]} for s in corpus.indices}
            contribution = {s: library_contributions(decisions[s], contract, groups[s]) for s in corpus.indices}
            save_json(output/f'{identifier}_history.json', history)
            save_json(output/f'{identifier}_library_contributions.json', contribution)
            checkpoint = {**trained, 'configuration': name, 'seed': seed, 'v34_plan': plan, 'v36_contract': contract,
                'source_hashes': prereg['scientific_sources'], 'normalizer_mean': torch.from_numpy(mean), 'normalizer_scale': torch.from_numpy(scale),
                'common_motion_state_digest': common_digest, 'data_summary_sha256': sha(data/'summary.json'),
                'data_audit_summary_sha256': sha(audit/'summary.json'), 'population_weights_sha256': sha(output/'population_weights.json'),
                'canonical_cache_manifest_sha256': sha(cache.directory/'manifest.json'), 'online_promoted': False,
                'baseline_weights_included': False, 'common_motion_in_response_optimizer': False}
            torch.save(checkpoint, output/f'{identifier}.pt')
            rows.append({'configuration': name, 'seed': seed, 'checkpoint_sha256': sha(output/f'{identifier}.pt'),
                'initial_state_digest': initial_digest, 'common_motion_state_digest': common_digest,
                'parameters': sum(p.numel() for p in model.parameters()), 'train': measured['train'], 'development': measured['development_validation']})
            selected_decisions[identifier] = decisions['development_validation']['bounded_shared']
            all_decisions[identifier] = {s: decisions[s]['bounded_shared'].count for s in decisions}; contributions[identifier] = contribution
            print(json.dumps({'status': 'fixed_final80_model_measured', 'model': identifier}), flush=True)
        if len(response_initializations) != 2 or len(set(response_initializations)) != 1:
            raise ValueError('Matched prefix/full-query initialization differs')
    selected, gates = qualify(rows, measured_controls, selected_decisions, contract, plan,
        {corpus.records[i]['group'] for i in corpus.indices['development_validation']})
    corpus.reverify()
    verify_used_sources(prereg, output, capsule, native_manifest)
    if sha(capsule) != BASELINE_SHA or any(sha(ROOT/n) != value for n, value in prereg['scientific_sources'].items()):
        raise ValueError('Original capsule or actual used scientific sources changed')
    result = {'status': 'complete_fixed_streamed_nine_model_training_pending_repeat_reload',
        'v34_plan': plan, 'v36_contract': contract, 'source_hashes': prereg['scientific_sources'], 'models': rows,
        'controls': measured_controls, 'selected_median_seeds': selected, 'qualification': gates,
        'primary_research_eligible': gates[contract['primary_configuration']]['research_eligible'],
        'data_summary_sha256': sha(data/'summary.json'), 'data_audit_summary_sha256': sha(audit/'summary.json'),
        'decision_calls': all_decisions, 'library_contributions': contributions,
        'enhanced_control_enabled': False, 'holdout_used': False, 'prior_gate_overridden': False,
        'common_motion_in_response_optimizer': False, 'two_complete_runs_verified': False,
        'independent_full_reload_verified': False, 'scope': contract['scope']}
    save_json(output/'summary.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'audit', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    existed = args.output.exists()
    try:
        result = train(args.data, args.audit, args.output)
        print(json.dumps({'status': result['status'], 'primary_research_eligible': result['primary_research_eligible']}), flush=True)
    except Exception as error:
        if not existed and args.output.is_dir() and not (args.output/'summary.json').exists():
            save_json(args.output/'implementation_failure.json', {'status': 'incomplete_no_scientific_qualification',
                'exception_type': type(error).__name__, 'exception': str(error)[:300],
                'enhanced_control_enabled': False, 'holdout_used': False})
        raise
