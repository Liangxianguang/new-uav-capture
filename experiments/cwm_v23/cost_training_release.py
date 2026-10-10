"""Full data/cost/cache/reinference audit of BOTH independent V23 research runs."""
import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_cost_response import (audit_data, normalization, cost_population_weights, control_metrics,
    prepare_contexts, FrozenLocalScore, make_cost_caches, cache_arrays, cost_decision_records,
    evaluate_cost_model, contrast_metrics, CalibratedOrigin, FrozenOriginResponse, LocalTwoHead,
    state_digest, frozen_configs, sha, BASELINE_SHA, decompose)
from cost_qualification import validate_cost_training, qualify_cost_models
from frozen_training_release import tensor_tree_equal, verify_manifest
from geometry_release import arrays, compare_arrays
sys.path.insert(0, str(ROOT.parent / 'cwm_v21'))
from artifact_transport import split_archive, resolve_artifact


def read_json(read, name):
    return json.loads(read(name))


def reconstruct_model(row, protocol):
    name, seed, origin = row['configuration'], row['seed'], row['origin']
    if origin not in ('gru', 'cv') or origin != name.split('_')[0]:
        raise ValueError('Checkpoint/model public origin differs')
    config = protocol['training']
    motion = name.endswith('_motion_only')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        core = LocalTwoHead('motion_only' if motion else 'plain', config['motion_scale_m'], config['response_scale_m'])
        if state_digest(core) != row['initial_state_digest']:
            raise ValueError('Fixed matched per-seed initialization differs')
        common = CalibratedOrigin(core if motion else LocalTwoHead('motion_only',
            config['motion_scale_m'], config['response_scale_m']), origin)
        model = common if motion else FrozenOriginResponse(common, core)
    return model


def audit_saved_sources(read, stage, report):
    for name, digest in report['source_hashes'].items():
        if hashlib.sha256(read(stage + '/source/' + name)).hexdigest() != digest or sha(ROOT.parent / name) != digest:
            raise ValueError('Saved/current run-used deployed-cost source differs')
    if read(stage + '/source/cwm_v23/training_protocol.json') != (ROOT / 'training_protocol.json').read_bytes():
        raise ValueError('Saved preregistered protocol bytes changed')


def audit_optimizer_history(read, stage, identifier, checkpoint, model, train_count, protocol):
    config = protocol['training']
    motion = checkpoint['configuration'].endswith('_motion_only')
    history = read_json(read, f'{stage}/{identifier}_history.json')
    epochs = config['motion_epochs' if motion else 'response_epochs']
    if ([h['epoch'] for h in history] != list(range(1, epochs + 1)) or
        any(len(h['mean_batch_loss_terms']) != (3 if motion else 4) for h in history) or
        not np.isfinite([h['mean_batch_loss_terms'] for h in history]).all()):
        raise ValueError('Fixed final-epoch budget/loss history differs')
    optimizer = checkpoint['optimizer_state']
    expected = [(n, p) for n, p in (model.core if motion else model.response_core).named_parameters()
                if not n.startswith(('response_head.' if motion else 'motion_head.'))]
    groups = optimizer['param_groups']
    ids = [i for g in groups for i in g['params']]
    if len(groups) != 1 or len(ids) != len(expected) or len(set(ids)) != len(expected) or set(ids) != set(optimizer['state']):
        raise ValueError('Frozen/response-only optimizer parameter budget differs')
    if groups[0]['lr'] != config['learning_rate'] or groups[0]['weight_decay'] != 0.:
        raise ValueError('Preregistered Adam configuration differs')
    steps = epochs * ((train_count + config['batch_calls'] - 1) // config['batch_calls'])
    for i, (_, parameter) in zip(ids, expected):
        state = optimizer['state'][i]
        if int(state['step'].item()) != steps:
            raise ValueError('Optimizer update count differs from fixed budget')
        if any(state[k].shape != parameter.shape or not torch.isfinite(state[k]).all() for k in ('exp_avg', 'exp_avg_sq')):
            raise ValueError('Optimizer moments shape/finite contract differs')
    # No randomness is consumed by these deterministic cores after initialization.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(checkpoint['seed'])
        LocalTwoHead('motion_only' if motion else 'plain', config['motion_scale_m'], config['response_scale_m'])
        expected_rng = torch.get_rng_state()
    if not torch.equal(checkpoint['torch_rng_state'], expected_rng):
        raise ValueError('Final Torch RNG differs from deterministic core initialization')
    sampler = np.random.default_rng(checkpoint['seed'])
    for _ in range(epochs):
        sampler.permutation(np.arange(train_count))
    if not tensor_tree_equal(checkpoint['sampler_rng_state'], sampler.bit_generator.state):
        raise ValueError('Final sampler RNG differs from fixed call-order budget')


def audit_cost_training(read, configs, restored, progress=False):
    calls, _ = audit_data(read, configs, restored)
    protocol = json.loads((ROOT / 'training_protocol.json').read_text(encoding='utf8'))
    validate_cost_training(protocol)
    config = protocol['training']
    indices = {split: [i for i, c in enumerate(calls) if c['record']['split'] == split]
               for split in ('train', 'development_validation')}
    all_indices = list(range(len(calls)))
    mean, scale = normalization(calls, indices['train'])
    weights, population = cost_population_weights(calls, indices['train'])
    serialized_population = json.loads(json.dumps({'population': population, 'weights': weights}))
    controls = control_metrics(calls, indices['development_validation'])
    contexts = prepare_contexts(calls, all_indices, read, configs)
    scores = {i: FrozenLocalScore(context, calls[i]['values']['proposed'][:, :, calls[i]['record']['agent']])
              for i, context in contexts.items()}
    diagnostic = json.loads((ROOT.parent / 'cwm_v20/diagnostic_protocol.json').read_text(encoding='utf8'))
    expected_population = {(name, seed) for name in protocol['models'] for seed in config['seeds']}
    initial_audits = {}
    for origin in ('gru', 'cv'):
        outputs = []
        for c in calls:
            v = c['values']
            path = v['backbone'] if origin == 'gru' else v['reference'][None] + .1 * np.arange(1, 9)[:, None] * v['velocity'][None]
            outputs.append({'reference': np.repeat(path[None], len(v['proposed']), 0)})
        _, initial_audits[origin] = make_cost_caches(calls, all_indices, outputs, contexts, scores, True)
    checkpoint_sets, reports = [], []
    for stage in ('primary', 'retrained'):
        report = read_json(read, stage + '/summary.json')
        if report['status'] != 'offline_fresh_deployed_cost_contrast_finished_not_promoted' or report['protocol'] != protocol:
            raise ValueError('Published deployed-cost protocol/status differs')
        if any(report[k] for k in ('enhanced_control_enabled', 'baseline_weights_included', 'holdout_used',
            'prior_gate_overridden', 'common_motion_in_response_optimizer', 'actual_unique_is_promotion_evidence')):
            raise ValueError('Optional offline original-only contract differs')
        audit_saved_sources(read, stage, report)
        for source in ('data', 'public'):
            if report[source + '_summary_sha256'] != hashlib.sha256(read(source + '/summary.json')).hexdigest():
                raise ValueError('Data/public training provenance differs')
        if (report['split_calls'] != {k: len(v) for k, v in indices.items()} or
            report['split_groups'] != {k: sorted({calls[i]['record']['group'] for i in v}) for k, v in indices.items()} or
            report['full_cost_calls'] != {k: sum(i in contexts for i in v) for k, v in indices.items()}):
            raise ValueError('Train/dev/complete cost population differs')
        if report['controls'] != controls:
            raise ValueError('Original GRU/UNCALIBRATED public CV controls differ')
        population_raw = read(stage + '/population_weights.json')
        if json.loads(population_raw) != serialized_population or hashlib.sha256(population_raw).hexdigest() != report['population_weights_sha256']:
            raise ValueError('Fixed train-only population weights differ')
        if not compare_arrays({'mean': mean, 'scale': scale}, arrays(read(stage + '/normalization.npz'))):
            raise ValueError('Train-only normalizer differs')
        for origin, recorded in initial_audits.items():
            if read_json(read, f'{stage}/initial_{origin}_score_audit.json') != recorded:
                raise ValueError('Initial original-engine cost/gradient audit differs')
        rows = report['models']
        if len(rows) != 18 or {(r['configuration'], r['seed']) for r in rows} != expected_population:
            raise ValueError('Fixed final18-model population differs')
        checkpoints, models = {}, {}
        for row in rows:
            name, seed, origin = row['configuration'], row['seed'], row['origin']
            identifier = f'{name}_seed{seed}'
            raw = read(stage + '/' + identifier + '.pt')
            if hashlib.sha256(raw).hexdigest() != row['checkpoint_sha256']:
                raise ValueError('Final checkpoint hash differs')
            cp = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            if cp['configuration'] != name or cp['seed'] != seed or cp['origin'] != origin or cp['protocol'] != protocol:
                raise ValueError('Checkpoint identity/protocol differs')
            if any(cp[k] for k in ('online_promoted', 'baseline_weights_included', 'mediator_optimized', 'common_motion_in_response_optimizer')):
                raise ValueError('Checkpoint optional frozen contract differs')
            if any(cp[k] != report[k] for k in ('source_hashes', 'data_summary_sha256', 'public_summary_sha256', 'population_weights_sha256')):
                raise ValueError('Checkpoint provenance differs')
            if not np.array_equal(cp['normalizer_mean'].numpy(), mean) or not np.array_equal(cp['normalizer_scale'].numpy(), scale):
                raise ValueError('Checkpoint normalizer differs')
            if cp['initial_state_digest'] != row['initial_state_digest'] or cp['deployed_cost_cache_sha256'] != hashlib.sha256(
                read(f'{stage}/{origin}_seed{seed}_deployed_cost_cache.npz')).hexdigest():
                raise ValueError('Checkpoint initialization/cache provenance differs')
            model = reconstruct_model(row, protocol)
            model.load_state_dict(cp['model_state'], strict=True)
            model.eval().requires_grad_(False)
            common = model if name.endswith('_motion_only') else model.common
            if state_digest(common) != row['common_motion_state_digest'] or state_digest(common) != cp['common_motion_state_digest']:
                raise ValueError('Frozen common origin digest differs')
            if row['parameters'] != sum(p.numel() for p in model.parameters()):
                raise ValueError('Matched model parameter budget differs')
            audit_optimizer_history(read, stage, identifier, cp, model, len(indices['train']), protocol)
            if name.endswith('_motion_only'):
                saved_stage = torch.load(io.BytesIO(read(f'{stage}/{origin}_seed{seed}_motion_stage.pt')),
                                         map_location='cpu', weights_only=True)
                if any(not tensor_tree_equal(saved_stage[k], cp[k]) for k in
                       ('model_state', 'optimizer_state', 'initial_state_digest', 'torch_rng_state',
                        'sampler_rng_state', 'origin', 'seed', 'protocol', 'source_hashes', 'online_promoted')):
                    raise ValueError('Completed pretraining-stage checkpoint differs from final motion control')
                if read(f'{stage}/{origin}_seed{seed}_motion_stage_history.json') != read(f'{stage}/{identifier}_history.json'):
                    raise ValueError('Completed pretraining-stage history differs from final control')
            checkpoints[identifier], models[identifier] = cp, model
        caches_by_origin_seed = {}
        for seed in config['seeds']:
            for origin in ('gru', 'cv'):
                common = models[f'{origin}_motion_only_seed{seed}']
                canonical = {}
                for chosen in indices.values():
                    _, outputs = evaluate_cost_model(common, calls, chosen, mean, scale, config['batch_calls'])
                    canonical.update(zip(chosen, outputs))
                caches, score_audit = make_cost_caches(calls, all_indices, [canonical[i] for i in all_indices], contexts, scores, True)
                if not compare_arrays(cache_arrays(caches), arrays(read(f'{stage}/{origin}_seed{seed}_deployed_cost_cache.npz'))):
                    raise ValueError('Independent deployed reference/scale/label/gradient cache differs')
                if score_audit != read_json(read, f'{stage}/{origin}_seed{seed}_deployed_score_audit.json'):
                    raise ValueError('Calibrated full original cost/piecewise gradient audit differs')
                caches_by_origin_seed[(origin, seed)] = caches
                for suffix in ('l2', 'cost_l2'):
                    model = models[f'{origin}_{suffix}_seed{seed}']
                    if not tensor_tree_equal(common.state_dict(), model.common.state_dict()):
                        raise ValueError('Response altered matched frozen calibrated common motion')
            response_cps = [checkpoints[f'{name}_seed{seed}'] for name in protocol['response_configurations']]
            if len({cp['initial_state_digest'] for cp in response_cps}) != 1 or any(not tensor_tree_equal(
                cp['sampler_rng_state'], response_cps[0]['sampler_rng_state']) for cp in response_cps):
                raise ValueError('Matched response initialization/order differs')
            if checkpoints[f'gru_motion_only_seed{seed}']['initial_state_digest'] != checkpoints[f'cv_motion_only_seed{seed}']['initial_state_digest']:
                raise ValueError('Matched independent motion initialization differs')
        decisions, actual_decisions = {}, {}
        for row in rows:
            name, seed, origin = row['configuration'], row['seed'], row['origin']
            identifier = f'{name}_seed{seed}'
            model, caches = models[identifier], caches_by_origin_seed[(origin, seed)]
            measured_diagnostics, split_decisions = {}, {}
            for split, chosen in indices.items():
                measured, outputs = evaluate_cost_model(model, calls, chosen, mean, scale, config['batch_calls'])
                if any(not np.array_equal(o['reference'], caches[i]['reference'].numpy())
                       for i, o in zip(chosen, outputs) if i in caches):
                    raise ValueError('Final forecast deployed reference differs from loss cache')
                measured.update(contrast_metrics(calls, chosen, outputs, caches))
                expected = {f'call{j}_{k}': v for j, o in enumerate(outputs) for k, v in o.items()}
                if measured != row['train' if split == 'train' else 'development'] or not compare_arrays(
                    expected, arrays(read(f'{stage}/{identifier}_{split}_predictions.npz'))):
                    raise ValueError('Independent train/dev forecast or metrics differ')
                measured_diagnostics[split] = decompose(calls, chosen, outputs, diagnostic)
                split_decisions[split] = cost_decision_records(calls, chosen, outputs, contexts)
                if split == 'development_validation':
                    decisions[identifier] = split_decisions[split]['geometry_union']
                    actual_decisions[identifier] = split_decisions[split]['actual_unique']
            if measured_diagnostics != read_json(read, f'{stage}/{identifier}_diagnostics.json'):
                raise ValueError('Independent response bins/offsets/weighting diagnostics differ')
            if split_decisions != read_json(read, f'{stage}/{identifier}_decisions.json'):
                raise ValueError('Independent complete actual/union original costs/choices/regrets differ')
            if progress:
                print(json.dumps({'stage': stage, 'model': identifier, 'status': 'fully_reinferred_and_recomputed'}), flush=True)
        selected, gates, factorial = qualify_cost_models(rows, controls, decisions, protocol)
        _, actual_gates, actual_factorial = qualify_cost_models(rows, controls, actual_decisions, protocol)
        if (selected != report['selected_median_seeds'] or gates != report['qualification'] or factorial != report['factorial_comparisons'] or
            actual_gates != report['actual_unique_diagnostic_qualification'] or actual_factorial != report['actual_unique_factorial_diagnostics'] or
            report['primary_research_eligible'] != gates[protocol['primary_configuration']]['research_eligible']):
            raise ValueError('Fixed primary selection/qualification/factorial diagnostics differ')
        checkpoint_sets.append(checkpoints)
        reports.append(report)
    for identifier, cp in checkpoint_sets[0].items():
        if not tensor_tree_equal(cp, checkpoint_sets[1][identifier]) or read(f'primary/{identifier}_history.json') != read(f'retrained/{identifier}_history.json'):
            raise ValueError('Independent full weights/optimizer/RNG/history differ')
    if reports[0] != reports[1]:
        raise ValueError('Independent complete research summaries differ')
    return {'status': 'passed_independent_fresh_deployed_cost_contrast_audit', 'models_per_run': 18, 'independent_runs': 2,
        'split_calls': reports[0]['split_calls'], 'full_cost_calls': reports[0]['full_cost_calls'],
        'all_train_development_forecasts_reinferred': True, 'weights_optimizer_rng_history_equal': True,
        'public252_frames_and_original_actual_union_costs_verified': True,
        'initial_and_calibrated_full_scores_piecewise_gradients_recomputed': True,
        'deployed_reference_scale_label_gradient_caches_recomputed': True,
        'matched_frozen_common_motion_equal': True, 'train_only_population_weights_verified': True,
        'all_bins_offsets_diagnostics_recomputed': True, 'selected_median_seeds': selected,
        'qualification': gates, 'factorial_comparisons': factorial,
        'primary_research_eligible': reports[0]['primary_research_eligible'],
        'enhanced_control_enabled': False, 'holdout_used': False, 'scope': protocol['later_required']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'public', 'primary', 'retrained', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--parts-output', type=Path)
    parser.add_argument('--restored-output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    capsule = ROOT.parent / 'cwm_v1/baseline/capsule.zip'
    if sha(capsule) != BASELINE_SHA:
        raise ValueError('Original baseline capsule changed')
    configs = frozen_configs(capsule, args.restored_output)
    if args.verify:
        artifact = resolve_artifact(args.verify, args.restored_output.parent / 'assembled')
        integrity = verify_manifest(artifact)
        with zipfile.ZipFile(artifact) as archive:
            result = audit_cost_training(archive.read, configs, args.restored_output, True)
            if read_json(archive.read, 'release_summary.json') != result:
                raise ValueError('Archived release summary differs')
        print(json.dumps({'status': result['status'], **integrity}), flush=True)
        return
    if any(getattr(args, n) is None for n in ('data', 'public', 'primary', 'retrained', 'output')):
        parser.error('Fresh data/public plus BOTH complete research runs required')
    roots = {n: getattr(args, n) for n in ('data', 'public', 'primary', 'retrained')}
    def read(name):
        stage, relative = name.split('/', 1)
        return (roots[stage] / relative).read_bytes()
    result = audit_cost_training(read, configs, args.restored_output, True)
    members = {}
    for stage, root in roots.items():
        for p in root.rglob('*'):
            relative = p.relative_to(root)
            if p.is_file() and not {'restored', '__pycache__'}.intersection(relative.parts):
                members[stage + '/' + relative.as_posix()] = p.read_bytes()
    members['release_summary.json'] = json.dumps(result, indent=2).encode('utf8')
    for p in ROOT.glob('*.py'):
        members['verification_source/' + p.name] = p.read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        archive.writestr('ARTIFACT_MANIFEST.json', json.dumps({n: hashlib.sha256(v).hexdigest() for n, v in members.items()}, indent=2))
    integrity = verify_manifest(args.output)
    with zipfile.ZipFile(args.output) as archive:
        recomputed = audit_cost_training(archive.read, configs, args.restored_output, True)
        if read_json(archive.read, 'release_summary.json') != recomputed or result != recomputed:
            raise ValueError('Archived evidence differs from prearchive audit')
    transport = None
    if args.parts_output:
        transport = split_archive(args.output, args.parts_output)
        joined = resolve_artifact(args.parts_output, args.restored_output.parent / 'assembled')
        if sha(joined) != integrity['sha256']:
            raise ValueError('Lossless published transport differs from audited archive')
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    for name in ('data', 'public', 'primary'):
        path = reports / ('training_summary.json' if name == 'primary' else name + '_summary.json')
        with path.open('xb') as file:
            file.write(read(name + '/summary.json'))
    with (reports / 'release_manifest.json').open('x', encoding='utf8') as file:
        json.dump({**result, 'artifact': integrity, 'transport': transport}, file, indent=2)
    print(json.dumps({'status': result['status'], **integrity}), flush=True)


if __name__ == '__main__':
    main()
