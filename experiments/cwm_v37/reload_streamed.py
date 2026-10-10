"""ALL two-run18model cold predictions, original costs and fixed gates replay.

No new training, seed/epoch switching, active policy or qualification override.
Original native data replay prerequisite remains separate and mandatory.
"""
import argparse
import itertools
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent/'cwm_v36'), str(HERE.parent/'cwm_v35'), str(HERE.parent/'cwm_v28')]
from contract import load_contract
from audited_stream import AuditedCorpus, population_weights, base_values, safe_file, sha
from stream_objective import public_batch
from checkpoint_reload import reconstruct
from evidence import summary, inventory, require_same_summary
from repeatability_identity import exact_tree
from train_ranking_response import (frozen_configs, save_json, BASELINE_SHA, verify, state_digest,
    restore_public_call, FrozenLocalScore, original_scores, contrast_cache, audit_deployed_score)
from ranking_loss import ranking_cache, ranking_kl
from stream_metrics import PredictionMetrics
from decision_gate import DecisionRows, qualify, library_contributions
from input_precision import audit_command_precision
from qualification_gate import preserved_cpu_rng

KEYS = ('reference', 'label_costs', 'temperature', 'logq', 'q')
IDENTITY = ('episode_index', 'step', 'ordinal', 'agent', 'group', 'variant')


def arrays(path):
    with np.load(path, allow_pickle=False) as archive: return {key: archive[key].copy() for key in archive.files}


def array_equal(actual, saved):
    return actual.dtype == saved.dtype and actual.shape == saved.shape and actual.tobytes() == saved.tobytes()


def require_arrays(actual, saved, label):
    if set(actual) != set(saved) or any(not array_equal(actual[k], saved[k]) for k in actual):
        raise ValueError('ALL cold-reloaded tensor bytes differ: '+label)


def native_context(corpus, call, configs):
    row = call['record']; path = safe_file(corpus.data, row['context_path'])
    if sha(path) != row['context_sha256']: raise ValueError('Audited public context changed')
    snapshot = json.loads(path.read_bytes())
    return restore_public_call(snapshot, *configs, call['values']['backbone'])


def independent_score(corpus, call, configs, reference):
    """New full original-engine invocation, not cached declared costs."""
    row, v = call['record'], base_values(call['values']); n = len(v['proposed'])
    context = native_context(corpus, call, configs)
    actions = v['proposed'][:, :, row['agent']]; score = FrozenLocalScore(context, actions)
    cached = ranking_cache(score, reference, v)
    truth = original_scores(context, actions, v['target'])
    native = corpus.cost_records[tuple(row[k] for k in IDENTITY[:4])]
    if (native['complete_indices'][:n] != list(range(n)) or
        not np.allclose(truth, native['true_costs'][:n], rtol=1e-10, atol=1e-8) or
        not np.allclose(truth, cached['label_costs'].numpy(), rtol=1e-10, atol=1e-8)):
        raise ValueError('Independent reloaded original true-cost labels differ')
    derivative = audit_deployed_score(context, v, reference, score, contrast_cache(score, reference, v))
    return cached, derivative, context


def independent_decisions(call, prediction, context):
    """Separate population/path/choice reconstruction using original engine."""
    row, v = call['record'], call['values']; count = int(v['base_candidate_count'])
    actions = v['proposed'][:count, :, row['agent']]
    cv = v['reference']+np.arange(1, 9)[:, None]*.1*v['velocity']
    paths = [('model', prediction['prediction'][:count]), ('own_motion_component', prediction['reference'][:count]),
        ('original_gru', np.broadcast_to(v['backbone'], (count, 8, 3))),
        ('constant_velocity', np.broadcast_to(cv, (count, 8, 3))), ('action_specific_truth', v['target'][:count])]
    costs = {name: original_scores(context, actions, path) for name, path in paths}
    actual = []; raw = []
    for candidate in v['actual_candidates']:
        matched = [i for i, action in enumerate(actions) if np.array_equal(action, candidate)]
        if len(matched) != 1: raise ValueError('Original actual candidate must map uniquely to BASE')
        raw.append(matched[0])
        if matched[0] not in actual: actual.append(matched[0])
    if not np.allclose(costs['original_gru'][raw], v['original_local_costs'], rtol=1e-10, atol=1e-8):
        raise ValueError('Independent original solver full costs differ')
    from local_shadow import rank_metrics
    result = {}
    for population, chosen in (('actual_unique', actual), ('public_geometry', v['public_geometry_indices'].tolist()),
                               ('bounded_shared', list(range(count)))):
        vectors = {key: vector[chosen].tolist() for key, vector in costs.items()}
        metrics = {key: rank_metrics(vector, vectors['action_specific_truth'], 1e-9) for key, vector in vectors.items()}
        anchor = row['actual_choice_pool_index']
        if anchor not in chosen or (population == 'actual_unique' and metrics['original_gru']['choice'] != chosen.index(anchor)):
            raise ValueError('Independent actual original anchor/selection differs')
        result[population] = {**{key: row[key] for key in IDENTITY}, 'population': population,
            'indices': chosen, 'costs': vectors, 'metrics': metrics,
            'original_anchor_realized_cost': float(costs['action_specific_truth'][anchor])}
    return result


def saved_rows(directory, identifier, corpus):
    base = directory/'predictions'/identifier
    manifest = json.loads((base/'decision_manifest.json').read_bytes())
    return {split: {population: DecisionRows(base/row['path'], row['sha256'], row['calls'])
        for population, row in populations.items()} for split, populations in manifest.items()}


def cold_controls(corpus):
    measures = {name: PredictionMetrics() for name in ('frozen_gru', 'constant_velocity')}
    for index in corpus.indices['development_validation']:
        call = corpus.load(index); v = call['values']; n = len(v['proposed']); gru = np.repeat(v['backbone'][None], n, 0)
        cv = v['reference'][None]+.1*np.arange(1, 9)[:, None]*v['velocity'][None]
        for name, target in (('frozen_gru', gru), ('constant_velocity', np.repeat(cv[None], n, 0))):
            measures[name].add(call['record'], v, {'prediction': target, 'motion': target-gru, 'response': np.zeros_like(gru)}, v['bounded_shared_indices'])
    groups = {corpus.records[i]['group'] for i in corpus.indices['development_validation']}
    return {name: measured.result(groups) for name, measured in measures.items()}


def replay_initial_audits(directory, corpus, configs):
    certificates = json.loads((directory/'initial_optimizer_prerequisites.json').read_bytes())
    if set(certificates) != {'gru', 'cv'}: raise ValueError('BOTH full original-reference audits required')
    for origin in ('gru', 'cv'):
        row = certificates[origin]; records = json.loads((directory/row['audit_path']).read_bytes())
        if sha(directory/row['audit_path']) != row['audit_sha256']: raise ValueError('Initial native cost audit changed')
        cursor = 0; supported = {}
        for index in range(len(corpus)):
            call = corpus.load(index); v = base_values(call['values'])
            if not (v['valid'].all() and v['anchor_valid'].all()): continue
            path = v['backbone'] if origin == 'gru' else v['reference'][None]+.1*np.arange(1, 9)[:, None]*v['velocity'][None]
            _, derivative, _ = independent_score(corpus, call, configs, np.repeat(path[None], len(v['proposed']), 0))
            if cursor >= len(records) or not exact_tree({'call_index': index, **derivative}, records[cursor]):
                raise ValueError('ALL independently recomputed initial original-cost gradients differ')
            cursor += 1; supported.setdefault(call['record']['group'], set()).add(call['record']['variant'])
        expected_axes = {v['name'] for v in corpus.protocol['variants']}
        population = {'complete_calls': cursor, 'groups': len(supported), 'family_axes': {g: sorted(v) for g, v in sorted(supported.items())}}
        if cursor != len(records) or len(supported) != 64 or any(axes != expected_axes for axes in supported.values()) or any(not exact_tree(row[k], v) for k, v in population.items()):
            raise ValueError('Complete initial score ALL-family/axis population differs')


def replay_model(model, directory, identifier, corpus, configs, mean, scale):
    base = directory/'predictions'/identifier
    manifest = json.loads((base/'manifest.json').read_bytes())
    decision_rows = saved_rows(directory, identifier, corpus)
    cursors = {s: {p: iter(rows) for p, rows in populations.items()} for s, populations in decision_rows.items()}
    metrics = {s: PredictionMetrics() for s in corpus.indices}; augmented = {s: PredictionMetrics() for s in corpus.indices}
    axes = {s: {a['name']: PredictionMetrics() for a in corpus.protocol['variants']} for s in corpus.indices}
    seed = int(identifier.rsplit('seed', 1)[1]); cache_root = directory/f'cache{seed}'
    cache_manifest = json.loads((cache_root/'manifest.json').read_bytes())
    derivative_rows = iter(json.loads((cache_root/'score_audit.json').read_bytes())); complete_ids = []
    for start in range(0, len(corpus), 16):
        ids = list(range(start, min(start+16, len(corpus)))); calls = corpus.batch(ids); inputs, slices = public_batch(calls, mean, scale)
        with torch.no_grad():
            tensors = [value.numpy() for value in model(*inputs)]
            common = model.common if hasattr(model, 'common') else model
            reference = common(*inputs)[0].numpy() if hasattr(model, 'common') else tensors[0]
        for index, call, (a, b) in zip(ids, calls, slices):
            output = {key: value[a:b].copy() for key, value in zip(('prediction', 'motion', 'response', 'reference'), [*tensors, reference])}
            path = base/manifest[str(index)]['path']
            if sha(path) != manifest[str(index)]['sha256']: raise ValueError('Final prediction file changed during replay')
            require_arrays(output, arrays(path), identifier+'/'+str(index))
            cached_row = cache_manifest['files'][str(index)]; cached_path = cache_root/cached_row['path']
            if sha(cached_path) != cached_row['sha256']: raise ValueError('Canonical reference cache changed')
            saved = arrays(cached_path)
            if not array_equal(saved['reference_augmented'], output['reference']): raise ValueError('Cold common canonical ALL-call reference differs')
            v = base_values(call['values']); complete = bool(v['valid'].all() and v['anchor_valid'].all()); kl = None
            if complete != cached_row['complete_base']: raise ValueError('ALL native complete BASE cost mask differs')
            if complete:
                complete_ids.append(index)
                cache, derivative, context = independent_score(corpus, call, configs, output['reference'][:len(v['proposed'])])
                if not exact_tree(next(derivative_rows, None), {'call_index': index, **derivative}):
                    raise ValueError('ALL independently reexecuted deployed cost/derivative rows differ')
                require_arrays({k: cache[k].numpy() for k in KEYS}, {k: saved[k] for k in KEYS}, 'BASE ranking cache')
                with torch.no_grad(): kl = float(ranking_kl(cache, torch.from_numpy(output['response'][:len(v['proposed'])])))
                rows = independent_decisions(call, output, context)
                for population, actual in rows.items():
                    expected = next(cursors[call['record']['split']][population], None)
                    if expected is None or not exact_tree(actual, expected): raise ValueError('ALL reexecuted original-cost/choice JSONL rows differ')
            else:
                if set(saved) != {'reference_augmented'}: raise ValueError('Incomplete BASE call carries forbidden ranking arrays')
            row = call['record']; values = call['values']; split = row['split']
            metrics[split].add(row, values, output, values['bounded_shared_indices'], kl)
            augmented[split].add(row, values, output, np.arange(len(values['proposed'])))
            axes[split][row['variant']].add(row, values, output, values['bounded_shared_indices'], kl)
        if start % 512 == 0: print(json.dumps({'status': 'cold_all_call_prediction_original_cost_replay', 'model': identifier, 'calls': start+len(ids)}), flush=True)
    for populations in cursors.values():
        if any(next(cursor, None) is not None for cursor in populations.values()): raise ValueError('Saved original decisions have omitted/extra native rows')
    if next(derivative_rows, None) is not None: raise ValueError('Extra/omitted deployed derivative evidence')
    families = {}
    for i in complete_ids:
        row = corpus.records[i]; families.setdefault(row['group'], set()).add(row['variant'])
    declared = {'complete_calls': len(complete_ids), 'groups': len(families),
        'family_axes': {g: sorted(v) for g, v in sorted(families.items())}}
    if any(not exact_tree(cache_manifest[k], v) for k, v in declared.items()):
        raise ValueError('Whole independently recomputed deployed-cost population differs')
    measured = {}
    for split in corpus.indices:
        groups = {corpus.records[i]['group'] for i in corpus.indices[split]}
        measured[split] = metrics[split].result(groups, require_ranking=True)
        measured[split]['augmented_prediction_diagnostic_not_selection'] = augmented[split].result(groups)
        measured[split]['by_geometry_axis'] = {axis: metric.result(groups, require_ranking=True) for axis, metric in axes[split].items()}
    return measured, decision_rows


def complete_history(path):
    result = json.loads(path.read_bytes())
    if len(result) != 80 or [r['epoch'] for r in result] != list(range(1, 81)) or not np.isfinite([r['mean_batch_loss_terms'] for r in result]).all():
        raise ValueError('ALL fixed80 finite epoch history rows required')
    return result


def replay_run(directory, report, corpus, configs, output):
    contract, plan = load_contract(); mean, scale, frames = corpus.normalization()
    saved_norm = arrays(directory/'normalization.npz')
    require_arrays({'mean': mean, 'scale': scale, 'train_frames': np.asarray(frames, dtype=np.int64)}, saved_norm, 'TRAIN normalizer')
    weights, population = population_weights(corpus)
    if not exact_tree(json.loads((directory/'population_weights.json').read_bytes()), json.loads(json.dumps({'population': population, 'weights': weights}))):
        raise ValueError('Fixed full TRAIN physical/BASE cost weights differ')
    if not exact_tree(audit_command_precision(corpus), json.loads((directory/'public_input_precision_audit.json').read_bytes())):
        raise ValueError('ALL original public float32 command identities differ')
    controls = cold_controls(corpus)
    if not exact_tree(controls, report['controls']): raise ValueError('Cold public GRU/CV control metrics differ')
    replay_initial_audits(directory, corpus, configs)
    stage_checks = {}; stages = {}
    for seed in plan['training']['seeds']:
        path = directory/f'cv_seed{seed}_motion_stage.pt'; checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['source_hashes'] != report['source_hashes']: raise ValueError('Motion-stage used source identity differs')
        model, checked = reconstruct(checkpoint, 'cv_motion_only', seed, corpus.indices['train'], stage=True)
        complete_history(directory/f'cv_seed{seed}_motion_stage_history.json')
        stages[seed] = checkpoint; stage_checks[str(seed)] = checked
    decisions, rows, contributions = {}, [], {}
    for row in report['models']:
        name, seed = row['configuration'], row['seed']; identifier = f'{name}_seed{seed}'; path = directory/(identifier+'.pt')
        if sha(path) != row['checkpoint_sha256']: raise ValueError('Individual final checkpoint identity differs')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        cache_digest = sha(directory/f'cache{seed}/manifest.json')
        for key, expected in (('source_hashes', report['source_hashes']), ('data_summary_sha256', report['data_summary_sha256']),
            ('data_audit_summary_sha256', report['data_audit_summary_sha256']), ('population_weights_sha256', sha(directory/'population_weights.json')),
            ('canonical_cache_manifest_sha256', cache_digest)):
            if not exact_tree(checkpoint[key], expected): raise ValueError('Final model provenance differs: '+key)
        if not array_equal(checkpoint['normalizer_mean'].numpy(), mean) or not array_equal(checkpoint['normalizer_scale'].numpy(), scale):
            raise ValueError('Checkpoint TRAIN-only normalizer differs')
        model, checked = reconstruct(checkpoint, name, seed, corpus.indices['train'])
        common = model.common if hasattr(model, 'common') else model
        if not exact_tree(common.state_dict(), stages[seed]['model_state']): raise ValueError('ALL final models must share frozen per-seed motion stage')
        if name == 'cv_motion_only':
            for key in ('model_state', 'optimizer_state', 'torch_rng_state', 'sampler_rng_state', 'initial_state_digest'):
                if not exact_tree(checkpoint[key], stages[seed][key]): raise ValueError('Final motion/stage complete training state differs')
        history = complete_history(directory/f'{identifier}_history.json')
        if name == 'cv_motion_only' and history != complete_history(directory/f'cv_seed{seed}_motion_stage_history.json'):
            raise ValueError('Final motion/stage80epoch histories differ')
        if row['parameters'] != sum(p.numel() for p in model.parameters()) or row['initial_state_digest'] != checkpoint['initial_state_digest'] or row['common_motion_state_digest'] != state_digest(common):
            raise ValueError('Final parameter/initialization/frozen common summary differs')
        measured, records = replay_model(model, directory, identifier, corpus, configs, mean, scale)
        if not exact_tree(measured['train'], row['train']) or not exact_tree(measured['development_validation'], row['development']):
            raise ValueError('ALL cold predictive/group/axis/ranking metrics differ')
        contribution = {s: library_contributions(records[s], contract, {corpus.records[i]['group'] for i in corpus.indices[s]}) for s in corpus.indices}
        if not exact_tree(contribution, json.loads((directory/f'{identifier}_library_contributions.json').read_bytes())):
            raise ValueError('Independent original-cost library contributions differ')
        decisions[identifier] = records['development_validation']['bounded_shared']; rows.append(row); contributions[identifier] = contribution
        if {s: records[s]['bounded_shared'].count for s in corpus.indices} != report['decision_calls'][identifier]:
            raise ValueError('ALL complete native decision counts differ')
        save_json(output/(identifier+'.json'), checked)
    selected, gates = qualify(rows, controls, decisions, contract, plan, {corpus.records[i]['group'] for i in corpus.indices['development_validation']})
    if (not exact_tree(selected, report['selected_median_seeds']) or not exact_tree(gates, report['qualification']) or
        not exact_tree(contributions, report['library_contributions']) or report['primary_research_eligible'] is not gates[contract['primary_configuration']]['research_eligible']):
        raise ValueError('ALL fixed ADE-median development gains/gates/contributions differ')
    return {'models': 9, 'motion_stages': stage_checks, 'all_predictions_original_costs_gates_reexecuted': True,
        'primary_research_eligible': report['primary_research_eligible']}


def replay(data, audit, primary, repeated, output):
    if output.exists(): raise ValueError('Exclusive NEW complete independent reload output required')
    source_files = sorted([*HERE.glob('*.py')])
    reload_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files}
    corpus = AuditedCorpus(data, audit)
    directories = [primary, repeated]; reports = [summary(d, corpus) for d in directories]
    manifests = [inventory(d, r, corpus) for d, r in zip(directories, reports)]
    require_same_summary(*reports)
    # ALL state contents, not only the checkpoint serialization SHA.
    checkpoints = [f"{r['configuration']}_seed{r['seed']}" for r in reports[0]['models']]
    checkpoints += [f'cv_seed{s}_motion_stage' for s in reports[0]['v34_plan']['training']['seeds']]
    equal_serializations = 0
    for name in checkpoints:
        paths = [d/(name+'.pt') for d in directories]
        if not exact_tree(*[torch.load(p, map_location='cpu', weights_only=True) for p in paths]):
            raise ValueError('Two full runs differ in model/optimizer/RNG/source contents: '+name)
        if (primary/(name+'_history.json')).read_bytes() != (repeated/(name+'_history.json')).read_bytes():
            raise ValueError('Two full fixed80epoch history bytes differ: '+name)
        equal_serializations += sha(paths[0]) == sha(paths[1])
    # Only checkpoint serialization and summary wrappers are allowed to differ.
    excluded = {'summary.json', 'preregistration.json', *[name+'.pt' for name in checkpoints]}
    if set(manifests[0]) != set(manifests[1]) or any(manifests[0][name] != manifests[1][name] for name in manifests[0] if name not in excluded):
        raise ValueError('Two full-run source/cache/prediction/decision/audit evidence bytes differ')
    output.mkdir(parents=True, exist_ok=False)
    capsule = HERE.parent/'cwm_v1/baseline/capsule.zip'
    if sha(capsule) != BASELINE_SHA: raise ValueError('Original capsule changed')
    with zipfile.ZipFile(capsule) as archive: native_manifest = json.loads(archive.read('CAPSULE_MANIFEST.json'))
    configs = frozen_configs(capsule, output/'r')
    results = []
    for label, directory, report in zip(('primary', 'repeated'), directories, reports):
        destination = output/label; destination.mkdir()
        results.append(replay_run(directory, report, corpus, configs, destination))
    corpus.reverify(); verify(output/'r', native_manifest)
    for directory, report, before in zip(directories, reports, manifests):
        if not exact_tree(summary(directory, corpus), report) or inventory(directory, report, corpus) != before:
            raise ValueError('Complete declared run evidence changed during independent ALL-model replay')
    if sha(capsule) != BASELINE_SHA: raise ValueError('Original capsule changed during full replay')
    if any(sha(ROOT/name) != digest for name, digest in reload_hashes.items()):
        raise ValueError('Actual reload sources changed during full independent replay')
    for name in reload_hashes:
        path = output/'source'/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes((ROOT/name).read_bytes())
    result = {'status': 'two_complete_streamed_runs_all18models_cold_reloaded_original_costs_verified',
        'reload_source_hashes': reload_hashes,
        'data_summary_sha256': sha(data/'summary.json'), 'data_audit_summary_sha256': sha(audit/'summary.json'),
        'primary_summary_sha256': sha(primary/'summary.json'), 'repeated_summary_sha256': sha(repeated/'summary.json'),
        'models_per_run': 9, 'complete_runs': 2, 'all12_checkpoint_contents_and_history_equal': True,
        'identical_serialized_checkpoint_pairs': equal_serializations, 'different_serialized_checkpoint_pairs': 12-equal_serializations,
        'all_predictions_original_costs_gates_reexecuted': True, 'all_optimizer_sampler_torch_rng_actually_restored': True,
        'all_declared_native_evidence_unchanged': True, 'runs': results,
        'primary_research_eligible': reports[0]['primary_research_eligible'], 'qualification': reports[0]['qualification'],
        'enhanced_control_enabled': False, 'holdout_used': False, 'new_training_performed_by_reload': False,
        'artifact_packaged_and_replayed': False,
        'scope': 'Corresponding Git checkout/history + original Python runtime and full audited V34 data. ALL18 final models/caches/original costs/gates reexecuted, no active control, holdout, original8Level/newscene capture/safety/latency or archive qualification.'}
    save_json(output/'summary.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'audit', 'primary', 'repeated', 'output'): parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    report = replay(args.data, args.audit, args.primary, args.repeated, args.output)
    print(json.dumps({'status': report['status'], 'primary_research_eligible': report['primary_research_eligible']}), flush=True)
