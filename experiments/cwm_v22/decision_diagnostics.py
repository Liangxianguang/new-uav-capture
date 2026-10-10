"""Posthoc V21 truth-replacement diagnosis using unchanged full original costs."""
import argparse
import collections
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol(protocol):
    if protocol != json.loads((ROOT / 'diagnostic_protocol.json').read_text(encoding='utf8')):
        raise ValueError('Pinned diagnostic protocol differs')
    if any(protocol[key] for key in ('enhanced_control_enabled', 'new_model_trained',
                                    'holdout_used', 'prior_gates_overridden')):
        raise ValueError('Diagnosis cannot train, promote or access holdout')
    if (protocol['models'] != ['motion_only', 'raw_call_mse', 'raw_point_mse', 'raw_call_l2', 'raw_point_l2']
            or protocol['seeds'] != [990101, 990102, 990103]
            or protocol['splits'] != ['train', 'development_validation']
            or protocol['candidate_populations'] != ['actual_unique', 'geometry_union']
            or not protocol['full_v21_audit_required_before_diagnosis']):
        raise ValueError('Fixed diagnosis population or full audit requirement differs')


def repair_contributions(r00, r10, r01, r11):
    values = np.asarray([r00, r10, r01, r11], dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or r11 > 1e-9:
        raise ValueError('Finite nonnegative regrets with zero full-truth regret required')
    motion = .5 * ((r00 - r10) + (r01 - r11))
    response = .5 * ((r00 - r01) + (r10 - r11))
    if not np.isclose(motion + response, r00 - r11, rtol=0, atol=1e-10):
        raise ValueError('Order-averaged repair contribution identity failed')
    return {'motion_repair': float(motion), 'response_repair': float(response),
            'total_repair': float(r00 - r11)}


def diagnostic_paths(values, output):
    truth = np.asarray(values['target'], dtype=np.float64)
    n = len(truth)
    if truth.shape != (n, 8, 3) or not n:
        raise ValueError('Fixed horizon candidate truth shape required')
    if not (values['valid'].all() and values['anchor_valid'].all()):
        raise ValueError('Complete-union decision support required')
    reference = np.repeat(np.asarray(values['anchor_target'], dtype=np.float64)[None], n, 0)
    backbone = np.repeat(np.asarray(values['backbone'], dtype=np.float64)[None], n, 0)
    prediction, motion, response = [np.asarray(output[k], dtype=np.float64)
                                    for k in ('prediction', 'motion', 'response')]
    if any(v.shape != truth.shape or not np.isfinite(v).all()
           for v in (truth, reference, backbone, prediction, motion, response)):
        raise ValueError('Finite complete candidate paths required')
    if not np.array_equal(motion, np.repeat(motion[:1], n, 0)):
        raise ValueError('Common motion must be action independent')
    equal = (values['proposed'] == values['anchor'][None]).all((1, 2, 3))
    if equal.sum() != 1 or not np.array_equal(response[equal], np.zeros_like(response[equal])):
        raise ValueError('Unique exact anchor and exact-zero response required')
    own_motion = backbone + motion
    if not np.allclose(prediction, own_motion + response, rtol=0, atol=1e-10):
        raise ValueError('Prediction differs from frozen common plus response')
    cv = values['reference'][None] + .1 * np.arange(1, 9)[:, None] * values['velocity'][None]
    cv = np.repeat(cv[None], n, 0)
    delta = truth - reference
    return {'model': prediction, 'own_motion': own_motion, 'fixed_reference_truth': reference,
            'action_specific_truth': truth, 'motion_plus_exact_response': own_motion + delta,
            'reference_truth_plus_learned_response': reference + response,
            'constant_velocity': cv, 'cv_plus_learned_response': cv + response,
            'cv_plus_exact_response': cv + delta, 'original_gru': backbone}


def local_diagnostics(costs, metrics, anchor_index, true_response_norm, tolerance):
    names = list(costs)
    arrays = {k: np.asarray(costs[k], dtype=np.float64) for k in names}
    truth = arrays['action_specific_truth']
    if (truth.ndim != 1 or not len(truth) or not 0 <= anchor_index < len(truth)
            or any(v.shape != truth.shape or not np.isfinite(v).all() for v in arrays.values())):
        raise ValueError('Finite equal-length complete cost vectors required')
    magnitude = np.asarray(true_response_norm, dtype=np.float64)
    if magnitude.shape != truth.shape or not np.isfinite(magnitude).all() or (magnitude < 0).any():
        raise ValueError('Candidate diagnostic response magnitudes differ')
    for name, values in arrays.items():
        choice = int(np.argmin(values))
        if metrics[name]['choice'] != choice or not np.isclose(
                metrics[name]['regret_in_diagnostic_cost'], truth[choice] - truth.min(), rtol=0, atol=1e-10):
            raise ValueError('Cost/choice/regret record inconsistent')
    regret = lambda key: metrics[key]['regret_in_diagnostic_cost']
    choice = metrics['model']['choice']
    own_choice = metrics['own_motion']['choice']
    gain = regret('own_motion') - regret('model')
    optimal = int(np.argmin(truth))
    centered_truth = truth - truth[anchor_index]
    centered_model = arrays['model'] - arrays['model'][anchor_index]
    nonanchor = np.arange(len(truth)) != anchor_index
    predicted_effect = arrays['model'] - arrays['own_motion']
    true_effect = truth - arrays['fixed_reference_truth']
    return {
        'repairs': repair_contributions(regret('model'), regret('reference_truth_plus_learned_response'),
                                       regret('motion_plus_exact_response'), regret('action_specific_truth')),
        'gain_vs_own_motion': float(gain),
        'choice_changed_vs_own_motion': bool(choice != own_choice),
        'helps_vs_own_motion': bool(gain > tolerance),
        'harms_vs_own_motion': bool(gain < -tolerance),
        'same_regret_vs_own_motion': bool(abs(gain) <= tolerance),
        'predicted_best_two_margin': float(np.diff(np.sort(arrays['model'])[:2])[0]) if len(truth) > 1 else None,
        'true_best_two_margin': float(np.diff(np.sort(truth)[:2])[0]) if len(truth) > 1 else None,
        'centered_model_cost_mae': float(np.abs(centered_model - centered_truth).mean()),
        'response_cost_contrast_mae_nonanchor': float(np.abs(predicted_effect - true_effect)[nonanchor].mean()) if nonanchor.any() else None,
        'chosen_true_response_mean_m': float(magnitude[choice]),
        'optimal_true_response_mean_m': float(magnitude[optimal]),
        'own_motion_choice_true_response_mean_m': float(magnitude[own_choice]),
        'truth_optimal_pool_choice': optimal
    }


def summarize(records, protocol):
    if not records:
        raise ValueError('Empty diagnostic decision support')
    groups = sorted({r['group'] for r in records})
    def grouped(getter):
        values = {g: [getter(r) for r in records if r['group'] == g] for g in groups}
        supported = {g: [v for v in vs if v is not None] for g, vs in values.items()}
        supported = {g: vs for g, vs in supported.items() if vs}
        return {'calls_with_support': sum(map(len, supported.values())), 'groups_with_support': len(supported),
                'group_equal_mean': float(np.mean([np.mean(vs) for vs in supported.values()])) if supported else None}
    paths = {name: {'regret': grouped(lambda r, name=name: r['metrics'][name]['regret_in_diagnostic_cost']),
                    'tie_optimal_fraction': grouped(lambda r, name=name: float(r['metrics'][name]['realized_tie_optimal']))}
             for name in protocol['paths']}
    keys = ['gain_vs_own_motion', 'choice_changed_vs_own_motion', 'helps_vs_own_motion', 'harms_vs_own_motion',
            'same_regret_vs_own_motion', 'predicted_best_two_margin', 'true_best_two_margin', 'centered_model_cost_mae',
            'response_cost_contrast_mae_nonanchor', 'chosen_true_response_mean_m', 'optimal_true_response_mean_m',
            'own_motion_choice_true_response_mean_m']
    details = {key: grouped(lambda r, key=key: r['diagnostics'][key]) for key in keys}
    repairs = {key: grouped(lambda r, key=key: r['diagnostics']['repairs'][key])
               for key in ('motion_repair', 'response_repair', 'total_repair')}
    by_group = {g: {'calls': sum(r['group'] == g for r in records),
                    'model_regret': float(np.mean([r['metrics']['model']['regret_in_diagnostic_cost'] for r in records if r['group'] == g]))}
                for g in groups}
    return {'calls': len(records), 'groups': len(groups), 'paths': paths,
            'diagnostics': details, 'repairs': repairs, 'by_group': by_group}


def paired_summary(changed, reference, protocol):
    identity = lambda rows: [tuple(r[k] for k in ('episode_index', 'step', 'agent', 'group', 'population')) for r in rows]
    if not changed or identity(changed) != identity(reference):
        raise ValueError('Paired diagnosis support differs')
    # Pure fixed group bootstrap, no new hypothesis test or promotion gate.
    differences = collections.defaultdict(list)
    for new, old in zip(changed, reference):
        differences[new['group']].append(old['metrics']['model']['regret_in_diagnostic_cost'] - new['metrics']['model']['regret_in_diagnostic_cost'])
    means = np.asarray([np.mean(v) for _, v in sorted(differences.items())])
    rng = np.random.default_rng(protocol['bootstrap']['seed'])
    boot = means[rng.integers(0, len(means), size=(protocol['bootstrap']['draws'], len(means)))].mean(1)
    return {'groups': len(means), 'group_equal_regret_gain': float(means.mean()),
            'percentile_95_interval': np.quantile(boot, [.025, .975]).tolist(),
            'interpretation': protocol['bootstrap']['interpretation']}


def run(artifact_path, output):
    sys.path.insert(0, str(ROOT.parent / 'cwm_v21'))
    import torch
    from loss_training_release import audit, audit_data, frozen_configs, BASELINE_SHA, verify_manifest, resolve_artifact, arrays
    from train_loss_factorial import prepare_contexts
    from task_score import original_scores
    from support_audit import map_actual
    from local_shadow import rank_metrics
    protocol_path = ROOT / 'diagnostic_protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf8'))
    validate_protocol(protocol)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    artifact = resolve_artifact(artifact_path, ROOT.parent.parent / 'results/cwm_v21/assembled')
    if sha(artifact) != protocol['source_artifact_sha256']:
        raise ValueError('Not the pinned V21 full archive')
    capsule = ROOT.parent / 'cwm_v1/baseline/capsule.zip'
    if sha(capsule) != protocol['baseline_capsule_sha256'] or BASELINE_SHA != protocol['baseline_capsule_sha256']:
        raise ValueError('Frozen original capsule changed')
    output.mkdir(parents=True, exist_ok=False)
    restored = output / 'restored'
    configs = frozen_configs(capsule, restored)
    integrity = verify_manifest(artifact)
    with zipfile.ZipFile(artifact) as archive:
        if hashlib.sha256(archive.read('primary/summary.json')).hexdigest() != protocol['source_training_summary_sha256']:
            raise ValueError('Published V21 training result changed')
        verification = audit(archive.read, configs, restored)
        if json.loads(archive.read('release_summary.json')) != verification:
            raise ValueError('Published V21 complete audit differs')
        print(json.dumps({'status': 'full_pinned_v21_audit_passed'}), flush=True)
        calls, _ = audit_data(archive.read, configs, restored)
        training = json.loads(archive.read('primary/summary.json'))
        if (training['protocol']['models'] != protocol['models']
                or training['protocol']['training']['seeds'] != protocol['seeds']):
            raise ValueError('All final models and seeds required')
        indices = {split: [i for i, c in enumerate(calls) if c['record']['split'] == split] for split in protocol['splits']}
        contexts = {split: prepare_contexts(calls, chosen, archive.read, configs) for split, chosen in indices.items()}
        if {k: len(v) for k, v in contexts.items()} != {'train': 833, 'development_validation': 272}:
            raise ValueError('Same complete-union decision support required')
        record_sets, results = {}, {}
        for split, chosen in indices.items():
            score_cache = collections.defaultdict(list)
            # Contexts and source predictions are pinned; no new optimization.
            for row in training['models']:
                identifier = f"{row['configuration']}_seed{row['seed']}"
                saved = arrays(archive.read(f'primary/{identifier}_{split}_predictions.npz'))
                records = {population: [] for population in protocol['candidate_populations']}
                prior = json.loads(archive.read(f'primary/{identifier}_decisions.json')) if split == 'development_validation' else None
                for j, i in enumerate(chosen):
                    if i not in contexts[split]:
                        continue
                    call, context = calls[i], contexts[split][i]
                    record, values = call['record'], call['values']
                    output_values = {key: saved[f'call{j}_{key}'] for key in ('prediction', 'motion', 'response')}
                    paths = diagnostic_paths(values, output_values)
                    if set(paths) != set(protocol['paths']):
                        raise ValueError('Fixed diagnostic path population differs')
                    actions = values['proposed'][:, :, record['agent']]
                    costs, cached = {}, score_cache[i]
                    for name, path in paths.items():
                        old = next((cost for previous, cost in cached if np.array_equal(previous, path)), None)
                        cost = original_scores(context, actions, path) if old is None else old.copy()
                        if old is None:
                            cached.append((path, cost))
                        costs[name] = cost
                    # Reuse exact union costs for subsets; same original engine/action paths.
                    actual, duplicates = map_actual(values, record['agent'])
                    raw_actual_indices = [int(np.flatnonzero((actions == a[None]).all((1, 2)))[0])
                                          for a in values['actual_candidates']]
                    if not np.allclose(costs['original_gru'][raw_actual_indices], values['original_local_costs'],
                                       rtol=1e-10, atol=1e-8):
                        raise ValueError('Actual original copied solver costs differ')
                    for population, selected in (('actual_unique', actual), ('geometry_union', np.arange(len(actions)))):
                        subset = {name: c[selected].tolist() for name, c in costs.items()}
                        metrics = {name: rank_metrics(c, subset['action_specific_truth'], protocol['tie_absolute_cost_tolerance'])
                                   for name, c in subset.items()}
                        anchor_locations = np.flatnonzero(selected == record['actual_choice_pool_index'])
                        if len(anchor_locations) != 1:
                            raise ValueError('Actual source anchor missing or duplicated')
                        if population == 'actual_unique' and metrics['original_gru']['choice'] != int(anchor_locations[0]):
                            raise ValueError('Actual original solver selection changed')
                        magnitude = np.linalg.norm(values['target'][selected] - values['anchor_target'][None], axis=-1).mean(1)
                        diagnostics = local_diagnostics(subset, metrics, int(anchor_locations[0]), magnitude,
                                                        protocol['tie_absolute_cost_tolerance'])
                        records[population].append({**{k: record[k] for k in ('episode_index', 'step', 'agent', 'group', 'split')},
                                                    'population': population, 'candidate_indices': selected.tolist(),
                                                    'actual_duplicate_candidates': duplicates, 'costs': subset,
                                                    'metrics': metrics, 'diagnostics': diagnostics})
                if prior is not None:
                    union = records['geometry_union']
                    if len(prior) != len(union):
                        raise ValueError('Published V21 union decision support differs')
                    for new, old in zip(union, prior):
                        if any(new[k] != old[k] for k in ('episode_index', 'step', 'agent', 'group')):
                            raise ValueError('Published V21 decision identity differs')
                        for old_key, new_key in (('model', 'model'), ('own_motion_component', 'own_motion'),
                                                ('constant_velocity', 'constant_velocity'), ('original_gru', 'original_gru'),
                                                ('action_specific_truth', 'action_specific_truth')):
                            if old['costs'][old_key] != new['costs'][new_key] or old['metrics'][old_key] != new['metrics'][new_key]:
                                raise ValueError('Published V21 full original costs/decisions changed')
                for population, rows in records.items():
                    key = (split, population, identifier)
                    record_sets[key] = rows
                    with (output / f'{identifier}_{split}_{population}_decisions.json').open('x', encoding='utf8', newline='\n') as stream:
                        json.dump(rows, stream, indent=2, allow_nan=False)
                    results.setdefault(identifier, {}).setdefault(split, {})[population] = summarize(rows, protocol)
                print(json.dumps({'status': 'full_cost_truth_repairs_diagnosed', 'model': identifier, 'split': split,
                                  'calls': len(records['geometry_union'])}), flush=True)
        paired = {}
        for split in protocol['splits']:
            paired[split] = {}
            for population in protocol['candidate_populations']:
                pairs = {}
                for changed, control in training['protocol']['factorial_comparisons']:
                    pairs[f'{changed}_vs_{control}'] = {str(seed): paired_summary(
                        record_sets[(split, population, f'{changed}_seed{seed}')],
                        record_sets[(split, population, f'{control}_seed{seed}')], protocol) for seed in protocol['seeds']}
                paired[split][population] = pairs
    sources = {p.name: sha(p) for p in (ROOT / 'decision_diagnostics.py', protocol_path)}
    report = {'status': 'posthoc_v21_decision_failure_diagnosis_not_new_validation', 'protocol': protocol,
              'source_artifact': integrity, 'source_hashes': sources, 'models': results,
              'paired_factorial_decision_gains': paired, 'prior_qualification': training['qualification'],
              'published_v21_full_audit_passed': True, 'published_v21_dev_decisions_exactly_reproduced': True,
              'complete_union_support': {k: len(v) for k, v in contexts.items()},
              'enhanced_control_enabled': False, 'new_model_trained': False, 'holdout_used': False,
              'prior_gates_overridden': False}
    with (output / 'summary.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({'status': report['status'], 'summary_sha256': sha(output / 'summary.json')}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    protocol = json.loads((ROOT / 'diagnostic_protocol.json').read_text(encoding='utf8'))
    parser.add_argument('--artifact', type=Path, default=ROOT.parent.parent / protocol['source_artifact'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact, args.output)


if __name__ == '__main__':
    main()
