"""Descriptive, gate-compatible V19 response failure decomposition; no training."""
import argparse
import collections
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def response_support(values):
    equal = (values['proposed'] == values['anchor'][None]).all((1, 2, 3))
    return values['valid'] & values['anchor_valid'][None] & ~equal[:, None]


def group_mean(values, groups, selection):
    selected = np.unique(groups[selection])
    if not len(selected):
        return None
    return float(np.mean([values[selection & (groups == g)].mean() for g in selected]))


def conditional_summary(points, selection, threshold):
    groups = points['group']
    population = np.unique(groups)
    count = int(selection.sum())
    result = {'points': count, 'groups_with_support': int(len(np.unique(groups[selection]))),
              'population_groups': int(len(population))}
    # Empty bins do NOT disappear from the denominator of additive contributions.
    for key, values in (('population_point_fraction', np.ones(len(groups))),
                        ('population_error_change_contribution_m', points['error'] - points['truth_norm'])):
        result[key] = float(np.mean([values[selection & (groups == g)].sum() / (groups == g).sum()
                                     for g in population]))
    measurements = {'truth_norm_m': points['truth_norm'], 'predicted_norm_m': points['predicted_norm'],
                    'response_error_m': points['error'],
                    'response_gain_vs_zero_m': points['truth_norm'] - points['error'],
                    'coordinate_mse_m2': points['squared'] / 3.,
                    'zero_coordinate_mse_m2': points['truth_norm'] ** 2 / 3.,
                    'coordinate_mse_gain_vs_zero_m2': (points['truth_norm'] ** 2 - points['squared']) / 3.,
                    'predicted_over_5cm_fraction': (points['predicted_norm'] > threshold).astype(float)}
    for key, values in measurements.items():
        result['group_equal_' + key] = group_mean(values, groups, selection)
    result['group_equal_direction_cosine_nonzero_only'] = group_mean(
        points['cosine'], groups, selection & points['direction_valid'])
    return result


def magnitude_masks(norms, edges):
    bucket = np.searchsorted(np.asarray(edges), norms, side='left')
    return [bucket == i for i in range(len(edges) + 1)]


def build_points(calls, indices, outputs):
    if len(indices) != len(outputs) or not len(indices):
        raise ValueError('Incomplete response output support')
    rows = collections.defaultdict(list)
    objectives = collections.defaultdict(list)
    for i, output in zip(indices, outputs):
        record, values = calls[i]['record'], calls[i]['values']
        mask = response_support(values)
        truth = values['target'] - values['anchor_target'][None]
        predicted = np.asarray(output['response'])
        if predicted.shape != truth.shape or not np.isfinite(predicted).all():
            raise ValueError('Nonfinite or mismatched response forecast')
        if not mask.any():
            raise ValueError('Empty response call support')
        truth, predicted = truth[mask], predicted[mask]
        truth_norm, predicted_norm = np.linalg.norm(truth, axis=-1), np.linalg.norm(predicted, axis=-1)
        squared = ((predicted - truth) ** 2).sum(-1)
        direction_valid = (truth_norm > 1e-9) & (predicted_norm > 1e-9)
        cosine = np.zeros(len(truth))
        cosine[direction_valid] = np.clip((predicted[direction_valid] * truth[direction_valid]).sum(-1) /
                                         (truth_norm[direction_valid] * predicted_norm[direction_valid]), -1., 1.)
        group = record['group']
        arrays = {'group': np.repeat(group, len(truth)), 'truth_norm': truth_norm, 'predicted_norm': predicted_norm,
                  'squared': squared, 'error': np.sqrt(squared), 'cosine': cosine, 'direction_valid': direction_valid,
                  'offset': np.broadcast_to(np.arange(1, 9), mask.shape)[mask]}
        for key, value in arrays.items():
            rows[key].append(value)
        objectives[group].append([float(squared.mean() / 3.), float((truth_norm ** 2).mean() / 3.),
                                  float((predicted_norm ** 2).mean() / 3.)])
    points = {key: np.concatenate(values) for key, values in rows.items()}
    objective = np.mean([np.mean(values, axis=0) for values in objectives.values()], axis=0)
    return points, {'group_equal_call_mean_coordinate_mse_m2': float(objective[0]),
                    'group_equal_call_mean_zero_coordinate_mse_m2': float(objective[1]),
                    'group_equal_call_mean_response_energy_m2': float(objective[2])}


def decompose(calls, indices, outputs, protocol):
    points, objective = build_points(calls, indices, outputs)
    threshold = protocol['false_response_threshold_m']
    overall = conditional_summary(points, np.ones(len(points['group']), dtype=bool), threshold)
    masks = magnitude_masks(points['truth_norm'], protocol['truth_magnitude_bins_m'])
    bins = {name: conditional_summary(points, mask, threshold)
            for name, mask in zip(protocol['bin_names'], masks)}
    if sum(row['points'] for row in bins.values()) != overall['points'] or not np.isclose(
            sum(row['population_error_change_contribution_m'] for row in bins.values()),
            overall['population_error_change_contribution_m'], rtol=0, atol=1e-12):
        raise ValueError('Diagnostic partition/contribution identity failed')
    offsets = {str(offset): conditional_summary(points, points['offset'] == offset, threshold)
               for offset in protocol['reported_horizon_offsets']}
    return {'overall': overall, 'by_true_response_magnitude': bins, 'by_horizon_offset': offsets,
            'fixed_v19_training_objective_components': objective}


def mediator_contrasts(calls, indices, protocol):
    rows = collections.defaultdict(list)
    absolute = collections.defaultdict(list)
    for i in indices:
        record, values = calls[i]['record'], calls[i]['values']
        commands = calls[i]['estimated_commands']
        truth = np.concatenate([values['anchor_commanded'][None], values['commanded']], 0)
        proposed = np.concatenate([values['anchor'][None], values['proposed']], 0)
        observed = np.concatenate([(values['anchor_termination'] != 'after_terminal')[None],
                                   values['termination'] != 'after_terminal'], 0)
        if commands.shape != truth.shape or not np.isfinite(commands).all() or not observed.any():
            raise ValueError('Invalid frozen mediator estimates/support')
        absolute[record['group']].append([
            float(np.linalg.norm(commands - truth, axis=-1).mean(-1)[observed].mean()),
            float(np.linalg.norm(proposed - truth, axis=-1).mean(-1)[observed].mean())])
        mask = response_support(values)
        true_delta = truth[1:] - truth[0]
        estimated_error = np.linalg.norm(commands[1:] - commands[0] - true_delta, axis=-1)
        proposed_error = np.linalg.norm(proposed[1:] - proposed[0] - true_delta, axis=-1)
        response_norm = np.linalg.norm(values['target'] - values['anchor_target'][None], axis=-1)[mask]
        for key, data in {'group': np.repeat(record['group'], len(response_norm)), 'truth_norm': response_norm,
                          'estimated_delta_all_agents_error_mps': estimated_error.mean(-1)[mask],
                          'raw_delta_all_agents_error_mps': proposed_error.mean(-1)[mask],
                          'estimated_delta_own_agent_error_mps': estimated_error[:, :, record['agent']][mask],
                          'raw_delta_own_agent_error_mps': proposed_error[:, :, record['agent']][mask]}.items():
            rows[key].append(data)
    points = {key: np.concatenate(data) for key, data in rows.items()}
    def aggregate(mask):
        result = {'points': int(mask.sum()), 'groups_with_support': int(len(np.unique(points['group'][mask])))}
        for key, values in points.items():
            if key not in ('group', 'truth_norm'):
                result['group_equal_' + key] = group_mean(values, points['group'], mask)
        return result
    absolute_mean = np.mean([np.mean(values, 0) for values in absolute.values()], 0)
    return {'observed_absolute_commands': {'group_equal_call_mean_estimated_error_mps': float(absolute_mean[0]),
                                          'group_equal_call_mean_raw_error_mps': float(absolute_mean[1])},
            'nonanchor_paired_command_deltas': aggregate(np.ones(len(points['group']), dtype=bool)),
            'paired_deltas_by_true_response': {name: aggregate(mask) for name, mask in zip(protocol['bin_names'],
                magnitude_masks(points['truth_norm'], protocol['truth_magnitude_bins_m']))}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(artifact, output):
    # Legacy helpers only loaded inside this fresh-process entry point.
    sys.path.insert(0, str(ROOT.parent / 'cwm_v19'))
    import torch
    from frozen_training_release import (verify_manifest, audit, frozen_configs, BASELINE_SHA,
        FrozenCommonResponse, LocalTwoHead, FrozenMediator, attach_public_estimates, audit_data, evaluate, arrays)
    protocol = json.loads((ROOT / 'diagnostic_protocol.json').read_text())
    if sha(artifact) != protocol['source_artifact_sha256']:
        raise ValueError('Not the published immutable V19 artifact')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    capsule = ROOT.parent / 'cwm_v1/baseline/capsule.zip'
    if sha(capsule) != BASELINE_SHA:
        raise ValueError('Frozen original capsule changed')
    configs = frozen_configs(capsule, output / 'restored')
    integrity = verify_manifest(artifact)
    with zipfile.ZipFile(artifact) as archive:
        verification = audit(archive.read, configs, output / 'restored')
        if json.loads(archive.read('release_summary.json')) != verification:
            raise ValueError('Published release audit differs')
        print(json.dumps({'status': 'published_v19_audit_passed'}), flush=True)
        calls, _ = audit_data(archive.read, configs, output / 'restored')
        training = json.loads(archive.read('primary/summary.json'))
        training_protocol = training['protocol']
        if protocol['models'] != training_protocol['models'] or protocol['seeds'] != training_protocol['training']['seeds']:
            raise ValueError('Diagnostic population differs from all V19 models')
        mediator = FrozenMediator(ROOT.parent / 'cwm_v14/artifacts/public_mechanism_training_20261010.zip', training_protocol)
        attach_public_estimates(calls, mediator)
        estimates = arrays(archive.read('primary/public_estimates.npz'))
        if any(not np.array_equal(c['estimated_commands'], estimates[f'call{i}']) for i, c in enumerate(calls)):
            raise ValueError('Published frozen mediator estimates differ')
        indices = {split: [i for i, c in enumerate(calls) if c['record']['split'] == split] for split in protocol['splits']}
        records = {}
        for row in training['models']:
            name, seed = row['configuration'], row['seed']
            identifier = f'{name}_seed{seed}'
            raw = archive.read(f'primary/{identifier}.pt')
            if hashlib.sha256(raw).hexdigest() != row['checkpoint_sha256']:
                raise ValueError('Published checkpoint differs')
            checkpoint = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            scales = training_protocol['training']
            with torch.random.fork_rng(devices=[]):
                common = LocalTwoHead('motion_only', scales['motion_scale_m'], scales['response_scale_m'])
                model = common if name == 'motion_only' else FrozenCommonResponse(common,
                    LocalTwoHead('plain', scales['motion_scale_m'], scales['response_scale_m']), name == 'mediated_response')
            model.load_state_dict(checkpoint['model_state'], strict=True)
            model.eval().requires_grad_(False)
            normalizer = [checkpoint[k].numpy() for k in ('normalizer_mean', 'normalizer_scale')]
            result = {}
            for split, chosen in indices.items():
                measured, outputs = evaluate(model, name, calls, chosen, *normalizer)
                if split == 'development_validation':
                    saved = arrays(archive.read(f'primary/{identifier}_predictions.npz'))
                    if measured != row['development'] or any(not np.array_equal(o[k], saved[f'call{j}_{k}'])
                            for j, o in enumerate(outputs) for k in ('prediction', 'motion', 'response')):
                        raise ValueError('Diagnostic inference differs from published development result')
                diagnostic = decompose(calls, chosen, outputs, protocol)
                if not np.isclose(diagnostic['overall']['group_equal_response_error_m'],
                                  measured['group_equal_paired_response_error_m'], rtol=0, atol=1e-12):
                    raise ValueError('Gate-compatible aggregation differs')
                result[split] = {'v19_metrics': measured, 'response_decomposition': diagnostic}
            records[identifier] = result
            print(json.dumps({'status': 'all_splits_diagnosed', 'model': identifier,
                              'response_error': {k: v['v19_metrics']['group_equal_paired_response_error_m'] for k, v in result.items()}}), flush=True)
        mediator_report = {split: mediator_contrasts(calls, chosen, protocol) for split, chosen in indices.items()}
        if not mediator.unchanged():
            raise ValueError('Frozen mediator changed during diagnosis')
    sources = {p.name: sha(p) for p in (ROOT / 'response_diagnostics.py', ROOT / 'diagnostic_protocol.json')}
    report = {'status': 'descriptive_v19_failure_diagnosis_not_new_validation', 'protocol': protocol,
              'source_artifact': integrity, 'source_hashes': sources,
              'models': records, 'mediator': mediator_report, 'prior_qualification': training['qualification'],
              'published_v19_independent_audit_passed': True,
              'published_development_predictions_exactly_reproduced': True,
              'enhanced_control_enabled': False, 'new_model_trained': False, 'holdout_used': False,
              'prior_gates_overridden': False}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({'status': report['status']}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, default=ROOT.parent.parent / json.loads((ROOT / 'diagnostic_protocol.json').read_text())['source_artifact'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.artifact, args.output)


if __name__ == '__main__':
    main()
