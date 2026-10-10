"""Preregistered final-epoch/median-seed offline qualification; never promotion."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'cwm_v8'))
from s4_value import descriptive_bootstrap


def validate_cost_training(protocol):
    if protocol != json.loads((ROOT / 'training_protocol.json').read_text(encoding='utf8')):
        raise ValueError('Published V23 training protocol changed')
    if any(protocol[k] for k in ('enhanced_control_enabled', 'original_gru_in_optimizer',
        'common_motion_in_response_optimizer', 'mediator_in_optimizer', 'private_labels_model_inputs',
        'holdout_used', 'prior_gate_override_allowed')):
        raise ValueError('Frozen optional research contract violated')
    if protocol['primary_configuration'] != 'cv_cost_l2' or len(protocol['models']) != 6:
        raise ValueError('Fixed calibrated 18-model population changed')


def select_cost_seeds(models, protocol):
    if len(models) != len(protocol['models']) * len(protocol['training']['seeds']):
        raise ValueError('Incomplete fixed final model population')
    chosen = {}
    for name in protocol['models']:
        rows = [r for r in models if r['configuration'] == name]
        if len(rows) != 3 or {r['seed'] for r in rows} != set(protocol['training']['seeds']):
            raise ValueError('Incomplete fixed three-seed population')
        if any(not np.isfinite(r['development']['group_equal_ade_m']) for r in rows):
            raise ValueError('Nonfinite final development ADE')
        chosen[name] = sorted(rows, key=lambda r: (r['development']['group_equal_ade_m'], r['seed']))[1]['seed']
    return chosen


def compare_cost_decisions(changed, baseline, protocol, baseline_key='model'):
    identity = lambda rows: [tuple(r[k] for k in ('episode_index', 'step', 'agent', 'group')) for r in rows]
    if not changed or identity(changed) != identity(baseline):
        raise ValueError('Complete original-cost decision supports differ or empty')
    gains = [b['metrics'][baseline_key]['regret_in_diagnostic_cost'] -
             r['metrics']['model']['regret_in_diagnostic_cost'] for r, b in zip(changed, baseline)]
    if not np.isfinite(gains).all():
        raise ValueError('Nonfinite decision gains')
    result = descriptive_bootstrap(gains, [r['group'] for r in changed],
        protocol['decision_bootstrap']['draws'], protocol['decision_bootstrap']['seed'])
    return {**result, 'interpretation': protocol['decision_bootstrap']['interpretation']}


def qualify_cost_models(models, controls, decisions, protocol):
    validate_cost_training(protocol)
    selected = select_cost_seeds(models, protocol)
    middle = lambda name, key: float(np.median([r['development'][key] for r in models if r['configuration'] == name]))
    thresholds = protocol['development_gate']
    response_key = 'group_equal_paired_response_error_m'
    cost_key = 'group_equal_normalized_cost_contrast_l1'
    gates = {}
    for name, config in protocol['response_configurations'].items():
        rows = [r for r in models if r['configuration'] == name]
        keys = ('group_equal_ade_m', response_key, cost_key)
        if any(not np.isfinite(r['development'][key]) for r in rows for key in keys):
            raise ValueError('Nonfinite final development research metrics')
        checks = {
            'each_seed_ade_vs_gru': all(r['development']['group_equal_ade_m'] <=
                controls['frozen_gru']['group_equal_ade_m'] * thresholds['maximum_each_seed_ade_ratio_vs_gru'] for r in rows),
            'median_ade_vs_cv': middle(name, 'group_equal_ade_m') <=
                controls['constant_velocity']['group_equal_ade_m'] * thresholds['maximum_median_ade_ratio_vs_cv'],
            'median_response_vs_zero': middle(name, response_key) <=
                controls['frozen_gru'][response_key] * thresholds['maximum_median_response_error_ratio_vs_zero'],
            'anchor_response_exact_zero': all(r['development']['anchor_response_exact_zero'] for r in rows)}
        chosen = decisions[f'{name}_seed{selected[name]}']
        motion = protocol['motion_controls'][config['motion_reference']]
        gains = {}
        for reference in ('own_motion', 'cv', 'motion_only'):
            if reference == 'motion_only':
                baseline, key = decisions[f'{motion}_seed{selected[motion]}'], 'model'
            else:
                baseline, key = chosen, 'own_motion_component' if reference == 'own_motion' else 'constant_velocity'
            gains[reference] = compare_cost_decisions(chosen, baseline, protocol, key)
            checks['gain_vs_' + reference] = gains[reference]['percentile_95_interval'][0] > thresholds[
                'minimum_selected_gain_lower_bound_vs_' + reference]
        if name == protocol['primary_configuration']:
            checks['median_response_vs_cv_l2'] = middle(name, response_key) <= middle('cv_l2', response_key)
            checks['median_cost_contrast_vs_cv_l2'] = middle(name, cost_key) <= .95 * middle('cv_l2', cost_key)
            for other in ('cv_l2', 'gru_cost_l2'):
                gains[other] = compare_cost_decisions(chosen, decisions[f'{other}_seed{selected[other]}'], protocol)
                checks['gain_vs_' + other] = gains[other]['percentile_95_interval'][0] > 0.
        gates[name] = {'research_eligible': bool(all(checks.values())),
            'checks': {k: bool(v) for k, v in checks.items()}, 'selected_seed': selected[name],
            'matching_motion_control': motion, 'selected_gains': gains, 'online_promoted': False}
    factorial = {}
    for changed, reference in protocol['factorial_comparisons']:
        factorial[f'{changed}_vs_{reference}'] = {
            'all_seed_response_gain_m': {str(seed): next(r for r in models if r['configuration'] == reference and r['seed'] == seed)['development'][response_key] -
                next(r for r in models if r['configuration'] == changed and r['seed'] == seed)['development'][response_key]
                for seed in protocol['training']['seeds']},
            'selected_decision_gain': compare_cost_decisions(decisions[f'{changed}_seed{selected[changed]}'],
                                                            decisions[f'{reference}_seed{selected[reference]}'], protocol),
            'descriptive_only': True}
    return selected, gates, factorial
