import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cost_qualification import validate_cost_training, select_cost_seeds, qualify_cost_models


def measured(response=.1, ade=.1, cost=.1):
    return {'group_equal_paired_response_error_m': response, 'group_equal_ade_m': ade,
            'group_equal_normalized_cost_contrast_l1': cost, 'anchor_response_exact_zero': True}


def decision(regret):
    return [{'episode_index': i, 'step': 1, 'agent': 0, 'group': f'g{i}',
             'metrics': {k: {'regret_in_diagnostic_cost': r} for k, r in
                         (('model', regret), ('own_motion_component', 1.), ('constant_velocity', 1.))}}
            for i in range(8)]


def results():
    protocol = json.loads((ROOT / 'training_protocol.json').read_text(encoding='utf8'))
    rows, decisions = [], {}
    for name in protocol['models']:
        for j, seed in enumerate(protocol['training']['seeds']):
            primary = name == protocol['primary_configuration']
            motion = name.endswith('_motion_only')
            rows.append({'configuration': name, 'seed': seed,
                'development': measured(1. if motion else .1, .1 + j * .01, .1 if primary else .2)})
            decisions[f'{name}_seed{seed}'] = decision(1. if motion else 0. if primary else .2)
    controls = {'frozen_gru': measured(1., 1.), 'constant_velocity': measured(1., 1.)}
    return protocol, rows, decisions, controls


def test_all_preregistered_gates_and_no_online_promotion():
    p, rows, decisions, controls = results()
    selected, gates, comparisons = qualify_cost_models(rows, controls, decisions, p)
    assert gates['cv_cost_l2']['research_eligible']
    assert all(not g['online_promoted'] for g in gates.values())
    assert selected['cv_cost_l2'] == 993102
    assert len(comparisons) == 4


@pytest.mark.parametrize('key,value,check', [
    ('group_equal_paired_response_error_m', 1.1, 'median_response_vs_zero'),
    ('group_equal_ade_m', .99, 'median_ade_vs_cv'),
    ('group_equal_normalized_cost_contrast_l1', .2, 'median_cost_contrast_vs_cv_l2'),
    ('anchor_response_exact_zero', False, 'anchor_response_exact_zero')])
def test_primary_cannot_pass_failed_metric_gate(key, value, check):
    p, rows, decisions, controls = results()
    for row in rows:
        if row['configuration'] == 'cv_cost_l2':
            row['development'][key] = value
    _, gates, _ = qualify_cost_models(rows, controls, decisions, p)
    assert not gates['cv_cost_l2']['research_eligible']
    assert not gates['cv_cost_l2']['checks'][check]


@pytest.mark.parametrize('comparison', ['cv_l2', 'gru_cost_l2'])
def test_primary_requires_strict_positive_gain_over_both_response_controls(comparison):
    p, rows, decisions, controls = results()
    for seed in p['training']['seeds']:
        decisions[f'{comparison}_seed{seed}'] = decision(0.)
    _, gates, _ = qualify_cost_models(rows, controls, decisions, p)
    assert not gates['cv_cost_l2']['research_eligible']
    assert not gates['cv_cost_l2']['checks']['gain_vs_' + comparison]


def test_gru_cannot_use_cv_motion_control_and_cv_remains_uncalibrated():
    p, rows, decisions, controls = results()
    for seed in p['training']['seeds']:
        decisions[f'gru_motion_only_seed{seed}'] = decision(0.)
    _, gates, _ = qualify_cost_models(rows, controls, decisions, p)
    assert gates['gru_cost_l2']['matching_motion_control'] == 'gru_motion_only'
    assert not gates['gru_cost_l2']['checks']['gain_vs_motion_only']
    assert gates['cv_cost_l2']['checks']['gain_vs_motion_only']
    for records in decisions.values():
        for row in records:
            row['metrics']['constant_velocity']['regret_in_diagnostic_cost'] = 0.
    _, gates, _ = qualify_cost_models(rows, controls, decisions, p)
    assert not gates['cv_cost_l2']['checks']['gain_vs_cv']


def test_missing_model_seed_changed_support_and_changed_protocol_rejected():
    p, rows, decisions, controls = results()
    with pytest.raises(ValueError, match='population'):
        select_cost_seeds(rows[:-1], p)
    changed = copy.deepcopy(p)
    changed['training']['cost_contrast_weight'] = 1.
    with pytest.raises(ValueError, match='protocol'):
        validate_cost_training(changed)
    decisions['cv_l2_seed993102'][0]['episode_index'] = 999
    with pytest.raises(ValueError, match='supports differ'):
        qualify_cost_models(rows, controls, decisions, p)
