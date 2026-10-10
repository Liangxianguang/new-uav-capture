import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from decision_diagnostics import (diagnostic_paths, repair_contributions, local_diagnostics,
                                  validate_protocol, summarize, paired_summary)


def fixture_paths():
    anchor = np.zeros((8, 4, 3))
    proposed = np.repeat(anchor[None], 2, 0)
    proposed[1, :, 0, 0] = 1
    truth = np.zeros((2, 8, 3))
    truth[1, :, 0] = .5
    values = {'target': truth, 'valid': np.ones((2, 8), bool), 'anchor_valid': np.ones(8, bool),
              'anchor_target': truth[0], 'backbone': np.ones((8, 3)),
              'proposed': proposed, 'anchor': anchor,
              'reference': np.zeros(3), 'velocity': np.zeros(3)}
    motion = np.full_like(truth, .2)
    response = np.zeros_like(truth)
    response[1, :, 0] = .1
    output = {'prediction': values['backbone'][None] + motion + response,
              'motion': motion, 'response': response}
    return values, output


def fake_metrics(costs):
    truth = np.asarray(costs['action_specific_truth'])
    result = {}
    for name, values in costs.items():
        choice = int(np.argmin(values))
        regret = float(truth[choice] - truth.min())
        result[name] = {'choice': choice, 'regret_in_diagnostic_cost': regret,
                        'realized_tie_optimal': regret <= 1e-9}
    return result


def test_truth_replacements_keep_motion_response_separate():
    values, output = fixture_paths()
    paths = diagnostic_paths(values, output)
    assert np.array_equal(paths['model'], output['prediction'])
    assert np.array_equal(paths['own_motion'], values['backbone'][None] + output['motion'])
    assert np.array_equal(paths['reference_truth_plus_learned_response'], values['anchor_target'][None] + output['response'])
    assert np.array_equal(paths['motion_plus_exact_response'], paths['own_motion'] + values['target'] - values['anchor_target'][None])
    assert np.array_equal(paths['cv_plus_learned_response'], output['response'])


@pytest.mark.parametrize('bad', ['padding', 'candidate_motion', 'anchor_response', 'prediction', 'nan'])
def test_invalid_diagnostic_support_rejected(bad):
    values, output = fixture_paths()
    if bad == 'padding':
        values['valid'][1, 7] = False
    elif bad == 'candidate_motion':
        output['motion'][1, 0, 0] += .01
    elif bad == 'anchor_response':
        output['response'][0, 0, 0] = .01
    elif bad == 'prediction':
        output['prediction'][1, 1, 1] += .01
    else:
        output['response'][1, 0, 0] = np.nan
    with pytest.raises(ValueError):
        diagnostic_paths(values, output)


def test_order_averaged_repairs_sum_to_regret_and_keep_negative_contributions():
    report = repair_contributions(2., 8., 1., 0.)
    assert report == {'motion_repair': -2.5, 'response_repair': 4.5, 'total_repair': 2.}
    for r00, r10, r01 in [(0, 0, 0), (4, 1, 2), (1, 3, 6)]:
        row = repair_contributions(r00, r10, r01, 0)
        assert row['motion_repair'] + row['response_repair'] == r00
    with pytest.raises(ValueError):
        repair_contributions(1, 1, 1, .01)


def test_contrast_error_excludes_anchor_and_help_harm_are_truth_regret():
    costs = {'model': [10, 9], 'own_motion': [10, 11], 'fixed_reference_truth': [6, 8],
              'action_specific_truth': [6, 5], 'motion_plus_exact_response': [10, 8],
              'reference_truth_plus_learned_response': [6, 6]}
    result = local_diagnostics(costs, fake_metrics(costs), 0, [0, .5], 1e-9)
    assert result['gain_vs_own_motion'] == 1
    assert result['helps_vs_own_motion'] and not result['harms_vs_own_motion']
    assert result['response_cost_contrast_mae_nonanchor'] == 1
    assert result['predicted_best_two_margin'] == 1
    assert result['chosen_true_response_mean_m'] == .5
    assert result['repairs']['total_repair'] == 0
    stale = fake_metrics(costs)
    stale['model']['choice'] = 0
    with pytest.raises(ValueError, match='inconsistent'):
        local_diagnostics(costs, stale, 0, [0, .5], 1e-9)


def test_single_anchor_has_no_imputed_margin_or_nonanchor_contrast():
    keys = ['model', 'own_motion', 'fixed_reference_truth', 'action_specific_truth',
            'motion_plus_exact_response', 'reference_truth_plus_learned_response']
    costs = {k: [10] for k in keys}
    result = local_diagnostics(costs, fake_metrics(costs), 0, [0], 1e-9)
    assert result['predicted_best_two_margin'] is None
    assert result['response_cost_contrast_mae_nonanchor'] is None
    assert result['same_regret_vs_own_motion']


def test_pinned_protocol_rejects_gate_or_population_changes():
    protocol = json.loads((ROOT / 'diagnostic_protocol.json').read_text(encoding='utf8'))
    validate_protocol(protocol)
    for key, value in [('holdout_used', True), ('models', ['raw_call_l2']),
                       ('full_v21_audit_required_before_diagnosis', False)]:
        changed = copy.deepcopy(protocol)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_protocol(changed)


def test_group_summary_not_call_weighted_and_empty_contrast_not_zero():
    protocol = json.loads((ROOT / 'diagnostic_protocol.json').read_text(encoding='utf8'))
    keys = protocol['paths']
    def row(group, regret):
        costs = {k: [10, 9] for k in keys}
        costs['action_specific_truth'] = [6, 5]
        metrics = fake_metrics(costs)
        details = local_diagnostics(costs, metrics, 0, [0, .5], 1e-9)
        for value in metrics.values():
            value['regret_in_diagnostic_cost'] = regret
        details['response_cost_contrast_mae_nonanchor'] = None
        return {'group': group, 'metrics': metrics, 'diagnostics': details}
    report = summarize([row('a', 0), row('a', 0), row('a', 0), row('b', 4)], protocol)
    assert report['paths']['model']['regret']['group_equal_mean'] == 2
    assert report['diagnostics']['response_cost_contrast_mae_nonanchor']['groups_with_support'] == 0
    assert report['diagnostics']['response_cost_contrast_mae_nonanchor']['group_equal_mean'] is None


def test_paired_bootstrap_retains_negative_gain_and_rejects_misalignment():
    protocol = json.loads((ROOT / 'diagnostic_protocol.json').read_text(encoding='utf8'))
    def row(group, regret):
        return {'episode_index': 1, 'step': 2, 'agent': 0, 'group': group, 'population': 'actual_unique',
                'metrics': {'model': {'regret_in_diagnostic_cost': regret}}}
    changed = [row('a', 2), row('b', 2)]
    reference = [row('a', 1), row('b', 1)]
    measured = paired_summary(changed, reference, protocol)
    assert measured['group_equal_regret_gain'] == -1
    assert measured['percentile_95_interval'] == [-1, -1]
    with pytest.raises(ValueError):
        paired_summary(changed, reference[::-1], protocol)
