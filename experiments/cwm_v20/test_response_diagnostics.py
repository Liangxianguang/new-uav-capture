import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from response_diagnostics import response_support, magnitude_masks, decompose, mediator_contrasts, build_points


def protocol():
    return json.loads((ROOT / 'diagnostic_protocol.json').read_text())


def sample(group='a', true_magnitude=0., predicted_magnitude=0.):
    proposed = np.zeros((2, 8, 4, 3))
    proposed[1, :, 0, 0] = 1.
    target = np.zeros((2, 8, 3))
    target[1, :, 0] = true_magnitude
    values = {'proposed': proposed, 'anchor': proposed[0], 'target': target,
              'anchor_target': target[0], 'valid': np.ones((2, 8), dtype=bool),
              'anchor_valid': np.ones(8, dtype=bool), 'commanded': proposed.copy(),
              'anchor_commanded': proposed[0].copy(),
              'termination': np.full((2, 8), 'running'), 'anchor_termination': np.full(8, 'running')}
    response = np.zeros_like(target)
    response[1, :, 0] = predicted_magnitude
    return {'record': {'group': group, 'agent': 0}, 'values': values,
            'estimated_commands': np.concatenate([proposed[0:1], proposed], 0)}, {'response': response}


def test_bins_are_disjoint_exhaustive_with_inclusive_upper_bound():
    norms = np.array([0., 1e-10, .005, .01, .02, .05, .15, .2, .3])
    masks = magnitude_masks(norms, protocol()['truth_magnitude_bins_m'])
    assert np.stack(masks).sum(0).tolist() == [1] * len(norms)
    assert [int(m.sum()) for m in masks] == [2, 2, 2, 2, 1]


def test_anchor_and_invalid_prefix_points_excluded_not_imputed():
    call, output = sample(true_magnitude=.2, predicted_magnitude=.1)
    call['values']['valid'][1, 4:] = False
    call['values']['anchor_valid'][2] = False
    output['response'][0] = 999.
    output['response'][1, 4:] = 999.
    assert response_support(call['values']).sum() == 3
    result = decompose([call], [0], [output], protocol())
    assert result['overall']['points'] == 3
    assert result['overall']['group_equal_response_error_m'] == pytest.approx(.1)
    assert result['by_horizon_offset']['8']['group_equal_response_error_m'] is None


def test_equal_groups_not_pooled_points_and_contributions_sum():
    a, oa = sample('a', 0., .2)
    b, ob = sample('b', .3, .1)
    b['values']['valid'][1, 1:] = False
    result = decompose([a, b], [0, 1], [oa, ob], protocol())
    assert result['overall']['group_equal_response_error_m'] == pytest.approx(.2)
    assert result['overall']['group_equal_truth_norm_m'] == pytest.approx(.15)
    bins = result['by_true_response_magnitude']
    assert bins['numerically_zero']['population_point_fraction'] == .5
    assert bins['numerically_zero']['groups_with_support'] == 1
    assert bins['numerically_zero']['population_error_change_contribution_m'] == pytest.approx(.1)
    assert bins['over_20cm']['population_error_change_contribution_m'] == pytest.approx(-.05)
    assert sum(v['population_error_change_contribution_m'] for v in bins.values()) == pytest.approx(.05)
    assert sum(v['population_point_fraction'] for v in bins.values()) == pytest.approx(1.)


def test_coordinate_mse_and_vector_error_can_disagree():
    a, oa = sample('a', 0., .05)
    b, ob = sample('b', .3, .26)
    result = decompose([a, b], [0, 1], [oa, ob], protocol())
    assert result['overall']['group_equal_response_gain_vs_zero_m'] > 0.
    assert result['overall']['group_equal_coordinate_mse_gain_vs_zero_m2'] > 0.
    # Mostly zero points can raise vector error despite a rare strong-point gain.
    calls, outputs = [a] * 10 + [b], [oa] * 10 + [ob]
    b['record']['group'] = 'a'
    result = decompose(calls, list(range(len(calls))), outputs, protocol())
    assert result['overall']['group_equal_response_gain_vs_zero_m'] < 0.
    assert result['overall']['group_equal_coordinate_mse_gain_vs_zero_m2'] > 0.


def test_call_balanced_objective_distinct_from_point_gate():
    a, oa = sample('a', .3, 0.)
    b, ob = sample('a', 0., .1)
    b['values']['valid'][1, 1:] = False
    result = decompose([a, b], [0, 1], [oa, ob], protocol())
    assert result['fixed_v19_training_objective_components']['group_equal_call_mean_coordinate_mse_m2'] == pytest.approx(.05 / 3.)
    assert result['overall']['group_equal_coordinate_mse_m2'] == pytest.approx((8 * .09 + .01) / 9 / 3.)


def test_mediator_absolute_error_can_cancel_in_contrast():
    call, output = sample(true_magnitude=.2)
    call['estimated_commands'] += .5
    result = mediator_contrasts([call], [0], protocol())
    assert result['observed_absolute_commands']['group_equal_call_mean_estimated_error_mps'] > 0.
    assert result['nonanchor_paired_command_deltas']['group_equal_estimated_delta_own_agent_error_mps'] == 0.


def test_invalid_prediction_rejected():
    call, output = sample()
    output['response'][1, 0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        build_points([call], [0], [output])


def test_no_anchor_only_call_silent_zero():
    call, output = sample()
    call['values']['proposed'][:] = 0.
    with pytest.raises(ValueError, match='Empty response'):
        build_points([call], [0], [output])
