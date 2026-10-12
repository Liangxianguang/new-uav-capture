"""Synthetic scalar metric/contract checks, NOT native model results."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent/'cwm_v35')]
from contract import load_contract, validate_contract
from stream_metrics import PredictionMetrics
from test_audited_stream import toy_call
from train_two_head import metrics as old_metrics


def test_pinned_plan_exact_full_budget_and_no_promotion():
    contract, plan = load_contract()
    assert plan['training']['seeds'] == [1034101, 1034102, 1034103]
    assert plan['training']['motion_epochs'] == plan['training']['response_epochs'] == 80
    assert plan['training']['independent_complete_runs'] == 2
    assert contract['enhanced_control_enabled'] is False


@pytest.mark.parametrize('kind', ['epoch', 'seed', 'primary', 'holdout', 'gate', 'reload'])
def test_fixed_contract_or_old_gates_cannot_be_changed(kind):
    contract, plan = load_contract()
    if kind == 'epoch': plan['training']['response_epochs'] = 1
    elif kind == 'seed': plan['training']['seeds'][0] = 1
    elif kind == 'primary': contract['primary_configuration'] = 'cv_full_query_rank_l2'
    elif kind == 'holdout': plan['holdout_used'] = True
    elif kind == 'gate': plan['development_gate']['minimum_selected_gain_lower_bound_vs_full_query'] = -1
    else: contract['independent_full_reload_required'] = False
    with pytest.raises(ValueError): validate_contract(contract, plan)


def test_scalar_metrics_match_old_point_and_group_formula_without_full_arrays():
    calls = [toy_call('a'), toy_call('a', tail_invalid=True), toy_call('b')]
    outputs = []
    scalar = PredictionMetrics()
    for i, call in enumerate(calls):
        v = call['values']; n = len(v['proposed'])
        response = .2*(v['target']-v['anchor_target'][None])
        output = {'prediction': v['target']*.8, 'motion': np.zeros((n, 8, 3)), 'response': response}
        scalar.add(call['record'], v, output, np.arange(n)); outputs.append(output)
    measured = scalar.result({'a', 'b'}); expected = old_metrics(calls, [0, 1, 2], outputs)
    for key in ('group_equal_ade_m', 'group_equal_reference_ade_m', 'group_equal_paired_response_error_m'):
        assert np.isclose(measured[key], expected[key], rtol=1e-15, atol=1e-15)
    assert measured['by_group'] == expected['by_group']
    assert all(not isinstance(v, np.ndarray) for g in scalar.groups.values() for v in g.values())


def test_extra_probe_prediction_metrics_do_not_change_base_selection():
    call = toy_call(); v = call['values']; n = len(v['proposed'])
    output = {'prediction': v['target'].copy(), 'motion': np.zeros((n, 8, 3)), 'response': v['target'].copy()}
    first = PredictionMetrics(); first.add(call['record'], v, output, v['bounded_shared_indices'])
    output['prediction'][3:] += 100.
    second = PredictionMetrics(); second.add(call['record'], v, output, v['bounded_shared_indices'])
    assert first.result({'g'}) == second.result({'g'})


@pytest.mark.parametrize('kind', ['group', 'ranking', 'nonfinite', 'population'])
def test_missing_or_invalid_metric_population_refused(kind):
    call = toy_call(); v = call['values']; n = len(v['proposed'])
    output = {'prediction': v['target'].copy(), 'motion': np.zeros((n, 8, 3)), 'response': v['target'].copy()}
    metric = PredictionMetrics()
    if kind == 'nonfinite':
        output['prediction'][0, 0, 0] = np.nan
        with pytest.raises(ValueError): metric.add(call['record'], v, output, np.arange(n))
    elif kind == 'population':
        with pytest.raises(ValueError): metric.add(call['record'], v, output, [0, 0])
    else:
        metric.add(call['record'], v, output, np.arange(n))
        with pytest.raises(ValueError): metric.result({'missing'} if kind == 'group' else {'g'}, require_ranking=kind == 'ranking')
