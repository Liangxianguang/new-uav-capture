"""Saved-cost report helpers are not substitutes for full research verification."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_completed_results import METHODS, describe_decisions, saved_cost_metrics


def decision(group='first', step=1, costs=(4., 1.), motion=(1., 4.), truth=(3., 1.)):
    vectors = {k: list(costs) for k in METHODS}
    vectors['own_motion_component'] = list(motion)
    vectors['action_specific_truth'] = list(truth)
    return {'episode_index': 1, 'step': step, 'agent': 0, 'group': group,
            'population': 'geometry_union', 'indices': [0, 2], 'costs': vectors,
            'metrics': {k: saved_cost_metrics(v, list(truth)) for k, v in vectors.items()}}


def test_first_tie_and_regret():
    assert saved_cost_metrics([2., 2.], [5., 1.]) == {
        'choice': 0, 'regret_in_diagnostic_cost': 4.,
        'realized_cost_at_choice': 5., 'realized_tie_optimal': False}


@pytest.mark.parametrize('costs,truth', [([], []), ([1.], [1., 2.]), ([float('nan')], [1.]),
                                       ([1.], [float('inf')])])
def test_invalid_costs_rejected(costs, truth):
    with pytest.raises(ValueError, match='Finite nonempty'):
        saved_cost_metrics(costs, truth)


def test_group_equal_not_call_equal():
    rows = [decision(), decision(step=2), decision('second', costs=(1., 4.), motion=(4., 1.))]
    result = describe_decisions(rows, 'geometry_union')
    assert result['choice_change_count'] == 3
    assert result['help_count'] == 2 and result['harm_count'] == 1
    assert result['group_equal_gain_vs_own_motion'] == 0.
    assert result['group_equal_help_rate'] == .5
    assert result['candidate_count_histogram'] == {'2': 3}


def test_no_changes_is_not_declared_success():
    result = describe_decisions([decision(costs=(1., 4.))], 'geometry_union')
    assert result['choice_change_count'] == result['help_count'] == result['harm_count'] == 0
    assert result['group_equal_gain_vs_own_motion'] == 0.
    assert 'research_eligible' not in result


@pytest.mark.parametrize('forgery', ['metric', 'dimension', 'method', 'index', 'population', 'duplicate'])
def test_saved_semantics_reject_forgery(forgery):
    row = copy.deepcopy(decision())
    rows = [row]
    if forgery == 'metric':
        row['metrics']['model']['choice'] = 0
    elif forgery == 'dimension':
        row['costs']['action_specific_truth'].append(3.)
    elif forgery == 'method':
        row['metrics'].pop('constant_velocity')
    elif forgery == 'index':
        row['indices'] = [0, 0]
    elif forgery == 'population':
        row['population'] = 'actual_unique'
    else:
        rows.append(copy.deepcopy(row))
    with pytest.raises(ValueError):
        describe_decisions(rows, 'geometry_union')
