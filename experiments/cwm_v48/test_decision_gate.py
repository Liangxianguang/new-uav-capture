"""Toy original-cost vectors and scalar bootstrap; not model efficacy."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contract import load_contract
from decision_gate import (DecisionRows, FamilyAxisAccumulator, family_axis_stats, compare_decisions,
    qualify, rank_metrics, sha)


def toy_decisions(kind='prefix'):
    contract, _ = load_contract(); result = []
    chosen = {'prefix': [3., 2., 0.], 'full': [3., 0., 2.], 'motion': [0., 2., 3.]}[kind]
    for group in range(16):
        for index, axis in enumerate(contract['geometry_axes']):
            costs = {'model': chosen.copy(), 'own_motion_component': [0., 2., 3.],
                'original_gru': [0., 2., 3.], 'constant_velocity': [0., 2., 3.], 'action_specific_truth': [9., 4., 1.]}
            result.append({'episode_index': group*4+index, 'step': 2, 'ordinal': 0, 'agent': 0, 'group': str(group),
                'variant': axis, 'population': 'bounded_shared', 'indices': [0, 1, 2], 'costs': costs,
                'metrics': {k: rank_metrics(v, costs['action_specific_truth'], 1e-9) for k, v in costs.items()}})
    return result


def qualification_fixture():
    contract, plan = load_contract(); models = []; decisions = {}
    for name in plan['models']:
        for i, seed in enumerate(plan['training']['seeds']):
            models.append({'configuration': name, 'seed': seed, 'development': {'group_equal_ade_m': .1+.1*i,
                'group_equal_paired_response_error_m': .1, 'group_equal_ranking_kl': .01, 'anchor_response_exact_zero': True}})
            decisions[f'{name}_seed{seed}'] = toy_decisions('motion' if name == 'cv_motion_only' else 'full' if name == 'cv_full_query_rank_l2' else 'prefix')
    controls = {'frozen_gru': {'group_equal_ade_m': 1., 'group_equal_paired_response_error_m': 1.},
        'constant_velocity': {'group_equal_ade_m': 1.}}
    return contract, plan, models, decisions, controls


def test_uses_final_ADE_median_and_strict_positive_matched_control_gain():
    contract, plan, models, decisions, controls = qualification_fixture()
    selected, gates = qualify(models, controls, decisions, contract, plan, set(map(str, range(16))))
    assert all(v == 1034102 for v in selected.values())
    primary = gates['cv_prefix_rank_l2']; assert primary['research_eligible'] is True
    assert primary['selected_gains']['full_query']['group_equal_mean'] == 3.
    assert primary['selected_gains']['full_query']['percentile_95_interval'] == [3., 3.]
    assert primary['online_promoted'] is False


def test_lucky_other_seed_cannot_rescue_failed_fixed_median():
    contract, plan, models, decisions, controls = qualification_fixture()
    decisions['cv_prefix_rank_l2_seed1034102'] = toy_decisions('motion')
    selected, gates = qualify(models, controls, decisions, contract, plan, set(map(str, range(16))))
    assert selected['cv_prefix_rank_l2'] == 1034102
    assert gates['cv_prefix_rank_l2']['research_eligible'] is False
    assert gates['cv_prefix_rank_l2']['checks']['gain_vs_full_query'] is False


def test_zero_gain_never_passes_strict_lower_bound():
    contract, plan, models, decisions, controls = qualification_fixture()
    decisions['cv_prefix_rank_l2_seed1034102'] = toy_decisions('full')
    _, gates = qualify(models, controls, decisions, contract, plan, set(map(str, range(16))))
    assert gates['cv_prefix_rank_l2']['selected_gains']['full_query']['percentile_95_interval'] == [0., 0.]
    assert gates['cv_prefix_rank_l2']['research_eligible'] is False


def test_all_seeds_equal_ADE_ties_choose_middle_sorted_seed():
    contract, plan, models, decisions, controls = qualification_fixture()
    for row in models: row['development']['group_equal_ade_m'] = .1
    selected, _ = qualify(models, controls, decisions, contract, plan, set(map(str, range(16))))
    assert selected['cv_prefix_rank_l2'] == 1034102


@pytest.mark.parametrize('kind', ['truth', 'identity', 'library', 'choice', 'missing_axis', 'partial'])
def test_unmatched_saved_decisions_never_bootstrapped(kind):
    changed, baseline = toy_decisions(), toy_decisions('full'); contract, _ = load_contract()
    if kind == 'truth': baseline[0]['costs']['action_specific_truth'][0] += 1
    elif kind == 'identity': baseline[0]['ordinal'] = 2
    elif kind == 'library': baseline[0]['indices'] = [0, 1]
    elif kind == 'choice': baseline[0]['metrics']['model']['choice'] = 2
    elif kind == 'missing_axis': changed = changed[1:]; baseline = baseline[1:]
    else: baseline.pop()
    with pytest.raises(ValueError): compare_decisions(changed, baseline, contract, set(map(str, range(16))))


def test_equal_axis_means_not_favorable_call_count_weighting():
    contract, _ = load_contract(); records = []; values = []
    for axis_index, axis in enumerate(contract['geometry_axes']):
        for _ in range(100 if axis_index == 0 else 1):
            records.append({'group': 'one', 'variant': axis}); values.append(4. if axis_index == 0 else 0.)
    result = family_axis_stats(values, records, contract['geometry_axes'], {'one'})
    assert result['group_equal_mean'] == 1. and result['percentile_95_interval'] == [1., 1.]
    assert np.mean(values) > 3.8


def test_JSONL_decisions_repeatable_checked_and_not_cached(tmp_path):
    path = tmp_path/'toy.jsonl'; rows = toy_decisions()
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf8')
    iterable = DecisionRows(path, sha(path), len(rows)); assert list(iterable) == rows and list(iterable) == rows
    assert set(iterable.__dict__) == {'path', 'digest', 'count'}
    path.write_bytes(b'changed')
    with pytest.raises(ValueError): list(iterable)


def test_scalar_family_accumulator_never_retains_per_call_vectors():
    contract, _ = load_contract(); acc = FamilyAxisAccumulator(contract['geometry_axes'])
    for row in toy_decisions(): acc.add(1., row)
    assert acc.result(set(map(str, range(16))))['group_equal_mean'] == 1.
    assert all(len(v) == 2 for axes in acc.grouped.values() for v in axes.values())
