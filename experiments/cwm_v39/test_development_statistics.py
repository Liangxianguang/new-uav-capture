"""Full-population synthetic outcomes only, never native capture/latency."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import development_statistics as gate
from test_closed_loop_gates import synthetic_rows


@pytest.fixture(scope='module')
def plan(): return gate.load_plan()


@pytest.fixture(scope='module')
def expected(plan): return {c: gate.specification(c, plan) for c in gate.COHORTS}


def rows(cohort, expected, plan):
    return synthetic_rows(expected[cohort], gate.arithmetic_view(plan))


def test_both_complete_cohorts_all8controls_never_authorize_native_or_holdout(plan, expected):
    report = gate.evaluate_development({c: rows(c, expected, plan) for c in gate.COHORTS}, plan)
    assert report['all_development_checks_passed'] is True
    for key in ('holdout_authorized_by_this_report', 'enhanced_control_enabled', 'holdout_used',
        'native_policy_execution_verified', 'native_policy_replay_verified',
        'active_model_fault_fallback_verified', 'artifact_packaged_and_replayed'):
        assert report[key] is False
    assert report['cohorts']['original']['episodes_each_method'] == 1280
    assert report['cohorts']['fresh_geometry']['episodes_each_method'] == 64
    for cohort in gate.COHORTS:
        assert set(report['cohorts'][cohort]['restricted_time_gains']) == set(gate.COMPARATORS)
        assert report['cohorts'][cohort]['restricted_time_gains']['cv_full_query_rank_l2_shared']['percentile_95_interval'] == [1., 1.]


@pytest.mark.parametrize('comparator', gate.COMPARATORS)
def test_equal_time_against_any_required_comparator_fails(plan, expected, comparator):
    data = rows('fresh_geometry', expected, plan)
    for row in data:
        if row['scoring_method'] == comparator:
            row['capture_time_seconds'] = row['restricted_safe_capture_time_seconds'] = 4.
    report = gate.summarize_cohort(data, 'fresh_geometry', expected['fresh_geometry'], plan)
    assert report['checks']['restricted_time_gain_vs_'+comparator] is False
    assert report['all_development_checks_passed'] is False


@pytest.mark.parametrize('kind', ['remove_full_query', 'remove_l2', 'duplicate', 'drop_axis', 'sample_only'])
def test_missing_methods_or_population_cannot_rescue_result(plan, expected, kind):
    data = rows('fresh_geometry', expected, plan)
    if kind == 'remove_full_query': data = [r for r in data if r['scoring_method'] != 'cv_full_query_rank_l2_shared']
    elif kind == 'remove_l2': data = [r for r in data if r['scoring_method'] != 'cv_l2_shared']
    elif kind == 'duplicate': data[-1] = data[0]
    elif kind == 'drop_axis': data = [r for r in data if r['variant'] != 'shorter_wall']
    else: data = data[:8]
    with pytest.raises(ValueError): gate.summarize_cohort(data, 'fresh_geometry', expected['fresh_geometry'], plan)


@pytest.mark.parametrize('kind', ['full_query_failure', 'legacy_l2_failure', 'latency', 'safety', 'all_primary_failed'])
def test_original_operational_safety_latency_failure_gates_retained(plan, expected, kind):
    data = rows('fresh_geometry', expected, plan)
    for row in data:
        if kind == 'full_query_failure' and row['scoring_method'] == 'cv_full_query_rank_l2_shared': row['optional_failure_calls'] = 1
        if kind == 'legacy_l2_failure' and row['scoring_method'] == 'cv_l2_shared': row['optional_failure_calls'] = 1
        if row['scoring_method'] == gate.PRIMARY:
            if kind == 'latency': row['end_to_end_control_latency_ms'] = [101.]
            if kind in ('safety', 'all_primary_failed'):
                row.update(safe_capture_success=False, capture_time_seconds=None, restricted_safe_capture_time_seconds=25.)
                if kind == 'safety': row['collision'] = True
                else: row['timeout'] = True
    report = gate.summarize_cohort(data, 'fresh_geometry', expected['fresh_geometry'], plan)
    assert report['all_development_checks_passed'] is False
    if kind == 'latency': assert report['checks']['primary_end_to_end_latency'] is False
    if kind == 'safety': assert report['additional_paired_failures']['collision'] == 64
    if kind == 'all_primary_failed': assert report['restricted_time_gains']['cv_full_query_rank_l2_shared']['group_aware_mean'] == -20.
    if kind.endswith('failure'): assert report['checks']['all_fixed_scoring_controls_available'] is False


@pytest.mark.parametrize('kind', ['primary', 'drop_comparator', 'gain_threshold', 'holdout', 'deploy', 'sample', 'source'])
def test_changed_fixed_plan_refused(plan, kind):
    changed = copy.deepcopy(plan)
    if kind == 'primary': changed['primary_method'] = 'cv_full_query_rank_l2_shared'
    elif kind == 'drop_comparator': changed['required_restricted_time_comparators'].remove('cv_l2_shared')
    elif kind == 'gain_threshold': changed['additional_matched_full_query_gain_lower95_strictly_above_seconds'] = -.1
    elif kind == 'holdout': changed['holdout_used'] = True
    elif kind == 'deploy': changed['deployment_control_enabled'] = True
    elif kind == 'sample': changed['methods'] = changed['methods'][:1]
    else: changed['v30_closed_loop_protocol_sha256'] = '0'*64
    with pytest.raises(ValueError): gate.validate_plan(changed)


def test_inherited_gates_population_bootstrap_physics_have_no_changes(plan):
    base = gate.read_json(gate.HERE.parent/'cwm_v30/closed_loop_protocol.json')
    view = gate.arithmetic_view(plan)
    assert {k for k in base if not gate.exact_tree(base[k], view[k])} == {'methods', 'primary_method'}
    assert view['development_gates']['maximum_end_to_end_control_latency_p95_ms'] == 100
    assert view['reserved_holdout_not_collected']['seed_start'] == 966010


def test_fixed8method_rotation_and_sealed_holdout(plan, expected):
    orders = {tuple(gate.method_order('original', i, plan)) for i in range(100)}
    assert len(orders) == 8 and all(set(o) == set(gate.METHODS) for o in orders)
    assert gate.method_order('fresh_geometry', 1, plan) == gate.method_order('fresh_geometry', 1, plan)
    with pytest.raises(ValueError): gate.specification('holdout', plan)
    with pytest.raises(ValueError): gate.evaluate_development({'fresh_geometry': rows('fresh_geometry', expected, plan)}, plan)


def test_l0_easier_diagnostic_cannot_hide_primary_original_regression(plan, expected):
    data = rows('original', expected, plan)
    for row in data:
        if row['scoring_method'] == gate.PRIMARY:
            time = 0. if row['level'] == 0 else 6.
            row['capture_time_seconds'] = row['restricted_safe_capture_time_seconds'] = time
    report = gate.summarize_cohort(data, 'original', expected['original'], plan)
    assert report['restricted_time_gains']['original']['group_aware_mean'] == -1.
    assert report['all_development_checks_passed'] is False
    assert set(report['all_level_absolute_and_group_aware_metrics']) == set(map(str, range(9)))
