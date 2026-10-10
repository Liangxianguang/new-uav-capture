"""Synthetic gate arithmetic, not native model/capture/latency qualification."""
import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import closed_loop_gates as gates


@pytest.fixture(scope='module')
def protocol():
    return json.loads((HERE/'closed_loop_protocol.json').read_bytes())


@pytest.fixture(scope='module')
def expected(protocol):
    return {cohort: gates.specification(cohort, protocol) for cohort in gates.COHORTS}


def synthetic_rows(expected, protocol):
    return [{**r, 'scoring_method': m, 'safe_capture_success': True, 'collision': False,
        'boundary_violation': False, 'timeout': False, 'target_invalid_episode': False,
        'capture_time_seconds': 4. if m == protocol['primary_method'] else 5.,
        'restricted_safe_capture_time_seconds': 4. if m == protocol['primary_method'] else 5.,
        'control_steps': 1, 'local_solver_failure_steps': 0, 'planner_local_solver_failures': 0,
        'optional_failure_calls': 0, 'native_control_latency_ms': [10.], 'end_to_end_control_latency_ms': [12.]}
        for r in expected for m in protocol['methods']]


def test_complete_pure_arithmetic_never_authorizes_holdout_or_deploy(protocol, expected):
    rows = {c: synthetic_rows(e, protocol) for c, e in expected.items()}
    report = gates.evaluate_development(rows, protocol)
    assert report['all_development_checks_passed'] is True
    for name in ('holdout_authorized_by_this_report', 'enhanced_control_enabled', 'holdout_used',
        'native_policy_replay_verified', 'artifact_packaged_and_replayed'):
        assert report[name] is False
    assert report['cohorts']['original']['episodes_each_method'] == 1280
    assert report['cohorts']['fresh_geometry']['episodes_each_method'] == 64
    assert report['cohorts']['original']['restricted_time_gains']['original']['percentile_95_interval'] == [1., 1.]
    assert set(report['cohorts']['original']['all_level_absolute_and_group_aware_metrics']) == set(map(str, range(9)))


@pytest.mark.parametrize('cohort', ['original', 'fresh_geometry'])
def test_missing_method_or_duplicate_episode_refused(protocol, expected, cohort):
    rows = synthetic_rows(expected[cohort], protocol)
    with pytest.raises(ValueError, match='Every preassigned'):
        gates.validate_rows(rows[:-1], expected[cohort], protocol)
    rows[-1] = rows[0]
    with pytest.raises(ValueError, match='duplicate'):
        gates.validate_rows(rows, expected[cohort], protocol)


@pytest.mark.parametrize('key,value', [('level', 99), ('mirror_group_id', 'fabricated'),
    ('mirror_pair_member', 'fabricated'), ('scoring_method', 'best_seed'), ('safe_capture_success', 1),
    ('capture_time_seconds', True), ('timeout', True), ('restricted_safe_capture_time_seconds', float('nan')),
    ('control_steps', 0), ('local_solver_failure_steps', 2), ('planner_local_solver_failures', -1),
    ('native_control_latency_ms', [float('nan')]), ('end_to_end_control_latency_ms', [9.]),
    ('end_to_end_control_latency_ms', [12., 12.]), ('native_control_latency_ms', [0.]),
    ('optional_failure_calls', -1)])
def test_wrong_identity_physical_or_complete_timing_support_refused(protocol, expected, key, value):
    rows = synthetic_rows(expected['fresh_geometry'], protocol)
    rows[0][key] = value
    with pytest.raises(ValueError):
        gates.validate_rows(rows, expected['fresh_geometry'], protocol)


def test_strict_gain_zero_cannot_pass(protocol, expected):
    rows = synthetic_rows(expected['fresh_geometry'], protocol)
    for row in rows:
        row['capture_time_seconds'] = row['restricted_safe_capture_time_seconds'] = 5.
    result = gates.summarize_cohort(rows, 'fresh_geometry', expected['fresh_geometry'], protocol)
    assert result['all_development_checks_passed'] is False
    assert result['checks']['restricted_time_gain_vs_original'] is False


def test_all_failed_cases_kept_capped_not_success_only(protocol, expected):
    rows = synthetic_rows(expected['fresh_geometry'], protocol)
    for row in rows:
        if row['scoring_method'] == protocol['primary_method']:
            row.update(safe_capture_success=False, timeout=True, capture_time_seconds=None,
                restricted_safe_capture_time_seconds=25.)
    result = gates.summarize_cohort(rows, 'fresh_geometry', expected['fresh_geometry'], protocol)
    assert result['restricted_time_gains']['original']['group_aware_mean'] == -20.
    assert result['absolute_outcomes_including_L0_if_present'][protocol['primary_method']]['timeout'] == 64
    assert result['all_development_checks_passed'] is False


def test_additional_paired_failures_cannot_cancel_between_episodes(protocol, expected):
    rows = synthetic_rows(expected['fresh_geometry'], protocol)
    by_id = {(r['episode_index'], r['scoring_method']): r for r in rows}
    first, second = expected['fresh_geometry'][:2]
    for record, method in ((first, 'original'), (second, protocol['primary_method'])):
        row = by_id[record['episode_index'], method]
        row.update(collision=True, safe_capture_success=False, capture_time_seconds=None,
            restricted_safe_capture_time_seconds=25., local_solver_failure_steps=1, planner_local_solver_failures=1)
    result = gates.summarize_cohort(rows, 'fresh_geometry', expected['fresh_geometry'], protocol)
    assert result['additional_paired_failures']['collision'] == 1
    assert result['additional_paired_failures']['local_solver_failure_steps'] == 1
    assert result['checks']['no_additional_collision'] is False


def test_fixed_latency_limit_and_optional_failure_not_relaxed(protocol, expected):
    rows = synthetic_rows(expected['fresh_geometry'], protocol)
    for row in rows:
        if row['scoring_method'] == protocol['primary_method']:
            row['end_to_end_control_latency_ms'] = [101.]
    rows[0]['optional_failure_calls'] = 1
    result = gates.summarize_cohort(rows, 'fresh_geometry', expected['fresh_geometry'], protocol)
    assert result['checks']['primary_end_to_end_latency'] is False
    assert result['checks']['all_fixed_scoring_controls_available'] is False
    assert result['end_to_end_latency'][protocol['primary_method']]['steps'] == 64
    assert result['end_to_end_latency'][protocol['primary_method']]['first_and_warmup_cycles_included'] is True


def test_shared_fresh_layout_bootstrap_preserves_axis_correlation(protocol):
    parts = {(None, 'a'): {'g1': 1., 'g2': -1.}, (None, 'b'): {'g1': -1., 'g2': 1.}}
    result = gates.statistic(parts, 'fresh_geometry', protocol)
    assert result['group_aware_mean'] == 0.
    assert result['percentile_95_interval'] == [0., 0.]


def test_level_variant_groups_equal_not_episode_or_window_weights(protocol):
    parts = {(7, 'fast'): {'singleton': 2., 'mirror': 10.}, (7, 'agile'): {'group': 25.}}
    result = gates.statistic(parts, 'original', protocol)
    assert result['group_aware_mean'] == 15.5
    # An easier Level0 diagnostic must not raise the primary8Level statistic.
    records = [{'episode_index': 0, 'level': 0, 'variant': 'warmup', 'mirror_group_id': 'g0'},
               {'episode_index': 1, 'level': 1, 'variant': 'v1', 'mirror_group_id': 'g1'}]
    assert set(gates.strata(records, 'original', {0: 100., 1: 1.})) == {(1, 'v1')}


def test_holdout_stats_or_single_cohort_cannot_be_requested(protocol, expected):
    with pytest.raises(ValueError, match='sealed'):
        gates.specification('holdout', protocol)
    with pytest.raises(ValueError, match='BOTH'):
        gates.evaluate_development({'original': synthetic_rows(expected['original'], protocol)}, protocol)


def test_fixed_rotation_is_complete_repeatable_not_best_method(protocol):
    one = gates.method_order('original', 123, protocol)
    assert one == gates.method_order('original', 123, protocol)
    assert sorted(one) == sorted(protocol['methods'])
    assert len({tuple(gates.method_order('original', i, protocol)) for i in range(100)}) == 7
    with pytest.raises(ValueError):
        gates.method_order('holdout', 123, protocol)
