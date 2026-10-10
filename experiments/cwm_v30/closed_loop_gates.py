"""Frozen group-aware DEVELOPMENT closed-loop statistics, never deployment.

Pure saved-outcome calculations are not native execution, archive replay or a
holdout authorization. Both complete cohorts and every fixed control required.
"""
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from closed_loop_populations import original_population, validate_protocol, restricted_safe_capture_time

COHORTS = ('original', 'fresh_geometry')
BOOL_FIELDS = ('safe_capture_success', 'collision', 'boundary_violation', 'timeout', 'target_invalid_episode')


def specification(cohort, protocol):
    validate_protocol(protocol)
    if cohort == 'original':
        records = original_population(ROOT/'experiments/cwm_v1/baseline/capsule.zip', protocol)
        return [{k: r[k] for k in ('episode_index', 'level', 'variant', 'mirror_group_id', 'mirror_pair_member')} for r in records]
    if cohort != 'fresh_geometry':
        raise ValueError('Development cohorts only; holdout remains sealed')
    spec = protocol['fresh_new_scene_development']
    return [{'episode_index': v['seed_start']+2*g+m, 'level': None, 'variant': v['name'],
        'mirror_group_id': f"cwm-newscene-{spec['layout_seed_start']+g}", 'mirror_pair_member': member}
        for v in spec['variants'] for g in range(spec['groups']) for m, member in enumerate(('upper', 'lower'))]


def method_order(cohort, episode_index, protocol):
    if cohort not in COHORTS or type(episode_index) is not int:
        raise ValueError('Fixed development cohort/episode identity required')
    methods = protocol['methods']
    offset = int(hashlib.sha256(f'{cohort}/{episode_index}'.encode()).hexdigest(), 16) % len(methods)
    return methods[offset:]+methods[:offset]


def finite_positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_rows(rows, expected, protocol):
    methods = protocol['methods']
    identities = {r['episode_index']: r for r in expected}
    if len(identities) != len(expected) or len(rows) != len(expected)*len(methods):
        raise ValueError('Every preassigned episode and fixed method required')
    index = {}
    for row in rows:
        identifier, method = row['episode_index'], row['scoring_method']
        if type(identifier) is not int or identifier not in identities or method not in methods or (identifier, method) in index:
            raise ValueError('Unexpected/duplicate closed-loop identity or scoring method')
        if any(row[k] != identities[identifier][k] for k in ('level', 'variant', 'mirror_group_id', 'mirror_pair_member')):
            raise ValueError('Original/fresh Level, group, variant or mirror changed')
        if any(type(row[k]) is not bool for k in BOOL_FIELDS):
            raise ValueError('Exact physical outcome flags required')
        if row['safe_capture_success'] and (row['timeout'] or type(row['capture_time_seconds']) not in (int, float)):
            raise ValueError('Safe capture cannot be timeout or use a boolean time')
        if row['restricted_safe_capture_time_seconds'] != restricted_safe_capture_time(row):
            raise ValueError('Failed/unsafe episodes cannot be dropped or timed as success')
        count = row['control_steps']
        if type(count) is not int or not 1 <= count <= 250:
            raise ValueError('Complete actual control-step support required')
        for name in ('local_solver_failure_steps', 'planner_local_solver_failures', 'optional_failure_calls'):
            if type(row[name]) is not int or row[name] < 0:
                raise ValueError('Actual native/optional failure counters required')
        if row['local_solver_failure_steps'] > count:
            raise ValueError('Local failure-step support inconsistent')
        if row['planner_local_solver_failures'] < row['local_solver_failure_steps']:
            raise ValueError('Native failure count cannot be smaller than failure steps')
        native, end = row['native_control_latency_ms'], row['end_to_end_control_latency_ms']
        if (type(native) is not list or type(end) is not list or len(native) != count or len(end) != count or any(not finite_positive(v) for v in native+end) or
            any(b+1e-6 < a for a, b in zip(native, end))):
            raise ValueError('Full positive native/end-to-end timing support required; no subtraction')
        index[identifier, method] = row
    if set(index) != {(r['episode_index'], m) for r in expected for m in methods}:
        raise ValueError('Incomplete paired policy population')
    return index


def strata(expected, cohort, values):
    grouped = defaultdict(lambda: defaultdict(list))
    for r in expected:
        if cohort == 'original' and r['level'] == 0:
            continue
        grouped[(r['level'], r['variant'])][r['mirror_group_id']].append(values[r['episode_index']])
    return {key: {g: float(np.mean(v)) for g, v in sorted(groups.items())} for key, groups in sorted(grouped.items(), key=str)}


def statistic(parts, cohort, protocol, subset_level=None, subset_variant=None):
    """Paired group bootstrap; all fresh axes share each resampled layout."""
    if subset_level is not None:
        parts = {k: v for k, v in parts.items() if k[0] == subset_level}
    if subset_variant is not None:
        parts = {k: v for k, v in parts.items() if k[1] == subset_variant}
    if not parts:
        raise ValueError('Nonempty complete stratified group support required')
    draws = protocol['bootstrap']['draws']
    rng = np.random.default_rng(protocol['bootstrap']['seed'])
    sampled, means = {}, {}
    common = None
    if cohort == 'fresh_geometry':
        groups = sorted(next(iter(parts.values())))
        if any(sorted(v) != groups for v in parts.values()):
            raise ValueError('Fresh axes require identical shared layout groups')
        common = rng.integers(0, len(groups), size=(draws, len(groups)))
    for key, group_values in parts.items():
        values = np.asarray([v for _, v in sorted(group_values.items())], dtype=np.float64)
        if not len(values) or not np.isfinite(values).all():
            raise ValueError('Finite complete group outcomes required')
        choices = common if common is not None else rng.integers(0, len(values), size=(draws, len(values)))
        sampled[key] = values[choices].mean(axis=1)
        means[key] = float(values.mean())
    if cohort == 'original':
        levels = sorted({k[0] for k in parts})
        point = np.mean([np.mean([v for k, v in means.items() if k[0] == level]) for level in levels])
        distribution = np.mean([np.mean([v for k, v in sampled.items() if k[0] == level], axis=0) for level in levels], axis=0)
    else:
        point = np.mean(list(means.values()))
        distribution = np.mean(list(sampled.values()), axis=0)
    return {'group_aware_mean': float(point), 'percentile_95_interval': np.percentile(distribution, [2.5, 97.5]).tolist(),
        'strata': {str(k): {'groups': len(v), 'group_equal_mean': means[k]} for k, v in parts.items()},
        'interpretation': 'Fixed development population group-resampling descriptive interval, not untouched holdout or formal safety/generalization proof.'}


def variant_group_mean(records, index, method, field):
    variants = defaultdict(lambda: defaultdict(list))
    for record in records:
        variants[record['variant']][record['mirror_group_id']].append(float(index[record['episode_index'], method][field]))
    return float(np.mean([np.mean([np.mean(values) for values in groups.values()]) for groups in variants.values()]))


def summarize_cohort(rows, cohort, expected, protocol):
    index = validate_rows(rows, expected, protocol)
    primary = protocol['primary_method']
    gates = protocol['development_gates']
    checks, gains, rates, noninferiority = {}, {}, {}, {}
    for reference in ('original', 'own_motion_shared', 'cv_l2_shared'):
        differences = {r['episode_index']: index[r['episode_index'], reference]['restricted_safe_capture_time_seconds']-
            index[r['episode_index'], primary]['restricted_safe_capture_time_seconds'] for r in expected}
        gains[reference] = statistic(strata(expected, cohort, differences), cohort, protocol)
        checks['restricted_time_gain_vs_'+reference] = gains[reference]['percentile_95_interval'][0] > gates['restricted_safe_capture_time_gain_lower95_vs_original_own_motion_and_l2_strictly_above_seconds']
    for method in protocol['methods']:
        rates[method] = {}
        for metric in BOOL_FIELDS+('restricted_safe_capture_time_seconds',):
            values = {r['episode_index']: float(index[r['episode_index'], method][metric]) for r in expected}
            parts = strata(expected, cohort, values)
            rates[method][metric] = statistic(parts, cohort, protocol)
            rates[method][metric]['absolute_episode_mean_including_L0_if_present'] = float(np.mean(list(values.values())))
    differences = {r['episode_index']: float(index[r['episode_index'], primary]['safe_capture_success'])-
        float(index[r['episode_index'], 'original']['safe_capture_success']) for r in expected}
    parts = strata(expected, cohort, differences)
    if cohort == 'original':
        for level in range(1, 9):
            value = statistic(parts, cohort, protocol, subset_level=level)
            noninferiority[str(level)] = value
            checks[f'level{level}_safe_capture_noninferiority'] = value['percentile_95_interval'][0] >= gates['original_each_level_safe_capture_rate_gain_lower95_minimum']
    else:
        for variant in protocol['fresh_new_scene_development']['variants']:
            name = variant['name']
            value = statistic(parts, cohort, protocol, subset_variant=name)
            noninferiority[name] = value
            checks[name+'_safe_capture_noninferiority'] = value['percentile_95_interval'][0] >= gates['new_scene_each_variant_safe_capture_rate_gain_lower95_minimum']
    # Pair-wise additional failures cannot cancel regressions at another episode.
    additional = {}
    for field, gate in (('collision', 'maximum_new_collision_episodes_vs_paired_original'),
        ('boundary_violation', 'maximum_new_boundary_violation_episodes_vs_paired_original'),
        ('target_invalid_episode', 'maximum_new_target_invalid_episodes_vs_paired_original'),
        ('local_solver_failure_steps', 'maximum_new_local_solver_failure_steps_vs_paired_original')):
        count = sum(max(0, int(index[r['episode_index'], primary][field])-int(index[r['episode_index'], 'original'][field])) for r in expected)
        additional[field] = count
        checks['no_additional_'+field] = count <= gates[gate]
    latency = {}
    for method in protocol['methods']:
        samples = [v for r in expected for v in index[r['episode_index'], method]['end_to_end_control_latency_ms']]
        latency[method] = {'steps': len(samples), 'p50_ms': float(np.percentile(samples, 50)),
            'p95_ms': float(np.percentile(samples, 95)), 'p99_ms': float(np.percentile(samples, 99)),
            'maximum_ms': float(max(samples)), 'first_and_warmup_cycles_included': True}
    checks['primary_end_to_end_latency'] = latency[primary]['p95_ms'] <= gates['maximum_end_to_end_control_latency_p95_ms']
    # Operational validity, not a new scientific threshold: an unavailable model
    # must not silently turn a control into the original and still qualify CWM.
    optional_failures = {m: sum(index[r['episode_index'], m]['optional_failure_calls'] for r in expected) for m in protocol['methods']}
    checks['all_fixed_scoring_controls_available'] = not any(optional_failures.values())
    absolute = {m: {'episodes': len(expected), **{k: sum(int(index[r['episode_index'], m][k]) for r in expected) for k in BOOL_FIELDS},
        'local_solver_failure_steps': sum(index[r['episode_index'], m]['local_solver_failure_steps'] for r in expected),
        'planner_local_solver_failures': sum(index[r['episode_index'], m]['planner_local_solver_failures'] for r in expected)} for m in protocol['methods']}
    # Report all original variants including L0 and both L7 variants separately.
    per_variant = {}
    for variant in sorted({r['variant'] for r in expected}):
        selected = [r for r in expected if r['variant'] == variant]
        per_variant[variant] = {m: {'episodes': len(selected),
            **{k+'_rate': float(np.mean([index[r['episode_index'], m][k] for r in selected])) for k in BOOL_FIELDS},
            'restricted_safe_capture_time_seconds': float(np.mean([index[r['episode_index'], m]['restricted_safe_capture_time_seconds'] for r in selected])),
            'group_equal_restricted_safe_capture_time_seconds': variant_group_mean(selected, index, m, 'restricted_safe_capture_time_seconds'),
            'local_solver_failure_steps': sum(index[r['episode_index'], m]['local_solver_failure_steps'] for r in selected),
            'planner_local_solver_failures': sum(index[r['episode_index'], m]['planner_local_solver_failures'] for r in selected),
            'optional_failure_calls': sum(index[r['episode_index'], m]['optional_failure_calls'] for r in selected),
            'end_to_end_latency_p95_ms': float(np.percentile([v for r in selected for v in index[r['episode_index'], m]['end_to_end_control_latency_ms']], 95))}
            for m in protocol['methods']}
    per_level = {}
    if cohort == 'original':
        for level in range(9):
            selected = [r for r in expected if r['level'] == level]
            per_level[str(level)] = {m: {'episodes': len(selected),
                **{k+'_absolute_episode_rate': float(np.mean([index[r['episode_index'], m][k] for r in selected])) for k in BOOL_FIELDS},
                **{k+'_variant_and_group_equal_mean': variant_group_mean(selected, index, m, k)
                   for k in BOOL_FIELDS+('restricted_safe_capture_time_seconds',)}}
                for m in protocol['methods']}
    return {'cohort': cohort, 'episodes_each_method': len(expected), 'checks': {k: bool(v) for k, v in checks.items()},
        'all_development_checks_passed': bool(all(checks.values())), 'restricted_time_gains': gains,
        'group_aware_method_metrics': rates, 'safe_capture_noninferiority': noninferiority,
        'additional_paired_failures': additional, 'end_to_end_latency': latency,
        'optional_failure_calls': optional_failures, 'absolute_outcomes_including_L0_if_present': absolute,
        'all_variant_absolute_metrics': per_variant, 'all_level_absolute_and_group_aware_metrics': per_level}


def evaluate_development(rows_by_cohort, protocol):
    validate_protocol(protocol)
    if set(rows_by_cohort) != set(COHORTS):
        raise ValueError('BOTH original ALL1280 and fresh ALL64 development cohorts required')
    cohorts = {c: summarize_cohort(rows_by_cohort[c], c, specification(c, protocol), protocol) for c in COHORTS}
    return {'status': 'complete_saved_development_policy_statistics_pending_native_replay', 'cohorts': cohorts,
        'all_development_checks_passed': all(c['all_development_checks_passed'] for c in cohorts.values()),
        'holdout_authorized_by_this_report': False, 'enhanced_control_enabled': False, 'holdout_used': False,
        'native_policy_replay_verified': False, 'artifact_packaged_and_replayed': False,
        'scope': 'Complete saved DEVELOPMENT outcomes only. Does not authorize holdout, active faults or deployment; requires original native source-bound execution and complete independent evidence replay.'}
