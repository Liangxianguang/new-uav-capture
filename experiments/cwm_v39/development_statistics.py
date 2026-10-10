"""Prefix-specific fixed development statistics, never native authorization.

Reuse unchanged V30 arithmetic and populations, retaining ALL old gates and
adding the matched full-query comparator. No model, environment or optimizer.
"""
import copy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent/'cwm_v38'), str(HERE.parent/'cwm_v30'), str(HERE.parent/'cwm_v28')]
from archive_io import read_json, sha
import closed_loop_gates as prior
from repeatability_identity import exact_tree

METHODS = ['original', 'original_gru_shared', 'constant_velocity_shared', 'own_motion_shared',
           'cv_motion_only_shared', 'cv_l2_shared', 'cv_full_query_rank_l2_shared', 'cv_prefix_rank_l2_shared']
COMPARATORS = ['original', 'own_motion_shared', 'cv_l2_shared', 'cv_full_query_rank_l2_shared']
PRIMARY = 'cv_prefix_rank_l2_shared'
COHORTS = ('original', 'fresh_geometry')


def load_plan():
    plan = read_json(HERE/'protocol.json')
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if not exact_tree(plan, read_json(HERE/'protocol.json')):
        raise ValueError('Exact preregistered prefix evaluation plan required')
    for relative, key in (
        ('experiments/cwm_v30/closed_loop_protocol.json', 'v30_closed_loop_protocol_sha256'),
        ('experiments/cwm_v36/protocol.json', 'v36_training_contract_sha256'),
        ('experiments/cwm_v28/training_protocol.json', 'v28_training_protocol_sha256')):
        if sha(ROOT/relative) != plan[key]:
            raise ValueError('Unchanged inherited evaluation/training contract required')
    if (plan['methods'] != METHODS or plan['primary_method'] != PRIMARY or
        plan['primary_configuration'] != 'cv_prefix_rank_l2' or
        plan['new_model_configurations'] != ['cv_motion_only', 'cv_full_query_rank_l2', 'cv_prefix_rank_l2'] or
        plan['required_restricted_time_comparators'] != COMPARATORS or
        type(plan['additional_matched_full_query_gain_lower95_strictly_above_seconds']) is not int or
        plan['additional_matched_full_query_gain_lower95_strictly_above_seconds'] != 0 or
        any(plan[k] is not False for k in ('new_training_or_tuning_allowed', 'deployment_control_enabled',
            'active_research_experiment_started', 'native_model_loader_implemented', 'native_paired_runner_implemented',
            'holdout_entry_implemented', 'holdout_used', 'prior_gate_override_allowed'))):
        raise ValueError('Fixed primary/allcontrols/default-off/noholdout contract required')
    base = read_json(HERE.parent/'cwm_v30/closed_loop_protocol.json')
    prior.validate_protocol(base)


def arithmetic_view(plan):
    validate_plan(plan)
    base = read_json(HERE.parent/'cwm_v30/closed_loop_protocol.json')
    view = copy.deepcopy(base)
    # These are the ONLY two fields changed in the view consumed by unchanged
    # V30 row validation/aggregation/gates. V39 owns the actual model provenance
    # and future authorization; unused old descriptive strings are not reused.
    view['methods'] = list(plan['methods'])
    view['primary_method'] = plan['primary_method']
    return view


def specification(cohort, plan):
    validate_plan(plan)
    if cohort not in COHORTS:
        raise ValueError('Development populations only; holdout remains sealed')
    base = read_json(HERE.parent/'cwm_v30/closed_loop_protocol.json')
    # This enumerates pinned DEVELOPMENT identities, not a physical rollout or
    # new/holdout scene generation. All1280 original +64 fresh identities remain.
    return prior.specification(cohort, base)


def method_order(cohort, episode_index, plan):
    return prior.method_order(cohort, episode_index, arithmetic_view(plan))


def summarize_cohort(rows, cohort, expected, plan):
    view = arithmetic_view(plan)
    if cohort not in COHORTS or not exact_tree(expected, specification(cohort, plan)):
        raise ValueError('Complete preassigned development population required')
    report = prior.summarize_cohort(rows, cohort, expected, view)
    # Unchanged V30 validation also requires every eighth control and all its
    # native timing/failure support. The new check augments the old three gains.
    index = prior.validate_rows(rows, expected, view)
    reference = 'cv_full_query_rank_l2_shared'
    values = {r['episode_index']: index[r['episode_index'], reference]['restricted_safe_capture_time_seconds']-
        index[r['episode_index'], PRIMARY]['restricted_safe_capture_time_seconds'] for r in expected}
    gain = prior.statistic(prior.strata(expected, cohort, values), cohort, view)
    report['restricted_time_gains'][reference] = gain
    report['checks']['restricted_time_gain_vs_'+reference] = bool(
        gain['percentile_95_interval'][0] > plan['additional_matched_full_query_gain_lower95_strictly_above_seconds'])
    report['all_development_checks_passed'] = bool(all(report['checks'].values()))
    report['legacy_l2_interpretation'] = plan['legacy_l2_control']['interpretation']
    return report


def evaluate_development(rows_by_cohort, plan):
    validate_plan(plan)
    if set(rows_by_cohort) != set(COHORTS):
        raise ValueError('BOTH complete original1280/fresh64 development cohorts required')
    cohorts = {c: summarize_cohort(rows_by_cohort[c], c, specification(c, plan), plan) for c in COHORTS}
    return {'status': 'complete_saved_prefix_development_statistics_pending_native_execution_replay',
        'protocol': plan, 'protocol_sha256': sha(HERE/'protocol.json'), 'cohorts': cohorts,
        'all_development_checks_passed': bool(all(c['all_development_checks_passed'] for c in cohorts.values())),
        'holdout_authorized_by_this_report': False, 'enhanced_control_enabled': False, 'holdout_used': False,
        'native_policy_execution_verified': False, 'native_policy_replay_verified': False,
        'active_model_fault_fallback_verified': False, 'artifact_packaged_and_replayed': False,
        'scope': 'Saved DEVELOPMENT arithmetic only; no model reconstruction, physics, active policy, holdout or deployment authorization. All native/source/model/certificate/independent replay and fault evidence remain required.'}
