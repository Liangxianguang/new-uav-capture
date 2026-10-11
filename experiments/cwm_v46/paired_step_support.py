"""ALL10752 saved-development episode/step support, never native replay.

Checks both complete fixed cohorts and all eight methods. Retains only offsets
and per-block counts, not the entire step JSONL. No models, optimizer, physics,
fault injection, holdout, timing overrides or scientific permission.
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent/'cwm_v39'), str(HERE.parent/'cwm_v38')]
import development_statistics as statistics
from archive_io import decode_json, regular, sha


def expected_order(rows, cohort, plan):
    expected = statistics.specification(cohort, plan)
    view = statistics.arithmetic_view(plan)
    assigned = statistics.prior.validate_rows(rows, expected, view)
    order = [(r['episode_index'], method) for r in expected
             for method in statistics.method_order(cohort, r['episode_index'], plan)]
    if [(r['episode_index'], r['scoring_method']) for r in rows] != order:
        raise ValueError('Complete preregistered native method/episode rotation required')
    return order, assigned


def index_cohort(root, rows, cohort, plan):
    """Bind EVERY step to its saved episode timing samples and failure totals.

    Counts/timing support do not prove native execution or safe outcomes.
    Additional per-step diagnostics are retained in the original JSONL; a later
    full policy replayer must compare them, arrays and public call evidence.
    """
    order, assigned = expected_order(rows, cohort, plan)
    path = regular(Path(root).resolve(), cohort+'/steps.jsonl')
    before = sha(path)
    index, ordinal, count, failures, failure_steps = {}, 0, 0, 0, 0
    start = 0
    with path.open('rb') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if ordinal >= len(order):
                raise ValueError('Extra native step after the complete fixed population')
            step = decode_json(line)
            key = order[ordinal]; row = assigned[key]
            if (type(step) is not dict or step.get('cohort') != cohort or
                type(step.get('episode_index')) is not int or step['episode_index'] != key[0] or
                step.get('scoring_method') != key[1] or
                type(step.get('step')) is not int or step['step'] != count+1):
                raise ValueError('Missing/reordered/duplicate or wrong native paired step')
            if count == 0:
                start = offset
            for field, samples in (('total_control_latency_ms', row['native_control_latency_ms']),
                ('end_to_end_control_latency_ms', row['end_to_end_control_latency_ms'])):
                value = step[field]
                if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0 or
                    float(value) != samples[count]):
                    raise ValueError('Actual step timing differs from the complete saved episode samples')
            value = step['planner_local_solver_failures']
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or int(value) != value:
                raise ValueError('Finite nonnegative integer-valued native solver counter required')
            failures += int(value); failure_steps += value > 0
            count += 1
            if count == row['control_steps']:
                if (failures != row['planner_local_solver_failures'] or
                    failure_steps != row['local_solver_failure_steps']):
                    raise ValueError('All native failure steps/totals must match the episode')
                index[key] = {'start': start, 'stop': stream.tell(), 'count': count}
                ordinal += 1; count = failures = failure_steps = 0
    if ordinal != len(order) or count or list(index) != order:
        raise ValueError('Incomplete full native step support')
    if sha(path) != before:
        raise ValueError('Native steps changed during complete indexing')
    return index


def index_both(root, rows_by_cohort):
    """Both fixed cohorts mandatory. Offset indexes are not a certificate."""
    if type(rows_by_cohort) is not dict or set(rows_by_cohort) != set(statistics.COHORTS):
        raise ValueError('BOTH original1280 and fresh64 development cohorts required')
    plan = statistics.load_plan()
    return {cohort: index_cohort(root, rows_by_cohort[cohort], cohort, plan)
            for cohort in statistics.COHORTS}
