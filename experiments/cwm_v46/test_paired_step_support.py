"""Full pinned DEVELOPMENT identities with synthetic one/two-step records."""
import copy
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent/'cwm_v30')]
import paired_step_support as support
from test_closed_loop_gates import synthetic_rows


@pytest.fixture(scope='module')
def plan(): return support.statistics.load_plan()


def fixture(root, cohort, plan):
    expected = support.statistics.specification(cohort, plan)
    rows = synthetic_rows(expected, support.statistics.arithmetic_view(plan))
    keyed = {(r['episode_index'], r['scoring_method']): r for r in rows}
    rows = [keyed[r['episode_index'], method] for r in expected
            for method in support.statistics.method_order(cohort, r['episode_index'], plan)]
    # Exercise a genuine multi-step block and nonzero counters without physics.
    rows[0].update(control_steps=2, native_control_latency_ms=[10., 11.],
        end_to_end_control_latency_ms=[12., 13.], local_solver_failure_steps=1, planner_local_solver_failures=2)
    steps = [{ 'cohort': cohort, 'episode_index': row['episode_index'], 'scoring_method': row['scoring_method'],
        'step': i+1, 'total_control_latency_ms': row['native_control_latency_ms'][i],
        'end_to_end_control_latency_ms': row['end_to_end_control_latency_ms'][i],
        'planner_local_solver_failures': 2. if row is rows[0] and i == 0 else 0.,
        'synthetic_not_native': True} for row in rows for i in range(row['control_steps'])]
    folder = root/cohort; folder.mkdir()
    write(folder/'steps.jsonl', steps)
    return rows, steps


def write(path, steps):
    path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in steps), encoding='utf8')


def test_both_full_populations_all10752_method_episodes_and_every_step_index(tmp_path, plan):
    rows, steps = {}, {}
    for cohort in support.statistics.COHORTS:
        rows[cohort], steps[cohort] = fixture(tmp_path, cohort, plan)
    index = support.index_both(tmp_path, rows)
    assert len(index['original']) == 10240 and len(index['fresh_geometry']) == 512
    assert sum(v['count'] for by_cohort in index.values() for v in by_cohort.values()) == 10754
    for cohort in support.statistics.COHORTS:
        with (tmp_path/cohort/'steps.jsonl').open('rb') as stream:
            for key, item in index[cohort].items():
                stream.seek(item['start']); observed = []
                while stream.tell() < item['stop']:
                    observed.append(json.loads(stream.readline()))
                assert stream.tell() == item['stop'] and len(observed) == item['count']
                assert all((s['episode_index'], s['scoring_method']) == key for s in observed)


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'reorder', 'extra', 'step', 'identity_bool',
    'cohort', 'method', 'native_time', 'end_time', 'time_bool', 'solver_bool', 'solver_fraction',
    'solver_negative', 'solver_total', 'duplicate_json_key'])
def test_changed_incomplete_unsupported_native_step_evidence_refused(kind, tmp_path, plan):
    rows, steps = fixture(tmp_path, 'fresh_geometry', plan)
    if kind == 'missing': steps.pop(0)
    elif kind == 'duplicate': steps[1] = copy.deepcopy(steps[0])
    elif kind == 'reorder': steps[0], steps[1] = steps[1], steps[0]
    elif kind == 'extra': steps.append(copy.deepcopy(steps[-1]))
    elif kind == 'step': steps[0]['step'] = True
    elif kind == 'identity_bool': steps[0]['episode_index'] = True
    elif kind == 'cohort': steps[0]['cohort'] = 'original'
    elif kind == 'method': steps[0]['scoring_method'] = 'best_seed'
    elif kind == 'native_time': steps[0]['total_control_latency_ms'] += .1
    elif kind == 'end_time': steps[0]['end_to_end_control_latency_ms'] += .1
    elif kind == 'time_bool': steps[0]['total_control_latency_ms'] = True
    elif kind == 'solver_bool': steps[0]['planner_local_solver_failures'] = True
    elif kind == 'solver_fraction': steps[0]['planner_local_solver_failures'] = .5
    elif kind == 'solver_negative': steps[0]['planner_local_solver_failures'] = -1
    elif kind == 'solver_total': steps[0]['planner_local_solver_failures'] = 3
    path = tmp_path/'fresh_geometry/steps.jsonl'
    write(path, steps)
    if kind == 'duplicate_json_key':
        text = path.read_text(); path.write_text(text.replace('"step": 1', '"step": 1, "step": 1', 1), encoding='utf8')
    with pytest.raises(ValueError): support.index_cohort(tmp_path, rows, 'fresh_geometry', plan)


@pytest.mark.parametrize('kind', ['missing_row', 'reordered_rows', 'missing_method', 'wrong_group', 'holdout'])
def test_episode_or_population_changes_refused_before_step_scan(kind, tmp_path, plan):
    rows, _ = fixture(tmp_path, 'fresh_geometry', plan)
    cohort = 'fresh_geometry'
    if kind == 'missing_row': rows.pop()
    elif kind == 'reordered_rows': rows[0], rows[1] = rows[1], rows[0]
    elif kind == 'missing_method': rows = [r for r in rows if r['scoring_method'] != 'cv_l2_shared']
    elif kind == 'wrong_group': rows[0]['mirror_group_id'] = 'invented'
    else: cohort = 'holdout'
    with pytest.raises(ValueError): support.index_cohort(tmp_path, rows, cohort, plan)


def test_single_cohort_cannot_be_complete_paired_support(tmp_path, plan):
    rows, _ = fixture(tmp_path, 'fresh_geometry', plan)
    with pytest.raises(ValueError): support.index_both(tmp_path, {'fresh_geometry': rows})
