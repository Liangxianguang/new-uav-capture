import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import audit_full_baseline as audit


def test_npz_full_numeric_bytes_not_zip_container_timestamps(tmp_path):
    a, b = tmp_path/'a.npz', tmp_path/'b.npz'
    np.savez(a, commanded=np.zeros((2, 4, 3)), planned=np.zeros((2, 4, 8, 3)))
    np.savez_compressed(b, commanded=np.zeros((2, 4, 3)), planned=np.zeros((2, 4, 8, 3)))
    audit.array_equal(a, b)
    np.savez(b, commanded=np.zeros((2, 4, 3), dtype=np.float32), planned=np.zeros((2, 4, 8, 3)))
    with pytest.raises(ValueError, match='bytes differ'):
        audit.array_equal(a, b)


@pytest.mark.parametrize('fault', ['values', 'shape', 'missing_key', 'signed_zero'])
def test_npz_real_numeric_changes_refused(tmp_path, fault):
    a, b = tmp_path/'a.npz', tmp_path/'b.npz'
    values = {'commanded': np.zeros((2, 4, 3)), 'planned': np.zeros((2, 4, 8, 3))}
    np.savez(a, **values)
    if fault == 'values':
        values['commanded'][0, 0, 0] = 1.
    elif fault == 'shape':
        values['commanded'] = np.zeros((1, 4, 3))
    elif fault == 'missing_key':
        values.pop('planned')
    else:
        values['commanded'][0, 0, 0] = -0.
    np.savez(b, **values)
    with pytest.raises(ValueError):
        audit.array_equal(a, b)


def test_only_explicit_wall_clock_fields_excluded_diagnostics_not_safety():
    left = {'step': 1., 'planner_latency_ms': 1., 'safety_solver_success': True, 'minimum': {'native_nonfinite_float': 'positive_infinity'}}
    right = {'step': 1., 'planner_latency_ms': 2., 'safety_solver_success': True, 'minimum': float('inf')}
    audit.require_equal(left, right, audit.STEP_TIMING, 'step')
    right['safety_solver_success'] = False
    with pytest.raises(ValueError, match='safety_solver_success'):
        audit.require_equal(left, right, audit.STEP_TIMING, 'step')
    right = {**left, 'new_latency_ms': 3.}
    with pytest.raises(ValueError, match='new_latency_ms'):
        audit.require_equal(left, right, audit.STEP_TIMING, 'step')


@pytest.mark.parametrize('left,right', [(True, 1), (1., 1), (0., -0.)])
def test_native_json_scalar_type_and_sign_not_hidden_by_python_equality(left, right):
    with pytest.raises(ValueError, match='native'):
        audit.require_equal({'native': left}, {'native': right}, frozenset(), 'diagnostics')


def test_nested_native_integer_keys_follow_actual_json_emission_not_python_mapping():
    audit.require_equal({'native': {'1': True}}, {'native': {1: np.bool_(True)}}, frozenset(), 'diagnostics')


@pytest.mark.parametrize('name', ['../outside', '/absolute', 'C:/outside'])
def test_evidence_inventory_paths_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError):
        audit.verify_inventory(tmp_path, {name: '0'*64})


def test_inventory_rechecks_hashes_not_existence_only(tmp_path):
    target = tmp_path/'evidence.json'
    target.write_bytes(b'original')
    hashes = {'evidence.json': audit.sha(target)}
    audit.verify_inventory(tmp_path, hashes)
    target.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash/path'):
        audit.verify_inventory(tmp_path, hashes)


def test_streamed_digest_matches_full_sha_with_multiple_chunks(tmp_path):
    path = tmp_path/'multi.bin'
    raw = b'original-evidence'*160000
    path.write_bytes(raw)
    assert audit.sha(path) == hashlib.sha256(raw).hexdigest()


def test_step_index_is_complete_and_decodes_only_requested_episode(tmp_path):
    records = [{'episode_index': i, 'level': 1, 'variant': 'v'} for i in (10, 20)]
    rows = [{'episode_index': r['episode_index'], 'level': 1, 'variant': 'v', 'step': s, 'diagnostics': [1, 2, 3]}
            for r in records for s in (1., 2.)]
    path = tmp_path/'steps.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf8')
    index = audit.index_steps(path, records)
    assert list(index) == [10, 20]
    assert all(v['count'] == 2 for v in index.values())
    assert all(set(v) == {'start', 'stop', 'count'} for v in index.values())
    assert audit.episode_steps(path, index[20]) == rows[2:]


@pytest.mark.parametrize('fault', ['missing', 'reorder', 'interleave', 'repeat_step', 'wrong_level'])
def test_streaming_cannot_drop_reorder_or_interleave_any_native_steps(tmp_path, fault):
    records = [{'episode_index': i, 'level': 1, 'variant': 'v'} for i in (10, 20)]
    rows = [{'episode_index': r['episode_index'], 'level': 1, 'variant': 'v', 'step': s}
            for r in records for s in (1., 2.)]
    if fault == 'missing':
        rows = rows[:2]
    elif fault == 'reorder':
        rows = rows[2:]+rows[:2]
    elif fault == 'interleave':
        rows = [rows[0], rows[2], rows[1], rows[3]]
    elif fault == 'repeat_step':
        rows[1]['step'] = 1.
    else:
        rows[0]['level'] = 2
    path = tmp_path/'steps.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf8')
    with pytest.raises(ValueError):
        audit.index_steps(path, records)


@pytest.fixture
def full_fixture(tmp_path, monkeypatch):
    """Synthetic inventory ONLY, explicitly not original-engine replay proof."""
    protocol = audit.read_json(HERE/'closed_loop_protocol.json')
    capsule = ROOT/'experiments/cwm_v1/baseline/capsule.zip'
    records = audit.original_population(capsule, protocol)
    monkeypatch.setattr(audit, 'original_population', lambda *_: copy.deepcopy(records))
    prereg = {'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'remote': audit.REMOTE, 'branch': audit.BRANCH, 'exact_pushed_sources_verified': True}
    source_hashes = {}
    for name in ('closed_loop_protocol.json', 'closed_loop_populations.py', 'baseline_full_levels.py'):
        relative = 'experiments/cwm_v30/'+name
        path = tmp_path/'source'/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT/relative).read_bytes())
        source_hashes[relative] = audit.sha(path)
    rows, steps = [], []
    for r in records:
        path = tmp_path/'trajectories'/f"level{r['level']}_{r['episode_index']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture-not-native-npz')
        command = path.with_suffix('.commands.npz')
        command.write_bytes(b'fixture-not-native-command-npz')
        rows.append({k: r[k] for k in ('episode_index', 'level', 'variant', 'mirror_group_id', 'mirror_pair_member')})
        rows[-1].update(safe_capture_success=False, capture_time_seconds=None, collision=False, boundary_violation=False,
            target_invalid_episode=False, timeout=True, restricted_safe_capture_time_seconds=25.,
            trajectory_sha256=audit.sha(path), commands_sha256=audit.sha(command))
        steps.append({'level': r['level'], 'variant': r['variant'], 'episode_index': r['episode_index'], 'step': 1.})
    audit.write_json(tmp_path/'scenes.json', records)
    audit.write_json(tmp_path/'preregistration.json', prereg)
    for name, values in (('episodes.jsonl', rows), ('steps.jsonl', steps)):
        (tmp_path/name).write_text(''.join(json.dumps(r)+'\n' for r in values), encoding='utf8')
    by_level = audit.summarize(rows)
    report = {'status': audit.STATUS, 'protocol': protocol, 'episodes': 1280, 'archived_original_episodes_reproduced': 256,
        'warmup_level_excluded_from_primary_macro': True, 'enhanced_control_enabled': False, 'holdout_used': False,
        'new_model_trained': False, 'baseline_capsule_sha256': audit.sha(capsule), 'preregistration': prereg,
        'source_hashes': source_hashes, 'scene_sha256': audit.sha(tmp_path/'scenes.json'),
        'episodes_sha256': audit.sha(tmp_path/'episodes.jsonl'), 'steps_sha256': audit.sha(tmp_path/'steps.jsonl'),
        'by_level': by_level, 'original8Level_equal_macro': audit.macro(by_level)}
    audit.write_json(tmp_path/'summary.json', report)
    return tmp_path, capsule, report


def test_complete_inventory_validation_does_not_claim_engine_reexecution(full_fixture):
    root, capsule, _ = full_fixture
    report, records, rows, steps, hashes = audit.validate_saved(root, capsule)
    assert len(records) == len(rows) == len(steps) == 1280
    assert len(hashes) == 2568
    assert report['status'] == audit.STATUS  # Still pending REAL independent replay.


@pytest.mark.parametrize('key,value', [('episodes', 1279), ('holdout_used', True),
    ('enhanced_control_enabled', True), ('archived_original_episodes_reproduced', 255)])
def test_incomplete_or_active_report_cannot_authorize_replay(full_fixture, key, value):
    root, capsule, report = full_fixture
    report[key] = value
    audit.write_json(root/'summary.json', report)
    with pytest.raises(ValueError, match='Complete original'):
        audit.validate_saved(root, capsule)


@pytest.mark.parametrize('fault', ['drop_episode', 'group_changed', 'biased_time', 'level_aggregate', 'step_order', 'source_snapshot', 'command_bytes'])
def test_honestly_rehashed_saved_evidence_forgery_refused(full_fixture, fault):
    root, capsule, report = full_fixture
    if fault in ('drop_episode', 'group_changed', 'biased_time'):
        rows = audit.read_lines(root/'episodes.jsonl')
        if fault == 'drop_episode':
            rows.pop()
        elif fault == 'group_changed':
            rows[0]['mirror_group_id'] = 'fabricated'
        else:
            rows[0]['restricted_safe_capture_time_seconds'] = 0.
        (root/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf8')
        report['episodes_sha256'] = audit.sha(root/'episodes.jsonl')
    elif fault == 'level_aggregate':
        report['by_level']['1']['variant_equal_safe_capture_success_rate'] = 1.
    elif fault == 'step_order':
        values = audit.read_lines(root/'steps.jsonl')
        values[0]['step'] = 2.
        (root/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in values), encoding='utf8')
        report['steps_sha256'] = audit.sha(root/'steps.jsonl')
    elif fault == 'source_snapshot':
        (root/'source/experiments/cwm_v30/baseline_full_levels.py').write_bytes(b'forged')
    else:
        row = audit.read_lines(root/'episodes.jsonl')[0]
        (root/'trajectories'/f"level{row['level']}_{row['episode_index']}.commands.npz").write_bytes(b'forged')
    audit.write_json(root/'summary.json', report)
    with pytest.raises(ValueError):
        audit.validate_saved(root, capsule)
