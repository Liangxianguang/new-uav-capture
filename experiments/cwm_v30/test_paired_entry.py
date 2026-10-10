"""Negative entry/unit checks only; no fake qualified-model success evidence."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import paired_closed_loop as entry
from measured_original_cycle import MeasuredOriginalCycle


def test_fresh_generation_refused_before_generator_or_env_access(monkeypatch):
    def forbidden(*args):
        raise AssertionError('Fresh scene generator must not run before qualification')
    monkeypatch.setattr(entry, 'scene_records', forbidden)
    protocol = json.loads((HERE/'closed_loop_protocol.json').read_bytes())
    with pytest.raises(ValueError, match='BEFORE'):
        entry.fresh_records(object(), protocol, SimpleNamespace(authorized=False, deployment_control_eligible=False))


def test_captured_public_ordinal_added_without_changing_action_or_legacy_source(monkeypatch):
    call = {'agent': 1, 'selected': np.zeros((8, 3))}
    boundary = entry.CompleteSequentialEntry.__new__(entry.CompleteSequentialEntry)
    boundary.pending = [(4, call, {'evidence': None})]
    captured = []
    monkeypatch.setattr(entry.SequentialResearchEntry, 'flush', lambda self: captured.append(self.pending[0][1]['ordinal']))
    boundary.flush()
    assert captured == [4]
    assert np.array_equal(call['selected'], np.zeros((8, 3)))


def test_failed_primary_does_not_construct_active_entry_or_generate_scenes(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, 'published_sources', lambda: {'commit': 'fixture-negative-refusal-only'})
    monkeypatch.setattr(entry, 'Baseline', lambda *_: object())
    monkeypatch.setattr(entry, 'load_bundle', lambda *_: SimpleNamespace(authorized=False,
        status='offline_qualification_failed', archive_sha256='fixture', certificate_sha256='fixture'))
    def forbidden(*args):
        raise AssertionError('No active entry or fresh generation on failed primary')
    monkeypatch.setattr(entry, 'fresh_records', forbidden)
    monkeypatch.setattr(entry, 'CompleteSequentialEntry', forbidden)
    certificate = tmp_path/'certificate.json'
    certificate.write_text('{}', encoding='utf8')
    output = tmp_path/'out'
    entry.run(tmp_path/'artifact.zip', certificate, output)
    report = json.loads((output/'refusal.json').read_bytes())
    assert report['research_actions_exercised'] is False
    assert report['fresh_scenes_generated'] is False and report['holdout_used'] is False
    assert not (output/'scenes.json').exists()


def fake_base():
    class Env:
        def step(self, action):
            return ('fixture', action)
    def evaluator(env, action):
        control_started = time.perf_counter()
        result = env.step(action)
        return result
    return SimpleNamespace(evaluator=SimpleNamespace(CaptureRadiusPursuit3DEnv=Env, run_episode=evaluator))


def test_cycle_observer_rejects_wrong_caller_not_just_plausible_timestamp():
    base = fake_base()
    measured = MeasuredOriginalCycle(base)
    measured.install()
    try:
        with pytest.raises(ValueError, match='unchanged evaluator'):
            base.evaluator.CaptureRadiusPursuit3DEnv().step(np.zeros((4, 3)))
        assert measured.samples == []
    finally:
        measured.close()


def test_cycle_observer_unit_fixture_reads_actual_clock_and_restores_class():
    base = fake_base()
    measured = MeasuredOriginalCycle(base)
    original = base.evaluator.CaptureRadiusPursuit3DEnv
    measured.install()
    env = base.evaluator.CaptureRadiusPursuit3DEnv()
    result = base.evaluator.run_episode(env, 1)
    assert result == ('fixture', 1)
    assert len(measured.samples) == 1 and measured.samples[0] > 0
    measured.close()
    assert base.evaluator.CaptureRadiusPursuit3DEnv is original


@pytest.mark.parametrize('values,steps', [([], []), ([12.], []), ([9.], [{'total_control_latency_ms': 10.}]),
    ([12.], [{'total_control_latency_ms': float('nan')}])])
def test_measurement_missing_nonfinite_or_subtracted_native_support_refused(values, steps):
    measured = MeasuredOriginalCycle(fake_base())
    measured.samples = values
    with pytest.raises(ValueError):
        measured.validate(steps)


@pytest.mark.parametrize('value', [-1., 0.5, float('nan'), float('inf')])
def test_native_local_failure_count_not_silently_truncated(value):
    measured = MeasuredOriginalCycle(fake_base())
    measured.samples = [12.]
    with pytest.raises(ValueError, match='integer local-failure'):
        measured.validate([{'total_control_latency_ms': 10., 'planner_local_solver_failures': value}])
