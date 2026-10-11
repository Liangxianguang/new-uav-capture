"""Synthetic selectors/hooks and negative runner checks, never native efficacy."""
import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prefix_selector as selector
import prefix_entry as hook
import paired_prefix as runner
from test_research_selector import fixture as legacy_fixture
from benchmark_loader import BenchmarkScorers
from research_loader import PrefixResearchBundle


def fixture():
    call, legacy, expand, score = legacy_fixture()
    adapters = {n: legacy.scorers['cv_l2'] for n in selector.NEURAL}
    prefix = PrefixResearchBundle(authorized=True)
    bundle = BenchmarkScorers(authorized=True, prefix_bundle=prefix, scorers=adapters, proposer=legacy.proposer)
    return call, bundle, expand, score


@pytest.mark.parametrize('mode', ['off', 'refusal', 'deploy', 'guarded', 'unknown'])
def test_disabled_refusal_never_inspects_optional_public_context(mode):
    call = {'selected': np.ones((8, 3)), 'selected_costs': np.array([7.])}
    def forbidden(*a, **k): pytest.fail('No optional read on off/refusal')
    result = selector.select_prefix(call, object(), object(), object(), mode=mode, expand=forbidden, scorer=forbidden)
    assert result['selected'].tobytes() == call['selected'].tobytes()
    assert result['selected_costs'].tobytes() == call['selected_costs'].tobytes()
    assert result['evidence'] is None and result['info']['deployment_control_eligible'] is False


@pytest.mark.parametrize('score', selector.SCORES)
def test_all_seven_scoring_controls_preserve_original_cost_row_and_delayed_peer_plans(score):
    call, bundle, expand, scoring = fixture()
    result = selector.select_prefix(call, np.zeros((8, 252)), None, bundle,
        mode='research_eval', score=score, expand=expand, scorer=scoring)
    assert result['info']['status'] == 'research_selection_ready' and result['info']['selected_index'] == 2
    assert result['selected'][0, 0] == .2 and result['selected_costs'][0] == 100.2
    assert result['info']['deployment_control_eligible'] is False
    for peer in (1, 2, 3):
        assert all(np.array_equal(p[:, peer], call['peer_sequences'][peer]) for p in result['evidence']['proposed'])
    if score == 'own_motion': assert not result['evidence']['scoring_paths'].any()
    if score == 'constant_velocity': assert np.all(result['evidence']['scoring_paths'] == .5)


def test_own_motion_uses_fixed_prefix_common_not_full_query_common():
    call, bundle, expand, scoring = fixture()
    class NoAccess:
        def forecast(self, *a): pytest.fail('Own motion must use fixed prefix primary common')
    bundle.scorers['cv_full_query_rank_l2'] = NoAccess()
    result = selector.select_prefix(call, np.zeros((8, 252)), None, bundle,
        mode='research_eval', score='own_motion', expand=expand, scorer=scoring)
    assert result['info']['status'] == 'research_selection_ready'


def test_shadow_and_first_tie_keep_original_actions():
    call, bundle, expand, scoring = fixture()
    result = selector.select_prefix(call, np.zeros((8, 252)), None, bundle,
        mode='shadow', expand=expand, scorer=scoring)
    assert result['selected'].tobytes() == call['selected'].tobytes()
    assert result['selected_costs'].tobytes() == call['selected_costs'].tobytes()
    assert result['info']['research_action_selected'] is False
    assert result['evidence']['proposed_selection'][0, 0] == .2
    result = selector.select_prefix(call, np.zeros((8, 252)), None, bundle,
        mode='research_eval', expand=expand, scorer=lambda *a: np.ones(3))
    assert result['info']['selected_index'] == 0 and not result['info']['choice_changed']


@pytest.mark.parametrize('fault', ['proposal', 'forecast', 'score', 'speed', 'anchor', 'peer', 'k2', 'shape', 'nonfinite', 'unknown_score'])
def test_runtime_failures_discard_expansion_revoke_both_bundles_and_preserve_anchor(fault):
    call, bundle, expand, scoring = fixture(); original_expand = expand; score_name = 'cv_prefix_rank_l2'
    if fault == 'proposal': expand = lambda *a, **k: {'inputs': None, 'info': {'status': 'candidate_expansion_failed'}}
    elif fault == 'forecast': bundle.scorers['cv_prefix_rank_l2'].predictor = lambda *a: {'broken': True}
    elif fault == 'score': scoring = lambda *a: (_ for _ in ()).throw(ValueError('Synthetic score failure'))
    elif fault == 'nonfinite': scoring = lambda *a: np.array([1., float('nan'), 3.])
    elif fault == 'k2': call['scenarios'].trajectories = np.zeros((2, 8, 3))
    elif fault == 'unknown_score': score_name = 'best_model'
    else:
        def changed(*args, **kwargs):
            result = original_expand(*args, **kwargs)
            if fault == 'speed': result['actions'][1, 0, 0] = 6.
            elif fault == 'anchor': result['actions'] += .05
            elif fault == 'shape': result['actions'] = result['actions'][:, :7]
            else: result['inputs'][2][1, :, 1, 0] += .01
            return result
        expand = changed
    result = selector.select_prefix(call, np.zeros((8, 252)), None, bundle,
        mode='research_eval', score=score_name, expand=expand, scorer=scoring)
    assert result['info']['status'] == 'selection_failed' and result['evidence'] is None
    assert result['selected'].tobytes() == call['selected'].tobytes()
    assert result['selected_costs'].tobytes() == call['selected_costs'].tobytes()
    assert bundle.authorized is False and bundle.prefix_bundle.authorized is False


def test_missing_received_peers_preserve_original_without_fabricated_joint_context():
    call, bundle, _, _ = fixture()
    result = selector.select_prefix(call, bundle=bundle, mode='research_eval',
        expand=lambda *a, **k: {'inputs': None, 'info': {'status': 'missing_received_peer_context'}})
    assert result['info']['status'] == 'missing_received_peer_context' and bundle.authorized
    assert result['selected'].tobytes() == call['selected'].tobytes()


def test_failed_scoring_preserves_inputs_and_all_caller_cpu_rng():
    call, bundle, expand, _ = fixture(); saved = copy.deepcopy(call)
    def failure(public, actions, paths):
        random.random(); np.random.rand(); torch.rand(1)
        public['known'][1].position[:] = 100.; actions[:] = 100.; paths[:] = 100.
        raise ValueError('Synthetic failure')
    py, ns, ts = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    result = selector.select_prefix(call, np.zeros((8, 252)), None, bundle,
        mode='research_eval', expand=expand, scorer=failure)
    assert result['info']['status'] == 'selection_failed'
    assert np.array_equal(call['known'][1].position, saved['known'][1].position)
    assert random.getstate() == py and torch.equal(ts, torch.get_rng_state())
    after = np.random.get_state(); assert np.array_equal(ns[1], after[1]) and ns[2:] == after[2:]


def fake_base():
    class Runtime:
        def __init__(self): self.history = []
        def _append_current_frame(self, observation): self.history.append(np.ones((4, 63))*observation)
    class Planner:
        config = SimpleNamespace(horizon_steps=8, dt_seconds=.1, max_speed_mps=5.)
        def _select_local_sequence(self, *a): return np.zeros((8, 3)), np.array([7.])
        def plan(self, observation, scenarios, **kwargs):
            for peer_value in (1., 2.):
                self._select_local_sequence(observation, scenarios, 0, [np.zeros((8, 3))], {},
                    {p: np.ones((8, 3))*peer_value for p in (1, 2, 3)})
            return 'synthetic_plan'
    return SimpleNamespace(evaluator=SimpleNamespace(DistributedMinimaxDNMPC=Planner,
        PredictionRuntime=Runtime, belief_reference=lambda *a: None))


def test_entry_off_refused_or_old_bundle_install_exact_parent_classes():
    base = fake_base(); parents = (base.evaluator.DistributedMinimaxDNMPC, base.evaluator.PredictionRuntime)
    entry = hook.PrefixSequentialEntry(base)
    for mode in ('plain', 'off', 'refusal', 'shadow', 'research_eval'):
        entry.configure(mode, bundle=object())
        assert (base.evaluator.DistributedMinimaxDNMPC, base.evaluator.PredictionRuntime) == parents
        assert not entry.events and not entry.history and not entry.pending
    entry.close()


def test_actual_hook_fixture_updates_public252_history_and_fresh_peers_each_best_response(monkeypatch):
    base = fake_base(); entry = hook.PrefixSequentialEntry(base); _, bundle, _, _ = fixture()
    captured = []
    def selected(call, history, belief, supplied, **kwargs):
        captured.append((call, history.copy()))
        return {'selected': call['selected'], 'selected_costs': call['selected_costs'], 'evidence': None,
            'info': {'status': 'research_selection_ready', 'score': kwargs['score']}}
    monkeypatch.setattr(hook, 'select_prefix', selected)
    entry.configure('research_eval', bundle=bundle); entry.episode = 1
    runtime = base.evaluator.PredictionRuntime()
    runtime._append_current_frame(1); runtime._append_current_frame(2)
    assert base.evaluator.DistributedMinimaxDNMPC().plan({}, {}, step_index=3) == 'synthetic_plan'
    assert len(captured) == 2 and len(entry.events) == 2 and not entry.pending
    assert [c['ordinal'] for c, _ in captured] == [0, 1]
    assert np.all(captured[0][0]['peer_sequences'][1] == 1.) and np.all(captured[1][0]['peer_sequences'][1] == 2.)
    assert captured[0][1].shape == (8, 252) and np.all(captured[0][1][-1] == 2.)
    entry.close()


def test_public_capture_failure_returns_original_and_revokes_without_changed_classes(monkeypatch):
    base = fake_base(); entry = hook.PrefixSequentialEntry(base); _, bundle, _, _ = fixture()
    entry.configure('research_eval', bundle=bundle); entry.episode = 1
    # Missing belief function fails only after the original local choice exists.
    del base.evaluator.belief_reference
    base.evaluator.DistributedMinimaxDNMPC().plan({}, {}, step_index=2)
    assert entry.events[0]['status'] == 'public_capture_failed'
    assert entry.events[1]['status'] == 'research_refused'
    assert not bundle.authorized and not bundle.prefix_bundle.authorized
    entry.close()


def test_failed_primary_runner_refuses_before_baseline_or_fresh_scenes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'published_sources', lambda: {'commit': 'NEGATIVE_FIXTURE'})
    monkeypatch.setattr(runner, 'load_benchmark', lambda *a, **k: BenchmarkScorers(status='prefix_primary_failed_no_legacy_access'))
    def forbidden(*a, **k): pytest.fail('No simulation without qualified primary')
    monkeypatch.setattr(runner, 'Baseline', forbidden); monkeypatch.setattr(runner, 'fresh_records', forbidden)
    monkeypatch.setattr(runner, 'PrefixSequentialEntry', forbidden)
    output = tmp_path/'new'
    runner.run(None, None, None, None, output)
    report = json.loads((output/'refusal.json').read_bytes())
    assert report['research_actions_exercised'] is False and report['fresh_scenes_generated'] is False
    assert not (output/'scenes.json').exists() and not (output/'r').exists()


def test_fresh_generation_rejects_unqualified_bundle_before_generator(monkeypatch):
    monkeypatch.setattr(runner, 'scene_records', lambda *a: pytest.fail('No new scenes before qualification'))
    with pytest.raises(ValueError, match='BEFORE'): runner.fresh_records(object(), runner.load_plan(), BenchmarkScorers())


@pytest.mark.parametrize('change', ['none', 'omit', 'duplicate', 'axis'])
def test_fresh_generator_support_is_all64_fixed_development_identities(change, monkeypatch):
    plan = runner.load_plan(); expected = runner.specification('fresh_geometry', plan)
    rows = copy.deepcopy(expected)
    if change == 'omit': rows.pop()
    elif change == 'duplicate': rows[-1] = rows[0]
    elif change == 'axis': rows[0]['variant'] = 'best_geometry'
    captured = []
    monkeypatch.setattr(runner, 'scene_records', lambda base, protocol: captured.append(protocol) or rows)
    _, bundle, _, _ = fixture()
    if change == 'none':
        result = runner.fresh_records(object(), plan, bundle)
        assert result == expected and len(result) == 64
        assert captured[0]['layout_seed_start'] == 2002010
        assert 'not_holdout' in captured[0]['stage']
    else:
        with pytest.raises(ValueError): runner.fresh_records(object(), plan, bundle)


def test_original_population_requires_unchanged_v30_protocol_not_arithmetic_view():
    inherited = runner.read_json(runner.ROOT/'experiments/cwm_v30/closed_loop_protocol.json')
    capsule = runner.ROOT/'experiments/cwm_v1/baseline/capsule.zip'
    actual = runner.original_population(capsule, inherited)
    assert len(actual) == 1280
    expected = runner.specification('original', runner.load_plan())
    assert [{k: r[k] for k in expected[0]} for r in actual] == expected
    with pytest.raises(ValueError): runner.original_population(capsule, runner.arithmetic_view(runner.load_plan()))


@pytest.mark.parametrize('kind', ['original', 'supported', 'unsupported', 'wrong_score', 'missing'])
def test_no_silent_original_policy_substitution_in_research_support(kind):
    row = {}; method = 'original' if kind == 'original' else 'cv_prefix_rank_l2_shared'
    events = [] if kind in ('original', 'missing') else [{'status': 'missing_received_peer_context'}] if kind == 'unsupported' else [
        {'status': 'research_selection_ready', 'score': 'cv_prefix_rank_l2' if kind == 'supported' else 'cv_l2'}]
    if kind in ('wrong_score', 'missing'):
        with pytest.raises(ValueError): runner.validate_support(row, events, method)
    else:
        runner.validate_support(row, events, method)
        assert row['optional_failure_calls'] == (1 if kind == 'unsupported' else 0)


def test_saved_evidence_manifest_includes_all_public_context_arrays(tmp_path):
    names = ['preregistration.json', 'prefix_certificate.json', 'legacy_certificate.json', 'scenes.json',
        'benchmark_preload.json', 'original/episodes.jsonl', 'original/steps.jsonl', 'source/test.py',
        'original/1/original/trajectory.npz', 'original/1/original/commands.npz',
        'original/1/original/calls/one.npz', 'original/1/original/calls/one.json']
    for name in names:
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'SYNTHETIC')
    event = {k+'_path': 'calls/one'+('.npz' if k == 'arrays' else '.json') for k in ('arrays', 'context')}
    event.update({k+'_sha256': runner.sha(tmp_path/'original/1/original'/event[k+'_path']) for k in ('arrays', 'context')})
    path = tmp_path/'original/1/original/events.json'; path.write_text(json.dumps([event]), encoding='utf8')
    row = {'trajectory_path': names[8], 'commands_path': names[9], 'events_path': path.relative_to(tmp_path).as_posix(), 'events_sha256': runner.sha(path)}
    result = runner.save_evidence_manifest(tmp_path, {'original': [row]}, {'test.py': 'unused-fixture'})
    assert set(result) == set(names) | {row['events_path']}
    (tmp_path/names[-1]).write_bytes(b'CHANGED')
    with pytest.raises(ValueError): runner.save_evidence_manifest(tmp_path, {'original': [row]}, {'test.py': 'unused-fixture'})


@pytest.mark.parametrize('kind', ['existing', 'input_parent', 'source'])
def test_output_must_not_overwrite_input_evidence_or_sources(tmp_path, kind):
    if kind == 'existing': output, inputs = tmp_path, []
    elif kind == 'input_parent': output, inputs = tmp_path/'new', [tmp_path/'new/evidence.zip']
    else: output, inputs = runner.ROOT/'experiments/cwm_v42/NEW_SIMULATION', []
    with pytest.raises(ValueError): runner.validate_output(output, inputs)
