"""Synthetic interface fixtures are NOT qualified models/capture evidence."""
import copy
import io
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from local_selector import select_research, SCORES
from research_bundle import ResearchBundle, predictor_from_checkpoint, load_bundle
from qualification_gate import OptionalResponseAdapter, QualificationGate
from sequential_entry import SequentialResearchEntry


class FixturePlanner:
    config = SimpleNamespace(horizon_steps=8, dt_seconds=.1, max_speed_mps=5.)
    def _local_scenario_cost_matrix(self, observation, scenarios, agent, actions, known, peers):
        return 100 + actions[:, 0, :1]


def fixture():
    actions = np.zeros((3, 8, 3), dtype=np.float64)
    actions[1, :, 0] = .1
    actions[2, :, 0] = .2
    anchor = actions[0].copy()
    call = {'selected': anchor.copy(), 'selected_costs': np.array([100.]),
        'local_candidates': actions[:1].copy(), 'agent': 0, 'planner': FixturePlanner(),
        'observation': {'defender_positions': np.zeros((4, 3)), 'defender_velocities': np.zeros((4, 3))},
        'known': {p: SimpleNamespace(position=np.ones(3)*p, velocity=np.zeros(3)) for p in (1, 2, 3)},
        'peer_sequences': {p: np.ones((8, 3))*p/10 for p in (1, 2, 3)},
        'scenarios': SimpleNamespace(trajectories=np.zeros((1, 8, 3)))}
    def predict(h, relative, proposed, joint_anchor, backbone, cv):
        reference = np.zeros((len(proposed), 8, 3), dtype=np.float64)
        response = proposed[:, :, 0].astype(np.float32).copy()
        return {'prediction': reference+response, 'reference': reference, 'response': response}
    adapters = {n: OptionalResponseAdapter(QualificationGate(mode='shadow'), lambda: predict)
                for n in ('cv_rank_l2', 'cv_l2', 'cv_motion_only')}
    bundle = ResearchBundle(authorized=True, status='SYNTHETIC_IMPLEMENTATION_FIXTURE',
                            proposer=OptionalResponseAdapter(QualificationGate(mode='shadow'), lambda: predict), scorers=adapters)
    def expand(public, history, belief, proposer, **kwargs):
        proposed = np.empty((len(actions), 8, 4, 3), dtype=np.float64)
        proposed[:, :, 0] = actions
        joint_anchor = np.empty((8, 4, 3), dtype=np.float64)
        joint_anchor[:, 0] = public['selected']
        for peer in (1, 2, 3):
            proposed[:, :, peer] = public['peer_sequences'][peer]
            joint_anchor[:, peer] = public['peer_sequences'][peer]
        values = (np.asarray(history).copy(), np.zeros((4, 6)), proposed, joint_anchor,
                  np.zeros((8, 3)), np.ones((8, 3))*.5)
        return {'actions': actions.copy(), 'inputs': values,
                'info': {'status': 'shadow_candidates_ready', 'control_eligible': False}}
    def score(public, offered, paths):
        assert paths.shape == (len(offered), 8, 3)
        return np.array([3., 2., 1.])
    return call, bundle, expand, score


@pytest.mark.parametrize('mode', ['off', 'guarded', 'unknown', 'refusal'])
def test_off_refusal_never_inspects_optional_context(mode):
    call = {'selected': np.arange(24., dtype=np.float64).reshape(8, 3), 'selected_costs': np.array([7.])}
    def forbidden(*a, **k):
        pytest.fail('Optional input/model/score must not be read')
    result = select_research(call, object(), object(), object(), mode=mode, expand=forbidden, scorer=forbidden)
    assert result['selected'].tobytes() == call['selected'].tobytes()
    assert result['selected_costs'].tobytes() == call['selected_costs'].tobytes()
    assert result['evidence'] is None and not result['info']['deployment_control_eligible']
    result['selected'][:] = 0
    assert call['selected'].sum() > 0


@pytest.mark.parametrize('score', SCORES)
def test_research_selects_action_but_not_deployment_and_retains_original_cost(score):
    call, bundle, expand, scoring = fixture()
    result = select_research(call, np.zeros((8, 252), dtype=np.float32), None, bundle,
                            mode='research_eval', score=score, expand=expand, scorer=scoring)
    assert result['info']['status'] == 'research_selection_ready'
    assert result['info']['choice_changed'] and result['info']['selected_index'] == 2
    assert result['selected'][0, 0] == .2 and result['selected_costs'][0] == 100.2
    assert result['info']['deployment_control_eligible'] is False
    assert result['evidence']['history'].dtype == np.float32
    for peer in (1, 2, 3):
        assert all(np.array_equal(a[:, peer], call['peer_sequences'][peer]) for a in result['evidence']['proposed'])
    assert not np.array_equal(result['selected'], result['evidence']['scoring_paths'][2])


def test_shadow_records_different_choice_but_keeps_original_action_and_costs():
    call, bundle, expand, scoring = fixture()
    result = select_research(call, np.zeros((8, 252)), None, bundle, mode='shadow', expand=expand, scorer=scoring)
    assert result['info']['status'] == 'shadow_selection_only'
    assert result['info']['research_action_selected'] is False
    assert np.array_equal(result['selected'], call['selected'])
    assert np.array_equal(result['selected_costs'], call['selected_costs'])
    assert result['evidence']['proposed_selection'][0, 0] == .2


def test_first_index_tie_rule_retained():
    call, bundle, expand, _ = fixture()
    result = select_research(call, np.zeros((8, 252)), None, bundle, mode='research_eval', expand=expand,
                            scorer=lambda *a: np.ones(3))
    assert result['info']['selected_index'] == 0
    assert np.array_equal(result['selected'], call['selected'])


@pytest.mark.parametrize('fault', ['generation', 'forecast', 'score', 'nonfinite', 'speed', 'anchor', 'joint', 'shape', 'k2'])
def test_failures_discard_all_proposals_disable_research_and_preserve_original(fault):
    call, bundle, expand, scoring = fixture()
    if fault == 'generation':
        expand = lambda *a, **k: {'inputs': None, 'info': {'status': 'candidate_expansion_failed'}}
    elif fault == 'forecast':
        bundle.scorers['cv_rank_l2'].loader = lambda: (_ for _ in ()).throw(OSError())
    elif fault in ('score', 'nonfinite'):
        scoring = (lambda *a: (_ for _ in ()).throw(RuntimeError())) if fault == 'score' else lambda *a: np.array([1., np.nan, 3.])
    elif fault == 'k2':
        call['scenarios'].trajectories = np.zeros((2, 8, 3))
    else:
        original = expand
        def broken(*a, **k):
            result = original(*a, **k)
            if fault == 'speed':
                result['actions'][2] = 10
            elif fault == 'anchor':
                result['actions'][0] = .7
            elif fault == 'joint':
                result['inputs'][2][1, :, 0] = .3
            else:
                result['actions'] = np.zeros((3, 8, 4, 3))
            return result
        expand = broken
    result = select_research(call, np.zeros((8, 252)), None, bundle, mode='research_eval', expand=expand, scorer=scoring)
    assert result['info']['status'] == 'selection_failed'
    assert result['evidence'] is None and bundle.authorized is False
    assert result['selected'].tobytes() == call['selected'].tobytes()
    assert result['selected_costs'].tobytes() == call['selected_costs'].tobytes()
    again = select_research(call, object(), object(), bundle, mode='research_eval',
                            expand=lambda *a, **k: pytest.fail('Failed research cannot retry'))
    assert again['info']['status'] == 'research_refused'


def test_missing_received_peer_skips_model_and_does_not_permanently_disable():
    call, bundle, _, _ = fixture()
    del call['peer_sequences'][2]
    result = select_research(call, np.zeros((8, 252)), lambda *a: (np.zeros(3), np.zeros(3)), bundle, mode='research_eval')
    assert result['info']['status'] == 'missing_received_peer_context'
    assert bundle.authorized is True
    assert all(a.load_calls == a.prediction_calls == 0 for a in bundle.scorers.values())
    assert bundle.proposer.load_calls == bundle.proposer.prediction_calls == 0
    assert np.array_equal(result['selected'], call['selected'])


def test_rng_and_public_input_isolation_even_when_injected_score_mutates_then_fails():
    call, bundle, expand, _ = fixture()
    original = copy.deepcopy(call)
    h = np.zeros((8, 252), dtype=np.float32)
    def bad_score(public, actions, paths):
        random.random(); np.random.rand(); torch.rand(1)
        public['known'][1].position[:] = 100
        public['observation']['defender_positions'][:] = 100
        actions[:] = 100; paths[:] = 100
        raise RuntimeError()
    py, ns, ts = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    result = select_research(call, h, None, bundle, mode='research_eval', expand=expand, scorer=bad_score)
    assert result['info']['status'] == 'selection_failed'
    assert random.getstate() == py and torch.equal(torch.get_rng_state(), ts)
    after = np.random.get_state()
    assert np.array_equal(after[1], ns[1]) and after[2:] == ns[2:]
    assert np.array_equal(call['known'][1].position, original['known'][1].position)
    assert np.array_equal(call['observation']['defender_positions'], original['observation']['defender_positions'])
    assert not h.any()


def test_entry_off_refusal_installs_exact_parent_classes_without_history_or_model_access():
    class Planner:
        pass
    class Runtime:
        pass
    evaluator = SimpleNamespace(DistributedMinimaxDNMPC=Planner, PredictionRuntime=Runtime)
    entry = SequentialResearchEntry(SimpleNamespace(evaluator=evaluator))
    for mode in ('plain', 'off', 'refusal', 'research_eval', 'shadow'):
        entry.configure(mode, bundle=ResearchBundle(authorized=False))
        assert evaluator.DistributedMinimaxDNMPC is Planner
        assert evaluator.PredictionRuntime is Runtime
        assert not entry.history and not entry.pending and not entry.events
    entry.close()


def test_new_checkpoint_predictor_reconstruction_and_public_six_array_contract():
    """Untrained initialized V28 fixture, not a completed80epoch checkpoint."""
    from release_ranking_training import CalibratedOrigin, LocalTwoHead, FrozenOriginResponse, state_digest
    protocol = json.loads((HERE.parent / 'cwm_v28/training_protocol.json').read_bytes())
    config = protocol['training']
    torch.manual_seed(994101)
    common = CalibratedOrigin(LocalTwoHead('motion_only', config['motion_scale_m'], config['response_scale_m']), 'cv')
    model = FrozenOriginResponse(common, LocalTwoHead('plain', config['motion_scale_m'], config['response_scale_m']))
    cp = {'protocol': protocol, 'configuration': 'cv_rank_l2', 'seed': 994101, 'online_promoted': False,
          'baseline_weights_included': False, 'common_motion_in_response_optimizer': False,
          'source_hashes': {}, 'data_summary_sha256': 'FIXTURE', 'data_audit_summary_sha256': 'FIXTURE',
          'model_state': model.state_dict(), 'common_motion_state_digest': state_digest(common),
          'normalizer_mean': torch.zeros(252, dtype=torch.float64), 'normalizer_scale': torch.ones(252, dtype=torch.float64)}
    stream = io.BytesIO(); torch.save(cp, stream); raw = stream.getvalue()
    import hashlib
    row = {'configuration': cp['configuration'], 'seed': cp['seed'], 'checkpoint_sha256': hashlib.sha256(raw).hexdigest()}
    predictor = predictor_from_checkpoint(raw, row, cp)
    call, _, expand, _ = fixture()
    values = expand(call, np.zeros((8, 252), dtype=np.float32), None, None)['inputs']
    adapter = OptionalResponseAdapter(QualificationGate(mode='shadow'), lambda: predictor)
    out, info = adapter.forecast(*values)
    assert info['forecast_available'] and not info['control_eligible']
    assert out['prediction'].shape == (3, 8, 3)
    assert not out['response'][0].any()
    assert out['reference'][0].tobytes() == out['reference'][2].tobytes()
    with pytest.raises(ValueError, match='identity'):
        predictor_from_checkpoint(raw+b'changed', row, cp)


def test_incomplete_release_certificate_refuses_before_artifact_or_checkpoint_access(tmp_path):
    certificate = tmp_path / 'bad.json'
    certificate.write_text(json.dumps({'status': 'training_pending'}))
    with pytest.raises(ValueError, match='Complete real'):
        load_bundle(tmp_path/'nonexistent.zip', certificate, tmp_path/'cache')
    assert not (tmp_path/'cache').exists()
