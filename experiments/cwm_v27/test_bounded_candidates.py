import copy
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bounded_candidates import expand_candidates, MAX_CANDIDATES, stable_unique
from qualification_gate import OptionalResponseAdapter, QualificationGate


def fixture():
    actual = np.zeros((2, 8, 3), dtype=np.float64)
    call = {'local_candidates': actual, 'selected': actual[0].copy(), 'agent': 0,
            'observation': {'defender_positions': np.zeros((4, 3)), 'defender_velocities': np.zeros((4, 3))},
            'known': {p: SimpleNamespace(position=np.ones(3)*p, velocity=np.zeros(3)) for p in (1, 2, 3)},
            'peer_sequences': {p: np.ones((8, 3)) * p / 10 for p in (1, 2, 3)},
            'planner': SimpleNamespace(config=object()),
            'scenarios': SimpleNamespace(trajectories=np.zeros((1, 8, 3)))}
    def belief(*_):
        return np.zeros(3), np.zeros(3)
    def predict(h, r, p, a, b, cv):
        reference = np.zeros((len(p), 8, 3), dtype=np.float64)
        response = np.zeros_like(reference)
        response[..., 0] = p[:, :, 0, 0]
        return {'reference': reference, 'response': response, 'prediction': reference+response}
    adapter = OptionalResponseAdapter(QualificationGate(mode='shadow'), lambda: predict)
    def geometry(c, *_):
        action = np.ones((8, 3)) / 10
        return stable_unique([*c['local_candidates'], action])
    def generate(c, path):
        result = np.zeros((8, 3))
        result[:, 0] = path[:, 0] + .1
        return [result]
    return call, belief, adapter, geometry, generate


def run(**kwargs):
    call, belief, adapter, geometry, generate = fixture()
    result = expand_candidates(call, np.zeros((8, 252)), belief, adapter, mode='shadow',
        geometry_factory=geometry, path_factory=generate, **kwargs)
    return result, call, adapter


@pytest.mark.parametrize('mode', ['off', 'guarded', 'unknown'])
def test_disabled_preserves_duplicates_without_context(mode):
    actual = np.arange(48.).reshape(2, 8, 3)
    result = expand_candidates({'local_candidates': actual}, object(), object(), object(), mode=mode)
    assert result['actions'].tobytes() == actual.tobytes()
    assert result['forecast'] is None and result['inputs'] is None
    assert not result['info']['control_eligible']
    result['actions'][0] = 0
    assert actual[0].sum() != 0


def test_reforecast_new_joint_actions_and_fixed_delayed_peers():
    result, call, adapter = run(source='response')
    assert result['info']['status'] == 'shadow_candidates_ready'
    assert len(result['actions']) == 5
    assert len(result['info']['rounds']) == 2 and adapter.prediction_calls == 3
    h, r, proposed, anchor, *_ = result['inputs']
    assert np.array_equal(proposed[:, :, 0], result['actions'])
    assert np.array_equal(anchor[:, 0], call['selected'])
    for peer in (1, 2, 3):
        assert all(np.array_equal(row[:, peer], call['peer_sequences'][peer]) for row in proposed)
        assert np.array_equal(anchor[:, peer], call['peer_sequences'][peer])
        assert np.array_equal(r[peer, :3], call['known'][peer].position)
    assert np.array_equal(result['forecast']['prediction'][..., 0], proposed[:, :, 0, 0])
    assert not result['info']['control_eligible']


def test_original_history_dtype_and_bytes_preserved():
    call, belief, adapter, geometry, generate = fixture()
    h = np.arange(8*252, dtype=np.float32).reshape(8,252)
    result = expand_candidates(call,h,belief,adapter,mode='shadow',geometry_factory=geometry,path_factory=generate)
    assert result['inputs'][0].dtype == h.dtype and result['inputs'][0].tobytes() == h.tobytes()


def test_missing_peer_skips_factories_and_model():
    call, belief, adapter, _, _ = fixture()
    del call['peer_sequences'][2]
    def forbidden(*_):
        raise AssertionError('Must not invoke proposal generation')
    result = expand_candidates(call, np.zeros((8, 252)), belief, adapter, mode='shadow',
        geometry_factory=forbidden, path_factory=forbidden)
    assert result['info']['status'] == 'missing_received_peer_context'
    assert adapter.load_calls == adapter.prediction_calls == 0
    assert np.array_equal(result['actions'], call['local_candidates'])


@pytest.mark.parametrize('fault', ['geometry', 'speed', 'shape', 'model', 'anchor', 'final'])
def test_any_failure_discards_all_proposals(fault):
    call, belief, adapter, geometry, generate = fixture()
    if fault == 'geometry':
        geometry = lambda *_: (_ for _ in ()).throw(RuntimeError())
    elif fault == 'speed':
        generate = lambda *_: [np.ones((8, 3)) * 5]
    elif fault == 'shape':
        generate = lambda *_: [np.zeros((8, 4, 3))]
    elif fault == 'anchor':
        call['selected'][:] = 1
    elif fault == 'model':
        adapter.loader = lambda: (_ for _ in ()).throw(OSError())
    else:
        normal = adapter.forecast
        def failed(*args):
            if adapter.prediction_calls == 2:
                return None, {'control_eligible': False}
            return normal(*args)
        adapter.forecast = failed
    result = expand_candidates(call, np.zeros((8, 252)), belief, adapter, mode='shadow', source='response',
        geometry_factory=geometry, path_factory=generate)
    assert result['info']['status'] == 'candidate_expansion_failed'
    assert result['actions'].tobytes() == call['local_candidates'].tobytes()
    assert result['forecast'] is None


def test_cap_stable_order_and_rng_input_isolation():
    call, belief, adapter, geometry, _ = fixture()
    original = copy.deepcopy(call)
    h = np.zeros((8, 252))
    def generate(c, path):
        random.random(); np.random.rand(); torch.rand(1)
        c['observation']['defender_positions'][:] = 99  # mutation stays on public copy
        return [np.ones((8, 3)) * i / 100 for i in range(1, 80)]
    py, ns, ts = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    result = expand_candidates(call, h, belief, adapter, mode='shadow', geometry_factory=geometry, path_factory=generate)
    assert len(result['actions']) == MAX_CANDIDATES and len(result['info']['rounds']) == 1
    assert random.getstate() == py and torch.equal(torch.get_rng_state(), ts)
    now = np.random.get_state()
    assert np.array_equal(now[1], ns[1]) and now[2:] == ns[2:]
    assert np.array_equal(call['observation']['defender_positions'], original['observation']['defender_positions'])
    assert np.array_equal(h, np.zeros((8, 252)))
    assert np.array_equal(result['inputs'][1][0], np.zeros(6))


@pytest.mark.parametrize('source', ['cv', 'motion', 'response', 'shared'])
def test_all_sources_are_finite_bounded_and_repeatable(source):
    first, _, _ = run(source=source)
    second, _, _ = run(source=source)
    assert first['info'] == second['info']
    assert np.array_equal(first['actions'], second['actions'])
    assert np.linalg.norm(first['actions'], axis=-1).max() <= 5
    assert len(first['actions']) <= MAX_CANDIDATES
    assert np.array_equal(first['actions'][0], np.zeros((8, 3)))
