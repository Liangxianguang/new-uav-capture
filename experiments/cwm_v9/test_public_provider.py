import copy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from public_provider import PublicResponseProvider, delayed_joint_context
from local_shadow import stable_union


def test_off_and_unqualified_modes_never_load_or_evaluate():
    def forbidden():
        raise AssertionError("Disabled provider must never load")
    for mode in ("off", "guarded"):
        provider = PublicResponseProvider(mode, forbidden, qualified=False)
        assert provider.forecast(None, None, None, None, None) is None
        assert provider.load_calls == 0 and provider.prediction_calls == 0
        assert not provider.control_eligible


def test_loader_failure_latches_without_retries_or_activation():
    calls = []
    def failed():
        calls.append(1)
        raise OSError("Unavailable checkpoint")
    provider = PublicResponseProvider("shadow", failed)
    for _ in range(3):
        assert provider.forecast(None, None, None, None, None) is None
    assert calls == [1] and provider.status == "load_failed" and not provider.control_eligible


def context_inputs():
    return np.zeros((8, 252)), np.zeros((4, 6)), np.zeros((3, 8, 4, 3)), np.zeros((8, 4, 3)), np.ones((8, 3), dtype=np.float64)


def test_shadow_can_forecast_alternatives_without_control_eligibility():
    def predictor(history, relative, proposed, anchor, backbone):
        response = np.zeros((len(proposed), 8, 3), dtype=np.float32)
        response[1] = .1
        return {"prediction": backbone[None] + response.astype(np.float64), "response": response}
    provider = PublicResponseProvider("shadow", lambda: predictor)
    h, r, p, a, b = context_inputs()
    p[1] = 1.
    result = provider.forecast(h, r, p, a, b)
    assert result is not None and result["prediction"][0].tobytes() == b.tobytes()
    assert not np.array_equal(result["prediction"][1], b)
    assert not provider.control_eligible and provider.prediction_calls == 1


@pytest.mark.parametrize("error", ["anchor_changed", "nonfinite", "shape", "bad_input"])
def test_prediction_failure_falls_back_and_latches(error):
    def predictor(history, relative, proposed, anchor, backbone):
        prediction = np.repeat(backbone[None], len(proposed), 0)
        response = np.zeros_like(prediction)
        if error == "anchor_changed":
            prediction[0, 0, 0] += .01
        elif error == "nonfinite":
            response[0, 0, 0] = np.nan
        elif error == "shape":
            prediction = prediction[:, :7]
        return {"prediction": prediction, "response": response}
    provider = PublicResponseProvider("shadow", lambda: predictor)
    values = list(context_inputs())
    if error == "bad_input":
        values[0] = np.zeros((7, 252))
    assert provider.forecast(*values) is None
    assert provider.status == "prediction_failed" and not provider.control_eligible
    count = provider.prediction_calls
    assert provider.forecast(*context_inputs()) is None
    assert provider.prediction_calls == count


def delayed_context():
    observation = {"defender_positions": np.arange(12).reshape(4, 3), "defender_velocities": np.arange(12).reshape(4, 3) * .1}
    known = {p: SimpleNamespace(position=np.full(3, p), velocity=np.full(3, .1 * p)) for p in (1, 2, 3)}
    peers = {p: np.full((8, 3), p) for p in known}
    return observation, known, peers


def test_joint_context_uses_only_received_peer_states_and_sequences():
    observation, known, peers = delayed_context()
    own, anchor = np.ones((3, 8, 3)), np.zeros((8, 3))
    a = delayed_joint_context(observation, known, peers, 0, own, anchor, np.zeros(3), np.zeros(3))
    changed = copy.deepcopy(observation)
    changed["defender_positions"][1:] += 100
    changed["defender_velocities"][1:] += 100
    b = delayed_joint_context(changed, known, peers, 0, own, anchor, np.zeros(3), np.zeros(3))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    joint, reference, relative = a
    assert np.array_equal(joint[:, :, 1], np.repeat(peers[1][None], 3, 0))
    assert np.array_equal(reference[:, 2], peers[2])
    assert np.array_equal(relative[3, :3], known[3].position)


def test_missing_or_malformed_peer_does_not_invent_a_plan():
    observation, known, peers = delayed_context()
    del known[3]
    assert delayed_joint_context(observation, known, peers, 0, np.zeros((2, 8, 3)), np.zeros((8, 3)), np.zeros(3), np.zeros(3)) is None
    observation, known, peers = delayed_context()
    peers[1] = np.zeros((7, 3))
    assert delayed_joint_context(observation, known, peers, 0, np.zeros((2, 8, 3)), np.zeros((8, 3)), np.zeros(3), np.zeros(3)) is None


def test_union_preserves_original_order_and_removes_only_exact_duplicates():
    a, b = np.zeros((8, 3)), np.ones((8, 3))
    c = b.copy()
    c[0, 0] += 1e-12
    union = stable_union([a, b, b.copy(), c])
    assert len(union) == 3
    assert np.array_equal(union[0], a) and np.array_equal(union[1], b)
    with pytest.raises(ValueError):
        stable_union([np.full((8, 3), 10.)])
