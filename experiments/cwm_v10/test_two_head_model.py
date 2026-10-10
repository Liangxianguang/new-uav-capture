import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from two_head_model import LocalTwoHead
from train_two_head import pack_inputs, batch_loss, normalization


def public_inputs():
    rng = np.random.default_rng(22)
    proposed = rng.normal(size=(3, 8, 4, 3))
    anchor = proposed[0].copy()
    return (torch.from_numpy(rng.normal(size=(3, 8, 252))).float(), torch.from_numpy(rng.normal(size=(3, 4, 6))).float(),
            torch.from_numpy(proposed).float(), torch.from_numpy(np.repeat(anchor[None], 3, 0)).float(), torch.from_numpy(rng.normal(size=(3, 8, 3))))


@pytest.mark.parametrize("kind", ["motion_only", "plain", "structured"])
def test_zero_initialized_model_preserves_supplied_backbone(kind):
    torch.set_num_threads(1)
    model = LocalTwoHead(kind)
    inputs = public_inputs()
    prediction, motion, response = model(*inputs)
    assert prediction.detach().numpy().tobytes() == inputs[-1].numpy().tobytes()
    assert not motion.any() and not response.any()


@pytest.mark.parametrize("kind", ["plain", "structured"])
def test_motion_and_response_are_separate_and_response_anchor_exact(kind):
    model = LocalTwoHead(kind)
    with torch.no_grad():
        model.motion_head.bias.fill_(.2)
        model.response_head.weight.normal_(0., .3)
    inputs = public_inputs()
    prediction, motion, response = model(*inputs)
    assert (response[0] == 0).all() and response[1:].abs().sum() > 0
    assert not torch.equal(prediction[0], inputs[-1][0])
    assert motion.abs().max() <= 2.5 and response.abs().max() <= 2.5
    assert torch.allclose(prediction[0], inputs[-1][0] + motion[0].double())


def sample_calls():
    rng = np.random.default_rng(41)
    result = []
    for i in range(3):
        anchor = rng.normal(size=(8, 4, 3))
        proposed = np.stack([anchor, anchor + .5])
        values = {"history": rng.normal(size=(8, 252)), "relative": rng.normal(size=(4, 6)), "proposed": proposed,
                  "anchor": anchor, "backbone": rng.normal(size=(8, 3)), "target": rng.normal(size=(2, 8, 3)),
                  "anchor_target": rng.normal(size=(8, 3)), "valid": np.ones((2, 8), bool), "anchor_valid": np.ones(8, bool)}
        result.append({"record": {"group": "a" if i < 2 else "b"}, "values": values})
    return result


def test_future_labels_do_not_change_any_public_input():
    calls = sample_calls()
    mean, scale = normalization(calls, [0, 1])
    original, _ = pack_inputs(calls, [0, 1, 2], mean, scale)
    altered = copy.deepcopy(calls)
    for call in altered:
        call["values"]["target"] += 1000.
        call["values"]["anchor_target"] -= 1000.
    changed, _ = pack_inputs(altered, [0, 1, 2], mean, scale)
    assert all(a.numpy().tobytes() == b.numpy().tobytes() for a, b in zip(original, changed))


def test_development_histories_never_affect_train_normalization():
    calls = sample_calls()
    a = normalization(calls, [0, 1])
    calls[2]["values"]["history"] += 1000.
    b = normalization(calls, [0, 1])
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_terminal_padding_cannot_change_two_head_loss():
    calls = sample_calls()
    for call in calls:
        call["values"]["valid"][:, 5:] = False
        call["values"]["anchor_valid"][5:] = False
    mean, scale = normalization(calls, [0, 1, 2])
    model = LocalTwoHead("plain")
    config = json.loads((ROOT / "training_protocol.json").read_text())["training"]
    a = batch_loss(model, calls, [0, 1, 2], mean, scale, {"a": 2, "b": 1}, config)
    altered = copy.deepcopy(calls)
    for call in altered:
        call["values"]["target"][:, 5:] += 1e6
        call["values"]["anchor_target"][5:] -= 1e6
    b = batch_loss(model, altered, [0, 1, 2], mean, scale, {"a": 2, "b": 1}, config)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_motion_only_is_independently_trainable_without_response():
    calls = sample_calls()
    mean, scale = normalization(calls, [0, 1, 2])
    model = LocalTwoHead("motion_only")
    config = json.loads((ROOT / "training_protocol.json").read_text())["training"]
    loss, _ = batch_loss(model, calls, [0, 1, 2], mean, scale, {"a": 2, "b": 1}, config)
    loss.backward()
    assert model.motion_head.weight.grad is not None
    assert model.response_head.weight.grad is None and not model.response_head.weight.requires_grad
    assert not model(*public_inputs())[2].any()


def test_plain_and_structured_parameter_counts_are_matched():
    counts = [sum(p.numel() for p in LocalTwoHead(kind).parameters()) for kind in ("plain", "structured")]
    assert abs(counts[1] / counts[0] - 1.) < .01
