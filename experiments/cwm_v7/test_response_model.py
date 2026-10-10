import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from response_model import ResponseModel, loss_terms
from train_response import normalization, validate_splits


def public_inputs():
    generator = torch.Generator().manual_seed(11)
    return [torch.randn(2, 8, 252, generator=generator), torch.randn(2, 4, 6, generator=generator),
            torch.randn(2, 8, 4, 3, generator=generator), torch.randn(2, 8, 4, 3, generator=generator),
            torch.randn(2, 8, 3, generator=generator, dtype=torch.float64)]


@pytest.mark.parametrize("kind", ["plain", "structured"])
def test_response_keeps_anchor_double_bytes_and_bounded_public_only_inputs(kind):
    model = ResponseModel(kind)
    with torch.no_grad():
        model.response_head.weight.fill_(.1)
    values = public_inputs()
    values[2] = values[3].clone()
    prediction, response, _ = model(*values)
    assert torch.equal(prediction, values[-1]) and (response == 0).all()
    values[2] += 2
    prediction, response, _ = model(*values)
    assert response.abs().max() <= 2.5
    assert list(inspect.signature(model.forward).parameters) == ["history", "relative", "proposed", "anchor", "backbone"]
    response.square().mean().backward()
    assert values[-1].grad is None
    assert not any("backbone" in name for name, _ in model.named_parameters())


def test_structured_action_features_are_agent_permutation_invariant_and_matched_size():
    torch.manual_seed(12)
    structured, plain = ResponseModel("structured"), ResponseModel("plain")
    _, relative, proposed, _, _ = public_inputs()
    permutation = [2, 0, 3, 1]
    assert torch.allclose(structured.action_features(relative, proposed), structured.action_features(relative[:, permutation], proposed[:, :, permutation]), atol=1e-6)
    sizes = [sum(p.numel() for p in model.parameters()) for model in (structured, plain)]
    assert max(sizes) / min(sizes) < 1.05


def test_loss_ignores_post_terminal_points_and_normalizer_ignores_validation():
    config = {"residual_penalty": .001, "branch_auxiliary_weight": .1}
    response = torch.zeros(2, 8, 8, 3)
    logits = torch.zeros(2, 8, 8)
    target = torch.ones_like(response)
    valid = torch.ones(2, 8, 8, dtype=torch.bool)
    valid[:, :, 4:] = False
    signs = torch.ones(2, 8, 8)
    weights = torch.ones(2)
    first = loss_terms(response, logits, target, valid, signs, weights, config)
    target[:, :, 4:] = 1e6
    second = loss_terms(response, logits, target, valid, signs, weights, config)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    data = {"history": np.arange(4 * 8 * 252).reshape(4, 8, 252).astype(float)}
    a = normalization(data, [0, 1])
    data["history"][2:] = 1e20
    b = normalization(data, [0, 1])
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_preassigned_split_rejects_scene_and_array_leakage():
    protocol = json.loads(Path(__file__).with_name("training_data_protocol.json").read_text())
    scenes = []
    for group in range(protocol["groups"]):
        for member, bias in enumerate(("upper", "lower")):
            scenes.append({"layout_seed": protocol["layout_seed_start"] + group, "mirror_group_id": f"g{group}",
                           "mirror_pair_member": bias, "episode_seed": protocol["seed_start"] + group * 2 + member,
                           "model_split": protocol["variant_cycles_split"][group // 4]})
    data = {"group": np.array([r["mirror_group_id"] for r in scenes]), "split": np.array([r["model_split"] for r in scenes])}
    splits, _ = validate_splits(data, scenes, protocol)
    assert len(splits["train"]) == 48 and len(splits["development_validation"]) == 16
    data["split"][0] = "development_validation"
    with pytest.raises(ValueError, match="dataset split"):
        validate_splits(data, scenes, protocol)
    data["split"][0] = "train"
    scenes[0]["model_split"] = "development_validation"
    with pytest.raises(ValueError):
        validate_splits(data, scenes, protocol)
