import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from residual_model import AnchoredResponse, paired_loss
from train_residual import split_groups


def example():
    return (torch.randn(3, 8, 252), torch.randn(3, 4, 6), torch.randn(3, 8, 4, 3),
            torch.randn(3, 8, 4, 3), torch.randn(3, 8, 3))


@pytest.mark.parametrize("kind", ["plain", "structured"])
def test_zero_init_preserves_backbone(kind):
    model = AnchoredResponse(kind)
    history, relative, proposed, anchor, backbone = example()
    prediction, response = model(history, relative, proposed, anchor, backbone)
    assert torch.equal(prediction, backbone)
    assert torch.count_nonzero(response) == 0


@pytest.mark.parametrize("kind", ["plain", "structured"])
def test_anchor_is_byte_exact_even_after_training_and_signed_zero(kind):
    model = AnchoredResponse(kind)
    with torch.no_grad():
        model.head[-1].weight.normal_()
        model.head[-1].bias.normal_()
    history, relative, proposed, _, backbone = example()
    backbone[0, 0, 0] = -0.
    saved = backbone.numpy().tobytes()
    prediction, response = model(history, relative, proposed, proposed, backbone)
    assert prediction.detach().numpy().tobytes() == saved
    assert torch.count_nonzero(response) == 0
    assert backbone.numpy().tobytes() == saved


def test_structural_node_permutation_and_bound():
    model = AnchoredResponse("structured")
    with torch.no_grad():
        model.head[-1].weight.normal_()
    h, r, u, a, b = example()
    _, first = model(h, r, u, a, b)
    permutation = [3, 1, 0, 2]
    _, permuted = model(h, r[:, permutation], u[:, :, permutation], a[:, :, permutation], b)
    assert torch.allclose(first, permuted, atol=1e-7, rtol=1e-6)
    assert torch.max(torch.abs(first)) <= .05
    assert torch.max(torch.linalg.vector_norm(first, dim=-1)) <= np.sqrt(3) * .05 + 1e-7


def test_split_groups_disjoint_and_deterministic():
    data = {"variant": np.repeat(["a", "b"], 12),
            "group": np.asarray([f"{v}{i}" for v in ("a", "b") for i in range(4) for _ in range(3)])}
    splits, assigned = split_groups(data, 930910)
    assert np.array_equal(splits, split_groups(data, 930910)[0])
    assert set(assigned.values()) == {"train", "calibration", "development_validation"}
    for group in assigned:
        assert len(set(splits[data["group"] == group])) == 1


def test_masked_invalid_labels_do_not_change_loss():
    response = torch.randn(2, 4, 8, 3, requires_grad=True)
    labels = torch.randn(2, 4, 8, 3)
    masks = torch.ones(2, 4, 8, dtype=torch.bool)
    masks[:, :, 4:] = False
    before = paired_loss(response, labels, masks)[0]
    labels[:, :, 4:] = 1e6
    after = paired_loss(response, labels, masks)[0]
    assert torch.equal(before, after)
    after.backward()
    assert torch.isfinite(response.grad).all()
    assert torch.count_nonzero(response.grad[:, :, 4:]) == 0


def test_matched_parameter_budget_and_equal_inputs():
    counts = [sum(p.numel() for p in AnchoredResponse(k).parameters()) for k in ("plain", "structured")]
    assert abs(counts[0] - counts[1]) / max(counts) < .1
    assert all("backbone" not in key for key in AnchoredResponse("structured").state_dict())
