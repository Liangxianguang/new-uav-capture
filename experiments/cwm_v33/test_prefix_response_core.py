"""Untrained synthetic architecture fixtures, not learned model efficacy."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prefix_response_core import PrefixQueryResponse


@pytest.fixture
def inputs():
    task_generator = torch.Generator().manual_seed(3301)
    history = torch.randn(3, 8, 252, generator=task_generator)
    relative = torch.randn(3, 4, 6, generator=task_generator)
    proposed = torch.randn(3, 8, 4, 3, generator=task_generator)
    anchor = torch.randn(3, 8, 4, 3, generator=task_generator)
    backbone = torch.randn(3, 8, 3, generator=task_generator).double()
    return history, relative, proposed, anchor, backbone


def randomized_pair():
    with torch.random.fork_rng():
        torch.manual_seed(3302)
        prefix = PrefixQueryResponse(True)
        torch.nn.init.normal_(prefix.response_head.weight, std=.1)
        full = PrefixQueryResponse(False); full.load_state_dict(prefix.state_dict())
    return prefix.eval(), full.eval()


def test_capacity_and_neural_tensors_are_exactly_matched():
    prefix, full = randomized_pair()
    assert sum(p.numel() for p in prefix.parameters()) == sum(p.numel() for p in full.parameters())
    assert sum(p.numel() for p in prefix.parameters() if p.requires_grad) == sum(p.numel() for p in full.parameters() if p.requires_grad)
    assert list(prefix.state_dict()) == list(full.state_dict())
    assert all(torch.equal(v, full.state_dict()[k]) for k, v in prefix.state_dict().items())
    assert not any(p.requires_grad for p in prefix.motion_head.parameters())


@pytest.mark.parametrize('causal', [True, False])
def test_zero_init_is_exact_backbone_and_no_input_mutation(inputs, causal):
    model = PrefixQueryResponse(causal)
    saved = [v.clone() for v in inputs]
    prediction, motion, response = model(*inputs)
    assert torch.equal(prediction, inputs[-1])
    assert not motion.any() and not response.any()
    assert all(torch.equal(v, before) for v, before in zip(inputs, saved))


@pytest.mark.parametrize('prefix_length', range(1, 9))
def test_arbitrary_future_proposed_and_anchor_changes_leave_prefix_invariant(inputs, prefix_length):
    prefix, _ = randomized_pair()
    expected = prefix(*inputs)[2]
    changed = [v.clone() for v in inputs]
    changed[2][:, prefix_length:] = 2.
    changed[3][:, prefix_length:] = -2.
    actual = prefix(*changed)[2]
    assert torch.equal(expected[:, :prefix_length], actual[:, :prefix_length])
    if prefix_length < 8: assert not torch.equal(expected[:, prefix_length:], actual[:, prefix_length:])


def test_matched_full_sequence_control_can_use_future_actions(inputs):
    prefix, full = randomized_pair()
    changed = [v.clone() for v in inputs]; changed[2][:, 1:] *= -1
    assert torch.equal(prefix(*inputs)[2][:, :1], prefix(*changed)[2][:, :1])
    assert not torch.equal(full(*inputs)[2][:, :1], full(*changed)[2][:, :1])


@pytest.mark.parametrize('causal', [True, False])
def test_anchor_and_prefix_anchor_responses_are_exact_zero(inputs, causal):
    pair = randomized_pair(); model = pair[0] if causal else pair[1]
    h, r, p, a, b = inputs
    prediction, _, response = model(h, r, a, a, b)
    assert not response.any() and torch.equal(prediction, b)
    p = p.clone(); p[:, :4] = a[:, :4]
    response = model(h, r, p, a, b)[2]
    assert not response[:, :4].any()


def test_future_action_gradients_are_zero_but_past_actions_are_trainable(inputs):
    prefix, full = randomized_pair()
    for model, future_expected in ((prefix, False), (full, True)):
        h, r, p, a, b = inputs; p = p.clone().requires_grad_(True); a = a.clone().requires_grad_(True)
        response = model(h, r, p, a, b)[2]
        response[:, 2].sum().backward()
        assert p.grad[:, :3].abs().sum() > 0 and a.grad[:, :3].abs().sum() > 0
        assert bool(p.grad[:, 3:].abs().sum() > 0) is future_expected
        assert bool(a.grad[:, 3:].abs().sum() > 0) is future_expected
        assert model.history_encoder.weight_ih_l0.grad is not None
        assert all(p.grad is None for p in model.motion_head.parameters())


@pytest.mark.parametrize('change', ['history', 'relative', 'actions', 'backbone', 'nonfinite'])
def test_invalid_public_input_refused(inputs, change):
    model = PrefixQueryResponse(True); values = [v.clone() for v in inputs]
    if change == 'history': values[0] = values[0][:, :7]
    elif change == 'relative': values[1] = values[1][:, :3]
    elif change == 'actions': values[2] = values[2][:, :7]
    elif change == 'backbone': values[4] = values[4][:, :7]
    else: values[2][0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError): model(*values)


@pytest.mark.parametrize('mode,scale', [(1, 2.5), ('true', 2.5), (True, 0.),
    (True, -1.), (True, float('nan')), (True, float('inf')), (True, True), (True, '2.5')])
def test_invalid_architecture_configuration_refused(mode, scale):
    with pytest.raises(ValueError): PrefixQueryResponse(mode, response_scale_m=scale)


def test_unchanged_frozen_origin_wrapper_interface_and_gradient_separation(inputs):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'cwm_v23'))
    from cost_origin_model import CalibratedOrigin, FrozenOriginResponse, LocalTwoHead
    common = CalibratedOrigin(LocalTwoHead('motion_only'), 'cv')
    response = PrefixQueryResponse(True)
    wrapper = FrozenOriginResponse(common, response)
    before = {k: v.clone() for k, v in common.state_dict().items()}
    h, r, p, a, b = inputs; cv = b+.1
    prediction, _, output = wrapper(h, r, p, a, b, cv)
    assert torch.equal(prediction, cv) and not output.any()
    wrapper.train()
    assert not common.training
    optimizer = torch.optim.Adam([p for p in wrapper.parameters() if p.requires_grad], lr=.001)
    prediction.sum().backward(); optimizer.step()
    assert all(p.grad is None and not p.requires_grad for p in common.parameters())
    assert all(torch.equal(v, common.state_dict()[k]) for k, v in before.items())
    assert response.response_head.weight.grad is not None
    assert all(p.grad is None for p in response.motion_head.parameters())
