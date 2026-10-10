import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cost_origin_model import (CalibratedOrigin, FrozenOriginResponse, LocalTwoHead,
    public_cost_inputs, calibrated_motion_loss)
from deployed_cost_loss import (contrast_cache, deployed_contrast, cost_population_weights,
    cost_response_loss)


def values(n=3, points=8, complete=True):
    anchor = np.zeros((8, 4, 3), dtype=np.float32)
    proposed = np.repeat(anchor[None], n, 0)
    proposed[1:] += 1.
    mask = np.zeros((n, 8), dtype=bool)
    mask[:, :points] = True
    if not complete:
        mask[1, -1] = False
    return {'history': np.zeros((8, 252)), 'relative': np.zeros((4, 6)),
            'reference': np.ones(3), 'velocity': np.ones(3),
            'backbone': np.zeros((8, 3)), 'proposed': proposed, 'anchor': anchor,
            'valid': mask, 'anchor_valid': np.ones(8, dtype=bool),
            'target': np.ones((n, 8, 3)), 'anchor_target': np.zeros((8, 3))}


def call(group, **kwargs):
    return {'record': {'group': group}, 'values': values(**kwargs)}


class SquareScore:
    def __init__(self, offsets=None):
        self.offsets = 0. if offsets is None else torch.as_tensor(offsets, dtype=torch.float64)

    def __call__(self, paths):
        return paths.square().sum((1, 2)) + self.offsets


def test_public_inputs_ignore_all_private_labels_and_cv_not_neural_feature():
    calls = [call('a')]
    mean, scale = np.zeros((1, 1, 252)), np.ones((1, 1, 252))
    a, slices = public_cost_inputs(calls, [0], mean, scale)
    calls[0]['values']['target'][:] = -999.
    calls[0]['values']['anchor_target'][:] = 888.
    calls[0]['values']['valid'][:] = False
    b, _ = public_cost_inputs(calls, [0], mean, scale)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert len(a) == 6 and slices == [(0, 3)]
    expected = 1. + .1 * np.arange(1, 9)
    assert np.allclose(a[-1][0, :, 0], expected)
    torch.manual_seed(993101)
    model = CalibratedOrigin(LocalTwoHead('motion_only'), 'cv')
    _, calibration = model.calibration(*a)
    changed = (*a[:5], a[-1] + 100.)
    _, other = model.calibration(*changed)
    assert torch.equal(calibration, other)


@pytest.mark.parametrize('origin', ['gru', 'cv'])
def test_zero_initial_reference_and_frozen_response_anchor(origin):
    calls = [call('a')]
    inputs, _ = public_cost_inputs(calls, [0], np.zeros((1, 1, 252)), np.ones((1, 1, 252)))
    common = CalibratedOrigin(LocalTwoHead('motion_only'), origin)
    prediction, _, response = common(*inputs)
    assert torch.equal(prediction, inputs[4 if origin == 'gru' else 5])
    assert (response == 0).all()
    model = FrozenOriginResponse(common, LocalTwoHead('plain'))
    frozen = copy.deepcopy(common.state_dict())
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    model.train()
    assert not common.training
    for _ in range(2):
        p, motion, r = model(*inputs)
        assert (r[0] == 0).all() and torch.equal(p[0], prediction[0])
        optimizer.zero_grad(set_to_none=True)
        (r - 1.).square().mean().backward()
        optimizer.step()
    assert all(torch.equal(v, common.state_dict()[k]) for k, v in frozen.items())
    assert all(p.grad is None and not p.requires_grad for p in common.parameters())
    assert all(p.grad is None for p in model.response_core.motion_head.parameters())


def test_calibration_energy_not_cv_minus_gru_offset():
    calls = [call('a')]
    model = CalibratedOrigin(LocalTwoHead('motion_only'), 'cv')
    _, terms = calibrated_motion_loss(model, calls, [0], np.zeros((1, 1, 252)),
                                     np.ones((1, 1, 252)), {'a': 1}, .001)
    assert terms[0] > 1. and terms[1] == 0.
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in LocalTwoHead('motion_only').parameters())


def test_deployed_not_true_reference_l1_and_detached_labels():
    v = values()
    v['target'][0] = v['anchor_target']
    ref = torch.full((3, 8, 3), 2., dtype=torch.float64, requires_grad=True)
    cache = contrast_cache(SquareScore(), ref, v)
    r = torch.zeros((3, 8, 3), dtype=torch.float64, requires_grad=True)
    loss = deployed_contrast(cache, r)
    expected = 24. / (4. * np.sqrt(24.))
    assert np.isclose(float(loss.detach()), expected)
    loss.backward()
    assert ref.grad is None
    assert r.grad is not None and (r.grad[1:] < 0).all() and (r.grad[0] == 0).all()
    assert all(not cache[k].requires_grad for k in ('reference', 'reference_cost', 'gradient', 'scale', 'label_effect'))
    # Zero anchor effect, not a cross-action baseline cost subtraction.
    other = contrast_cache(SquareScore([1., 7., 123.]), ref, v)
    assert torch.equal(cache['label_effect'], other['label_effect'])
    assert torch.equal(deployed_contrast(other, r), loss.detach())


def test_full_cost_rejects_terminal_padding_and_candidate_dependent_motion():
    v = values(complete=False)
    with pytest.raises(ValueError, match='terminal-incomplete'):
        contrast_cache(SquareScore(), np.zeros((3, 8, 3)), v)
    v = values()
    ref = np.zeros((3, 8, 3))
    ref[1] += .1
    with pytest.raises(ValueError, match='Action-independent'):
        contrast_cache(SquareScore(), ref, v)


def test_population_cost_excludes_incomplete_and_has_fixed_batch_denominator():
    calls = [call('a'), call('a', complete=False), call('b')]
    weights, report = cost_population_weights(calls, [0, 1, 2])
    assert weights['cost'] == {0: 1.5, 1: 0., 2: 1.5}
    assert report['group_complete_cost_calls'] == {'a': 1, 'b': 1}
    caches = {i: contrast_cache(SquareScore(), np.ones((3, 8, 3)), calls[i]['values']) for i in (0, 2)}
    response = torch.zeros((9, 8, 3), dtype=torch.float64, requires_grad=True)
    _, terms = cost_response_loss(response, calls, [0, 1, 2], [(0, 3), (3, 6), (6, 9)],
                                  weights, caches, .001, .1)
    _, singleton = cost_response_loss(response[:3], calls, [0], [(0, 3)], weights, caches, .001, .1)
    raw = deployed_contrast(caches[0], response[:3])
    assert torch.allclose(terms[-1], raw)
    assert torch.allclose(singleton[-1], 1.5 * raw)  # not renormalized to raw
    split_terms = []
    for j in range(3):
        _, part = cost_response_loss(response[j*3:(j+1)*3], calls, [j], [(0, 3)], weights, caches, .001, .1)
        split_terms.append(part)
    assert torch.allclose(torch.stack(split_terms).mean(0), terms)
    with pytest.raises(ValueError, match='every training group'):
        cost_population_weights([call('a'), call('b', complete=False)], [0, 1])


def test_unknown_origins_and_cost_weights_fail_closed():
    with pytest.raises(ValueError, match='origin'):
        CalibratedOrigin(LocalTwoHead('motion_only'), 'truth')
    calls = [call('a')]
    weights, _ = cost_population_weights(calls, [0])
    response = torch.zeros((3, 8, 3))
    with pytest.raises(ValueError, match='preregistered'):
        cost_response_loss(response, calls, [0], [(0, 3)], weights, {}, .001, 1.)
