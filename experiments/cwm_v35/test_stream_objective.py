"""Toy tensor gradient/math tests, not model optimization or efficacy."""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_audited_stream import toy_call
from audited_stream import base_values
from stream_objective import public_batch, batch_objective, factorial_response_loss, ranking_kl
from ranking_loss import ranking_cache


def fixture():
    calls = [toy_call('a', tail_invalid=True), toy_call('b')]
    ids = [17, 31]; weights = {'point': {17: .75, 31: 1.25}, 'cost': {17: .8, 31: 1.2}}
    _, slices = public_batch(calls, np.zeros((1, 1, 252)), np.ones((1, 1, 252)))
    caches = {i: ranking_cache(lambda target: target.square().sum((1, 2)), np.zeros((3, 8, 3)), base_values(call['values']))
        for i, call in zip(ids, calls)}
    response = torch.zeros((14, 8, 3), requires_grad=True)
    return calls, ids, slices, weights, caches, response


def test_matches_original_vectorL2_energy_and_base_KL_math_exactly():
    calls, ids, slices, weights, caches, response = fixture()
    loss, terms = batch_objective(response, calls, ids, slices, weights, caches)
    physical, original = factorial_response_loss(response, calls, [0, 1], slices,
        {'point': {0: .75, 1: 1.25}}, {'geometry': 'vector_l2', 'weighting': 'point'}, .001)
    auxiliary = (ranking_kl(caches[17], response[:3])*.8+ranking_kl(caches[31], response[7:10])*1.2)/2
    assert torch.equal(loss, physical+.1*auxiliary)
    assert torch.equal(terms, torch.cat([original.double(), auxiliary[None]]))
    loss.backward(); assert torch.isfinite(response.grad).all()


def test_augmented_probe_changes_physical_loss_but_NOT_base_ranking():
    calls, ids, slices, weights, caches, response = fixture()
    first, a = batch_objective(response, calls, ids, slices, weights, caches)
    modified = response.detach().clone(); modified[3:7] += .7
    second, b = batch_objective(modified, calls, ids, slices, weights, caches)
    assert torch.equal(a[-1], b[-1]) and not torch.equal(a[0], b[0]) and first != second


def test_private_labels_cannot_change_any_public_forward_tensor():
    calls, *_ = fixture(); altered = copy.deepcopy(calls)
    for call in altered:
        # Add a same shift to every branch and its anchor, preserving prefix equality.
        call['values']['target'] += 100.; call['values']['anchor_target'] += 100.
    normalizer = (np.zeros((1, 1, 252)), np.ones((1, 1, 252)))
    a, sa = public_batch(calls, *normalizer); b, sb = public_batch(altered, *normalizer)
    assert sa == sb and all(torch.equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize('kind', ['cost_support', 'cache_count', 'missing_cache', 'slice', 'duplicate_ids', 'nonfinite'])
def test_changed_population_or_partial_cache_refused(kind):
    calls, ids, slices, weights, caches, response = fixture()
    if kind == 'cost_support': weights['cost'][17] = 0.
    elif kind == 'cache_count': caches[17]['reference'] = torch.zeros((7, 8, 3))
    elif kind == 'missing_cache': del caches[17]
    elif kind == 'slice': slices[0] = (1, 7)
    elif kind == 'duplicate_ids': ids[1] = ids[0]
    else: response = response.detach(); response[0, 0, 0] = float('nan')
    with pytest.raises(ValueError): batch_objective(response, calls, ids, slices, weights, caches)


@pytest.mark.parametrize('kind', ['floor', 'dtype', 'shape', 'nonfinite'])
def test_normalizer_contract_refused(kind):
    calls, *_ = fixture(); mean = np.zeros((1, 1, 252)); scale = np.ones_like(mean)
    if kind == 'floor': scale[0, 0, 0] = .009
    elif kind == 'dtype': mean = mean.astype(np.float32)
    elif kind == 'shape': mean = mean[0]
    else: mean[0, 0, 0] = np.nan
    with pytest.raises(ValueError): public_batch(calls, mean, scale)
