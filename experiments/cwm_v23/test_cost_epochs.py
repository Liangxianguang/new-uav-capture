"""Small synthetic implementation/reproducibility checks, not research runs."""
import collections
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_cost_response import run_cost_epochs, state_digest
from cost_origin_model import (CalibratedOrigin, FrozenOriginResponse, LocalTwoHead,
    evaluate_cost_model)
from deployed_cost_loss import cost_population_weights, contrast_cache
from test_cost_learning import call, SquareScore


def tiny_training():
    torch.set_num_threads(1)
    protocol = json.loads((ROOT / 'training_protocol.json').read_text(encoding='utf8'))
    config = {**protocol['training'], 'seed': 993101, 'motion_epochs': 2, 'response_epochs': 2, 'batch_calls': 2}
    calls = [call('a'), call('a', complete=False), call('b')]
    indices = [0, 1, 2]
    mean, scale = np.zeros((1, 1, 252)), np.ones((1, 1, 252))
    weights, _ = cost_population_weights(calls, indices)
    counts = collections.Counter(c['record']['group'] for c in calls)
    outcomes = {}
    for origin in ('gru', 'cv'):
        torch.manual_seed(config['seed'])
        common = CalibratedOrigin(LocalTwoHead('motion_only'), origin)
        initial = state_digest(common.core)
        optimizer = torch.optim.Adam([p for p in common.parameters() if p.requires_grad], lr=.001)
        history, rng = run_cost_epochs(common, optimizer, origin + '_motion_only', calls, indices,
            mean, scale, weights, counts, {}, config, protocol)
        common.eval().requires_grad_(False)
        for p in common.parameters():
            p.grad = None
        outcomes[origin + '_motion_only'] = (initial, state_digest(common), history, rng)
        _, outputs = evaluate_cost_model(common, calls, indices, mean, scale, 2)
        caches = {i: contrast_cache(SquareScore(), o['reference'], calls[i]['values'])
                  for i, o in zip(indices, outputs) if weights['cost'][i] > 0}
        common_before = state_digest(common)
        for suffix in ('l2', 'cost_l2'):
            torch.manual_seed(config['seed'])
            response = LocalTwoHead('plain')
            initial = state_digest(response)
            model = FrozenOriginResponse(copy.deepcopy(common), response)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
            history, rng = run_cost_epochs(model, optimizer, origin + '_' + suffix, calls, indices,
                mean, scale, weights, counts, caches, config, protocol)
            assert state_digest(model.common) == common_before
            assert all(p.grad is None for p in model.common.parameters())
            _, outputs = evaluate_cost_model(model, calls, indices, mean, scale, 2)
            assert all(np.array_equal(o['reference'], caches[i]['reference'].numpy())
                       for i, o in zip(indices, outputs) if i in caches)
            outcomes[origin + '_' + suffix] = (initial, state_digest(model), history, rng)
    return outcomes


def test_two_stage_matched_initialization_and_two_independent_small_runs():
    a, b = tiny_training(), tiny_training()
    assert a == b
    assert len(a) == 6
    assert a['gru_motion_only'][0] == a['cv_motion_only'][0]
    assert len({a[k][0] for k in ('gru_l2', 'cv_l2', 'gru_cost_l2', 'cv_cost_l2')}) == 1
    assert all(len(v[2]) == 2 for v in a.values())
    assert a['gru_l2'][1] != a['gru_cost_l2'][1]
    assert a['cv_l2'][1] != a['cv_cost_l2'][1]
