import collections
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from cost_training_release import audit_optimizer_history, reconstruct_model
from train_cost_response import run_cost_epochs, state_digest
from cost_origin_model import CalibratedOrigin, FrozenOriginResponse, LocalTwoHead, evaluate_cost_model
from deployed_cost_loss import cost_population_weights, contrast_cache
from test_cost_learning import call, SquareScore


@pytest.fixture(scope='module', params=['gru_motion_only', 'cv_cost_l2'])
def small_checkpoint(request):
    torch.set_num_threads(1)
    p = json.loads((ROOT / 'training_protocol.json').read_text(encoding='utf8'))
    # Implementation test budgets only. Production validation compares against
    # immutable protocol; these are NOT new research runs or result evidence.
    p['training'] = {**p['training'], 'motion_epochs': 2, 'response_epochs': 2, 'batch_calls': 2}
    cfg = {**p['training'], 'seed': 993101}
    name = request.param
    origin = name.split('_')[0]
    motion = name.endswith('_motion_only')
    calls = [call('a'), call('a', complete=False), call('b')]
    indices = [0, 1, 2]
    mean, scale = np.zeros((1, 1, 252)), np.ones((1, 1, 252))
    weights, _ = cost_population_weights(calls, indices)
    counts = collections.Counter(c['record']['group'] for c in calls)
    torch.manual_seed(cfg['seed'])
    common = CalibratedOrigin(LocalTwoHead('motion_only'), origin)
    core = common.core
    if motion:
        model, caches = common, {}
    else:
        _, outputs = evaluate_cost_model(common, calls, indices, mean, scale, 2)
        caches = {i: contrast_cache(SquareScore(), o['reference'], calls[i]['values'])
                  for i, o in zip(indices, outputs) if weights['cost'][i] > 0}
        torch.manual_seed(cfg['seed'])
        core = LocalTwoHead('plain')
        model = FrozenOriginResponse(common, core)
    initial = state_digest(core)
    optimizer = torch.optim.Adam([param for param in model.parameters() if param.requires_grad], lr=.001)
    history, rng = run_cost_epochs(model, optimizer, name, calls, indices, mean, scale, weights,
                                   counts, caches, cfg, p)
    cp = {'configuration': name, 'seed': cfg['seed'], 'initial_state_digest': initial,
          'model_state': copy.deepcopy(model.state_dict()), 'optimizer_state': copy.deepcopy(optimizer.state_dict()),
          'torch_rng_state': torch.get_rng_state(), 'sampler_rng_state': rng}
    row = {'configuration': name, 'seed': cfg['seed'], 'origin': origin, 'initial_state_digest': initial}
    return p, cp, row, model, history


def reader(history):
    return lambda name: json.dumps(history).encode('utf8')


def test_optimizer_budget_rng_and_history_independently_reconstructed(small_checkpoint):
    p, cp, row, model, history = small_checkpoint
    audit_optimizer_history(reader(history), 'primary', 'example', cp, model, 3, p)
    rebuilt = reconstruct_model(row, p)
    rebuilt.load_state_dict(cp['model_state'], strict=True)
    assert state_digest(rebuilt) == state_digest(model)


@pytest.mark.parametrize('what', ['torch_rng', 'sampler_rng', 'step', 'moments', 'learning_rate', 'history'])
def test_forged_rng_budget_optimizer_or_history_rejected(small_checkpoint, what):
    p, original_cp, row, model, original_history = small_checkpoint
    cp, history = copy.deepcopy(original_cp), copy.deepcopy(original_history)
    state = next(iter(cp['optimizer_state']['state'].values()))
    if what == 'torch_rng':
        cp['torch_rng_state'][0] ^= 1
    elif what == 'sampler_rng':
        cp['sampler_rng_state']['state']['state'] += 1
    elif what == 'step':
        state['step'] += 1
    elif what == 'moments':
        state['exp_avg'] = torch.zeros(1)
    elif what == 'learning_rate':
        cp['optimizer_state']['param_groups'][0]['lr'] = .0001
    else:
        history.pop()
    with pytest.raises(ValueError):
        audit_optimizer_history(reader(history), 'primary', 'example', cp, model, 3, p)


def test_changed_initialization_or_truth_origin_rejected(small_checkpoint):
    p, cp, row, model, history = small_checkpoint
    bad = {**row, 'initial_state_digest': '0' * 64}
    with pytest.raises(ValueError, match='initialization'):
        reconstruct_model(bad, p)
    bad = {**row, 'origin': 'truth'}
    with pytest.raises(ValueError, match='origin'):
        reconstruct_model(bad, p)
