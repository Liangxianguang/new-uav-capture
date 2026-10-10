"""Synthetic80epoch optimizer restoration, never V34 scientific training."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent/'cwm_v36'),
    str(Path(__file__).resolve().parent.parent/'cwm_v35')]
import train_streamed as entry
from test_audited_stream import toy_call
from audited_stream import base_values
from ranking_loss import ranking_cache
from checkpoint_reload import reconstruct, exact_tree


@pytest.fixture(scope='module')
def toy_checkpoints():
    torch.set_num_threads(1); contract, plan = entry.load_contract(); seed = 1034101
    calls = [toy_call('a'), toy_call('b')]; indices = [0, 1]
    corpus = SimpleNamespace(indices={'train': indices}, batch=lambda ids: [calls[i] for i in ids])
    mean = np.zeros((1, 1, 252)); scale = np.ones_like(mean)
    config = {**plan['training'], 'seed': seed}
    torch.manual_seed(seed); common = entry.CalibratedOrigin(entry.LocalTwoHead('motion_only', 2.5, 2.5), 'cv')
    initial = entry.state_digest(common); optimizer = torch.optim.Adam(common.parameters(), lr=.001)
    _, rng = entry.run_epochs(common, optimizer, 'cv_motion_only', corpus, mean, scale, {'a': 1, 'b': 1}, None, None, config)
    fields = {'v34_plan': plan, 'v36_contract': contract, 'seed': seed, 'online_promoted': False,
        'baseline_weights_included': False, 'common_motion_in_response_optimizer': False}
    motion = {**fields, 'configuration': 'cv_motion_only', 'model_state': copy.deepcopy(common.state_dict()),
        'optimizer_state': copy.deepcopy(optimizer.state_dict()), 'torch_rng_state': torch.get_rng_state(),
        'sampler_rng_state': rng, 'initial_state_digest': initial, 'common_motion_state_digest': entry.state_digest(common)}
    common.eval().requires_grad_(False)
    for parameter in common.parameters(): parameter.grad = None
    torch.manual_seed(seed); core = entry.PrefixQueryResponse(True); initial = entry.state_digest(core)
    model = entry.FrozenOriginResponse(copy.deepcopy(common), core)
    optimizer = torch.optim.Adam(entry.checked_response_parameters(model), lr=.001)
    def cache_batch(ids, supplied):
        return {i: ranking_cache(lambda t: t.square().sum((1, 2)), np.zeros((3, 8, 3)), base_values(call['values']))
            for i, call in zip(ids, supplied)}
    cache = SimpleNamespace(batch=cache_batch); weights = {'point': {0: 1., 1: 1.}, 'cost': {0: 1., 1: 1.}}
    _, rng = entry.run_epochs(model, optimizer, 'cv_prefix_rank_l2', corpus, mean, scale, {}, weights, cache, config)
    response = {**fields, 'configuration': 'cv_prefix_rank_l2', 'model_state': copy.deepcopy(model.state_dict()),
        'optimizer_state': copy.deepcopy(optimizer.state_dict()), 'torch_rng_state': torch.get_rng_state(),
        'sampler_rng_state': rng, 'initial_state_digest': initial, 'common_motion_state_digest': entry.state_digest(common)}
    return motion, response


@pytest.mark.parametrize('index', [0, 1])
def test_real_optimizer_and_RNG_restored_without_update_or_global_rng_mutation(toy_checkpoints, index):
    checkpoint = toy_checkpoints[index]; before = torch.get_rng_state().clone()
    model, result = reconstruct(checkpoint, checkpoint['configuration'], 1034101, [0, 1])
    assert exact_tree(model.state_dict(), checkpoint['model_state'])
    assert exact_tree(before, torch.get_rng_state())
    assert result['expected_optimizer_steps'] == 80 and result['optimizer_updated_by_reload'] is False
    assert all(not p.requires_grad for p in model.parameters())


def test_motion_stage_state_actually_reloaded(toy_checkpoints):
    checkpoint = copy.deepcopy(toy_checkpoints[0]); del checkpoint['configuration']
    _, result = reconstruct(checkpoint, 'cv_motion_only', 1034101, [0, 1], stage=True)
    assert result['actual_optimizer_state_exact'] is True


@pytest.mark.parametrize('kind', ['step', 'missing_moment', 'moment_dtype', 'lr', 'rng', 'initial', 'common', 'promoted', 'nonfinite', 'weights_dtype', 'epoch_budget'])
def test_partial_or_modified_scientific_checkpoint_refused(toy_checkpoints, kind):
    checkpoint = copy.deepcopy(toy_checkpoints[1])
    state = checkpoint['optimizer_state']['state']; key = next(iter(state))
    if kind == 'step': state[key]['step'] = torch.tensor(79.)
    elif kind == 'missing_moment': del state[key]
    elif kind == 'moment_dtype': state[key]['exp_avg'] = state[key]['exp_avg'].double()
    elif kind == 'lr': checkpoint['optimizer_state']['param_groups'][0]['lr'] = .01
    elif kind == 'rng': checkpoint['sampler_rng_state']['state']['state'] += 1
    elif kind == 'initial': checkpoint['initial_state_digest'] = '0'*64
    elif kind == 'common': checkpoint['common_motion_state_digest'] = '0'*64
    elif kind == 'promoted': checkpoint['online_promoted'] = True
    elif kind == 'nonfinite': state[key]['exp_avg'].flatten()[0] = float('nan')
    elif kind == 'weights_dtype':
        name = next(iter(checkpoint['model_state'])); checkpoint['model_state'][name] = checkpoint['model_state'][name].double()
    else: checkpoint['v34_plan']['training']['response_epochs'] = 1
    with pytest.raises(ValueError): reconstruct(checkpoint, 'cv_prefix_rank_l2', 1034101, [0, 1])
