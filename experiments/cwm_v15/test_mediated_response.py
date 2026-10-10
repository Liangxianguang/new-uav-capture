import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from mediated_model import FrozenMediator, LocalTwoHead, pack, physical_loss, state_digest
from train_mediated_response import qualification, core_kind
from train_two_head import batch_loss, pack_inputs


def sample():
    rng = np.random.default_rng(978101)
    anchor = rng.normal(size=(8, 4, 3))
    proposed = np.stack([anchor, anchor + .2])
    values = {'history': rng.normal(size=(8, 252)), 'relative': rng.normal(size=(4, 6)),
              'anchor': anchor, 'proposed': proposed, 'backbone': rng.normal(size=(8, 3)),
              'target': rng.normal(size=(2, 8, 3)), 'anchor_target': rng.normal(size=(8, 3)),
              'valid': np.ones((2, 8), bool), 'anchor_valid': np.ones(8, bool),
              'commanded': proposed * .9, 'anchor_commanded': anchor * .9,
              'termination': np.full((2, 8), 'running'), 'anchor_termination': np.full(8, 'running')}
    return [{'record': {'group': 'a'}, 'values': values}]


def mediator():
    return FrozenMediator(ROOT.parent / 'cwm_v14/artifacts/public_mechanism_training_20261010.zip',
                          json.loads((ROOT / 'training_protocol.json').read_text()))


def test_frozen_public_mediator_has_no_private_input_or_rng_mutation():
    torch.set_num_threads(1)
    torch.manual_seed(979101)
    rng_before = torch.get_rng_state().clone()
    frozen = mediator()
    assert torch.equal(rng_before, torch.get_rng_state())
    values = sample()[0]['values']
    estimate = frozen.estimate(values)
    changed = copy.deepcopy(values)
    for key in ('target', 'anchor_target', 'commanded', 'anchor_commanded'):
        changed[key][:] = 1e6
    changed['valid'][:] = False
    changed['anchor_valid'][:] = False
    changed['termination'][:] = 'after_terminal'
    changed['branch_after_label_only'] = np.full((2, 8), -1)
    changed['executed'] = np.full((2, 8, 4, 3), 1e6)
    assert np.array_equal(estimate, frozen.estimate(changed))
    assert np.array_equal(estimate[0], estimate[1])
    assert frozen.unchanged()
    assert torch.equal(rng_before, torch.get_rng_state())


def test_wrong_frozen_archive_fails_closed():
    protocol = json.loads((ROOT / 'training_protocol.json').read_text())
    protocol['mediator_archive_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='archive mismatch'):
        FrozenMediator(ROOT.parent / 'cwm_v14/artifacts/public_mechanism_training_20261010.zip', protocol)


@pytest.mark.parametrize('kind', ['plain', 'motion_only'])
def test_raw_loss_and_gradients_equal_original_v10(kind):
    calls = sample()
    mean, scale = np.zeros(252), np.ones(252)
    config = json.loads((ROOT / 'training_protocol.json').read_text())['training']
    torch.manual_seed(979101)
    model = LocalTwoHead(kind)
    with torch.no_grad():
        model.motion_head.weight.normal_(0, .02)
        if kind != 'motion_only':
            model.response_head.weight.normal_(0, .02)
    clone = copy.deepcopy(model)
    actual, terms = physical_loss(model, 'raw_plain' if kind == 'plain' else kind,
                                  calls, [0], mean, scale, {'a': 1}, config)
    expected, original_terms = batch_loss(clone, calls, [0], mean, scale, {'a': 1}, config)
    assert torch.equal(actual, expected)
    assert torch.equal(terms, original_terms)
    actual.backward()
    expected.backward()
    for a, b in zip(model.parameters(), clone.parameters()):
        assert (a.grad is None and b.grad is None) or torch.equal(a.grad, b.grad)
    raw, slices = pack(calls, [0], mean, scale, False)
    original, original_slices = pack_inputs(calls, [0], mean, scale)
    assert slices == original_slices
    assert all(torch.equal(a, b) for a, b in zip(raw, original))


def test_mediated_loss_uses_raw_support_and_ignores_terminal_padding():
    calls = sample()
    values = calls[0]['values']
    values['valid'][:, 4:] = False
    values['anchor_valid'][4:] = False
    # A mediator may map distinct raw proposals onto the same command. Such
    # pairs still belong in the raw intervention population and response loss.
    calls[0]['estimated_commands'] = np.repeat(values['anchor'][None], 3, 0)
    config = json.loads((ROOT / 'training_protocol.json').read_text())['training']
    model = LocalTwoHead('plain')
    loss, terms = physical_loss(model, 'mediated_plain', calls, [0], np.zeros(252), np.ones(252), {'a': 1}, config)
    expected_response = np.mean((values['target'][1, :4] - values['anchor_target'][:4]) ** 2)
    assert np.isclose(float(terms[1]), expected_response)
    changed = copy.deepcopy(calls)
    changed[0]['values']['target'][:, 4:] = 1e6
    changed[0]['values']['anchor_target'][4:] = 1e6
    other, _ = physical_loss(model, 'mediated_plain', changed, [0], np.zeros(252), np.ones(252), {'a': 1}, config)
    assert torch.equal(loss, other)


def test_raw_mediated_core_initialization_and_sampler_match():
    protocol = json.loads((ROOT / 'training_protocol.json').read_text())
    for seed in protocol['training']['seeds']:
        models = []
        for name in ('raw_plain', 'mediated_plain'):
            torch.manual_seed(seed)
            models.append(LocalTwoHead(core_kind(name)))
        assert state_digest(models[0]) == state_digest(models[1])
        left, right = np.random.default_rng(seed), np.random.default_rng(seed)
        for _ in range(protocol['training']['epochs']):
            assert np.array_equal(left.permutation(1152), right.permutation(1152))


@pytest.mark.parametrize('failure', ['response', 'decision', None])
def test_motion_ade_alone_cannot_promote_and_gain_must_be_positive(failure):
    protocol = json.loads((ROOT / 'training_protocol.json').read_text())
    models, decisions = [], {}
    for name in protocol['models']:
        for seed in protocol['training']['seeds']:
            models.append({'configuration': name, 'seed': seed, 'development': {
                'group_equal_ade_m': .1, 'group_equal_paired_response_error_m':
                .3 if name == 'mediated_plain' and failure == 'response' else .1 if name == 'mediated_plain' else .2,
                'anchor_response_exact_zero': True}})
            decisions[f'{name}_seed{seed}'] = [
                {'episode_index': g, 'step': 2, 'agent': 0, 'group': f'g{g}', 'metrics': {
                    'model': {'regret_in_diagnostic_cost': 1. if name == 'mediated_plain' and failure != 'decision' else 2.},
                    'own_motion_component': {'regret_in_diagnostic_cost': 2.},
                    'constant_velocity': {'regret_in_diagnostic_cost': 2.}}} for g in range(8)]
    controls = {'frozen_gru': {'group_equal_ade_m': 1., 'group_equal_paired_response_error_m': .2},
                'constant_velocity': {'group_equal_ade_m': 1.}}
    _, gates = qualification(models, controls, decisions, protocol)
    assert gates['mediated_plain']['research_eligible'] == (failure is None)
    assert gates['mediated_plain']['online_promoted'] is False
