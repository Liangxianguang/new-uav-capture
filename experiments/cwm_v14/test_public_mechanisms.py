import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from public_mechanism_model import PublicMechanismModel, public_inputs, single_call_loss, batch_call_loss
from train_public_mechanisms import call_metrics, prior_probabilities, qualification
import json


def sample():
    rng = np.random.default_rng(44)
    anchor = rng.normal(size=(8, 4, 3))
    values = {'history': rng.normal(size=(8, 252)), 'relative': rng.normal(size=(4, 6)),
              'anchor': anchor, 'proposed': np.stack([anchor, anchor+.1]),
              'valid': np.ones((2, 8), bool), 'anchor_valid': np.ones(8, bool),
              'anchor_commanded': anchor*.9, 'commanded': np.stack([anchor*.9, (anchor+.1)*.9]),
              'termination': np.full((2, 8), 'running'), 'anchor_termination': np.full(8, 'running')}
    private = {'branch_after_label_only': np.ones((2, 8), int), 'anchor_branch_after_label_only': np.ones(8, int)}
    return values, private


@pytest.mark.parametrize('kind', ['branch_snapshot', 'branch_history', 'cbf_history'])
def test_labels_cannot_change_public_prediction(kind):
    torch.set_num_threads(1)
    torch.manual_seed(976101)
    model = PublicMechanismModel(kind).eval()
    values, private = sample()
    mean, scale = np.zeros(252), np.ones(252)
    before = model(*public_inputs(values, mean, scale))
    changed = copy.deepcopy(values)
    changed['commanded'][:] = 1000
    changed['target'] = np.full((2, 8, 3), 1e6)
    changed['branch_after_label_only'] = -np.ones((2, 8))
    changed['valid'][:] = False
    changed['termination'][:] = 'bad'
    assert torch.equal(before, model(*public_inputs(changed, mean, scale)))


@pytest.mark.parametrize('kind', ['branch_history', 'cbf_history'])
def test_terminal_padding_has_zero_supervised_influence(kind):
    values, private = sample()
    values['valid'][:, 4:] = False
    values['anchor_valid'][4:] = False
    values['termination'] = np.array([['running']*4 + ['after_terminal']*4]*2)
    values['anchor_termination'] = values['termination'][0].copy()
    model = PublicMechanismModel(kind)
    mean, scale = np.zeros(252), np.ones(252)
    initial = single_call_loss(model, values, private, mean, scale, .0001)
    altered = copy.deepcopy(values)
    labels = copy.deepcopy(private)
    altered['commanded'][:, 4:] = 1e6
    altered['anchor_commanded'][4:] = 1e6
    labels['branch_after_label_only'][:, 4:] = -1
    labels['anchor_branch_after_label_only'][4:] = -1
    assert torch.equal(initial, single_call_loss(model, altered, labels, mean, scale, .0001))


def test_train_prior_not_affected_by_development():
    values, private = sample()
    calls = [{'record': {'group': 'train', 'step': 2}, 'values': values, 'private': private},
             {'record': {'group': 'dev', 'step': 2}, 'values': values, 'private': copy.deepcopy(private)}]
    prior = prior_probabilities(calls, [0], 1.)
    calls[1]['private']['branch_after_label_only'][:] = -1
    assert prior == prior_probabilities(calls, [0], 1.)


def test_exact_anchor_has_no_paired_branch_effect():
    values, private = sample()
    probability = np.repeat(np.eye(3)[np.full((1, 8), 2)][None], 3, 0).reshape(3, 8, 3)
    metrics = call_metrics(values, private, probability, probability, None)
    assert metrics['paired_points'] == 8  # equal anchor candidate excluded
    assert metrics['model_paired_brier'] == metrics['zero_paired_brier'] == 0.


def test_nonfinite_or_wrong_public_input_rejected():
    values, _ = sample()
    values['history'][0, 0] = np.nan
    with pytest.raises(ValueError, match='Nonfinite'):
        public_inputs(values, np.zeros(252), np.ones(252))


def test_branch_accuracy_alone_cannot_promote_mechanism():
    protocol = json.loads((ROOT / 'training_protocol.json').read_text())
    rows = []
    for kind in protocol['models']:
        for seed in protocol['training']['seeds']:
            metrics = {'model_nll': .2, 'prior_nll': .5, 'model_paired_brier': .2, 'zero_paired_brier': .1,
                       'model_command_error_mps': .5, 'proposed_command_error_mps': .5,
                       'group_metrics': {f'g{i}': {'model_nll': .2, 'prior_nll': .5, 'model_paired_brier': .2, 'zero_paired_brier': .1,
                                                  'model_command_error_mps': .5, 'proposed_command_error_mps': .5} for i in range(8)}}
            rows.append({'kind': kind, 'seed': seed, 'development': metrics})
    gates = qualification(rows, protocol)
    assert not any(g['mechanism_research_eligible'] for g in gates.values())
    assert not any(g['online_promoted'] for g in gates.values())


def test_candidate_reference_probability_difference_is_required():
    values, private = sample()
    private['branch_after_label_only'][1] = -1
    probability = np.repeat(np.eye(3)[np.full((1, 8), 2)], 3, 0)
    measurements = call_metrics(values, private, probability, probability, None)
    assert measurements['zero_paired_brier'] == 2.
    assert measurements['model_paired_brier'] == 2.
    assert measurements['calibration']['model'][-1]['fraction'] == 1.


@pytest.mark.parametrize('kind',['branch_history','cbf_history'])
def test_vectorized_loss_matches_per_call_objective_and_gradient(kind):
    torch.set_num_threads(1)
    values, private = sample()
    calls = [{'record':{'group':'a'},'values':values,'private':private},
             {'record':{'group':'b'},'values':copy.deepcopy(values),'private':copy.deepcopy(private)}]
    calls[1]['values']['valid'][:,4:] = False
    mean, scale = np.zeros(252), np.ones(252)
    model = PublicMechanismModel(kind)
    clone = copy.deepcopy(model)
    actual = batch_call_loss(model,calls,[0,1],mean,scale,{'a':2,'b':1},.0001)
    expected = (single_call_loss(clone,values,private,mean,scale,.0001)*.5 + single_call_loss(clone,calls[1]['values'],calls[1]['private'],mean,scale,.0001))/1.5
    assert torch.allclose(actual,expected,atol=1e-7,rtol=1e-6)
    actual.backward()
    expected.backward()
    assert all(torch.allclose(a.grad,b.grad,atol=1e-7,rtol=1e-6) for a,b in zip(model.parameters(),clone.parameters()))
