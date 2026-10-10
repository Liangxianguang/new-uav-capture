import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualification_gate import OptionalResponseAdapter, QualificationGate, REQUIRED_ONLINE_CHECKS


def inputs():
    p = np.zeros((2, 8, 4, 3), dtype=np.float64)
    p[1, :, 0, 0] = 1.
    return (np.zeros((8, 252)), np.zeros((4, 6)), p, p[0].copy(), np.zeros((8, 3)), np.zeros((8, 3)))


def qualified():
    return {'research_eligible': True, 'online_checks': {key: True for key in REQUIRED_ONLINE_CHECKS}}


def predictor(*args):
    p = args[2]
    reference = np.full((len(p), 8, 3), .2, dtype=np.float64)
    response = np.zeros(reference.shape, dtype=np.float32)
    response[1, :, 0] = .1
    return {'reference': reference, 'response': response,
            'prediction': np.where(response == 0, reference, reference + response.astype(reference.dtype))}


@pytest.mark.parametrize('mode,qual', [('off', qualified()), ('guarded', {'research_eligible': False}),
                                       ('guarded', {'research_eligible': True})])
def test_off_refusal_pending_never_load_or_inspect(mode, qual):
    adapter = OptionalResponseAdapter(QualificationGate(mode=mode, qualification=qual),
        lambda: (_ for _ in ()).throw(AssertionError('must not load')))
    result, info = adapter.forecast(*([object()] * 6))
    assert result is None and not info['forecast_available'] and not info['control_eligible']
    assert adapter.load_calls == adapter.prediction_calls == 0


def test_valid_outputs_are_target_paths_not_actions_and_calibration_allowed():
    adapter = OptionalResponseAdapter(QualificationGate(mode='guarded', qualification=qualified()), lambda: predictor)
    values = inputs()
    result, info = adapter.forecast(*values)
    assert result['prediction'].shape == (2, 8, 3) != values[2].shape
    assert info['control_eligible'] and adapter.load_calls == adapter.prediction_calls == 1
    assert not np.array_equal(result['reference'][0], values[4])


def test_failed_model_may_only_forecast_in_shadow():
    gate = QualificationGate(mode='shadow', qualification={'research_eligible': False})
    result, info = OptionalResponseAdapter(gate, lambda: predictor).forecast(*inputs())
    assert result is not None and not info['control_eligible'] and not gate.control_eligible


@pytest.mark.parametrize('failure', ['load', 'predict', 'action_shape', 'anchor', 'reference', 'sum', 'nan'])
def test_failures_return_empty_forecast_and_disable_further_calls(failure):
    def load():
        if failure == 'load':
            raise OSError('missing')
        def predict(*args):
            if failure == 'predict':
                raise RuntimeError('broken')
            out = predictor(*args)
            if failure == 'action_shape':
                out['prediction'] = args[2]
            elif failure == 'anchor':
                out['response'][0, 0, 0] = .1
            elif failure == 'reference':
                out['reference'][1, 0, 0] += .1
            elif failure == 'sum':
                out['prediction'][1, 0, 0] += .1
            elif failure == 'nan':
                out['prediction'][0, 0, 0] = np.nan
            return out
        return predict
    adapter = OptionalResponseAdapter(QualificationGate(mode='guarded', qualification=qualified()), load)
    result, info = adapter.forecast(*inputs())
    assert result is None and not info['control_eligible']
    counts = adapter.load_calls, adapter.prediction_calls
    assert adapter.forecast(*([object()] * 6))[0] is None
    assert counts == (adapter.load_calls, adapter.prediction_calls)


def test_inputs_and_global_rng_preserved_even_with_mutating_random_predictor():
    values = inputs()
    saved = [x.copy() for x in values]
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    def load():
        random.random(); np.random.rand(); torch.rand(1)
        def predict(*args):
            out = predictor(*args)
            for arg in args:
                arg[...] = 100.
            random.random(); np.random.rand(); torch.rand(1)
            return out
        return predict
    adapter = OptionalResponseAdapter(QualificationGate(mode='shadow'), load)
    assert adapter.forecast(*values)[0] is not None
    for before, after in zip(saved, values):
        np.testing.assert_array_equal(before, after)
    assert random.getstate() == python_state and torch.equal(torch.get_rng_state(), torch_state)
    measured = np.random.get_state()
    assert measured[0] == numpy_state[0] and np.array_equal(measured[1], numpy_state[1]) and measured[2:] == numpy_state[2:]


@pytest.mark.parametrize('value', ['true', 1, [], None])
def test_truthy_nonboolean_qualification_cannot_control(value):
    qual = qualified()
    qual['research_eligible'] = value
    assert not QualificationGate(mode='guarded', qualification=qual).control_eligible


def test_missing_online_check_cannot_control():
    qual = qualified()
    qual['online_checks'].pop(REQUIRED_ONLINE_CHECKS[0])
    assert not QualificationGate(mode='guarded', qualification=qual).control_eligible
