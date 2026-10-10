import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from factorial_loss import paired_mask, population_weights, factorial_response_loss


def call(group, points, label):
    proposed = np.zeros((2,8,4,3))
    proposed[1,:,0,0] = 1.
    valid = np.ones((2,8),dtype=bool)
    valid[1,points:] = False
    target = np.zeros((2,8,3))
    target[1,:,0] = label
    return {'record': {'group':group}, 'values': {'proposed':proposed, 'anchor':proposed[0],
        'valid':valid, 'anchor_valid':np.ones(8,dtype=bool), 'target':target, 'anchor_target':target[0]}}


@pytest.mark.parametrize('geometry',['coordinate_mse','vector_l2'])
@pytest.mark.parametrize('weighting',['call','point'])
def test_exact_population_objective_and_minibatch_expectation(geometry,weighting):
    calls = [call('a',8,3.),call('a',1,1.),call('b',2,2.)]
    weights,_ = population_weights(calls,[0,1,2])
    response = torch.zeros((6,8,3),dtype=torch.float64,requires_grad=True)
    objective = {'geometry':geometry,'weighting':weighting}
    full,_ = factorial_response_loss(response,calls,[0,1,2],[(0,2),(2,4),(4,6)],weights,objective,0.)
    errors = [3.,1.,2.] if geometry == 'vector_l2' else [3.,1./3.,4./3.]
    a = (errors[0]+errors[1])/2 if weighting == 'call' else (8*errors[0]+errors[1])/9
    assert float(full.detach()) == pytest.approx((a+errors[2])/2)
    single = [factorial_response_loss(response[2*i:2*i+2],calls,[i],[(0,2)],weights,objective,0.)[0] for i in range(3)]
    assert float(torch.stack(single).mean().detach()) == pytest.approx(float(full.detach()))
    full.backward()
    assert torch.isfinite(response.grad).all()


def test_zero_norm_subgradient_and_invalid_anchor_padding_excluded():
    calls = [call('a',2,0.)]
    weights,_ = population_weights(calls,[0])
    response = torch.zeros((2,8,3),requires_grad=True)
    with torch.no_grad():
        response[0] = 999.
        response[1,2:] = 999.
    loss,_ = factorial_response_loss(response,calls,[0],[(0,2)],weights,
        {'geometry':'vector_l2','weighting':'point'},.001)
    loss.backward()
    assert float(loss) == 0. and torch.isfinite(response.grad).all() and (response.grad == 0).all()


def test_empty_support_and_duplicate_population_rejected():
    calls = [call('a',0,1.)]
    with pytest.raises(ValueError,match='Empty'):
        population_weights(calls,[0])
    with pytest.raises(ValueError,match='unique'):
        population_weights(calls,[0,0])


def test_energy_has_matching_weighting_and_zero_anchor_gradient():
    calls = [call('a',8,0.),call('a',1,0.)]
    weights,_ = population_weights(calls,[0,1])
    response = torch.zeros((4,8,3),dtype=torch.float64,requires_grad=True)
    with torch.no_grad():
        response[1,:,0] = 3.
        response[3,:,0] = 1.
    loss,terms = factorial_response_loss(response,calls,[0,1],[(0,2),(2,4)],weights,
        {'geometry':'vector_l2','weighting':'point'},.001)
    assert float(terms[1].detach()) == pytest.approx((8*3+1/3)/9)
    assert float(loss.detach()) == pytest.approx((8*3+1)/9+.001*float(terms[1].detach()))
    loss.backward()
    assert (response.grad[[0,2]] == 0).all() and (response.grad[3,1:] == 0).all()
