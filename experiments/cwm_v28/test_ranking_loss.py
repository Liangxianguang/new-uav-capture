import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from ranking_loss import ranking_cache, ranking_kl


def data():
    target = np.zeros((3,8,3)); target[:,0,0]=[0,1,2]
    return {'target':target,'valid':np.ones((3,8),dtype=bool),'anchor_valid':np.ones(8,dtype=bool)}


def test_full_cost_offsets_retained_and_response_gradient_not_common():
    v=data(); common=torch.zeros((3,8,3),dtype=torch.float64,requires_grad=True)
    offsets=torch.tensor([5.,-3.,2.],dtype=torch.float64)
    score=lambda p:p[:,0,0]+offsets
    cache=ranking_cache(score,common,v)
    assert torch.equal(cache['label_costs'],torch.tensor([5.,-2.,4.],dtype=torch.float64))
    response=torch.zeros_like(common,requires_grad=True)
    loss=ranking_kl(cache,response);loss.backward()
    assert response.grad is not None and response.grad.abs().sum()>0
    assert common.grad is None and not cache['temperature'].requires_grad
    assert not cache['q'].requires_grad and not cache['reference'].requires_grad


def test_exact_label_cost_distribution_has_zero_kl_and_shift_invariance():
    v=data(); m=np.zeros_like(v['target']); score=lambda p:p[:,0,0]
    cache=ranking_cache(score,m,v)
    response=torch.tensor(v['target'])
    assert ranking_kl(cache,response).abs()<1e-14
    shifted=ranking_cache(lambda p:score(p)+1000,m,v)
    assert torch.allclose(cache['q'],shifted['q'],rtol=0,atol=1e-14)
    assert torch.allclose(ranking_kl(cache,torch.zeros_like(response)),ranking_kl(shifted,torch.zeros_like(response)),rtol=0,atol=1e-14)


@pytest.mark.parametrize('fault',['incomplete','candidate_dependent','nonfinite','single'])
def test_invalid_cost_supervision_rejected(fault):
    v=data(); m=np.zeros_like(v['target']); score=lambda p:p[:,0,0]
    if fault=='incomplete': v['valid'][1,0]=False
    elif fault=='candidate_dependent': m[1,0,0]=1
    elif fault=='nonfinite': score=lambda p:p[:,0,0]*float('nan')
    else: v['target']=v['target'][:1];m=m[:1]
    with pytest.raises(ValueError): ranking_cache(score,m,v)


def test_cost_temperature_floor_and_tie_targets():
    v=data(); cache=ranking_cache(lambda p:torch.zeros(len(p),dtype=torch.float64),np.zeros_like(v['target']),v)
    assert cache['temperature']==1 and torch.allclose(cache['q'],torch.ones(3,dtype=torch.float64)/3)
    assert ranking_kl(cache,torch.zeros_like(cache['reference'])).abs()<1e-14
