"""Label-supervised candidate COST distribution KL at frozen deployed motion."""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'cwm_v21'))
from factorial_loss import factorial_response_loss


def ranking_cache(score, reference, values):
    if not (values['valid'].all() and values['anchor_valid'].all()):
        raise ValueError('Ranking requires complete common candidate horizon')
    m = torch.as_tensor(reference, dtype=torch.float64).detach().clone()
    if m.shape != values['target'].shape or len(m) < 2 or not torch.equal(m,m[:1].expand_as(m)):
        raise ValueError('Action-independent frozen public reference required')
    truth = torch.as_tensor(values['target'],dtype=torch.float64).detach()
    with torch.no_grad():
        costs = score(truth)
        if costs.shape != (len(m),) or not torch.isfinite(costs).all():
            raise ValueError('Finite complete original true-cost labels required')
        temperature = costs.std(correction=0).clamp_min(1.).detach()
        logq = torch.log_softmax(-costs/temperature,dim=0).detach()
    return {'score':score,'reference':m,'label_costs':costs.detach(),
            'temperature':temperature,'logq':logq,'q':logq.exp()}


def ranking_kl(cache, response):
    if response.shape != cache['reference'].shape or not torch.isfinite(response).all():
        raise ValueError('Finite paired target response required')
    costs = cache['score'](cache['reference']+response.double())
    if costs.shape != cache['label_costs'].shape or not torch.isfinite(costs).all():
        raise ValueError('Finite original deployed predicted costs required')
    logp = torch.log_softmax(-costs/cache['temperature'],dim=0)
    return (cache['q']*(cache['logq']-logp)).sum()


def ranking_response_loss(response,calls,indices,slices,weights,caches,penalty=.001,ranking_weight=.1):
    if ranking_weight not in (0.,.1) or penalty != .001:
        raise ValueError('Only fixed V28 L2/ranking objectives allowed')
    physical,terms = factorial_response_loss(response,calls,indices,slices,weights,
        {'geometry':'vector_l2','weighting':'point'},penalty)
    auxiliary = []
    for i,(start,end) in zip(indices,slices):
        if weights['cost'][i] > 0:
            if i not in caches:
                raise ValueError('Missing complete-call frozen ranking cache')
            auxiliary.append(ranking_kl(caches[i],response[start:end])*weights['cost'][i])
        else:
            if i in caches:
                raise ValueError('Incomplete call in ranking supervision')
            auxiliary.append(response[start:end].sum().double()*0.)
    measured = torch.stack(auxiliary).mean()
    return physical+ranking_weight*measured,torch.cat([terms.double(),measured[None]])
