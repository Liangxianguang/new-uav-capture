import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from train_loss_factorial import (FrozenCommonResponse, LocalTwoHead, qualification, selected_seeds,
    validate_training, state_digest)


def protocol():
    return json.loads((ROOT/'training_protocol.json').read_text())


def metrics(error=.1,ade=.1):
    return {'group_equal_paired_response_error_m':error,'group_equal_ade_m':ade,'anchor_response_exact_zero':True}


def decision(model_regret=0.,common_regret=1.):
    return [{'episode_index':1,'step':2,'agent':0,'group':f'g{i}',
             'metrics':{k:{'regret_in_diagnostic_cost':r} for k,r in (
                 ('model',model_regret),('own_motion_component',common_regret),('constant_velocity',1.))}} for i in range(8)]


def fixture_results():
    p = protocol()
    rows, decisions = [],{}
    for name in p['models']:
        for j,seed in enumerate(p['training']['seeds']):
            rows.append({'configuration':name,'seed':seed,'development':metrics(1. if name == 'motion_only' else .1,.1+.01*j)})
            decisions[f'{name}_seed{seed}'] = decision(1. if name == 'motion_only' else .2 if name == 'raw_call_mse' else 0.)
    controls = {'frozen_gru':metrics(1.,1.),'constant_velocity':metrics(1.,1.)}
    return p,rows,decisions,controls


def test_primary_cannot_pass_when_response_gate_fails_despite_good_ade():
    p,rows,decisions,controls = fixture_results()
    for row in rows:
        if row['configuration'] == p['primary_configuration']:
            row['development']['group_equal_paired_response_error_m'] = 1.01
    selected,gates,_ = qualification(rows,controls,decisions,p)
    assert not gates['raw_point_l2']['research_eligible']
    assert not gates['raw_point_l2']['checks']['median_response_vs_zero']
    assert gates['raw_point_l2']['checks']['each_seed_ade_vs_gru']
    assert selected['raw_point_l2'] == 990102
    assert not gates['raw_point_l2']['online_promoted']


def test_primary_requires_gain_over_control_not_only_zero_and_motion():
    p,rows,decisions,controls = fixture_results()
    for seed in p['training']['seeds']:
        decisions[f'raw_point_l2_seed{seed}'] = decision(.2)
    _,gates,_ = qualification(rows,controls,decisions,p)
    assert not gates['raw_point_l2']['research_eligible']
    assert not gates['raw_point_l2']['checks']['gain_vs_call_mse']
    assert gates['raw_call_l2']['research_eligible']
    assert p['primary_configuration'] == 'raw_point_l2'


def test_good_offline_result_still_never_online_promoted():
    p,rows,decisions,controls = fixture_results()
    _,gates,factorial = qualification(rows,controls,decisions,p)
    assert gates['raw_point_l2']['research_eligible']
    assert all(not gate['online_promoted'] for gate in gates.values())
    assert len(factorial) == 5 and all(v['descriptive_only'] for v in factorial.values())


def test_missing_seed_and_changed_primary_rejected():
    p,rows,decisions,controls = fixture_results()
    with pytest.raises(ValueError,match='population'):
        selected_seeds(rows[:-1],p)
    p['primary_configuration'] = 'raw_call_mse'
    with pytest.raises(ValueError,match='protocol'):
        validate_training(p)


def test_response_training_cannot_change_common_or_anchor():
    torch.manual_seed(990101)
    common = LocalTwoHead('motion_only')
    model = FrozenCommonResponse(common,LocalTwoHead('plain'),False)
    before = state_digest(common)
    history = torch.randn(2,8,252)
    relative = torch.randn(2,4,6)
    proposed,anchor = torch.randn(2,8,4,3),torch.randn(2,8,4,3)
    proposed[0] = anchor[0]
    backbone = torch.randn(2,8,3,dtype=torch.float64)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    model.train()
    assert not common.training
    for _ in range(2):
        prediction,motion,response = model(history,relative,proposed,anchor,backbone)
        assert (response[0] == 0).all()
        optimizer.zero_grad(set_to_none=True)
        (response-torch.ones_like(response)).square().mean().backward()
        optimizer.step()
    assert state_digest(common) == before
    assert all(p.grad is None and not p.requires_grad for p in common.parameters())
