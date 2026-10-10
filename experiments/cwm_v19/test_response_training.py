import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from train_frozen_response import model_inputs,evaluate,control_metrics,qualification
from frozen_motion_model import LocalTwoHead,FrozenCommonResponse,response_loss


def call():
    proposed = np.zeros((2,8,4,3))
    proposed[1,:,0,0] = 1.
    target = np.zeros((2,8,3))
    target[1,:,0] = .1
    values = {'history':np.zeros((8,252),np.float32),'relative':np.zeros((4,6)),
              'backbone':np.zeros((8,3)),'proposed':proposed,'anchor':proposed[0],
              'target':target,'anchor_target':target[0],'valid':np.ones((2,8),bool),
              'anchor_valid':np.ones(8,bool),'reference':np.zeros(3),'velocity':np.zeros(3)}
    return {'record':{'group':'test_group'},'values':values,
            'estimated_commands':np.concatenate([proposed[:1],proposed],0)*.5}


def test_response_loss_excludes_anchor_and_terminal_padding():
    calls = [call()]
    response = torch.zeros(2,8,3,requires_grad=True)
    loss,_ = response_loss(response,calls,[0],[(0,2)],{'test_group':1},{'residual_penalty':.001})
    loss.backward()
    assert (response.grad[0] == 0).all()
    assert (response.grad[1,:,0] != 0).all()
    calls[0]['values']['valid'][1,4:] = False
    response = torch.zeros(2,8,3,requires_grad=True)
    loss,_ = response_loss(response,calls,[0],[(0,2)],{'test_group':1},{'residual_penalty':.001})
    loss.backward()
    assert (response.grad[1,4:] == 0).all()


def test_no_nonanchor_support_rejected():
    c = call()
    c['values']['valid'][1] = False
    with pytest.raises(ValueError,match='nonanchor'):
        response_loss(torch.zeros(2,8,3),[c],[0],[(0,2)],{'test_group':1},{'residual_penalty':.001})


def test_model_pack_does_not_read_future_labels():
    c = call()
    mean,scale = np.zeros((1,1,252)),np.ones((1,1,252))
    expected,_ = model_inputs([c],[0],mean,scale,True)
    c['values']['target'].fill(1e5)
    c['values']['branch_sign_label_only'] = np.ones((2,8),np.int8)
    actual,_ = model_inputs([c],[0],mean,scale,True)
    assert all(torch.equal(a,b) for a,b in zip(expected,actual))


def test_zero_initialized_optional_response_matches_zero_control_metrics():
    c = call()
    torch.manual_seed(1)
    model = FrozenCommonResponse(LocalTwoHead('motion_only'),LocalTwoHead('plain'))
    measured,_ = evaluate(model,'raw_response',[c],[0],np.zeros((1,1,252)),np.ones((1,1,252)))
    controls = control_metrics([c],[0])
    assert measured == controls['frozen_gru']


def test_failed_response_and_zero_gain_cannot_promote():
    protocol = json.loads((ROOT/'training_protocol.json').read_text())
    measure = {'group_equal_ade_m':.1,'group_equal_paired_response_error_m':.2,'anchor_response_exact_zero':True}
    models = [{'configuration':name,'seed':seed,'development':measure} for name in protocol['models'] for seed in protocol['training']['seeds']]
    metric = {'regret_in_diagnostic_cost':0.}
    row = {'episode_index':1,'step':2,'agent':0,'group':'test_group',
           'metrics':{name:metric for name in ('model','own_motion_component','constant_velocity')}}
    decisions = {f'{r["configuration"]}_seed{r["seed"]}':[row] for r in models}
    controls = {k:{'group_equal_ade_m':1.,'group_equal_paired_response_error_m':.1} for k in ('frozen_gru','constant_velocity')}
    _,gates = qualification(models,controls,decisions,protocol)
    assert all(not g['research_eligible'] and not g['online_promoted'] for g in gates.values())
    assert all(not g['checks']['median_response_vs_zero'] for g in gates.values())
