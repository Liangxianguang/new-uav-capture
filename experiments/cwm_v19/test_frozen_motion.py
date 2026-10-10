import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from frozen_motion_model import FrozenCommonResponse,LocalTwoHead


def inputs():
    torch.manual_seed(42)
    history, relative = torch.randn(3,8,252),torch.randn(3,4,6)
    anchor = torch.randn(1,8,4,3).repeat(3,1,1,1)
    proposed = anchor.clone()
    proposed[1:,:,0,0] += 1.
    backbone = torch.randn(3,8,3,dtype=torch.float64)
    return history,relative,proposed,anchor,backbone


def model(mediated=False):
    torch.manual_seed(8)
    common = LocalTwoHead('motion_only')
    with torch.no_grad():
        common.motion_head.bias.fill_(.1)
    response = LocalTwoHead('plain')
    return FrozenCommonResponse(common,response,mediated)


def test_frozen_common_motion_excluded_from_gradients_and_optimizer():
    m = model().train()
    before = copy.deepcopy(m.common.state_dict())
    optimizer = torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.01)
    optimizer_ids = {id(p) for g in optimizer.param_groups for p in g['params']}
    assert not m.common.training
    assert not optimizer_ids.intersection(id(p) for p in m.common.parameters())
    prediction, motion, response = m(*inputs())
    assert not motion.requires_grad
    response.square().sum().add(response.sum()).backward()
    optimizer.step()
    assert all(p.grad is None for p in m.common.parameters())
    assert all(torch.equal(before[k],v) for k,v in m.common.state_dict().items())


def test_raw_and_mediated_share_exact_common_motion():
    raw,med = model(),model(True)
    data = inputs()
    changed = data[2] * .5
    out_raw = raw(*data)
    out_med = med(*data,changed,data[3]*.5)
    assert torch.equal(out_raw[1],out_med[1])
    assert (out_med[2][0] == 0).all()


def test_public_estimates_required_for_mediated_response():
    with pytest.raises(ValueError,match='estimated commands'):
        model(True)(*inputs())


def test_zero_response_preserves_frozen_motion_prediction_and_anchor():
    m = model().eval()
    data = inputs()
    expected,_,_ = m.common(*data)
    actual,_,response = m(*data)
    assert torch.equal(actual,expected)
    assert (response == 0).all()


def test_private_future_data_not_part_of_model_api():
    data = inputs()
    with pytest.raises(TypeError):
        model()(*data,target_branch_sign=1)
