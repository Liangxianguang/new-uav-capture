"""Structural tests, not model efficacy or complete native qualification."""
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from coordinate_audit import role_coordinate_isolated, FrozenLocalScore

@pytest.mark.parametrize('gap',[0.,1e-6,3.2430380124459646e-5,4e-5])
def test_first_coordinate_keeps_original_margin(gap):
    assert not role_coordinate_isolated(0,gap,1e-5)

@pytest.mark.parametrize('t',range(1,8))
def test_later_coordinate_isolated_at_exact_role_tie(t):
    score = object.__new__(FrozenLocalScore)
    score.peer_positions = torch.tensor([[-1.,0.,0.],[1.,0.,0.]],dtype=torch.float64)
    score.peer_ids = torch.tensor([3,2])
    paths = torch.zeros((3,8,3),dtype=torch.float64)
    expected = score.roles(paths).clone()
    assert score.role_gap(paths).tolist() == [0.,0.,0.]
    for xyz in range(3):
        for sign in (-1,1):
            perturbed = paths.clone(); perturbed[:,t,xyz] += sign*1e-5
            assert torch.equal(score.roles(perturbed),expected)
    assert role_coordinate_isolated(t,0.,1e-5)

def test_first_coordinate_really_can_switch_role():
    score = object.__new__(FrozenLocalScore)
    score.peer_positions = torch.tensor([[-1.,0.,0.],[1.,0.,0.]],dtype=torch.float64)
    score.peer_ids = torch.tensor([3,2])
    minus = torch.zeros((1,8,3),dtype=torch.float64); plus = minus.clone()
    minus[:,0,0] -= 1e-5; plus[:,0,0] += 1e-5
    assert int(score.roles(minus)[0]) != int(score.roles(plus)[0])
    assert not role_coordinate_isolated(0,0.,1e-5)
