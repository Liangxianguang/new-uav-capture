import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT),str(ROOT.parent.parent/'src')]
from geometry_probes import public_geometry_pool,stable_unique
from collect_probe_pilot import summarize


class Planner:
    def _local_candidate_sequences(self,*args):
        self.called = True
        return [np.zeros((8,3))]


def call():
    selected = np.ones((8,3))
    return {'local_candidates':np.stack([selected,selected]),'selected':selected,
            'planner':Planner(),'known':{},'agent':0,
            'observation':{'defender_positions':np.zeros((4,3))}}


def test_default_off_preserves_original_multiplicity_without_context():
    c = {'local_candidates':np.ones((2,8,3))}
    result = public_geometry_pool(c,None,None,mode='off')
    assert np.array_equal(result,c['local_candidates'])
    assert result is not c['local_candidates']


def test_shadow_probes_preserve_anchor_limit_and_original_planner():
    c = call()
    original = copy.deepcopy(c['selected'])
    before = vars(c['planner']).copy()
    result = public_geometry_pool(c,np.array([1.,0.,0.]),np.zeros(3))
    assert np.array_equal(result[0],original)
    assert np.array_equal(c['selected'],original)
    assert np.linalg.norm(result,axis=-1).max() <= 5.+1e-8
    assert vars(c['planner']) == before
    assert len(result) == 6


def test_private_future_labels_do_not_change_public_geometry_proposals():
    c = call()
    result = public_geometry_pool(c,np.array([1.,0.,0.]),np.zeros(3))
    c['future_target'] = np.full((8,3),1e6)
    c['target_branch_sign_label_only'] = -1
    assert np.array_equal(result,public_geometry_pool(c,np.array([1.,0.,0.]),np.zeros(3)))


def test_zero_xy_direction_has_deterministic_fallback():
    result = public_geometry_pool(call(),np.zeros(3),np.zeros(3))
    expected = public_geometry_pool(call(),np.array([1.,0.,0.]),np.zeros(3))
    assert np.array_equal(result,expected)


@pytest.mark.parametrize('mode',['guarded','enabled'])
def test_geometry_probe_never_implicitly_enables_controller(mode):
    with pytest.raises(ValueError,match='not an enabled controller'):
        public_geometry_pool(call(),np.ones(3),np.zeros(3),mode=mode)


def test_nonfinite_or_wrong_budget_rejected():
    with pytest.raises(ValueError,match='Frozen public geometry'):
        public_geometry_pool(call(),np.array([np.nan,0.,0.]),np.zeros(3))
    with pytest.raises(ValueError,match='Frozen public geometry'):
        public_geometry_pool(call(),np.ones(3),np.zeros(3),magnitude=1.)


def test_local_data_gate_retains_target_invalid_failure():
    p = json.loads((ROOT/'protocol.json').read_text())
    proposals = np.zeros((2,8,4,3))
    proposals[1,:,:,0] = 1.
    calls = []
    for i in range(8):
        for j in range(4):
            target = np.zeros((2,8,3))
            target[1,:,0] = .1
            calls.append({'record':{'episode_index':i,'step':2+j,'agent':0,'group':str(i),'split':'development_pilot'},
                          'values':{'proposed':proposals.copy(),'anchor':proposals[0].copy(),'actual_candidates':proposals[:1,:,0].copy(),
                                    'valid':np.ones((2,8),bool),'anchor_valid':np.ones(8,bool),
                                    'target':target,'anchor_target':target[0].copy(),'termination':np.full((2,8),'running')}})
    episodes = [{'target_invalid_episode':False} for _ in range(16)]
    result = summarize(calls,[],episodes,p)
    assert result['eligible_for_separate_fresh_training_protocol']
    calls[0]['values']['valid'][1,7] = False
    calls[0]['values']['termination'][1,7] = 'target_boundary_violation'
    bad = summarize(calls,[],episodes,p)
    assert not bad['eligible_for_separate_fresh_training_protocol']
    assert bad['probe_target_invalid_branches'] == 1
