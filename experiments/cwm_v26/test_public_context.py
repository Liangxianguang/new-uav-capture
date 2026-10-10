import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from real_shadow_entry import build_public_inputs


def call():
    peers={i:SimpleNamespace(position=np.full(3,float(i)+10),velocity=np.full(3,float(i)+1)) for i in (1,2,3)}
    return {'observation':{'defender_positions':np.arange(12).reshape(4,3).astype(float),
        'defender_velocities':np.zeros((4,3)), 'private_future':object()},
        'planner':SimpleNamespace(config=object()),'agent':0,'known':peers,
        'peer_sequences':{i:np.full((8,3),float(i)) for i in (1,2,3)},
        'local_candidates':np.zeros((2,8,3)),'selected':np.zeros((8,3)),
        'scenarios':SimpleNamespace(trajectories=np.ones((1,8,3)))}


def belief(observation,config):
    return np.array([1.,2.,3.]),np.array([.1,.2,.3])


def test_delayed_received_context_not_privileged_current_peers():
    c=call();history=np.ones((8,252))
    values=build_public_inputs(c,history,belief)
    assert len(values)==6
    for peer in (1,2,3):
        assert np.array_equal(values[1][peer,:3],c['known'][peer].position-belief(None,None)[0])
        assert np.array_equal(values[2][0,:,peer],c['peer_sequences'][peer])
    assert np.array_equal(values[-1],belief(None,None)[0][None]+.1*np.arange(1,9)[:,None]*belief(None,None)[1][None])
    values[0][:]=10
    assert (history==1).all()


def test_missing_delayed_peer_plan_is_skipped_not_zero_filled():
    c=call();c['peer_sequences'].pop(3)
    assert build_public_inputs(c,np.zeros((8,252)),belief) is None
