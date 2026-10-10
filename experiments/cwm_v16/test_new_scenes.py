import copy
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from collect_scenes import validate_protocol, transform, response_statistics, variant_summary


def protocol():
    return json.loads((ROOT/'scene_protocol.json').read_text())


@pytest.mark.parametrize('change',['control','target','multiple_axes','duplicate','holdout'])
def test_protocol_rejects_confounded_or_unprotected_scene_design(change):
    values = protocol()
    validate_protocol(values)
    if change == 'control':
        values['enhanced_control_enabled'] = True
    elif change == 'target':
        values['target_branch_rule_overrides'] = {'target_branch_commit_step':1}
    elif change == 'multiple_axes':
        values['variants'][1]['defender_y_multiplier'] = 1.4
    elif change == 'duplicate':
        values['variants'][1]['seed_start'] = values['variants'][0]['seed_start']
    else:
        values['layout_seed_start'] = 1966010
    with pytest.raises(ValueError):
        validate_protocol(values)


@dataclass
class Wall:
    center_xy:np.ndarray
    half_extents_xy:np.ndarray


@dataclass
class Scene:
    obstacles:tuple
    defender_positions:np.ndarray
    obstacle_zone_x:tuple
    name:str


def test_scene_transform_does_not_mutate_reference_or_mix_axes():
    scene = Scene((Wall(np.zeros(2),np.array([.5,4.])),),np.array([[-8.,2.,5.],[-8.,-2.,5.]]),(-1.,1.),'test')
    original = copy.deepcopy(scene)
    variants = protocol()['variants']
    transformed = [transform(scene,v) for v in variants]
    assert np.array_equal(scene.defender_positions,original.defender_positions)
    assert np.array_equal(scene.obstacles[0].half_extents_xy,original.obstacles[0].half_extents_xy)
    assert np.array_equal(scene.obstacles[0].center_xy,original.obstacles[0].center_xy)
    assert np.array_equal(transformed[1].obstacles[0].center_xy,[2.25,0.])
    assert np.array_equal(transformed[1].defender_positions,transformed[0].defender_positions)
    assert np.array_equal(transformed[2].defender_positions[:,1],[2.8,-2.8])
    assert np.array_equal(transformed[2].obstacles[0].half_extents_xy,[.5,4.])
    assert np.array_equal(transformed[3].obstacles[0].half_extents_xy,[.5,3.2])


def sample():
    valid = np.zeros((8,80),bool)
    valid[:,:4] = True
    termination = np.full((8,80),'after_terminal',dtype='U128')
    termination[:,:4] = 'running'
    target = np.zeros((8,80,3))
    target[1:,:4,0] = .1
    values = {'target':target,'valid':valid,'termination':termination,
              'branch_sign_label_only':np.ones((8,80),dtype=np.int8)}
    return values


def read_arrays(values):
    raw = io.BytesIO()
    np.savez_compressed(raw,**values)
    return lambda _:raw.getvalue()


def test_response_support_is_group_equal_and_ignores_terminal_padding():
    values = sample()
    windows = [{'arrays_path':'one','group':'a'}]
    before = response_statistics(windows,read_arrays(values),8,.05)
    values['target'][:,4:] = np.arange(8)[:,None,None]*1e6
    after = response_statistics(windows,read_arrays(values),8,.05)
    assert before == after
    assert before['valid_points'] == 28
    assert before['by_group']['a']['full_pairs'] == 0
    assert before['signal_groups'] == ['a']
    assert before['group_equal_over_threshold_fraction'] == 1.


def test_response_alone_does_not_qualify_invalid_stress_geometry():
    p = protocol()
    scenes = [{'variant':'reference','route_valid':True} for _ in range(16)]
    episodes = [{'variant':'reference',**{k:False for k in ('safe_capture_success','collision','boundary_violation','timeout','target_invalid_episode')}} for _ in range(16)]
    windows = [{'variant':'reference','group':f'g{i}','arrays_path':f'file{i}'} for i in range(8)]
    values = sample()
    clean = variant_summary('reference',scenes,episodes,windows,read_arrays(values),p)
    assert clean['eligible_for_fresh_confirmation']
    values['valid'][1,3] = False
    values['termination'][1,3] = 'target_invalid'
    bad = variant_summary('reference',scenes,episodes,windows,read_arrays(values),p)
    assert not bad['eligible_for_fresh_confirmation']
    assert bad['response_support_gate_passed']
    assert bad['stress_target_invalid_branches'] == 8
    assert not bad['model_training_authorized_by_this_report']
