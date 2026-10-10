import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scene_release import validate_rollout


def values():
    termination = np.full((8,80),'after_terminal',dtype='U128')
    termination[:,:3] = 'running'
    termination[:,3] = 'captured'
    target = np.zeros((8,80,3))
    target[:,:4] = [-4.,0.,5.]
    valid = np.zeros((8,80),bool)
    valid[:,:4] = True
    return {'target':target,'defenders':np.zeros((8,80,4,3)),
            'commanded':np.zeros((8,80,4,3)),'executed':np.zeros((8,80,4,3)),
            'proposed':np.zeros((8,80,4,3)),'valid':valid,
            'branch_sign_label_only':np.zeros((8,80),np.int8),
            'termination':termination,'public_reference':np.array([-4.,0.,5.])}


def scene():
    return {'scenario':{'obstacles':[{'shape':'wall','center_xy':[1.5,0.],
                                     'height':9.4,'half_extents_xy':[.5,4.]}]}}


@pytest.mark.parametrize('change',['padding','mask','boundary','gap','imputed','nonfinite'])
def test_independent_rollout_validation_rejects_invalid_labels_or_prefix(change):
    data = values()
    validate_rollout(data,scene())
    if change == 'padding':
        data['valid'][0,4] = True
    elif change == 'mask':
        data['valid'][0,2] = False
    elif change == 'boundary':
        data['target'][0,2] = [9.8,0.,5.]
    elif change == 'gap':
        data['termination'][0,2] = 'after_terminal'
    elif change == 'imputed':
        data['target'][0,5,0] = 1.
    else:
        data['executed'][0,2,0,0] = np.nan
    with pytest.raises(ValueError):
        validate_rollout(data,scene())


def test_valid_terminal_violation_is_retained_as_diagnostic_failure():
    data = values()
    data['target'][0,3] = [9.8,0.,5.]
    data['valid'][0,3] = False
    data['termination'][0,3] = 'target_boundary_violation'
    validate_rollout(data,scene())
