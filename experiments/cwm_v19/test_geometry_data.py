import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from collect_geometry_data import assigned_split, assign_scenes, validate_protocol


def protocol():
    return json.loads((ROOT/'data_protocol.json').read_text())


def test_fixed_train_development_mirror_assignment():
    p = protocol()
    validate_protocol(p)
    scenes = assign_scenes([{'member':i%2} for i in range(64)], p)
    assert sum(s['model_split'] == 'train' for s in scenes) == 48
    assert sum(s['model_split'] == 'development_validation' for s in scenes) == 16
    assert all(scenes[2*i]['model_split'] == scenes[2*i+1]['model_split'] for i in range(32))
    for split in ('train','development_validation'):
        groups = [i for i in range(32) if assigned_split(i,p) == split]
        assert {i%4 for i in groups} == {0,1,2,3}


def test_holdout_or_activated_protocol_rejected():
    p = protocol()
    p['holdout_used'] = True
    with pytest.raises(ValueError, match='original data contract'):
        validate_protocol(p)
    p = protocol()
    p['enhanced_control_enabled'] = True
    with pytest.raises(ValueError):
        validate_protocol(p)


def test_group_outside_frozen_population_rejected():
    for i in (-1,32):
        with pytest.raises(ValueError):
            assigned_split(i, protocol())


def test_changed_horizon_and_identity_rejected():
    p = protocol()
    p['seed_start'] = 988010
    with pytest.raises(ValueError, match='identity'):
        validate_protocol(p)
    p = protocol()
    p['horizon_steps'] = 80
    with pytest.raises(ValueError, match='horizon'):
        validate_protocol(p)
