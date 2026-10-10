import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from fresh_geometry_data import validate_protocol, assigned_split, assign_scenes


def protocol():
    return json.loads((ROOT/'data_protocol.json').read_text())


def test_fresh_preassigned_disjoint_population():
    p = protocol()
    validate_protocol(p)
    scenes = assign_scenes([{'member':i%2} for i in range(64)],p)
    assert sum(s['model_split'] == 'train' for s in scenes) == 48
    assert all(scenes[i]['model_split'] == scenes[i+1]['model_split'] for i in range(0,64,2))
    for split in ('train','development_validation'):
        assert {i%4 for i in range(32) if assigned_split(i,p) == split} == {0,1,2,3}
    training = json.loads((ROOT/'training_protocol.json').read_text())
    assert training['primary_configuration'] == 'raw_point_l2'
    assert len(training['response_configurations']) == 4
    assert set(tuple(v.values()) for v in training['response_configurations'].values()) == {
        ('coordinate_mse','call'), ('coordinate_mse','point'), ('vector_l2','call'), ('vector_l2','point')}


@pytest.mark.parametrize('key,value',[('seed_start',989010),('layout_seed_start',1966010),
    ('holdout_used',True),('enhanced_control_enabled',True),('horizon_steps',80),('train_groups',28)])
def test_changed_published_contract_rejected(key,value):
    p = protocol()
    p[key] = value
    with pytest.raises(ValueError):
        validate_protocol(p)


def test_outside_group_range_rejected():
    for i in (-1,32):
        with pytest.raises(ValueError):
            assigned_split(i,protocol())
