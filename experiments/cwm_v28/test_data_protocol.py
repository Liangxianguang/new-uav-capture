import copy
import json
import sys
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from collect_ranking_data import validate_data_protocol


def test_fresh_population_disjoint_from_all_saved_protocols_and_holdout():
    validate_data_protocol(json.loads((HERE/'data_protocol.json').read_text()))


@pytest.mark.parametrize('key,value', [('seed_start',966010),('layout_seed_start',1993010),
    ('train_groups',25),('holdout_used',True),('enhanced_control_enabled',True),
    ('target_branch_rule_overrides',{'speed':1})])
def test_changed_protocol_cannot_collect(key,value):
    p=json.loads((HERE/'data_protocol.json').read_text());p[key]=value
    with pytest.raises(ValueError): validate_data_protocol(p)


def test_fixed_new_ranking_hypothesis_retains_old_basic_thresholds():
    p=json.loads((HERE/'training_protocol.json').read_text())
    old=json.loads((HERE.parent/'cwm_v23/training_protocol.json').read_text())
    for key,value in old['development_gate'].items():
        assert p['development_gate'][key]==value
    assert p['primary_configuration']=='cv_rank_l2'
    assert p['training']['seeds']==[994101,994102,994103]
    assert p['training']['independent_complete_runs']==2
    assert not p['prior_gate_override_allowed'] and not p['enhanced_control_enabled'] and not p['holdout_used']
    assert p['models']==['cv_motion_only','cv_l2','cv_rank_l2']
