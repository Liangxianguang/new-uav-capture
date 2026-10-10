import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from support_audit import (map_actual,support_record,response_summary,information_paths,
                           decision_summary,validate_protocol,rank_metrics)


def protocol():
    return json.loads((ROOT/'protocol.json').read_text())


def call():
    proposed = np.zeros((3,8,4,3))
    proposed[1,:,0,0] = 1.
    proposed[2,:,0,1] = 1.
    target = np.zeros((3,8,3))
    target[1] = .1
    target[2] = .5
    return {'record':{'episode_index':1,'step':2,'agent':0,'group':'g','split':'train'},
            'values':{'proposed':proposed,'anchor':proposed[0].copy(),
                      'actual_candidates':np.stack([proposed[1,:,0],proposed[0,:,0],proposed[1,:,0]]),
                      'target':target,'anchor_target':target[0].copy(),'backbone':np.ones((8,3)),
                      'valid':np.ones((3,8),bool),'anchor_valid':np.ones(8,bool),
                      'reference':np.zeros(3),'velocity':np.ones(3)}}


def test_mapping_preserves_original_order_without_label_selection():
    values = call()['values']
    indices,duplicates = map_actual(values,0)
    assert indices.tolist() == [1,0] and duplicates == 1
    values['target'][:] = 1e6
    values['valid'][:] = False
    other,n = map_actual(values,0)
    assert np.array_equal(indices,other) and n == duplicates


def test_absent_or_duplicate_union_candidate_is_rejected():
    values = call()['values']
    values['actual_candidates'][0] += .2
    with pytest.raises(ValueError,match='mapping not unique'):
        map_actual(values,0)
    values = call()['values']
    values['proposed'] = np.concatenate([values['proposed'],values['proposed'][:1]])
    with pytest.raises(ValueError,match='mapping not unique'):
        map_actual(values,0)


def test_actual_and_expanded_support_are_not_conflated():
    c = call()
    c['values']['valid'][2,4:] = False
    actual = support_record(c,np.array([1,0]),'actual_unique',protocol())
    union = support_record(c,np.arange(3),'expanded_union',protocol())
    assert actual['all_candidates_full_horizon_valid']
    assert not union['all_candidates_full_horizon_valid']
    assert actual['valid_nonanchor_points'] == 8
    assert union['valid_nonanchor_points'] == 12
    c['values']['target'][2,4:] = 1e6
    assert union == support_record(c,np.arange(3),'expanded_union',protocol())


def test_group_equal_response_does_not_overweight_more_calls():
    c = call()
    one = support_record(c,np.array([1,0]),'actual_unique',protocol())
    two = copy.deepcopy(one)
    two['group'] = 'other'
    two['response_sum_m'] = 0.
    measured = response_summary([one,one,two])
    assert np.isclose(measured['group_equal_mean_response_m'],np.sqrt(.03)*.5)
    assert measured['valid_nonanchor_points'] == 24


def test_information_ceiling_has_anchored_exact_response():
    c = call()
    paths = information_paths(c['values'],np.array([1,0]))
    assert np.array_equal(paths['gru_plus_exact_response'][1],c['values']['backbone'])
    assert np.array_equal(paths['gru_plus_exact_response'],paths['original_gru']+
                          (paths['action_specific_truth']-paths['fixed_reference_truth']))
    assert set(paths) == set(protocol()['information_conditions'])


def test_diagnostic_bootstrap_does_not_claim_new_validation_or_promote():
    p = protocol()
    rows = []
    for i in range(8):
        costs = {name:[2.,1.] for name in p['information_conditions']}
        costs['original_gru'] = [1.,2.]
        metrics = {k:rank_metrics(v,costs['action_specific_truth']) for k,v in costs.items()}
        rows.append({'group':str(i),'metrics':metrics,'actual_solver_regret':1.,'actual_solver_local_choice':0,
                     'response_errors':{}})
    measured = decision_summary(rows,p)
    assert measured['comparisons']['original_gru__to__gru_plus_exact_response']['gain']['group_equal_mean'] == 1.
    assert 'Previously inspected' in measured['comparisons']['original_gru__to__gru_plus_exact_response']['gain']['interpretation']
    assert 'research_eligible' not in measured


@pytest.mark.parametrize('key',['enhanced_control_enabled','new_model_trained','holdout_used','prior_gate_overridden'])
def test_information_diagnostic_cannot_override_original_failed_gate(key):
    p = protocol()
    p[key] = True
    with pytest.raises(ValueError,match='diagnostic contract'):
        validate_protocol(p)
