import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import train_ranking_response as training
from ranking_loss import ranking_cache,ranking_response_loss
from ranking_qualification import qualify,compare_decisions,library_contributions
from factorial_loss import population_weights
from decision_value import rank_metrics


def synthetic_calls():
    calls=[]
    for i in range(3):
        proposed=np.zeros((3,8,4,3));proposed[1,:,0,0]=1;proposed[2,:,0,0]=2
        target=np.zeros((3,8,3));target[1,:,0]=.1;target[2,:,0]=.2
        calls.append({'record':{'group':f'g{i}','split':'train'},'values':{
            'history':np.ones((8,252),dtype=np.float32)*i/100,'relative':np.zeros((4,6)),
            'proposed':proposed,'anchor':proposed[0].copy(),'backbone':np.zeros((8,3)),
            'reference':np.zeros(3),'velocity':np.zeros(3),'target':target,'anchor_target':target[0].copy(),
            'valid':np.ones((3,8),dtype=bool),'anchor_valid':np.ones(8,dtype=bool)}})
    return calls


def test_actual_new_response_optimizer_changes_only_response_and_repeats():
    calls=synthetic_calls();indices=list(range(len(calls)));mean,scale=training.normalization(calls,indices)
    weights,_=population_weights(calls,indices);weights['cost']={i:1. for i in indices}
    config=json.loads((HERE/'training_protocol.json').read_text())['training']
    def one():
        torch.manual_seed(42)
        common=training.CalibratedOrigin(training.LocalTwoHead('motion_only',2.5,2.5),'cv').eval().requires_grad_(False)
        model=training.FrozenOriginResponse(common,training.LocalTwoHead('plain',2.5,2.5))
        before=training.state_digest(common);core=training.state_digest(model.response_core)
        _,outputs=training.evaluate_cost_model(common,calls,indices,mean,scale,16)
        score=lambda p:p[:,:,0].sum(1)+torch.tensor([2.,1.,0.],dtype=torch.float64)
        caches={i:ranking_cache(score,o['reference'],calls[i]['values']) for i,o in zip(indices,outputs)}
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        # ONE synthetic implementation epoch, NOT a production budget amendment.
        history,rng=training.run_epochs(model,optimizer,'cv_rank_l2',calls,indices,mean,scale,
            {c['record']['group']:1 for c in calls},weights,caches,{**config,'seed':42,'response_epochs':1})
        assert training.state_digest(common)==before and all(p.grad is None for p in common.parameters())
        assert training.state_digest(model.response_core)!=core
        return copy.deepcopy(model.state_dict()),history,rng
    first=one();second=one()
    assert first[1:]==second[1:] and all(torch.equal(first[0][k],second[0][k]) for k in first[0])


def decision(choice,group='g0',ordinal=0,truth=None,population='bounded_shared',indices=None):
    truth=[0.,1.,2.] if truth is None else truth
    count=len(truth);scores=[10.]*count;scores[choice]=0.
    base=[10.]*count;base[-1]=0.
    costs={'model':scores,'own_motion_component':base,'constant_velocity':base,
           'original_gru':base,'action_specific_truth':truth}
    return {'episode_index':1,'step':2,'ordinal':ordinal,'agent':0,'group':group,'population':population,
            'indices':list(range(count)) if indices is None else indices,'costs':costs,
            'metrics':{k:rank_metrics(v,truth) for k,v in costs.items()}}


def test_fixed_median_seeds_and_strict_positive_gates():
    protocol=json.loads((HERE/'training_protocol.json').read_text())
    models=[];decisions={}
    for name in protocol['models']:
        for seed in protocol['training']['seeds']:
            models.append({'configuration':name,'seed':seed,'development':{'group_equal_ade_m':.1,
                'group_equal_paired_response_error_m':.03 if name=='cv_rank_l2' else .04,
                'group_equal_ranking_kl':.01 if name=='cv_rank_l2' else .02,'anchor_response_exact_zero':True}})
            choice={'cv_motion_only':2,'cv_l2':1,'cv_rank_l2':0}[name]
            decisions[f'{name}_seed{seed}']=[decision(choice,f'g{g}',ordinal) for g in range(8) for ordinal in range(2)]
    controls={name:{'group_equal_ade_m':1.,'group_equal_paired_response_error_m':.1} for name in ('frozen_gru','constant_velocity')}
    selected,gates=qualify(models,controls,decisions,protocol)
    assert set(selected.values())=={994102} and gates['cv_rank_l2']['research_eligible']
    assert not gates['cv_rank_l2']['online_promoted']
    # Tie/no changed decisions cannot pass positive-gain requirement.
    decisions['cv_rank_l2_seed994102']=copy.deepcopy(decisions['cv_l2_seed994102'])
    assert not qualify(models,controls,decisions,protocol)[1]['cv_rank_l2']['research_eligible']


@pytest.mark.parametrize('fault',['ordinal','truth','metric','empty','population'])
def test_forged_or_mismatched_decision_support_rejected(fault):
    p=json.loads((HERE/'training_protocol.json').read_text());a=[decision(0)];b=[decision(1)]
    if fault=='ordinal': b[0]['ordinal']=1
    elif fault=='truth': b[0]['costs']['action_specific_truth'][0]=99
    elif fault=='metric': b[0]['metrics']['model']['realized_cost_at_choice']=-100
    elif fault=='empty': a=[]
    else: b[0]['population']='actual_unique'
    with pytest.raises(ValueError):compare_decisions(a,b,p)


def test_candidate_library_gain_uses_realized_cost_not_different_minima():
    p=json.loads((HERE/'training_protocol.json').read_text())
    shared=decision(1,truth=[100.,50.]);small=decision(0,truth=[100.],population='actual_unique',indices=[0])
    geometry=copy.deepcopy(small);geometry['population']='public_geometry'
    assert shared['metrics']['model']['regret_in_diagnostic_cost']==small['metrics']['model']['regret_in_diagnostic_cost']==0
    measured=library_contributions({'bounded_shared':[shared],'actual_unique':[small],'public_geometry':[geometry]},p)
    assert measured['actual_unique__to__bounded_shared__model']['group_equal_mean']==50.
    small['costs']['action_specific_truth'][0]=101.
    with pytest.raises(ValueError):library_contributions({'bounded_shared':[shared],'actual_unique':[small],'public_geometry':[geometry]},p)


def test_no_audit_no_optimizer_or_output_created(tmp_path,monkeypatch):
    used=[]
    monkeypatch.setattr(torch.optim,'Adam',lambda *a,**k:used.append(1))
    with pytest.raises(FileNotFoundError):training.run_training(tmp_path/'data',tmp_path/'no_audit',tmp_path/'training')
    assert not used and not (tmp_path/'training').exists()
