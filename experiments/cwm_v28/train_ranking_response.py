"""Fixed fresh V28 three-family ranking training, original route never changed."""
import argparse
import collections
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(HERE.parent/'cwm_v23'),str(HERE.parent/'cwm_v19')]
from audit_ranking_data import verified_calls
from collect_ranking_data import save_json
from ranking_loss import ranking_cache,ranking_kl,ranking_response_loss
from ranking_qualification import validate_training_protocol,qualify,library_contributions
from cost_origin_model import (LocalTwoHead,CalibratedOrigin,FrozenOriginResponse,normalization,
    public_cost_inputs,calibrated_motion_loss,evaluate_cost_model)
from deployed_cost_loss import cost_population_weights,contrast_cache,audit_deployed_score
from train_frozen_response import state_digest,control_metrics,frozen_configs
from local_shadow import restore_public_call,rank_metrics
from support_audit import map_actual
from task_score import FrozenLocalScore,original_scores
from freeze_baseline import sha,verify
from geometry_release import BASELINE_SHA


def complete_contexts(calls,configs,data):
    result={}
    for i,c in enumerate(calls):
        row,v=c['record'],c['values']
        if v['valid'].all() and v['anchor_valid'].all():
            result[i]=restore_public_call(json.loads((data/row['context_path']).read_text()),*configs,v['backbone'])
    if {calls[i]['record']['group'] for i in result}!={c['record']['group'] for c in calls}:
        raise ValueError('Complete shared-library cost support required in all train/dev groups')
    return result


def decision_records(calls,indices,outputs,contexts):
    result={name:[] for name in ('actual_unique','public_geometry','bounded_shared')}
    for i,output in zip(indices,outputs):
        if i not in contexts: continue  # One identical complete library support.
        row,v=calls[i]['record'],calls[i]['values'];count=len(v['proposed'])
        actions=v['proposed'][:,:,row['agent']]
        cv=v['reference'][None]+.1*np.arange(1,9)[:,None]*v['velocity'][None]
        paths={'model':output['prediction'],'own_motion_component':output['reference'],
            'original_gru':np.repeat(v['backbone'][None],count,axis=0),
            'constant_velocity':np.repeat(cv[None],count,axis=0),'action_specific_truth':v['target']}
        costs={name:original_scores(contexts[i],actions,path) for name,path in paths.items()}
        actual,_=map_actual(v,row['agent'])
        raw=[int(np.flatnonzero((actions==a[None]).all((1,2)))[0]) for a in v['actual_candidates']]
        if not np.allclose(costs['original_gru'][raw],v['original_local_costs'],rtol=1e-10,atol=1e-8):
            raise ValueError('Actual original full costs changed')
        for population,selected in (('actual_unique',actual),('public_geometry',v['public_geometry_indices']),('bounded_shared',np.arange(count))):
            subset={name:value[selected].tolist() for name,value in costs.items()}
            measured={name:rank_metrics(value,subset['action_specific_truth'],1e-9) for name,value in subset.items()}
            anchor=np.flatnonzero(selected==row['actual_choice_pool_index'])
            if len(anchor)!=1: raise ValueError('Original selected anchor absent from library')
            if population=='actual_unique' and measured['original_gru']['choice']!=int(anchor[0]):
                raise ValueError('Original actual local selection changed')
            result[population].append({**{key:row[key] for key in ('episode_index','step','ordinal','agent','group')},
                'population':population,'indices':selected.tolist(),'costs':subset,'metrics':measured,
                'original_anchor_realized_cost':float(costs['action_specific_truth'][row['actual_choice_pool_index']])})
    return result


def ranking_metrics(calls,indices,outputs,caches):
    grouped=collections.defaultdict(list)
    for i,output in zip(indices,outputs):
        if i not in caches: continue
        with torch.no_grad(): value=float(ranking_kl(caches[i],torch.from_numpy(output['response'])))
        grouped[calls[i]['record']['group']].append(value)
    if set(grouped)!={calls[i]['record']['group'] for i in indices}:
        raise ValueError('Ranking metrics missing complete group support')
    by_group={g:{'calls':len(v),'ranking_kl':float(np.mean(v))} for g,v in grouped.items()}
    return {'group_equal_ranking_kl':float(np.mean([v['ranking_kl'] for v in by_group.values()])),
            'ranking_by_group':by_group,'ranking_complete_calls':sum(len(v) for v in grouped.values())}


def run_epochs(model,optimizer,name,calls,indices,mean,scale,counts,weights,caches,config):
    motion=name=='cv_motion_only';epochs=config['motion_epochs' if motion else 'response_epochs']
    sampler=np.random.default_rng(config['seed']);history=[]
    for epoch in range(1,epochs+1):
        model.train();losses=[];order=sampler.permutation(indices)
        for start in range(0,len(order),config['batch_calls']):
            chosen=order[start:start+config['batch_calls']]
            if motion:
                loss,terms=calibrated_motion_loss(model,calls,chosen,mean,scale,counts,config['residual_penalty'])
            else:
                tensors,slices=public_cost_inputs(calls,chosen,mean,scale)
                _,_,response=model(*tensors)
                weight=0. if name=='cv_l2' else config['ranking_weight']
                loss,terms=ranking_response_loss(response,calls,chosen,slices,weights,caches,config['residual_penalty'],weight)
            if not torch.isfinite(loss): raise ValueError('Nonfinite fixed-budget loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config['gradient_clip_norm'])
            optimizer.step();losses.append([float(loss.detach()),*[float(v.detach()) for v in terms]])
        history.append({'epoch':epoch,'mean_batch_loss_terms':np.mean(losses,axis=0).tolist()})
        if epoch%20==0: print(json.dumps({'configuration':name,'seed':config['seed'],**history[-1]}),flush=True)
    return history,sampler.bit_generator.state


def audited_caches(calls,indices,outputs,contexts,scores):
    caches={};audits=[]
    for i,output in zip(indices,outputs):
        if i not in contexts: continue
        v=calls[i]['values'];reference=output['reference']
        contrast=contrast_cache(scores[i],reference,v)
        check=audit_deployed_score(contexts[i],v,reference,scores[i],contrast)
        cache=ranking_cache(scores[i],reference,v)
        true_cost=original_scores(contexts[i],v['proposed'][:,:,calls[i]['record']['agent']],v['target'])
        if not np.allclose(true_cost,cache['label_costs'].numpy(),rtol=1e-10,atol=1e-8):
            raise ValueError('Ranking labels differ from independent original full costs')
        caches[i]=cache;audits.append({'call_index':i,**check})
    if set(caches)!=set(contexts): raise ValueError('Incomplete frozen ranking cache/audit')
    return caches,audits


def run_training(data,audit_output,output):
    protocol=json.loads((HERE/'training_protocol.json').read_text());validate_training_protocol(protocol)
    # MUST occur before any new optimizer/normalizer or output/model is created.
    calls,data_audit=verified_calls(data,audit_output)
    output.mkdir(parents=True,exist_ok=False)
    capsule=HERE.parent/'cwm_v1/baseline/capsule.zip'
    if sha(capsule)!=BASELINE_SHA: raise ValueError('Original capsule changed')
    configs=frozen_configs(capsule,output/'restored')
    indices={split:[i for i,c in enumerate(calls) if c['record']['split']==split] for split in ('train','development_validation')}
    if {calls[i]['record']['group'] for i in indices['train']}&{calls[i]['record']['group'] for i in indices['development_validation']}:
        raise ValueError('Mirror-group split leakage')
    if tuple(len({calls[i]['record']['group'] for i in v}) for v in indices.values())!=(24,8):
        raise ValueError('Complete fixed train/dev groups required')
    mean,scale=normalization(calls,indices['train'])
    weights,population=cost_population_weights(calls,indices['train'])
    save_json(output/'population_weights.json',{'population':population,'weights':weights})
    np.savez_compressed(output/'normalization.npz',mean=mean,scale=scale)
    contexts=complete_contexts(calls,configs,data)
    scores={i:FrozenLocalScore(context,calls[i]['values']['proposed'][:,:,calls[i]['record']['agent']]) for i,context in contexts.items()}
    all_indices=list(range(len(calls)))
    sources={Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
        and Path(m.__file__).resolve().is_relative_to(ROOT/'experiments')}|{HERE/'training_protocol.json',HERE/'data_protocol.json',HERE.parent/'cwm_v11/protocol.json'}
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    for p in sources:
        destination=output/'source'/p.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(p.read_bytes())
    # Both original public references audited before ANY optimizer is instantiated.
    for origin in ('gru','cv'):
        initial=[]
        for c in calls:
            v=c['values'];path=v['backbone'] if origin=='gru' else v['reference'][None]+.1*np.arange(1,9)[:,None]*v['velocity'][None]
            initial.append({'reference':np.repeat(path[None],len(v['proposed']),axis=0)})
        _,checks=audited_caches(calls,all_indices,initial,contexts,scores)
        save_json(output/('initial_'+origin+'_score_audit.json'),checks)
        print(json.dumps({'status':'initial_public_original_cost_gradient_audit_passed','origin':origin,'complete_calls':len(checks)}),flush=True)
    counts=collections.Counter(calls[i]['record']['group'] for i in indices['train'])
    config=protocol['training'];models=[];decisions={};contributions={}
    controls=control_metrics(calls,indices['development_validation'])
    for seed in config['seeds']:
        torch.manual_seed(seed)
        common=CalibratedOrigin(LocalTwoHead('motion_only',config['motion_scale_m'],config['response_scale_m']),'cv')
        common_initial=state_digest(common)
        optimizer=torch.optim.Adam(common.parameters(),lr=config['learning_rate'])
        motion_history,motion_rng=run_epochs(common,optimizer,'cv_motion_only',calls,indices['train'],mean,scale,counts,weights,{}, {**config,'seed':seed})
        motion_trained={'model_state':copy.deepcopy(common.state_dict()),'optimizer_state':copy.deepcopy(optimizer.state_dict()),
                        'torch_rng_state':torch.get_rng_state(),'sampler_rng_state':motion_rng,'initial_state_digest':common_initial}
        torch.save({**motion_trained,'seed':seed,'protocol':protocol,'source_hashes':hashes,'online_promoted':False,
                    'baseline_weights_included':False},output/f'cv_seed{seed}_motion_stage.pt')
        save_json(output/f'cv_seed{seed}_motion_stage_history.json',motion_history)
        common.eval().requires_grad_(False)
        for p in common.parameters(): p.grad=None
        common_digest=state_digest(common)
        canonical={}
        for split,chosen in indices.items():
            _,predictions=evaluate_cost_model(common,calls,chosen,mean,scale,config['batch_calls'])
            canonical.update(zip(chosen,predictions))
        caches,checks=audited_caches(calls,all_indices,[canonical[i] for i in all_indices],contexts,scores)
        save_json(output/f'cv_seed{seed}_deployed_score_audit.json',checks)
        np.savez_compressed(output/f'cv_seed{seed}_ranking_cache.npz',**{f'call{i}_{k}':v.numpy() for i,c in caches.items() for k,v in c.items() if torch.is_tensor(v)})
        initial_states=[]
        for name in protocol['models']:
            if name=='cv_motion_only':
                model=common;trained=motion_trained;used_history=motion_history;initial=common_initial
            else:
                torch.manual_seed(seed)
                core=LocalTwoHead('plain',config['motion_scale_m'],config['response_scale_m']);initial=state_digest(core)
                initial_states.append(initial);model=FrozenOriginResponse(copy.deepcopy(common),core)
                parameters=[p for p in model.parameters() if p.requires_grad]
                if any(id(p) in {id(v) for v in model.common.parameters()} for p in parameters):
                    raise ValueError('Frozen common motion entered response optimizer')
                optimizer=torch.optim.Adam(parameters,lr=config['learning_rate'])
                used_history,rng=run_epochs(model,optimizer,name,calls,indices['train'],mean,scale,counts,weights,caches,{**config,'seed':seed})
                if state_digest(model.common)!=common_digest or any(p.grad is not None or p.requires_grad for p in model.common.parameters()):
                    raise ValueError('Frozen common motion changed during response stage')
                trained={'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'torch_rng_state':torch.get_rng_state(),
                         'sampler_rng_state':rng,'initial_state_digest':initial}
            identifier=f'{name}_seed{seed}';split_metrics={};split_decisions={};split_contributions={}
            for split,chosen in indices.items():
                measured,predictions=evaluate_cost_model(model,calls,chosen,mean,scale,config['batch_calls'])
                if any(not np.array_equal(o['reference'],caches[i]['reference'].numpy()) for i,o in zip(chosen,predictions) if i in caches):
                    raise ValueError('Canonical frozen deployed reference differs')
                measured.update(ranking_metrics(calls,chosen,predictions,caches))
                split_metrics[split]=measured
                np.savez_compressed(output/f'{identifier}_{split}_predictions.npz',**{f'call{j}_{k}':v for j,o in enumerate(predictions) for k,v in o.items()})
                split_decisions[split]=decision_records(calls,chosen,predictions,contexts)
                split_contributions[split]=library_contributions(split_decisions[split],protocol)
                if split=='development_validation':decisions[identifier]=split_decisions[split]['bounded_shared']
            save_json(output/f'{identifier}_decisions.json',split_decisions)
            save_json(output/f'{identifier}_library_contributions.json',split_contributions)
            save_json(output/f'{identifier}_history.json',used_history)
            checkpoint={**trained,'configuration':name,'seed':seed,'protocol':protocol,'source_hashes':hashes,
                'normalizer_mean':torch.from_numpy(mean),'normalizer_scale':torch.from_numpy(scale),'common_motion_state_digest':common_digest,
                'data_summary_sha256':sha(data/'summary.json'),'data_audit_summary_sha256':sha(audit_output/'summary.json'),
                'population_weights_sha256':sha(output/'population_weights.json'),'ranking_cache_sha256':sha(output/f'cv_seed{seed}_ranking_cache.npz'),
                'online_promoted':False,'baseline_weights_included':False,'common_motion_in_response_optimizer':False}
            torch.save(checkpoint,output/(identifier+'.pt'))
            row={'configuration':name,'seed':seed,'checkpoint_sha256':sha(output/(identifier+'.pt')),'initial_state_digest':initial,
                 'common_motion_state_digest':common_digest,'parameters':sum(p.numel() for p in model.parameters()),
                 'train':split_metrics['train'],'development':split_metrics['development_validation']}
            models.append(row);contributions[identifier]=split_contributions
            print(json.dumps({'status':'fixed_final_model_measured','model':identifier,'development':{k:row['development'][k] for k in ('group_equal_ade_m','group_equal_paired_response_error_m','group_equal_ranking_kl')}}),flush=True)
        if len(initial_states)!=2 or len(set(initial_states))!=1: raise ValueError('Matched response initialization differs')
    selected,gates=qualify(models,controls,decisions,protocol)
    if sha(capsule)!=BASELINE_SHA or any(sha(ROOT/n)!=digest for n,digest in hashes.items()):
        raise ValueError('Original capsule or run-used sources changed')
    # Recheck frozen audited data hashes after all optimizers have finished.
    verified_calls(data,audit_output)
    report={'status':'offline_fixed_fresh_ranking_training_finished_pending_two_run_release','protocol':protocol,
        'source_hashes':hashes,'models':models,'controls':controls,'selected_median_seeds':selected,'qualification':gates,
        'primary_research_eligible':gates['cv_rank_l2']['research_eligible'],'library_contributions':contributions,
        'split_calls':{k:len(v) for k,v in indices.items()},'split_groups':{k:sorted({calls[i]['record']['group'] for i in v}) for k,v in indices.items()},
        'full_cost_calls':{k:sum(i in contexts for i in v) for k,v in indices.items()},
        'data_summary_sha256':sha(data/'summary.json'),'data_audit_summary_sha256':sha(audit_output/'summary.json'),
        'enhanced_control_enabled':False,'holdout_used':False,'prior_gate_overridden':False,'common_motion_in_response_optimizer':False,
        'two_complete_runs_verified':False,'scope':protocol['later_required']}
    save_json(output/'summary.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('data','audit','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    output_existed=args.output.exists()
    try:
        report=run_training(args.data,args.audit,args.output)
        print(json.dumps({'status':report['status'],'qualification':report['qualification']}),flush=True)
    except Exception as error:
        failure=args.output/'implementation_failure.json'
        if not output_existed and args.output.is_dir() and not (args.output/'summary.json').exists() and not failure.exists():
            with failure.open('x',encoding='utf8') as file:
                json.dump({'status':'incomplete_no_training_qualification','exception_type':type(error).__name__,
                    'exception':str(error)[:300],'enhanced_control_enabled':False,'holdout_used':False},file,indent=2)
        raise
