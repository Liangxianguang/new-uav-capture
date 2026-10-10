"""Fresh sequential paired data on one fixed public bounded proposal library."""
import argparse
import copy
import hashlib
import json
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE),str(HERE.parent/'cwm_v27'),str(HERE.parent/'cwm_v26'),
               str(HERE.parent/'cwm_v25'),str(HERE.parent/'cwm_v18'),str(HERE.parent/'cwm_v17'),str(HERE.parent/'cwm_v9'),
               str(HERE.parent/'cwm_v1')]
from bounded_candidates import expand_candidates
from checkpoint_provider import checkpoint_predictor
from qualification_gate import OptionalResponseAdapter, QualificationGate
from baseline import Baseline
from freeze_baseline import sha, verify
from local_shadow import public_call_snapshot, restore_public_call, delayed_joint_context, local_cost, s4_rollout, fingerprint
from geometry import translated_records
from geometry_release import arrays,compare_arrays,target_bad
from support_audit import support_record,response_summary,map_actual

REMOTE = 'https://github.com/Liangxianguang/new-uav-capture.git'
BRANCH = 'refs/heads/causal-world-model-v1-20261009'


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')


def validate_data_protocol(protocol):
    if protocol != json.loads((HERE/'data_protocol.json').read_text()):
        raise ValueError('Published fresh V28 protocol differs')
    if ((protocol['groups'],protocol['train_groups'],protocol['development_groups']) != (32,24,8) or
        protocol['seed_start'] != 994010 or protocol['layout_seed_start'] != 1994010 or
        protocol['target_branch_rule_overrides'] or protocol['horizon_steps'] != 8 or
        any(protocol[k] for k in ('enhanced_control_enabled','new_model_trained','private_labels_model_inputs','holdout_used'))):
        raise ValueError('Frozen fresh public-only population contract violated')
    ep = set(range(protocol['seed_start'],protocol['seed_start']+2*protocol['groups']))
    layouts = set(range(protocol['layout_seed_start'],protocol['layout_seed_start']+protocol['groups']))
    # Audit all old protocol identities, including nested reserved populations.
    def check(node, inherited_groups=0):
        if isinstance(node,dict):
            count = node.get('groups',inherited_groups)
            for key,actual,multiplier in (('seed_start',ep,2),('episode_seed_start',ep,2),('layout_seed_start',layouts,1)):
                if key in node and count and actual & set(range(node[key],node[key]+multiplier*count)):
                    raise ValueError('Prior/reserved data identity overlap')
            for child in node.values(): check(child,count)
        elif isinstance(node,list):
            for child in node: check(child,inherited_groups)
    for path in sorted((ROOT/'experiments').glob('cwm_v*/*protocol*.json')):
        if path.parent != HERE:
            check(json.loads(path.read_text(encoding='utf8')))
    check(protocol['reserved_holdout_not_collected'])


def published_commit():
    """Fresh collection cannot precede an exact pushed protocol/source commit."""
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    commit = git('rev-parse','HEAD')
    remote = git('-c','http.sslBackend=schannel','ls-remote',REMOTE,BRANCH)
    if not remote or remote.split()[0] != commit:
        raise ValueError('Publish exact protocol/source HEAD before fresh collection')
    sources = [*HERE.glob('*.py'),HERE/'data_protocol.json',HERE/'training_protocol.json',
               HERE.parent/'cwm_v27/bounded_candidates.py',HERE.parent/'cwm_v27/protocol.json']
    for path in sources:
        relative = path.relative_to(ROOT).as_posix()
        raw = subprocess.check_output(['git','show',commit+':'+relative],cwd=ROOT)
        if raw != path.read_bytes():
            raise ValueError('Source/protocol changed after pushed commit: '+relative)
    return {'commit':commit,'remote':REMOTE,'branch':BRANCH,'exact_pushed_sources_verified':True}


class SequentialTap(Baseline):
    def __init__(self,capsule,restored,steps):
        super().__init__(capsule,restored)
        self.calls,self.current_step=[],None
        parent=self.evaluator.DistributedMinimaxDNMPC
        owner=self
        class Planner(parent):
            def plan(self,observation,scenarios,**kwargs):
                owner.calls=[]
                owner.current_step=int(kwargs.get('step_index',0))
                return super().plan(observation,scenarios,**kwargs)
            def _select_local_sequence(self,observation,scenarios,agent_id,local_candidates,known,peer_sequences):
                selected,costs=super()._select_local_sequence(observation,scenarios,agent_id,local_candidates,known,peer_sequences)
                if owner.current_step in steps:
                    owner.calls.append({'observation':copy.deepcopy(observation),'scenarios':copy.deepcopy(scenarios),
                        'agent':int(agent_id),'ordinal':len(owner.calls),'planner':SimpleNamespace(config=self.config),
                        'local_candidates':np.stack(local_candidates).copy(),'selected':selected.copy(),'selected_costs':costs.copy(),
                        'known':copy.deepcopy(known),'peer_sequences':copy.deepcopy(peer_sequences)})
                return selected,costs
        self.evaluator.DistributedMinimaxDNMPC=Planner


def inspect(base,env,captured,history,adapter,noise_key):
    from encirclement3d.minimax_mpc import aggregate_scenario_costs
    snapshot=public_call_snapshot(captured)
    call=restore_public_call(snapshot,base.planner_config,base.distributed_config,captured['scenarios'].trajectories[0])
    row={'agent':call['agent'],'ordinal':call['ordinal'],'noise_key':noise_key,
         'actual_candidate_count':len(call['local_candidates'])}
    result=expand_candidates(call,history,base.evaluator.belief_reference,adapter,mode='shadow',source='shared')
    if result['inputs'] is None:
        return {**row,'status':result['info']['status'],'proposal_info':result['info']},None,snapshot
    actual=call['local_candidates'];own=result['actions']
    reference,velocity=base.evaluator.belief_reference(call['observation'],call['planner'].config)
    backbone=call['scenarios'].trajectories[0].copy()
    original_costs=local_cost(call,actual,np.repeat(backbone[None],len(actual),axis=0))
    chosen=next(i for i,a in enumerate(actual) if np.array_equal(a,call['selected']))
    source_cost=aggregate_scenario_costs(call['selected_costs'],call['scenarios'].normalized_weights,
                                      call['planner'].config.risk_mode,call['planner'].config.cvar_alpha)
    if not np.isclose(original_costs[chosen],source_cost,rtol=1e-10,atol=1e-8) or int(np.argmin(original_costs)) != chosen:
        raise ValueError('Original costs/choice changed before branch labeling')
    h,relative,joint,anchor,b,cv=result['inputs']
    # All model inputs/proposals frozen BEFORE any private branch truth is read.
    branches=[s4_rollout(base,env,p,noise_key) for p in joint]
    anchored=s4_rollout(base,env,anchor,noise_key)
    if not compare_arrays(anchored,s4_rollout(base,env,anchor,noise_key)):
        raise ValueError('Repeated paired anchor differs')
    values={k:np.stack([branch[k] for branch in branches]) for k in branches[0]}
    values.update({'anchor_'+k:v for k,v in anchored.items()})
    values.update(history=h,relative=relative,proposed=joint,anchor=anchor,backbone=b,reference=reference,velocity=velocity,
                  actual_candidates=actual.copy(),original_local_costs=original_costs,
                  public_geometry_indices=np.arange(result['info']['seed_count'],dtype=np.int64),
                  proposal_prediction=result['forecast']['prediction'],proposal_motion_reference=result['forecast']['reference'],
                  proposal_response=result['forecast']['response'])
    anchor_index=next(i for i,a in enumerate(own) if np.array_equal(a,call['selected']))
    return {**row,'status':'collected_bounded_sequential_pairs','actual_choice_pool_index':anchor_index,
            'expanded_candidate_count':len(own),'proposal_info':result['info'],'original_local_costs_equal':True,
            'anchor_repeat_equal':True},values,snapshot


def summarize(calls,records,episodes,protocol):
    mappings=[]
    for c in calls:
        actual,_=map_actual(c['values'],c['record']['agent'])
        for name,indices in (('actual_unique',actual),('public_geometry',c['values']['public_geometry_indices']),
                             ('bounded_shared',np.arange(len(c['values']['proposed'])))):
            mappings.append(support_record(c,indices,name,protocol))
    response={name:response_summary([r for r in mappings if r['population']==name]) for name in protocol['candidate_populations']}
    gate=protocol['data_gate'];union=response['bounded_shared']
    invalid=sum(int(((~c['values']['valid']) & (c['values']['termination']!='after_terminal')).any(-1).sum()) for c in calls)
    failures=sum(r['status'] not in ('collected_bounded_sequential_pairs','missing_received_peer_context') for r in records)
    signals=[g for g,v in union['by_group'].items() if v['over_threshold_points']]
    checks={'original_target_valid':sum(e['target_invalid_episode'] for e in episodes)<=gate['maximum_original_target_invalid_episodes'],
            'branch_target_valid':invalid<=gate['maximum_probe_target_invalid_branches'],
            'all_groups_nonanchor_support':union['groups_with_nonanchor_support']>=gate['minimum_groups_with_nonanchor_support'],
            'signal_groups':len(signals)>=gate['minimum_signal_groups'],
            'response_fraction':union['group_equal_over_threshold_fraction'] is not None and union['group_equal_over_threshold_fraction']>=gate['minimum_group_equal_over_threshold_fraction'],
            'full_support':union['full_support_calls']>=gate['minimum_full_support_calls'],
            'proposal_failures':failures<=gate['maximum_proposal_failure_calls']}
    splits={}
    for split,short in (('train','train'),('development_validation','development')):
        selected=[m for m,c in zip([r for r in mappings if r['population']=='bounded_shared'],calls) if c['record']['split']==split]
        stats=response_summary(selected)
        groups=protocol['train_groups'] if split=='train' else protocol['development_groups']
        checks[short+'_all_groups']=stats['groups_with_nonanchor_support']==groups
        checks[short+'_full_support']=stats['full_support_calls']>=gate['minimum_full_support_'+short+'_calls']
        checks[short+'_paired_points']=stats['valid_nonanchor_points']>=gate['minimum_common_valid_nonanchor_points_each_split']
        checks[short+'_signals']=sum(g['over_threshold_points']>0 for g in stats['by_group'].values())>=gate['minimum_signal_groups_each_split']
        checks[short+'_response_fraction']=stats['group_equal_over_threshold_fraction'] is not None and stats['group_equal_over_threshold_fraction']>=gate['minimum_group_equal_over_threshold_fraction']
        splits[split]=stats
    return {'response_support':response,'split_support':splits,'target_invalid_branches':invalid,
            'proposal_failure_calls':failures,'data_checks':{k:bool(v) for k,v in checks.items()},
            'data_gate_passed':all(checks.values()),'training_authorized_by_this_report':False,
            'pending':'Independent public252/received-context/candidate/branch/full-cost audit before any training.'}


def main(output):
    protocol=json.loads((HERE/'data_protocol.json').read_text())
    validate_data_protocol(protocol)
    preregistration=published_commit()
    capsule=HERE.parent/'cwm_v1/baseline/capsule.zip'
    proposer=HERE.parent/'cwm_v26/artifacts/real_sequential_shadow_20261010.zip'
    if sha(capsule)!=protocol['baseline_capsule_sha256'] or sha(proposer)!=protocol['proposal_archive_sha256']:
        raise ValueError('Pinned baseline/proposal archive differs')
    output.mkdir(parents=True,exist_ok=False)
    save_json(output/'preregistration.json',preregistration)
    base=SequentialTap(capsule,output/'restored',set(protocol['snapshot_steps']))
    def load():
        with zipfile.ZipFile(proposer) as z: raw=z.read('model/primary.pt')
        return checkpoint_predictor(raw,protocol['proposal_checkpoint_sha256'])
    adapter=OptionalResponseAdapter(QualificationGate(mode='shadow',qualification={'research_eligible':False}),load)
    scenes=translated_records(base,protocol,protocol['wall_center_x_m'])
    for i,scene in enumerate(scenes):
        scene['model_split']='train' if i//2<protocol['train_groups'] else 'development_validation'
    if {s['episode_seed'] for s in scenes}!=set(range(994010,994074)) or {s['layout_seed'] for s in scenes}!=set(range(1994010,1994042)):
        raise ValueError('Fresh scene identities differ')
    (output/'scenes.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in scenes),encoding='utf8')
    sources={Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
             and Path(m.__file__).resolve().is_relative_to(ROOT/'experiments')}|{HERE/'data_protocol.json',HERE/'training_protocol.json',
             HERE.parent/'cwm_v23/cost_origin_model.py',HERE.parent/'cwm_v23/training_protocol.json',
             HERE.parent/'cwm_v10/two_head_model.py',HERE.parent/'cwm_v10/train_two_head.py'}
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    for p in sources:
        target=output/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    calls,records,episodes=[],[],[]
    for scene in scenes:
        history=[];commands=[];plans=[]
        def observer(env,observation,actions,sequence):
            history.append(base.evaluator.policy_observations(env,observation).reshape(-1).copy())
            commands.append(actions.copy());plans.append(sequence.copy())
            if env.step_count not in protocol['snapshot_steps']: return
            padded=np.stack([history[0]]*max(0,8-len(history))+history[-8:])
            parent_before=fingerprint(env)
            for captured in base.calls:
                captured_before=pickle.dumps(public_call_snapshot(captured),protocol=5)
                identity=f"{protocol['noise_seed']}/{scene['episode_index']}/{env.step_count}/{captured['agent']}/{captured['ordinal']}"
                noise_key=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4],'little')
                row,values,snapshot=inspect(base,env,captured,padded,adapter,noise_key)
                row.update(episode_index=scene['episode_index'],group=scene['mirror_group_id'],split=scene['model_split'],step=int(env.step_count))
                name=f"{scene['episode_index']}_{env.step_count}_{captured['ordinal']}_{captured['agent']}"
                path=output/'calls'/(name+'.npz');path.parent.mkdir(exist_ok=True)
                context=path.with_suffix('.json');save_json(context,snapshot)
                row.update(context_path=context.relative_to(output).as_posix(),context_sha256=sha(context))
                if values is not None:
                    np.savez_compressed(path,**values)
                    row.update(arrays_path=path.relative_to(output).as_posix(),arrays_sha256=sha(path))
                    calls.append({'record':row,'values':values})
                if pickle.dumps(public_call_snapshot(captured),protocol=5)!=captured_before:
                    raise ValueError('Research labeling changed captured public context')
                records.append(row)
            if fingerprint(env)!=parent_before: raise ValueError('Research labeling changed original environment')
            save_json(output/'records.json',records)
            print(json.dumps({'episode':scene['episode_index'],'step':int(env.step_count),'calls':len(calls),'records':len(records)}),flush=True)
        observed=output/'observed'/f"{scene['episode_index']}.npz"
        plain=output/'plain'/observed.name
        row,_=base.run(scene,observed,observer)
        observed_commands={'commanded':np.stack(commands),'planned':np.stack(plans)}
        np.savez_compressed(observed.with_suffix('.commands.npz'),**observed_commands)
        commands.clear();plans.clear()
        def plain_observer(env,observation,actions,sequence): commands.append(actions.copy());plans.append(sequence.copy())
        original,_=base.run(scene,plain,plain_observer)
        np.savez_compressed(plain.with_suffix('.commands.npz'),commanded=np.stack(commands),planned=np.stack(plans))
        if not compare_arrays(arrays(observed.read_bytes()),arrays(plain.read_bytes())) or not compare_arrays(observed_commands,{'commanded':np.stack(commands),'planned':np.stack(plans)}):
            raise ValueError('Optional collection changed original trajectory/plan/CBF commands')
        keys=('safe_capture_success','collision','boundary_violation','timeout','target_invalid_episode','termination_reason')
        if any(row[k]!=original[k] for k in keys): raise ValueError('Optional collection changed original outcomes')
        if bool(target_bad(arrays(observed.read_bytes())['target_positions'],scene['scenario']).any())!=row['target_invalid_episode']:
            raise ValueError('Original target validity differs')
        episodes.append({'episode_index':scene['episode_index'],'split':scene['model_split'],**{k:row[k] for k in keys},
                         'trajectory_commands_plans_equal':True,'observed_sha256':sha(observed),'plain_sha256':sha(plain)})
        save_json(output/'episodes.json',episodes)
        print(json.dumps({'completed_episodes':len(episodes),'calls':len(calls),'split':scene['model_split']}),flush=True)
    verify(base.root,base.capsule_manifest)
    if any(sha(ROOT/n)!=digest for n,digest in hashes.items()): raise ValueError('Collection sources changed during run')
    summary={'status':'fresh_bounded_sequential_data_collected_pending_independent_audit','protocol':protocol,
             'preregistration':preregistration,'source_hashes':hashes,'episodes':episodes,'calls':len(calls),'records':len(records),
             'scene_sha256':sha(output/'scenes.jsonl'),'records_sha256':sha(output/'records.json'),
             **summarize(calls,records,episodes,protocol),'enhanced_control_enabled':False,'new_model_trained':False,'holdout_used':False}
    save_json(output/'summary.json',summary)
    print(json.dumps({'status':summary['status'],'data_gate_passed':summary['data_gate_passed']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    main(args.output)
