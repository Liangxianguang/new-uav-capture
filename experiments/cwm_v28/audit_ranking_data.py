"""Independent original replay and complete fresh sequential paired-data audit.

Never uses collection hooks/inspect() or the saved history to create expected
observations. Re-observes the frozen original entry, regenerates public pools,
re-rolls every private branch on independent copies and checks full engine costs.
"""
import argparse
import copy
import hashlib
import json
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
               str(HERE.parent/'cwm_v25'),str(HERE.parent/'cwm_v23'),str(HERE.parent/'cwm_v10')]
from collect_ranking_data import validate_data_protocol, summarize, save_json, REMOTE, BRANCH
from bounded_candidates import expand_candidates
from checkpoint_provider import checkpoint_predictor
from qualification_gate import OptionalResponseAdapter, QualificationGate
from baseline import Baseline
from freeze_baseline import sha,verify
from local_shadow import public_call_snapshot,restore_public_call,s4_rollout,fingerprint
from geometry_release import arrays,compare_arrays,target_bad
from task_score import original_scores,FrozenLocalScore
from two_head_release import check_geometry


def _equal(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def check_scene_contract(scene,protocol):
    if (scene['execution']!={'enabled':False,'action_delay_steps':0,'command_noise_std':0.} or
        scene['obstacle_count']!=1 or scene['level'] is not None or scene['geometry_wall_center_x_m']!=1.5):
        raise ValueError('Original execution/scene contract differs')
    metadata=scene['scenario'];wall=metadata['obstacles'][0]
    if (metadata['target_escape_direction']!=[1.,0.,0.] or metadata['scenario_type']!='s4_branching' or
        metadata['defender_side']!='left' or metadata['layout_seed']!=scene['layout_seed'] or
        metadata['target_crossing_required'] is not False or metadata['required_defender_zone_entries']!=1 or
        metadata['require_target_zone_entry'] is not None or wall['orientation_degrees']!=90. or
        wall['radius']!=wall['half_extents_xy'][0] or
        metadata['obstacle_zone_x']!=[1.5-wall['radius'],1.5+wall['radius']]):
        raise ValueError('Frozen target/geometry metadata differs')
    condition=next(c for c in protocol['observation_conditions'] if c['overrides']==scene['pursuit_overrides'])
    if (scene['observation_condition']!=condition['name'] or
        scene['variant']!=f"s4_{condition['name']}_{scene['target_speed_scale']}" or
        metadata['name']!=f"s4_adaptive_branching_{scene['mirror_pair_member']}_{scene['layout_seed']}_wallx1.5"):
        raise ValueError('Frozen public variant/provenance differs')


def check_record_paths(row):
    name=f"calls/{row['episode_index']}_{row['step']}_{row['ordinal']}_{row['agent']}"
    if row['context_path']!=name+'.json' or ('arrays_path' in row and row['arrays_path']!=name+'.npz'):
        raise ValueError('Recorded public call path/identity differs')


def audit(data,output):
    protocol=json.loads((HERE/'data_protocol.json').read_text())
    validate_data_protocol(protocol)
    report=json.loads((data/'summary.json').read_text())
    if (report['protocol']!=protocol or report['status']!='fresh_bounded_sequential_data_collected_pending_independent_audit' or
        any(report[k] is not False for k in ('enhanced_control_enabled','new_model_trained','holdout_used')) or
        report['training_authorized_by_this_report'] is not False):
        raise ValueError('Complete original-only fresh dataset required')
    prereg=json.loads((data/'preregistration.json').read_text())
    if prereg!=report['preregistration'] or prereg['remote']!=REMOTE or prereg['branch']!=BRANCH or prereg['exact_pushed_sources_verified'] is not True:
        raise ValueError('Exact published preregistration differs')
    for name in ('data_protocol.json','training_protocol.json','collect_ranking_data.py','ranking_loss.py'):
        relative='experiments/cwm_v28/'+name
        committed=subprocess.check_output(['git','show',prereg['commit']+':'+relative],cwd=ROOT)
        if committed!=(HERE/name).read_bytes():
            raise ValueError('Preregistered training/collection source differs: '+name)
    for name,digest in report['source_hashes'].items():
        if sha(ROOT/name)!=digest or sha(data/'source'/name)!=digest:
            raise ValueError('Original collection source differs: '+name)
    for name,key in (('scenes.jsonl','scene_sha256'),('records.json','records_sha256')):
        if sha(data/name)!=report[key]: raise ValueError('Dataset digest differs: '+name)
    records=json.loads((data/'records.json').read_text())
    episodes=json.loads((data/'episodes.json').read_text())
    scenes=[json.loads(s) for s in (data/'scenes.jsonl').read_text().splitlines()]
    if (len(scenes)!=64 or len(episodes)!=64 or episodes!=report['episodes'] or len(records)!=report['records'] or
        [s['episode_index'] for s in scenes]!=list(range(994010,994074)) or
        {s['layout_seed'] for s in scenes}!=set(range(1994010,1994042)) or
        len({(r['episode_index'],r['step'],r['ordinal'],r['agent']) for r in records})!=len(records)):
        raise ValueError('Fixed fresh complete population or sequential identities differ')
    geometry_protocol={**protocol,'variant_cycles_split':['train']*6+['development_validation']*2}
    for scene in scenes:
        check_geometry(scene,geometry_protocol)
        check_scene_contract(scene,protocol)
        group=scene['layout_seed']-protocol['layout_seed_start']
        if scene['model_split']!=('train' if group<24 else 'development_validation'):
            raise ValueError('Preassigned mirror-group split differs')
    output.mkdir(parents=True,exist_ok=False)
    capsule=HERE.parent/'cwm_v1/baseline/capsule.zip'
    teacher=HERE.parent/'cwm_v26/artifacts/real_sequential_shadow_20261010.zip'
    if sha(capsule)!=protocol['baseline_capsule_sha256'] or sha(teacher)!=protocol['proposal_archive_sha256']:
        raise ValueError('Pinned original/proposal archive differs')
    base=Baseline(capsule,output/'restored')
    with zipfile.ZipFile(teacher) as archive: raw=archive.read('model/primary.pt')
    predictor=checkpoint_predictor(raw,protocol['proposal_checkpoint_sha256'])
    adapter=OptionalResponseAdapter(QualificationGate(mode='shadow',qualification={'research_eligible':False}),lambda:predictor)
    audit_sources={Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
        and Path(m.__file__).resolve().is_relative_to(ROOT/'experiments')}|{HERE/'data_protocol.json',HERE/'training_protocol.json'}
    source_hashes={p.relative_to(ROOT).as_posix():sha(p) for p in audit_sources}
    for p in audit_sources:
        destination=output/'source'/p.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(p.read_bytes())
    parent=base.evaluator.DistributedMinimaxDNMPC
    invocations=[];step=[None]
    class IndependentPlanner(parent):
        def plan(self,observation,scenarios,**kwargs):
            invocations.clear();step[0]=int(kwargs.get('step_index',0))
            return super().plan(observation,scenarios,**kwargs)
        def _select_local_sequence(self,observation,scenarios,agent_id,local_candidates,known,peer_sequences):
            chosen,costs=super()._select_local_sequence(observation,scenarios,agent_id,local_candidates,known,peer_sequences)
            if step[0] in protocol['snapshot_steps']:
                invocations.append({'agent':int(agent_id),'ordinal':len(invocations),
                    'observation':copy.deepcopy(observation),'scenarios':copy.deepcopy(scenarios),
                    'planner':SimpleNamespace(config=self.config),'local_candidates':np.stack(local_candidates).copy(),
                    'selected':chosen.copy(),'selected_costs':costs.copy(),
                    'known':copy.deepcopy(known),'peer_sequences':copy.deepcopy(peer_sequences)})
            return chosen,costs
    base.evaluator.DistributedMinimaxDNMPC=IndependentPlanner
    calls,checked,cost_checks=[],[],[]
    by_episode={s['episode_index']:[r for r in records if r['episode_index']==s['episode_index']] for s in scenes}
    try:
        for scene,episode in zip(scenes,episodes):
            if episode['episode_index']!=scene['episode_index'] or episode['split']!=scene['model_split']:
                raise ValueError('Episode order/split differs')
            history=[];frames=[];commands=[];plans=[];cursor=[0]
            selected=by_episode[scene['episode_index']]
            from encirclement3d.showcase import s4_branch_route_metrics,validate_s4_branching_scenario
            geometry_env=base.env_class(base.configuration(scene),obstacle_count=1,target_speed_scale=scene['target_speed_scale'])
            actual_scenario=base.scenario_from_metadata(scene['scenario'])
            validate_s4_branching_scenario(geometry_env,actual_scenario)
            if s4_branch_route_metrics(geometry_env,actual_scenario)!=scene['route_validation']:
                raise ValueError('Independent original route validation differs')
            def observer(env,observation,actions,sequence):
                feature=base.evaluator.policy_observations(env,observation).reshape(-1).copy()
                if feature.shape!=(252,): raise ValueError('Original public encoder differs')
                history.append(feature)
                frames.append(feature)
                commands.append(actions.copy());plans.append(sequence.copy())
                if env.step_count not in protocol['snapshot_steps']: return
                padded=np.stack([history[0]]*max(0,8-len(history))+history[-8:])
                before=fingerprint(env)
                for captured in invocations:
                    if cursor[0]>=len(selected): raise ValueError('Unrecorded original sequential call')
                    row=selected[cursor[0]];cursor[0]+=1
                    check_record_paths(row)
                    identity=(scene['episode_index'],int(env.step_count),captured['ordinal'],captured['agent'])
                    if tuple(row[k] for k in ('episode_index','step','ordinal','agent'))!=identity or row['split']!=scene['model_split'] or row['group']!=scene['mirror_group_id']:
                        raise ValueError('Recorded sequential identity/group differs')
                    path=data/row['context_path']
                    expected_snapshot=public_call_snapshot(captured)
                    if sha(path)!=row['context_sha256'] or json.loads(path.read_text())!=expected_snapshot:
                        raise ValueError('Actual received public sequential context differs')
                    call=restore_public_call(expected_snapshot,base.planner_config,base.distributed_config,captured['scenarios'].trajectories[0])
                    for peer,message in call['known'].items():
                        if (message.sender!=peer or message.receiver!=row['agent'] or message.delivery_step>env.step_count or
                            env.step_count-message.sent_step>base.distributed_config.max_message_age_steps):
                            raise ValueError('Received delayed message contract differs')
                    generated=np.stack(call['planner']._local_candidate_sequences(call['observation'],call['scenarios'],call['agent'],
                        call['observation']['defender_positions'][call['agent']],call['known']))
                    if not _equal(generated,call['local_candidates']): raise ValueError('Original candidates differ')
                    proposed=expand_candidates(call,padded,base.evaluator.belief_reference,adapter,mode='shadow',source='shared')
                    if proposed['inputs'] is None:
                        if ('arrays_path' in row or row['status']!=proposed['info']['status'] or row['proposal_info']!=proposed['info']):
                            raise ValueError('Missing/failure call improperly imputed')
                        checked.append({**dict(zip(('episode_index','step','ordinal','agent'),identity)),'status':row['status']})
                        continue
                    if row['status']!='collected_bounded_sequential_pairs' or row['proposal_info']!=proposed['info']:
                        raise ValueError('Public bounded proposal provenance differs')
                    path=data/row['arrays_path']
                    if sha(path)!=row['arrays_sha256']: raise ValueError('Saved paired array digest differs')
                    v=arrays(path.read_bytes())
                    expected=dict(zip(('history','relative','proposed','anchor','backbone','cv'),proposed['inputs']))
                    for key in ('history','relative','proposed','anchor','backbone'):
                        if not _equal(v[key],expected[key]): raise ValueError('Fresh public input differs: '+key)
                    reference,velocity=base.evaluator.belief_reference(call['observation'],call['planner'].config)
                    for key,value in (('reference',reference),('velocity',velocity),('actual_candidates',generated),
                        ('public_geometry_indices',np.arange(proposed['info']['seed_count'],dtype=np.int64))):
                        if not _equal(v[key],value): raise ValueError('Fresh public proposal feature differs: '+key)
                    for key,name in (('prediction','proposal_prediction'),('reference','proposal_motion_reference'),('response','proposal_response')):
                        if not _equal(v[name],proposed['forecast'][key]): raise ValueError('Fixed proposer reinference differs')
                    anchor_index=next(i for i,a in enumerate(proposed['actions']) if np.array_equal(a,call['selected']))
                    if row['actual_choice_pool_index']!=anchor_index: raise ValueError('Original anchor identity differs')
                    key=int.from_bytes(hashlib.sha256(f"{protocol['noise_seed']}/{row['episode_index']}/{row['step']}/{row['agent']}/{row['ordinal']}".encode()).digest()[:4],'little')
                    if key!=row['noise_key']: raise ValueError('Preassigned common random number differs')
                    branches=[s4_rollout(base,env,p,key) for p in expected['proposed']]
                    anchor=s4_rollout(base,env,expected['anchor'],key)
                    if not compare_arrays(anchor,s4_rollout(base,env,expected['anchor'],key)):
                        raise ValueError('Independent repeated anchor differs')
                    for name in anchor:
                        if not _equal(v[name],np.stack([b[name] for b in branches])) or not _equal(v['anchor_'+name],anchor[name]):
                            raise ValueError('Independent branch supervision differs: '+name)
                    raw_cost=original_scores(call,generated,np.repeat(expected['backbone'][None],len(generated),axis=0))
                    chosen=next(i for i,a in enumerate(generated) if np.array_equal(a,call['selected']))
                    if not np.allclose(raw_cost,v['original_local_costs'],rtol=1e-10,atol=1e-8) or int(np.argmin(raw_cost))!=chosen:
                        raise ValueError('Original full local costs/choice differ')
                    if v['valid'].all() and v['anchor_valid'].all():
                        actions=proposed['actions'];score=FrozenLocalScore(call,actions)
                        truth=original_scores(call,actions,v['target'])
                        if not np.allclose(score(torch.from_numpy(v['target'])).detach().numpy(),truth,rtol=1e-10,atol=1e-8):
                            raise ValueError('Full label cost surrogate differs from original engine')
                        cost_checks.append({'identity':list(identity),'true_costs':truth.tolist()})
                    calls.append({'record':row,'values':v})
                    checked.append({**dict(zip(('episode_index','step','ordinal','agent'),identity)),
                        'status':'public_context_proposal_all_branches_full_costs_checked'})
                if fingerprint(env)!=before: raise ValueError('Independent audit changed original parent')
            path=output/'trajectories'/f"{scene['episode_index']}.npz"
            row,_=base.run(scene,path,observer)
            if cursor[0]!=len(selected): raise ValueError('Recorded call omitted from original replay')
            trace={'commanded':np.stack(commands),'planned':np.stack(plans)}
            np.savez_compressed(path.with_suffix('.commands.npz'),**trace)
            np.savez_compressed(output/'trajectories'/f"{scene['episode_index']}.public252.npz",history=np.stack(frames))
            for kind in ('observed','plain'):
                original=data/kind/path.name
                if (sha(original)!=episode[kind+'_sha256'] or not compare_arrays(arrays(path.read_bytes()),arrays(original.read_bytes())) or
                    not compare_arrays(trace,arrays(original.with_suffix('.commands.npz').read_bytes()))):
                    raise ValueError('Independent original trajectory/plan/CBF commands differ')
            for key in ('safe_capture_success','collision','boundary_violation','timeout','target_invalid_episode','termination_reason'):
                if row[key]!=episode[key]: raise ValueError('Original episode outcome differs')
            if bool(target_bad(arrays(path.read_bytes())['target_positions'],scene['scenario']).any())!=row['target_invalid_episode']:
                raise ValueError('Original target geometry validity differs')
            save_json(output/'checked_calls.json',checked)
            print(json.dumps({'independently_audited_episodes':scene['episode_index']-994009,'checked_calls':len(checked)}),flush=True)
    finally:
        base.evaluator.DistributedMinimaxDNMPC=parent
    if len(checked)!=len(records) or len(calls)!=report['calls']: raise ValueError('Incomplete independent audit support')
    stats=summarize(calls,records,episodes,protocol)
    if any(report[key]!=value for key,value in stats.items()) or not stats['data_gate_passed']:
        raise ValueError('Recomputed complete data gate differs or failed')
    verify(base.root,base.capsule_manifest)
    if any(sha(ROOT/name)!=digest for name,digest in source_hashes.items()):
        raise ValueError('Audit-used sources changed during full replay')
    for name,digest in report['source_hashes'].items():
        if sha(ROOT/name)!=digest or sha(data/'source'/name)!=digest:
            raise ValueError('Collection source changed during audit')
    save_json(output/'full_label_cost_checks.json',cost_checks)
    used=['summary.json','preregistration.json','scenes.jsonl','records.json','episodes.json']
    used.extend(name for r in records for name in (r['context_path'],r.get('arrays_path')) if name is not None)
    used.extend('source/'+name for name in report['source_hashes'])
    save_json(output/'audited_data_manifest.json',{name:sha(data/name) for name in sorted(set(used))})
    audited={'status':'independent_fresh_sequential_public_branch_cost_audit_passed','data_summary_sha256':sha(data/'summary.json'),
        'protocol':protocol,'source_hashes':source_hashes,'checked_calls':len(checked),'supported_calls':len(calls),
        'audited_data_manifest_sha256':sha(output/'audited_data_manifest.json'),
        'checked_calls_sha256':sha(output/'checked_calls.json'),'full_label_cost_checks_sha256':sha(output/'full_label_cost_checks.json'),
        'complete_cost_calls':len(cost_checks),'data_gate_passed':True,'training_authorized_by_this_report':True,
        'enhanced_control_enabled':False,'new_model_trained':False,'holdout_used':False}
    save_json(output/'summary.json',audited)
    return calls,audited,base


def verified_calls(data,audit_output):
    """Reload only after the COMPLETE independently replayed audit passes."""
    p=json.loads((HERE/'data_protocol.json').read_text());validate_data_protocol(p)
    report=json.loads((audit_output/'summary.json').read_text())
    if (report['status']!='independent_fresh_sequential_public_branch_cost_audit_passed' or report['protocol']!=p or
        report['data_summary_sha256']!=sha(data/'summary.json') or report['training_authorized_by_this_report'] is not True or
        report['data_gate_passed'] is not True or any(report[k] is not False for k in ('enhanced_control_enabled','new_model_trained','holdout_used'))):
        raise ValueError('Independent complete data audit required before optimizer')
    for name,digest in report['source_hashes'].items():
        if sha(ROOT/name)!=digest or sha(audit_output/'source'/name)!=digest:
            raise ValueError('Auditor/source changed after complete audit')
    for name,key in (('checked_calls.json','checked_calls_sha256'),('full_label_cost_checks.json','full_label_cost_checks_sha256'),
                     ('audited_data_manifest.json','audited_data_manifest_sha256')):
        if sha(audit_output/name)!=report[key]: raise ValueError('Full completed audit evidence changed')
    manifest=json.loads((audit_output/'audited_data_manifest.json').read_text())
    for name,digest in manifest.items():
        if sha(data/name)!=digest: raise ValueError('Data changed after independent audit: '+name)
    data_report=json.loads((data/'summary.json').read_text())
    if (data_report['protocol']!=p or data_report['status']!='fresh_bounded_sequential_data_collected_pending_independent_audit' or
        sha(data/'records.json')!=data_report['records_sha256'] or sha(data/'scenes.jsonl')!=data_report['scene_sha256'] or
        json.loads((data/'episodes.json').read_text())!=data_report['episodes'] or
        any(data_report[k] is not False for k in ('enhanced_control_enabled','new_model_trained','holdout_used'))):
        raise ValueError('Audited original-only dataset contract differs')
    records=json.loads((data/'records.json').read_text());calls=[]
    required={'summary.json','preregistration.json','scenes.jsonl','records.json','episodes.json'}
    required.update(name for r in records for name in (r['context_path'],r.get('arrays_path')) if name is not None)
    required.update('source/'+name for name in data_report['source_hashes'])
    if set(manifest)!=required: raise ValueError('Audit manifest omitted or added dataset evidence')
    for row in records:
        check_record_paths(row)
        if sha(data/row['context_path'])!=row['context_sha256']: raise ValueError('Audited context changed')
        if 'arrays_path' in row:
            if sha(data/row['arrays_path'])!=row['arrays_sha256']: raise ValueError('Audited labels/input arrays changed')
            calls.append({'record':row,'values':arrays((data/row['arrays_path']).read_bytes())})
    if len(calls)!=report['supported_calls'] or len(records)!=report['checked_calls']:
        raise ValueError('Audited complete support changed')
    return calls,report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    _,report,_=audit(args.data,args.output)
    print(json.dumps({k:report[k] for k in ('status','checked_calls','supported_calls','complete_cost_calls')}),flush=True)
