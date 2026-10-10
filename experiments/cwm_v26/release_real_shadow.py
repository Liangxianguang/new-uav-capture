"""Independent original-entry/public-context/model reinference release audit."""
import argparse
import copy
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(HERE.parent/'cwm_v25'),str(HERE.parent/'cwm_v9'),str(HERE.parent/'cwm_v1')]
from checkpoint_provider import checkpoint_predictor, PinnedV23Loader, ARCHIVE_SHA
from artifact_transport import resolve_artifact,file_digest
from baseline import Baseline
from freeze_baseline import REFERENCE,sha,verify
from original_entry_boundary import compare_rows,BASELINE_SHA
from public_provider import delayed_joint_context

CHECKPOINT_SHA='907f6c8c7df68ee53c626d0b8ade973e6053be53f2973b30366df1d444ecfa14'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def arrays(raw):
    with np.load(io.BytesIO(raw),allow_pickle=False) as f:
        return {k:f[k].copy() for k in f.files}


def equal_arrays(a,b,keys):
    return all(a[k].shape==b[k].shape and a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes() for k in keys)


def independent_public_replay(base,protocol,output):
    """Public history from pre-step env observer, NOT the model runtime hook."""
    from types import SimpleNamespace
    from local_shadow import public_call_snapshot
    parent=base.evaluator.DistributedMinimaxDNMPC
    calls=[]
    class Probe(parent):
        def plan(self,observation,scenarios,**kwargs):
            calls.clear()
            return super().plan(observation,scenarios,**kwargs)
        def _select_local_sequence(self,observation,scenarios,agent_id,local_candidates,known,peer_sequences):
            selected,cost=super()._select_local_sequence(observation,scenarios,agent_id,local_candidates,known,peer_sequences)
            calls.append({'observation':copy.deepcopy(observation),'scenarios':copy.deepcopy(scenarios),
                'agent':int(agent_id),'ordinal':len(calls),'planner':SimpleNamespace(config=self.config),
                'local_candidates':np.stack(local_candidates).copy(),'known':copy.deepcopy(known),
                'peer_sequences':copy.deepcopy(peer_sequences),'selected':selected.copy(),'selected_costs':cost.copy()})
            return selected,cost
    base.evaluator.DistributedMinimaxDNMPC=Probe
    records={r['episode_index']:r for r in base.records()}
    expected=[]
    try:
        for key in protocol['record_ids']:
            history=[];commanded=[];planned=[]
            def observe(env,observation,actions,sequence):
                history.append(base.evaluator.policy_observations(env,observation).reshape(-1).copy())
                padded=np.stack([history[0]]*max(0,8-len(history))+history[-8:])
                commanded.append(actions.copy());planned.append(sequence.copy())
                for call in calls:
                    identity={'episode_index':key,'step':int(env.step_count),'agent':call['agent'],'ordinal':call['ordinal']}
                    reference,velocity=base.evaluator.belief_reference(call['observation'],base.planner_config)
                    context=delayed_joint_context(call['observation'],call['known'],call['peer_sequences'],call['agent'],
                        call['local_candidates'],call['selected'],reference,velocity)
                    if context is None:
                        expected.append((identity,None,None))
                    else:
                        proposed,anchor,relative=context
                        cv=reference[None]+.1*np.arange(1,9)[:,None]*velocity[None]
                        expected.append((identity,public_call_snapshot(call),dict(history=padded.copy(),relative=relative,
                            proposed=proposed,anchor=anchor,backbone=call['scenarios'].trajectories[0].copy(),cv=cv)))
            path=output/f'{key}.npz'
            output.mkdir(parents=True,exist_ok=True)
            row,_=base.run(records[key],path,observe)
            np.savez_compressed(path.with_suffix('.commands.npz'),commanded=np.stack(commanded),planned=np.stack(planned))
            with np.load(base.root/REFERENCE/'trajectories'/f"level{records[key]['level']}_{key}.npz") as original,np.load(path) as measured:
                if any(not np.array_equal(original[k],measured[k]) for k in ('defender_positions','target_positions')):
                    raise ValueError('Independent public observer changed original trajectory')
    finally:
        base.evaluator.DistributedMinimaxDNMPC=parent
    return expected


def audit(read,base,expected,independent_output):
    from local_shadow import restore_public_call
    protocol=json.loads((HERE/'protocol.json').read_text())
    summaries=[json.loads(read(stage+'/summary.json')) for stage in ('primary','repeated')]
    raw=read('model/primary.pt')
    predict=checkpoint_predictor(raw,CHECKPOINT_SHA)
    supported=sum(values is not None for _,_,values in expected)
    missing=len(expected)-supported
    if not supported or not any(i['ordinal']>=4 for i,_,_ in expected):
        raise ValueError('No complete sequential best-response coverage')
    historical={r['episode_index']:r for r in map(json.loads,(base.root/REFERENCE/'episodes.jsonl').read_text().splitlines())}
    for stage,summary in zip(('primary','repeated'),summaries):
        if (summary['status']!='real_checkpoint_sequential_shadow_finished_not_control_qualification' or
            summary['protocol']!=protocol or summary['source_hashes']!=summaries[0]['source_hashes'] or
            set(summary['modes'])!=set(protocol['modes']) or
            any(summary[k] is not False for k in ('enhanced_control_enabled','holdout_used','new_model_trained')) or
            summary['historical_capsule_integrity_after_replay'] is not True):
            raise ValueError('Not complete failed-model sequential shadow evidence')
        for name,sha256 in summary['source_hashes'].items():
            if digest((ROOT/name).read_bytes())!=sha256 or digest(read('source/'+name))!=sha256:
                raise ValueError('Saved/current real-shadow source differs')
        for mode in protocol['modes']:
            result=summary['modes'][mode]
            if [r['episode_index'] for r in result['rows']]!=protocol['record_ids']:
                raise ValueError('Fixed historical records differ')
            for row in result['rows']:
                if compare_rows(historical[row['episode_index']],row):
                    raise ValueError('Historical physical outcome differs')
                name=f"level{row['level']}_{row['episode_index']}.npz"
                current=arrays(read(f'{stage}/{mode}/{name}'))
                historical_arrays=arrays((base.root/REFERENCE/'trajectories'/name).read_bytes())
                if not equal_arrays(current,historical_arrays,('defender_positions','target_positions')):
                    raise ValueError('Historical physical trajectory differs')
                command_name=name[:-4]+'.commands.npz'
                commands=arrays(read(f'{stage}/{mode}/{command_name}'))
                independent=arrays((independent_output/f"{row['episode_index']}.commands.npz").read_bytes())
                if not equal_arrays(commands,independent,('commanded','planned')):
                    raise ValueError('DN-MPC plan or CBF command differs from independent original')
            events=result['events']
            if any(e['control_eligible'] is not False for e in events):
                raise ValueError('Failed shadow is controlling')
            if mode in ('plain','off','refusal'):
                if result['load_calls'] or result['prediction_calls'] or result['checkpoint_sha256'] is not None:
                    raise ValueError('Disabled mode read a checkpoint')
                continue
            if (result['supported_calls']!=supported or result['missing_peer_calls']!=missing or len(events)!=len(expected) or
                    result['checkpoint_sha256']!=CHECKPOINT_SHA or result['load_calls']!=1):
                raise ValueError('Sequential coverage or real checkpoint identity differs')
            fault={'load_failure':(0,'load_failed'),'prediction_failure':(1,'prediction_failed'),
                   'real_shadow':(supported,'shadow_ready')}
            if (result['prediction_calls'],result['status'])!=fault[mode]:
                raise ValueError('Real checkpoint mode/fault not exercised')
            for event,(identity,snapshot,values) in zip(events,expected):
                if any(event[k]!=v for k,v in identity.items()):
                    raise ValueError('Sequential call identity differs')
                if values is None:
                    if event['status']!='missing_received_peer_context' or event['forecast_available'] is not False or 'arrays_path' in event:
                        raise ValueError('Missing delayed peer plan silently filled')
                    continue
                context_raw=read(f"{stage}/{mode}/"+event['context_path'])
                array_raw=read(f"{stage}/{mode}/"+event['arrays_path'])
                if digest(context_raw)!=event['context_sha256'] or digest(array_raw)!=event['arrays_sha256']:
                    raise ValueError('Saved context/forecast hashes differ')
                if json.loads(context_raw)!=snapshot:
                    raise ValueError('Actual delayed sequential context differs from independent original')
                saved=arrays(array_raw)
                if not equal_arrays(saved,values,tuple(values)):
                    raise ValueError('Real public252 history/delayed-peer/GRU/CV input differs')
                call=restore_public_call(snapshot,base.planner_config,base.distributed_config,values['backbone'])
                for peer,message in call['known'].items():
                    if (message.sender!=peer or message.receiver!=identity['agent'] or message.delivery_step>identity['step'] or
                            identity['step']-message.sent_step>base.distributed_config.max_message_age_steps):
                        raise ValueError('Delayed message contract differs')
                generated=call['planner']._local_candidate_sequences(call['observation'],call['scenarios'],call['agent'],
                    call['observation']['defender_positions'][call['agent']],call['known'])
                if not np.array_equal(np.stack(generated),call['local_candidates']):
                    raise ValueError('Original candidate reconstruction differs')
                costs=call['planner']._local_scenario_cost_matrix(call['observation'],call['scenarios'],call['agent'],
                    np.stack(generated),call['known'],call['peer_sequences'])[:,0]
                choice=int(np.argmin(costs))
                if not np.array_equal(generated[choice],call['selected']) or not np.isclose(costs[choice],call['selected_costs'][0],atol=1e-8,rtol=1e-10):
                    raise ValueError('Original local cost/choice reconstruction differs')
                if mode=='real_shadow':
                    inferred=predict(*(values[k] for k in ('history','relative','proposed','anchor','backbone','cv')))
                    if not equal_arrays(saved,inferred,tuple(inferred)) or event['forecast_available'] is not True:
                        raise ValueError('Real checkpoint forecast reinference differs')
                elif event['forecast_available'] is not False or any(k in saved for k in ('prediction','response','reference')):
                    raise ValueError('Fault unexpectedly yielded a forecast')
    verify(base.root,base.capsule_manifest)
    return {'status':'passed_independent_real_public_sequential_shadow_audit','independent_runs':2,
        'episodes_per_mode_per_run':len(protocol['record_ids']),'sequential_calls_per_fault_shadow_mode':len(expected),
        'supported_calls':supported,'missing_received_peer_calls':missing,'real_forecasts_reinferred_per_run':supported,
        'checkpoint_sha256':CHECKPOINT_SHA,'baseline_capsule_sha256':BASELINE_SHA,'model_archive_sha256':ARCHIVE_SHA,
        'historical_trajectories_and_original_plans_commands_exact':True,'independent_public252_delayed_contexts_exact':True,
        'original_candidates_and_local_cost_choices_recomputed':True,'enhanced_control_enabled':False,
        'new_model_trained':False,'holdout_used':False,'scope':protocol['remaining']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('primary','repeated','output','report','verify'):
        parser.add_argument('--'+key,type=Path)
    parser.add_argument('--audit-output',type=Path,required=True)
    parser.add_argument('--artifact',type=Path,default=HERE.parent/'cwm_v23/artifacts/deployed_cost_contrast_20261010.parts.json')
    args=parser.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    capsule=HERE.parent/'cwm_v1/baseline/capsule.zip'
    if sha(capsule)!=BASELINE_SHA:
        raise ValueError('Historical capsule differs')
    args.audit_output.mkdir(parents=True,exist_ok=False)
    base=Baseline(capsule,args.audit_output/'restored')
    expected=independent_public_replay(base,json.loads((HERE/'protocol.json').read_text()),args.audit_output/'independent')
    if args.verify:
        with zipfile.ZipFile(args.verify) as z:
            manifest=json.loads(z.read('ARTIFACT_MANIFEST.json'))
            if set(z.namelist())!=set(manifest)|{'ARTIFACT_MANIFEST.json'} or any(digest(z.read(k))!=v for k,v in manifest.items()):
                raise ValueError('Archive member population/hash differs')
            result=audit(z.read,base,expected,args.audit_output/'independent')
            if result!=json.loads(z.read('release_summary.json')):
                raise ValueError('Archived independent audit summary differs')
        print(json.dumps({**result,'artifact_sha256':sha(args.verify)}),flush=True)
        return
    if any(x is None for x in (args.primary,args.repeated,args.output,args.report)):
        parser.error('Both completed shadow populations and output/report required')
    members={}
    for stage,directory in (('primary',args.primary),('repeated',args.repeated)):
        members[stage+'/summary.json']=(directory/'summary.json').read_bytes()
        protocol=json.loads(members[stage+'/summary.json'])['protocol']
        for mode in protocol['modes']:
            for p in sorted((directory/mode).rglob('*')):
                if p.is_file():
                    members[stage+'/'+p.relative_to(directory).as_posix()]=p.read_bytes()
    summary=json.loads(members['primary/summary.json'])
    for name in summary['source_hashes']:
        members['source/'+name]=(ROOT/name).read_bytes()
    archive=resolve_artifact(args.artifact,args.audit_output/'assembled')
    if file_digest(archive)!=ARCHIVE_SHA:
        raise ValueError('Pinned original model archive differs')
    with zipfile.ZipFile(archive) as z:
        members['model/primary.pt']=z.read('primary/cv_cost_l2_seed993102.pt')
    result=audit(members.__getitem__,base,expected,args.audit_output/'independent')
    members['release_summary.json']=json.dumps(result,indent=2,allow_nan=False).encode()
    members['verification_source/release_real_shadow.py']=Path(__file__).read_bytes()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in members.items():
            z.writestr(name,raw)
        z.writestr('ARTIFACT_MANIFEST.json',json.dumps({k:digest(v) for k,v in members.items()},indent=2))
    with zipfile.ZipFile(args.output) as z:
        if audit(z.read,base,expected,args.audit_output/'independent')!=result:
            raise ValueError('Post-archive independent audit differs')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x',encoding='utf8') as file:
        json.dump({**result,'artifact_sha256':sha(args.output),'artifact_bytes':args.output.stat().st_size},file,indent=2,allow_nan=False)
    print(json.dumps({**result,'artifact_bytes':args.output.stat().st_size}),flush=True)


if __name__=='__main__':
    main()
