"""Fresh public geometry probes at actual delayed local calls; original unchanged."""
import argparse
import hashlib
import json
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT),str(ROOT.parent/'cwm_v17'),str(ROOT.parent/'cwm_v9'),str(ROOT.parent/'cwm_v7')]
from geometry_probes import public_geometry_pool
from local_shadow import ActualLocalTap,delayed_joint_context,public_call_snapshot,local_cost,s4_rollout,fingerprint
from geometry import translated_records
from support_audit import support_record,response_summary,scored_record,decision_summary
from geometry_release import arrays,compare_arrays,target_bad,sha,BASELINE_SHA
from freeze_baseline import verify
from verify_release import check_archive


def inspect(base,env,call,history,protocol,noise_key):
    from encirclement3d.minimax_mpc import aggregate_scenario_costs
    row = {'agent':call['agent'],'ordinal':call['ordinal'],'noise_key':noise_key}
    observation,planner,agent = call['observation'],call['planner'],call['agent']
    reference,velocity = base.evaluator.belief_reference(observation,planner.config)
    actual = call['local_candidates']
    base_context = delayed_joint_context(observation,call['known'],call['peer_sequences'],agent,actual,call['selected'],reference,velocity)
    if base_context is None:
        return {**row,'status':'skipped_missing_delayed_peer_plan'},None
    costs = local_cost(call,actual,np.repeat(call['scenarios'].trajectories[0][None],len(actual),0))
    chosen = next(i for i,a in enumerate(actual) if np.array_equal(a,call['selected']))
    original = aggregate_scenario_costs(call['selected_costs'],call['scenarios'].normalized_weights,planner.config.risk_mode,planner.config.cvar_alpha)
    if not np.isclose(costs[chosen],original,rtol=1e-10,atol=1e-8) or int(np.argmin(costs)) != chosen:
        raise ValueError('Actual original local costs/choice changed')
    pool = public_geometry_pool(call,reference,velocity,protocol['probe_magnitude_mps'])
    joint,anchor,relative = delayed_joint_context(observation,call['known'],call['peer_sequences'],agent,pool,call['selected'],reference,velocity)
    branches = [s4_rollout(base,env,sequence,noise_key) for sequence in joint]
    anchored = s4_rollout(base,env,anchor,noise_key)
    if not compare_arrays(anchored,s4_rollout(base,env,anchor,noise_key)):
        raise ValueError('Repeated actual local anchor changed')
    values = {k:np.stack([b[k] for b in branches]) for k in branches[0]}
    values.update({**{'anchor_'+k:v for k,v in anchored.items()},'history':history,'relative':relative,
                   'backbone':call['scenarios'].trajectories[0].copy(),'proposed':joint,'anchor':anchor,
                   'actual_candidates':actual.copy(),'original_local_costs':costs,'reference':reference,'velocity':velocity})
    selected_pool = next(i for i,a in enumerate(pool) if np.array_equal(a,call['selected']))
    return {**row,'status':'collected_shadow_geometry_probes','actual_choice_pool_index':selected_pool,
            'original_local_costs_equal':True,'anchor_repeat_equal':True,'actual_candidate_count':len(actual),
            'expanded_candidate_count':len(pool)},values


def summarize(calls,decisions,episodes,protocol):
    mappings = []
    from support_audit import map_actual
    for c in calls:
        actual,_ = map_actual(c['values'],c['record']['agent'])
        for name,indices in (('actual_unique',actual),('geometry_union',np.arange(len(c['values']['proposed'])))):
            mappings.append(support_record(c,indices,name,protocol))
    response = {name:response_summary([r for r in mappings if r['population']==name]) for name in ('actual_unique','geometry_union')}
    invalid_branches = 0
    for call in calls:
        values = call['values']
        invalid_branches += int(((~values['valid']) & (values['termination'] != 'after_terminal')).any(-1).sum())
    gate = protocol['data_gate']
    candidate = response['geometry_union']
    signal_groups = [g for g,v in candidate['by_group'].items() if v['over_threshold_points']]
    checks = {'original_target_valid':sum(r['target_invalid_episode'] for r in episodes) <= gate['maximum_original_target_invalid_episodes'],
              'probe_target_valid':invalid_branches <= gate['maximum_probe_target_invalid_branches'],
              'all_groups_nonanchor_support':candidate['groups_with_nonanchor_support'] >= gate['minimum_groups_with_nonanchor_support'],
              'signal_groups':len(signal_groups) >= gate['minimum_signal_groups'],
              'response_fraction':candidate['group_equal_over_threshold_fraction'] is not None and candidate['group_equal_over_threshold_fraction'] >= gate['minimum_group_equal_over_threshold_fraction'],
              'full_support':candidate['full_support_calls'] >= gate['minimum_full_support_calls']}
    diagnostic_protocol = {**protocol,'models':[], 'comparisons':[['original_gru','gru_plus_exact_response'],
        ['constant_velocity','cv_plus_exact_response'],['fixed_reference_truth','action_specific_truth'],['original_gru','fixed_reference_truth']]}
    return {'response_support':response,'probe_target_invalid_branches':invalid_branches,'signal_groups':signal_groups,
            'data_checks':{k:bool(v) for k,v in checks.items()},'eligible_for_separate_fresh_training_protocol':bool(all(checks.values())),
            'decisions':{name:decision_summary([r for r in decisions if r['population']==name],diagnostic_protocol) for name in ('actual_unique','geometry_union')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--capsule',type=Path,default=ROOT.parent/'cwm_v1/baseline/capsule.zip')
    parser.add_argument('--geometry',type=Path,default=ROOT.parent/'cwm_v7/artifacts/geometry_qualification_20261010.zip')
    args = parser.parse_args()
    protocol = json.loads((ROOT/'protocol.json').read_text())
    if (any(protocol[k] for k in ('enhanced_control_enabled','new_model_trained','private_labels_model_inputs','holdout_used'))
            or protocol['target_branch_rule_overrides'] or protocol['horizon_steps'] != 8
            or sha(args.capsule) != BASELINE_SHA or sha(args.geometry) != protocol['geometry_archive_sha256']):
        raise ValueError('Frozen original/geometry/public-only local contract changed')
    check_archive(args.geometry,'ARTIFACT_MANIFEST.json',False)
    with zipfile.ZipFile(args.geometry) as archive:
        if json.loads(archive.read('qualification/summary.json'))['selected_wall_center_x_m'] != protocol['wall_center_x_m']:
            raise ValueError('Previously qualified scene distribution changed')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    base = ActualLocalTap(args.capsule,args.output/'restored',protocol['snapshot_steps'])
    sources = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
               and Path(m.__file__).resolve().is_relative_to(ROOT.parent)} | {Path(__file__).resolve(),ROOT/'protocol.json'}
    hashes = {p.relative_to(ROOT.parent).as_posix():sha(p) for p in sources}
    for source in sources:
        destination = args.output/'source'/source.relative_to(ROOT.parent)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(source.read_bytes())
    scenes = translated_records(base,protocol,protocol['wall_center_x_m'])
    if {r['layout_seed'] for r in scenes} != set(range(1988010,1988018)) or {r['episode_seed'] for r in scenes} != set(range(988010,988026)):
        raise ValueError('Fresh preassigned geometry pilot identity differs')
    (args.output/'scenes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scenes))
    from support_audit import map_actual
    calls,records,decisions,episodes = [],[],[],[]
    for scene in scenes:
        history = []
        def observer(env,observation,actions,sequence):
            history.append(base.evaluator.policy_observations(env,observation).reshape(-1).copy())
            if env.step_count not in protocol['snapshot_steps']:
                return
            padded = np.stack([history[0]]*max(0,8-len(history))+history[-8:])
            before = fingerprint(env)
            for call in base.calls:
                planner_before = hashlib.sha256(pickle.dumps(call['planner'].__dict__,protocol=5)).hexdigest()
                key = int.from_bytes(hashlib.sha256(f"{protocol['noise_seed']}/{scene['episode_index']}/{env.step_count}/{call['agent']}".encode()).digest()[:4],'little')
                row,values = inspect(base,env,call,padded,protocol,key)
                row.update(episode_index=scene['episode_index'],group=scene['mirror_group_id'],split='development_pilot',step=int(env.step_count))
                path = args.output/'calls'/f"{scene['episode_index']}_{env.step_count}_{call['agent']}.npz"
                path.parent.mkdir(exist_ok=True)
                if values is not None:
                    np.savez_compressed(path,**values)
                    row.update(arrays_path=path.relative_to(args.output).as_posix(),arrays_sha256=sha(path))
                    c = {'record':row,'values':values}
                    calls.append(c)
                    actual,_ = map_actual(values,row['agent'])
                    for name,indices in (('actual_unique',actual),('geometry_union',np.arange(len(values['proposed'])))):
                        if values['valid'][indices].all() and values['anchor_valid'].all():
                            decisions.append(scored_record(c,indices,name,call,{},protocol))
                context = path.with_suffix('.json')
                context.write_text(json.dumps(public_call_snapshot(call),indent=2))
                row.update(context_path=context.relative_to(args.output).as_posix(),context_sha256=sha(context))
                if hashlib.sha256(pickle.dumps(call['planner'].__dict__,protocol=5)).hexdigest() != planner_before:
                    raise AssertionError('Geometry pilot changed original captured planner')
                records.append(row)
            if fingerprint(env) != before:
                raise AssertionError('Geometry pilot changed baseline parent')
        observed = args.output/'observed'/f"{scene['episode_index']}.npz"
        plain = args.output/'plain'/observed.name
        row,_ = base.run(scene,observed,observer)
        original,_ = base.run(scene,plain)
        if not compare_arrays(arrays(observed.read_bytes()),arrays(plain.read_bytes())):
            raise AssertionError('Geometry probe collector changed original arrays')
        keys = ('safe_capture_success','collision','boundary_violation','timeout','target_invalid_episode','termination_reason')
        if any(row[k] != original[k] for k in keys):
            raise AssertionError('Geometry probes changed original physical outcomes')
        if bool(target_bad(arrays(observed.read_bytes())['target_positions'],scene['scenario']).any()) != row['target_invalid_episode']:
            raise AssertionError('Original target validity differs from geometry')
        episodes.append({'episode_index':scene['episode_index'],**{k:row[k] for k in keys},'trajectory_byte_equal':True,
                         'observed_sha256':sha(observed),'plain_sha256':sha(plain)})
        for name,data in (('records',records),('decisions',decisions),('episodes',episodes)):
            (args.output/(name+'.json')).write_text(json.dumps(data,indent=2))
        print(json.dumps({'episodes':len(episodes),'calls':len(calls),'records':len(records)}),flush=True)
    result = summarize(calls,decisions,episodes,protocol)
    verify(base.root,base.capsule_manifest)
    if any(sha(ROOT.parent/name) != digest for name,digest in hashes.items()):
        raise AssertionError('Run-used local geometry pilot sources changed')
    summary = {'status':'fresh_local_geometry_probe_pilot_finished_not_control_qualification','protocol':protocol,'source_hashes':hashes,
               'baseline_capsule_sha256':BASELINE_SHA,'scene_sha256':sha(args.output/'scenes.jsonl'),
               'episodes':episodes,'calls':len(calls),'records':len(records),'skipped_calls':len(records)-len(calls),**result,
               'enhanced_control_enabled':False,'new_model_trained':False,'holdout_used':False,'scope':protocol['scope']}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'status':summary['status'],'data_checks':summary['data_checks'],'eligible':summary['eligible_for_separate_fresh_training_protocol']}),flush=True)


if __name__ == '__main__':
    main()
