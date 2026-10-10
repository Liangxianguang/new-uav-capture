"""Fresh V23 split-preassigned geometry shadow data; original control unchanged."""
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
sys.path[:0] = [str(ROOT), str(ROOT.parent/'cwm_v18')]
from collect_probe_pilot import (inspect, summarize, ActualLocalTap, translated_records,
    fingerprint, public_call_snapshot, arrays, compare_arrays, target_bad, sha,
    BASELINE_SHA, verify, check_archive, scored_record)
from support_audit import map_actual
from release_verify import verify_archive


def validate_protocol(protocol):
    expected = json.loads((ROOT/'data_protocol.json').read_text())
    if protocol != expected:
        raise ValueError('Published fresh cost-contrast data protocol changed')
    if any(protocol[k] for k in ('enhanced_control_enabled', 'new_model_trained',
                                 'private_labels_model_inputs', 'holdout_used')):
        raise ValueError('Frozen public-only original data contract violated')
    if (protocol['groups'], protocol['train_groups'], protocol['development_groups']) != (32,24,8):
        raise ValueError('Fixed mirror-group population changed')
    if protocol['seed_start'] != 993010 or protocol['layout_seed_start'] != 1993010:
        raise ValueError('Fresh data identity changed')
    if protocol['horizon_steps'] != 8 or protocol['target_branch_rule_overrides']:
        raise ValueError('Frozen target/horizon contract changed')
    reserved = protocol['reserved_holdout_not_collected']
    episodes = set(range(protocol['seed_start'], protocol['seed_start']+2*protocol['groups']))
    layouts = set(range(protocol['layout_seed_start'], protocol['layout_seed_start']+protocol['groups']))
    if episodes & set(range(reserved['seed_start'], reserved['seed_start']+2*reserved['groups'])) or layouts & set(range(reserved['layout_seed_start'], reserved['layout_seed_start']+reserved['groups'])):
        raise ValueError('Reserved holdout overlap')
    for ranges, actual in ((protocol['prior_episodes_not_reused'], episodes), (protocol['prior_layouts_not_reused'], layouts)):
        if any(actual & set(range(a,b+1)) for a,b in ranges):
            raise ValueError('Previously inspected data overlap')


def assigned_split(group_number, protocol):
    if not 0 <= group_number < protocol['groups']:
        raise ValueError('Group index outside preassigned population')
    return 'train' if group_number < protocol['train_groups'] else 'development_validation'


def assign_scenes(scenes, protocol):
    for i, scene in enumerate(scenes):
        scene['model_split'] = assigned_split(i//2, protocol)
    return scenes


def split_statistics(calls, decisions, episodes, protocol):
    overall = summarize(calls, decisions, episodes, protocol)
    split_results = {}
    gate = protocol['split_data_gate']
    for split, short in (('train','train'), ('development_validation','development')):
        selected = [c for c in calls if c['record']['split'] == split]
        split_protocol = {**protocol, 'data_gate':{**protocol['data_gate'],
            'minimum_groups_with_nonanchor_support': gate['minimum_groups_with_nonanchor_support_'+short],
            'minimum_signal_groups':gate['minimum_signal_groups_each_split'],
            'minimum_group_equal_over_threshold_fraction':gate['minimum_group_equal_over_threshold_fraction_each_split'],
            'minimum_full_support_calls':gate['minimum_full_support_'+short+'_calls']}}
        result = summarize(selected, [r for r in decisions if r['split'] == split],
                           [r for r in episodes if r['split'] == split], split_protocol)
        result['data_checks']['minimum_common_valid_nonanchor_points'] = (
            result['response_support']['geometry_union']['valid_nonanchor_points'] >=
            gate['minimum_common_valid_nonanchor_points_each_split'])
        result['eligible_for_separate_fresh_training_protocol'] = bool(all(result['data_checks'].values()))
        split_results[split] = result
    return {'overall':overall, 'splits':split_results,
            'training_eligible':bool(overall['eligible_for_separate_fresh_training_protocol'] and
                all(s['eligible_for_separate_fresh_training_protocol'] for s in split_results.values()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT/'data_protocol.json').read_text())
    validate_protocol(protocol)
    capsule = ROOT.parent/'cwm_v1/baseline/capsule.zip'
    geometry = ROOT.parent/'cwm_v7/artifacts/geometry_qualification_20261010.zip'
    pilot = ROOT.parent/'cwm_v18/artifacts/public_local_geometry_pilot_audited_20261010.zip'
    for path, key in ((capsule,'baseline_capsule_sha256'),(geometry,'geometry_archive_sha256'),(pilot,'pilot_archive_sha256')):
        if sha(path) != protocol[key]:
            raise ValueError('Pinned original/qualification/pilot changed')
    verify_archive(pilot)
    check_archive(geometry,'ARTIFACT_MANIFEST.json',False)
    with zipfile.ZipFile(pilot) as archive:
        if not json.loads(archive.read('release_summary.json'))['pilot_runs'][0]['eligible_for_separate_fresh_training_protocol']:
            raise ValueError('V18 data pilot not passed')
        library = archive.read('pilot_first/source/cwm_v18/geometry_probes.py')
        if hashlib.sha256(library).hexdigest() != sha(ROOT.parent/'cwm_v18/geometry_probes.py'):
            raise ValueError('V18 frozen public geometry library changed')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    base = ActualLocalTap(capsule, args.output/'restored', protocol['snapshot_steps'])
    sources = sorted({Path(m.__file__).resolve() for m in list(sys.modules.values())
                      if getattr(m,'__file__',None) and Path(m.__file__).resolve().is_relative_to(ROOT.parent)} |
                     {Path(__file__).resolve(), ROOT/'data_protocol.json'})
    hashes = {p.relative_to(ROOT.parent).as_posix():sha(p) for p in sources}
    for source in sources:
        destination = args.output/'source'/source.relative_to(ROOT.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    scenes = assign_scenes(translated_records(base, protocol, protocol['wall_center_x_m']), protocol)
    if {s['layout_seed'] for s in scenes} != set(range(1993010,1993042)) or {s['episode_seed'] for s in scenes} != set(range(993010,993074)):
        raise ValueError('Fresh scene identity mismatch')
    (args.output/'scenes.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in scenes))
    calls, records, decisions, episodes = [], [], [], []
    for scene in scenes:
        history = []
        def observer(env, observation, actions, sequence):
            history.append(base.evaluator.policy_observations(env, observation).reshape(-1).copy())
            if env.step_count not in protocol['snapshot_steps']:
                return
            padded = np.stack([history[0]]*max(0,8-len(history))+history[-8:])
            before = fingerprint(env)
            for call in base.calls:
                planner_before = hashlib.sha256(pickle.dumps(call['planner'].__dict__, protocol=5)).hexdigest()
                key = int.from_bytes(hashlib.sha256(
                    f"{protocol['noise_seed']}/{scene['episode_index']}/{env.step_count}/{call['agent']}".encode()).digest()[:4],'little')
                row, values = inspect(base, env, call, padded, protocol, key)
                row.update(episode_index=scene['episode_index'], group=scene['mirror_group_id'],
                           split=scene['model_split'], step=int(env.step_count))
                path = args.output/'calls'/f"{scene['episode_index']}_{env.step_count}_{call['agent']}.npz"
                path.parent.mkdir(exist_ok=True)
                if values is not None:
                    np.savez_compressed(path, **values)
                    row.update(arrays_path=path.relative_to(args.output).as_posix(), arrays_sha256=sha(path))
                    current = {'record':row, 'values':values}
                    calls.append(current)
                    actual, _ = map_actual(values, row['agent'])
                    for name, indices in (('actual_unique',actual),('geometry_union',np.arange(len(values['proposed'])))):
                        if values['valid'][indices].all() and values['anchor_valid'].all():
                            decisions.append(scored_record(current, indices, name, call, {}, protocol))
                context = path.with_suffix('.json')
                context.write_text(json.dumps(public_call_snapshot(call), indent=2))
                row.update(context_path=context.relative_to(args.output).as_posix(), context_sha256=sha(context))
                if hashlib.sha256(pickle.dumps(call['planner'].__dict__, protocol=5)).hexdigest() != planner_before:
                    raise ValueError('Shadow data changed original captured planner')
                records.append(row)
            if fingerprint(env) != before:
                raise ValueError('Shadow data changed original parent')
        observed = args.output/'observed'/f"{scene['episode_index']}.npz"
        plain = args.output/'plain'/observed.name
        row, _ = base.run(scene, observed, observer)
        original, _ = base.run(scene, plain)
        if not compare_arrays(arrays(observed.read_bytes()), arrays(plain.read_bytes())):
            raise ValueError('Shadow data changed original trajectory arrays')
        keys = ('safe_capture_success','collision','boundary_violation','timeout','target_invalid_episode','termination_reason')
        if any(row[k] != original[k] for k in keys):
            raise ValueError('Shadow data changed original physical outcomes')
        if bool(target_bad(arrays(observed.read_bytes())['target_positions'],scene['scenario']).any()) != row['target_invalid_episode']:
            raise ValueError('Original target validity differs from geometry')
        episodes.append({'episode_index':scene['episode_index'], 'split':scene['model_split'],
                         **{k:row[k] for k in keys}, 'trajectory_byte_equal':True,
                         'observed_sha256':sha(observed),'plain_sha256':sha(plain)})
        for name, data in (('records',records),('decisions',decisions),('episodes',episodes)):
            (args.output/(name+'.json')).write_text(json.dumps(data, indent=2))
        print(json.dumps({'episodes':len(episodes), 'split':scene['model_split'],
                          'calls':len(calls),'records':len(records)}), flush=True)
    result = split_statistics(calls, decisions, episodes, protocol)
    verify(base.root, base.capsule_manifest)
    if any(sha(ROOT.parent/name) != digest for name,digest in hashes.items()):
        raise ValueError('Run-used collection sources changed')
    summary = {'status':'fresh_cost_contrast_data_finished_not_promoted', 'protocol':protocol,
               'source_hashes':hashes,'capsule_sha256':BASELINE_SHA,'scene_sha256':sha(args.output/'scenes.jsonl'),
               'records_sha256':sha(args.output/'records.json'),'decisions_sha256':sha(args.output/'decisions.json'),
               'episodes':episodes,'calls':len(calls),'records':len(records),'skipped_calls':len(records)-len(calls),
               **result,'enhanced_control_enabled':False,'new_model_trained':False,'holdout_used':False}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'status':summary['status'],'training_eligible':result['training_eligible']}), flush=True)


if __name__ == '__main__':
    main()
