"""Independent paired-collection and original public-replay archive audit."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from collect_scenes import validate_protocol,variant_summary,arrays,compare_arrays,target_bad,sha,BASELINE_SHA
from verify_release import check_archive


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def validate_rollout(values,scene):
    expected = {'target':(8,80,3),'defenders':(8,80,4,3),'commanded':(8,80,4,3),
                'executed':(8,80,4,3),'proposed':(8,80,4,3),'valid':(8,80),
                'branch_sign_label_only':(8,80),'termination':(8,80),'public_reference':(3,)}
    if set(values) != set(expected) or any(values[k].shape != v for k,v in expected.items()):
        raise ValueError('Actual eight-probe finite horizon shape contract mismatch')
    if values['valid'].dtype != bool or not np.isin(values['branch_sign_label_only'],[-1,0,1]).all():
        raise ValueError('Validity/branch label contract mismatch')
    for key in ('target','defenders','commanded','executed','proposed','public_reference'):
        if not np.isfinite(values[key]).all():
            raise ValueError('Nonfinite actual scene branch values')
    for probe in range(8):
        observed = values['termination'][probe] != 'after_terminal'
        n = int(observed.sum())
        if n == 0 or not observed[:n].all() or observed[n:].any() or values['valid'][probe,n:].any():
            raise ValueError('Noncontiguous observed branch or valid terminal padding')
        if n < 80 and values['termination'][probe,n-1] == 'running':
            raise ValueError('Truncated branch has no terminal reason')
        for key in ('target','defenders','commanded','executed','branch_sign_label_only'):
            if np.any(values[key][probe,n:] != 0):
                raise ValueError('Terminal padding contains imputed future')
        physical_bad = target_bad(values['target'][probe,:n],scene['scenario'])
        if not np.array_equal(physical_bad,~values['valid'][probe,:n]):
            raise ValueError('Actual target geometry differs from branch validity mask')


def audit(read):
    protocol = json.loads((ROOT/'scene_protocol.json').read_text())
    validate_protocol(protocol)
    data = json.loads(read('primary/summary.json'))
    repeated = json.loads(read('repeated/summary.json'))
    public = json.loads(read('public/summary.json'))
    if (data['status'] != 'finite_scene_diagnostic_finished_not_model_qualification'
            or repeated['status'] != data['status']
            or public['status'] != 'independent_original_and_public_joint_probes_equal'):
        raise ValueError('Diagnostic/public replay status mismatch')
    for stage,report in (('primary',data),('repeated',repeated),('public',public)):
        if (report['protocol'] != protocol or report['baseline_capsule_sha256'] != BASELINE_SHA
                or any(report[k] for k in ('enhanced_control_enabled','new_model_trained','holdout_used'))):
            raise ValueError('Frozen original/new-scene contract mismatch')
        for name,d in report['source_hashes'].items():
            if digest(read(stage+'/source/'+name)) != d or sha(ROOT.parent/name) != d:
                raise ValueError('Run-used independent scene source mismatch')
        if json.loads(read(stage+'/source/cwm_v16/scene_protocol.json')) != protocol:
            raise ValueError('Saved preregistered scene protocol mismatch')
    for key in data.keys()-{'elapsed_seconds'}:
        if data[key] != repeated[key]:
            raise ValueError('Independent paired-collection summaries differ')
    for name in ('scenes.jsonl','episodes.json','windows.json','skips.json'):
        if read('primary/'+name) != read('repeated/'+name):
            raise ValueError('Independent assigned scenes/probe records differ')
    if data['scene_sha256'] != digest(read('primary/scenes.jsonl')) or public['data_summary_sha256'] != digest(read('primary/summary.json')):
        raise ValueError('Original public replay/data provenance mismatch')
    scenes = [json.loads(line) for line in read('primary/scenes.jsonl').decode().splitlines()]
    scene_map = {r['episode_index']:r for r in scenes}
    expected_ids = {v['seed_start']+i for v in protocol['variants'] for i in range(protocol['groups']*2)}
    if len(scenes) != 64 or set(scene_map) != expected_ids:
        raise ValueError('Preassigned new-scene population mismatch')
    expected_layouts = set(range(protocol['layout_seed_start'],protocol['layout_seed_start']+8))
    if {r['layout_seed'] for r in scenes} != expected_layouts:
        raise ValueError('Actual diagnostic layouts changed')
    episodes = json.loads(read('primary/episodes.json'))
    rows = {r['episode_index']:r for r in episodes}
    valid_ids = {i for i,s in scene_map.items() if s['route_valid']}
    if set(rows) != valid_ids or len(rows) != len(episodes) or data['episodes'] != len(episodes) or data['assigned_scenes'] != 64:
        raise ValueError('Route failures were dropped/replaced or original replay missing')
    replay_rows = {r['episode_index']:r for r in public['episodes']}
    if set(replay_rows) != valid_ids or len(replay_rows) != len(public['episodes']):
        raise ValueError('Independent original trajectory population differs')
    windows = json.loads(read('primary/windows.json'))
    skips = json.loads(read('primary/skips.json'))
    identity = lambda r:(r['episode_index'],r['step'])
    identities = [identity(r) for r in windows]
    if len(set(identities)) != len(identities) or data['windows'] != len(windows) or data['skips'] != len(skips):
        raise ValueError('Duplicate/missing finite probe population')
    expected_snapshots = {(i,s) for i in valid_ids for s in protocol['snapshot_steps']}
    skip_ids = [identity(r) for r in skips]
    if (len(set(skip_ids)) != len(skip_ids) or set(identities)&set(skip_ids)
            or set(identities)|set(skip_ids) != expected_snapshots):
        raise ValueError('Fixed snapshots or terminated-prefix skips changed')
    checked = public['checked_windows']
    if {identity(r) for r in checked} != set(identities) or len(checked) != len(windows) or not all(r['public_probes_byte_equal'] for r in checked):
        raise ValueError('Original public joint proposals not fully verified')
    for scene in scenes:
        variant = next(v for v in protocol['variants'] if v['name'] == scene['variant'])
        offset = scene['episode_index']-variant['seed_start']
        if not 0 <= offset < 16:
            raise ValueError('Actual variant identity mismatch')
        reference_id = protocol['variants'][0]['seed_start']+offset
        if (scene['controller_sampling_identity'] != reference_id or scene['episode_seed'] != reference_id
                or scene['layout_seed'] != protocol['layout_seed_start']+offset//2 or scene['geometry_axis'] != variant):
            raise ValueError('Matched original stochastic input identity differs')
        index = scene['episode_index']
        if index not in valid_ids:
            continue
        observed = arrays(read(f'primary/observed/{index}.npz'))
        for stage,folder in (('primary','plain'),('repeated','observed'),('repeated','plain'),('public','trajectories')):
            if not compare_arrays(observed,arrays(read(f'{stage}/{folder}/{index}.npz'))):
                raise ValueError('Actual independent original trajectories differ')
        for key,folder in (('observed_sha256','observed'),('plain_sha256','plain')):
            if rows[index][key] != digest(read(f'primary/{folder}/{index}.npz')):
                raise ValueError('Actual original trajectory digest mismatch')
        replay = replay_rows[index]
        if replay['trajectory_sha256'] != digest(read(f'public/trajectories/{index}.npz')) or not replay['trajectory_byte_equal']:
            raise ValueError('Actual independent original replay digest mismatch')
        if not rows[index]['trajectory_byte_equal'] or bool(target_bad(observed['target_positions'],scene['scenario']).any()) != rows[index]['target_invalid_episode']:
            raise ValueError('Actual original target validity mismatch')
        for key in ('safe_capture_success','collision','boundary_violation','timeout','target_invalid_episode','termination_reason'):
            if replay[key] != rows[index][key]:
                raise ValueError('Independent original physical outcome differs')
        last_step = len(observed['target_positions'])-1
        if any(r['step'] < last_step for r in skips if r['episode_index'] == index):
            raise ValueError('Skipped live fixed snapshot')
    for row in windows:
        raw = read('primary/'+row['arrays_path'])
        if digest(raw) != row['arrays_sha256'] or not compare_arrays(arrays(raw),arrays(read('repeated/'+row['arrays_path']))):
            raise ValueError('Independent counterfactual branch arrays differ')
        scene = scene_map[row['episode_index']]
        if row['group'] != scene['mirror_group_id'] or row['variant'] != scene['variant'] or not row['parent_unchanged']:
            raise ValueError('Counterfactual source group/parent differs')
        key = int.from_bytes(hashlib.sha256(f"{protocol['noise_seed']}/{scene['controller_sampling_identity']}/{row['step']}".encode()).digest()[:4],'little')
        if row['noise_key'] != key:
            raise ValueError('Shared exogenous noise identity differs')
        validate_rollout(arrays(raw),scene)
    calculated = [variant_summary(v['name'],scenes,episodes,windows,lambda n:read('primary/'+n),protocol) for v in protocol['variants']]
    if calculated != data['variants']:
        raise ValueError('Independent finite geometry/response gate differs')
    return {'status':'passed_independent_two_collection_and_public_replay_audit',
            'assigned_scenes':64,'completed_episodes':len(episodes),'windows':len(windows),
            'counterfactual_arrays_equal':True,'original_trajectories_byte_equal':True,
            'public_joint_proposals_replayed':True,'target_geometry_and_padding_verified':True,
            'variants':calculated,'enhanced_control_enabled':False,'new_model_trained':False,'holdout_used':False,
            'scope':protocol['scope']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in ('primary','repeated','public'):
        parser.add_argument('--'+stage,type=Path)
    parser.add_argument('--verify',type=Path)
    args = parser.parse_args()
    if args.verify:
        integrity = check_archive(args.verify,'ARTIFACT_MANIFEST.json',False)
        with zipfile.ZipFile(args.verify) as archive:
            result = audit(archive.read)
        print(json.dumps({**result,'artifact_sha256':integrity['sha256']}),flush=True)
        return
    members = {}
    for stage in ('primary','repeated','public'):
        run = getattr(args,stage)
        if run is None:
            parser.error('Two finished collections and independent original replay required')
        for file in run.rglob('*'):
            relative = file.relative_to(run)
            if file.is_file() and 'restored' not in relative.parts:
                members[stage+'/'+relative.as_posix()] = file.read_bytes()
    for file in ROOT.glob('*.py'):
        members['verification_source/'+file.name] = file.read_bytes()
    result = audit(members.__getitem__)
    path = ROOT/'artifacts/new_scene_diagnostic_20261010.zip'
    path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,raw in members.items():
            archive.writestr(name,raw)
        archive.writestr('ARTIFACT_MANIFEST.json',json.dumps({n:digest(raw) for n,raw in members.items()},indent=2))
    integrity = check_archive(path,'ARTIFACT_MANIFEST.json',False)
    with zipfile.ZipFile(path) as archive:
        audit(archive.read)
    reports = ROOT/'reports'
    reports.mkdir(exist_ok=True)
    for stage in ('primary','repeated','public'):
        with (reports/(stage+'_summary.json')).open('xb') as file:
            file.write((getattr(args,stage)/'summary.json').read_bytes())
    with (reports/'release_manifest.json').open('x') as file:
        json.dump({**result,'artifact_sha256':integrity['sha256'],'artifact_bytes':path.stat().st_size},file,indent=2)
    print(json.dumps({**result,'artifact_sha256':integrity['sha256'],'artifact_bytes':path.stat().st_size}),flush=True)


if __name__ == '__main__':
    main()
