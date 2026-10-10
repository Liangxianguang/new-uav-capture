"""Independent original replay and public joint-proposal verification."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from collect_scenes import (BackboneTap,scene_records,validate_protocol,s4_commands,
                           sha,BASELINE_SHA,arrays,compare_arrays,verify)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--capsule',type=Path,default=ROOT.parent/'cwm_v1/baseline/capsule.zip')
    args = parser.parse_args()
    protocol = json.loads((ROOT/'scene_protocol.json').read_text())
    validate_protocol(protocol)
    if sha(args.capsule) != BASELINE_SHA:
        raise ValueError('Protected original baseline mismatch')
    data = json.loads((args.data/'summary.json').read_text())
    if data['protocol'] != protocol or data['enhanced_control_enabled'] or data['new_model_trained']:
        raise ValueError('Actual finite diagnostic contract mismatch')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    base = BackboneTap(args.capsule,args.output/'restored')
    scenes = [json.loads(line) for line in (args.data/'scenes.jsonl').read_text().splitlines()]
    if scene_records(base,protocol) != scenes:
        raise ValueError('Independent scene geometry/route regeneration differs')
    sources = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
               and Path(m.__file__).resolve().is_relative_to(ROOT.parent)} | {Path(__file__).resolve(),ROOT/'scene_protocol.json'}
    hashes = {p.relative_to(ROOT.parent).as_posix():sha(p) for p in sources}
    for source in sources:
        destination = args.output/'source'/source.relative_to(ROOT.parent)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(source.read_bytes())
    windows = json.loads((args.data/'windows.json').read_text())
    original_rows = {r['episode_index']:r for r in json.loads((args.data/'episodes.json').read_text())}
    episodes,checked = [],[]
    for scene in scenes:
        if not scene['route_valid']:
            continue
        index = scene['episode_index']
        selected = {r['step']:r for r in windows if r['episode_index'] == index}
        seen = set()
        def observer(env,observation,actions,sequence):
            if env.step_count not in selected:
                return
            record = selected[env.step_count]
            path = args.data/record['arrays_path']
            if sha(path) != record['arrays_sha256']:
                raise ValueError('Actual new-scene probe arrays changed')
            values = arrays(path.read_bytes())
            reference,_ = base.evaluator.belief_reference(observation,base.planner_config)
            proposed = s4_commands(sequence,observation,reference,protocol)
            if not np.array_equal(proposed,values['proposed']) or not np.array_equal(reference,values['public_reference']):
                raise ValueError('Independent public joint probes differ')
            seen.add(int(env.step_count))
            checked.append({'episode_index':index,'step':int(env.step_count),'public_probes_byte_equal':True})
        path = args.output/'trajectories'/f'{index}.npz'
        controller_record = {**scene,'episode_index':scene['controller_sampling_identity']}
        result,_ = base.run(controller_record,path,observer)
        if seen != set(selected):
            raise ValueError('Incomplete public new-scene proposal verification')
        if not compare_arrays(arrays(path.read_bytes()),arrays((args.data/'observed'/path.name).read_bytes())):
            raise ValueError('Independent new-scene original trajectory differs')
        expected = original_rows[index]
        keys = ('safe_capture_success','collision','boundary_violation','timeout','target_invalid_episode','termination_reason')
        if any(result[k] != expected[k] for k in keys):
            raise ValueError('Independent original physical outcome differs')
        episodes.append({'episode_index':index,**{k:result[k] for k in keys},'trajectory_byte_equal':True,'trajectory_sha256':sha(path)})
        print(json.dumps({'episodes':len(episodes),'checked_windows':len(checked)}),flush=True)
    if len(checked) != len(windows):
        raise ValueError('Incomplete finite public probe population')
    verify(base.root,base.capsule_manifest)
    if any(sha(ROOT.parent/n) != h for n,h in hashes.items()):
        raise ValueError('Run-used independent replay sources changed')
    summary = {'status':'independent_original_and_public_joint_probes_equal','source_hashes':hashes,
               'protocol':protocol,'baseline_capsule_sha256':BASELINE_SHA,'data_summary_sha256':sha(args.data/'summary.json'),
               'episodes':episodes,'checked_windows':checked,'enhanced_control_enabled':False,'new_model_trained':False,'holdout_used':False}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'status':summary['status'],'episodes':len(episodes),'checked_windows':len(checked)}),flush=True)


if __name__ == '__main__':
    main()
