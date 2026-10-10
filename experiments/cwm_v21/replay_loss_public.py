"""Original independent public252/GRU replay, retaining all public frames."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT),str(ROOT.parent/'cwm_v18'),str(ROOT.parent/'cwm_v10')]
from fresh_geometry_data import validate_protocol
from replay_probe_public import (BackboneTap, decode_public, restore_public_call, public_geometry_pool,
    compare_public, arrays, compare_arrays, sha, BASELINE_SHA, verify)
from local_shadow import encode_public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT/'data_protocol.json').read_text())
    validate_protocol(protocol)
    data = json.loads((args.data/'summary.json').read_text())
    capsule = ROOT.parent/'cwm_v1/baseline/capsule.zip'
    if data['protocol'] != protocol or data['enhanced_control_enabled'] or data['new_model_trained'] or sha(capsule) != BASELINE_SHA:
        raise ValueError('Frozen original/public replay contract changed')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    base = BackboneTap(capsule,args.output/'restored')
    configs = (base.planner_config,base.distributed_config)
    scenes = [json.loads(line) for line in (args.data/'scenes.jsonl').read_text().splitlines()]
    records = json.loads((args.data/'records.json').read_text())
    sources = sorted({Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None)
                      and Path(m.__file__).resolve().is_relative_to(ROOT.parent)} |
                     {Path(__file__).resolve(),ROOT/'data_protocol.json'})
    hashes = {p.relative_to(ROOT.parent).as_posix():sha(p) for p in sources}
    for source in sources:
        dest = args.output/'source'/source.relative_to(ROOT.parent)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(source.read_bytes())
    episodes, checked = [], []
    for scene in scenes:
        history, frames = [], []
        selected = [r for r in records if r['episode_index'] == scene['episode_index']]
        def observer(env,observation,actions,sequence):
            feature = base.evaluator.policy_observations(env,observation).reshape(-1).copy()
            history.append(feature)
            frames.append(encode_public({'step':int(env.step_count),'observation':observation,
                                         'feature':feature,'backbone':base.last_backbone[0]}))
            padded = np.stack([history[0]]*max(0,8-len(history))+history[-8:])
            for row in selected:
                if row['step'] != env.step_count:
                    continue
                path = args.data/row['context_path']
                if sha(path) != row['context_sha256']:
                    raise ValueError('Public original context digest changed')
                snapshot = json.loads(path.read_text())
                context = decode_public(snapshot)
                if any(not compare_public(observation[k],context['observation'][k]) for k in observation):
                    raise ValueError('Independent public original observation differs')
                if not np.array_equal(context['backbone'],base.last_backbone[0]):
                    raise ValueError('Independent original GRU differs')
                if 'arrays_path' in row:
                    path = args.data/row['arrays_path']
                    if sha(path) != row['arrays_sha256']:
                        raise ValueError('Saved local array digest changed')
                    values = arrays(path.read_bytes())
                    if not np.array_equal(values['history'],padded) or not np.array_equal(values['backbone'],base.last_backbone[0]):
                        raise ValueError('Independent public252 history/backbone differs')
                    call = restore_public_call(snapshot,*configs,values['backbone'])
                    reference,velocity = base.evaluator.belief_reference(observation,base.planner_config)
                    pool = public_geometry_pool(call,reference,velocity,protocol['probe_magnitude_mps'])
                    if not np.array_equal(pool,values['proposed'][:,:,row['agent']]):
                        raise ValueError('Independent public geometry candidates differ')
                checked.append({'episode_index':row['episode_index'],'step':row['step'],'agent':row['agent'],
                                'public_context_equal':True,'eligible_arrays':'arrays_path' in row})
        path = args.output/'trajectories'/f"{scene['episode_index']}.npz"
        result,_ = base.run(scene,path,observer)
        if not compare_arrays(arrays(path.read_bytes()),arrays((args.data/'observed'/path.name).read_bytes())):
            raise ValueError('Independent original trajectory differs')
        frame_path = args.output/'frames'/f"{scene['episode_index']}.json"
        frame_path.parent.mkdir(exist_ok=True)
        frame_path.write_text(json.dumps(frames,indent=2))
        episodes.append({'episode_index':scene['episode_index'],'trajectory_byte_equal':True,
                         'trajectory_sha256':sha(path),'frames_sha256':sha(frame_path),'frames':len(frames),
                         'safe_capture_success':result['safe_capture_success']})
        print(json.dumps({'episodes':len(episodes),'checked_calls':len(checked)}),flush=True)
    if len(checked) != len(records):
        raise ValueError('Incomplete independent actual local public replay')
    verify(base.root,base.capsule_manifest)
    if any(sha(ROOT.parent/n) != d for n,d in hashes.items()):
        raise ValueError('Run-used independent public replay sources changed')
    summary = {'status':'independent_original_public_geometry_replay_frames_equal','protocol':protocol,
               'source_hashes':hashes,'data_summary_sha256':sha(args.data/'summary.json'),
               'capsule_sha256':BASELINE_SHA,'episodes':episodes,'checked_calls':checked,
               'enhanced_control_enabled':False,'new_model_trained':False,'holdout_used':False}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'status':summary['status'],'episodes':len(episodes),'calls':len(checked)}),flush=True)


if __name__ == '__main__':
    main()
