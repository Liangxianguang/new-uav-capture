"""Independent original/public/paired replay adds supervision-only true labels."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent / 'cwm_v13'), str(ROOT.parent / 'cwm_v10')]
from mechanism_replay import branch_trace, verify_original_fields
from local_shadow import ActualLocalTap, public_call_snapshot, fingerprint
from replay_public import compare_public
from geometry_release import arrays, compare_arrays, sha, BASELINE_SHA
from freeze_baseline import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--capsule', type=Path, default=ROOT.parent / 'cwm_v1/baseline/capsule.zip')
    args = parser.parse_args()
    protocol = json.loads((ROOT / 'data_protocol.json').read_text())
    data_report = json.loads((args.data / 'summary.json').read_text())
    if data_report['protocol'] != protocol or not data_report['training_eligible'] or sha(args.capsule) != BASELINE_SHA:
        raise ValueError('Completed fixed eligible original collection required')
    for name in ('calls', 'scenes'):
        ext = 'json' if name == 'calls' else 'jsonl'
        if sha(args.data / f'{name}.{ext}') != data_report[name + '_sha256']:
            raise ValueError('Original collection digest mismatch')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    base = ActualLocalTap(args.capsule, args.output / 'restored', protocol['snapshot_steps'])
    scenes = [json.loads(line) for line in (args.data / 'scenes.jsonl').read_text().splitlines()]
    calls = json.loads((args.data / 'calls.json').read_text())
    sources = sorted({Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, '__file__', None)
                      and Path(m.__file__).resolve().is_relative_to(ROOT.parent)} | {ROOT / 'data_protocol.json'})
    hashes = {p.relative_to(ROOT.parent).as_posix(): sha(p) for p in sources}
    for p in sources:
        dest = args.output / 'source' / p.relative_to(ROOT.parent)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(p.read_bytes())
    records, episodes = [], []
    checked_public = 0
    started = time.perf_counter()
    for scene in scenes:
        selected = [r for r in calls if r['episode_index'] == scene['episode_index']]
        history = []
        def observer(env, observation, actions, sequence):
            nonlocal checked_public
            before = fingerprint(env)
            history.append(base.evaluator.policy_observations(env, observation).reshape(-1).copy())
            padded = np.stack([history[0]] * max(0, 8-len(history)) + history[-8:])
            for call in base.calls:
                row = next(r for r in selected if r['step'] == env.step_count and r['agent'] == call['agent'])
                snapshot = json.loads((args.data / row['context_path']).read_text())
                if not compare_public(public_call_snapshot(call), snapshot):
                    raise ValueError('True-label replay changed original public local input')
                checked_public += 1
                if 'arrays_path' not in row:
                    continue
                values = arrays((args.data / row['arrays_path']).read_bytes())
                if not np.array_equal(values['history'], padded) or not np.array_equal(values['backbone'], base.last_backbone[0]):
                    raise ValueError('True-label replay changed public252 history/backbone')
                branches = [branch_trace(base, env, proposed, row['noise_key']) for proposed in values['proposed']]
                anchor = branch_trace(base, env, values['anchor'], row['noise_key'])
                repeated = branch_trace(base, env, values['anchor'], row['noise_key'])
                if not compare_arrays(anchor, repeated):
                    raise ValueError('True-label reference replay differs')
                verify_original_fields(values, branches, anchor, 1e-12)
                labels = {key: np.stack([branch[key] for branch in branches]) for key in branches[0] if key.endswith('label_only')}
                labels.update({'anchor_' + key: value for key, value in anchor.items() if key.endswith('label_only')})
                labels['initial_branch_label_only'] = np.asarray(0 if env.target_branch_sign is None else int(env.target_branch_sign))
                path = args.output / 'labels' / Path(row['arrays_path']).name
                path.parent.mkdir(exist_ok=True)
                np.savez_compressed(path, **labels)
                records.append({'episode_index': row['episode_index'], 'step': row['step'], 'agent': row['agent'], 'group': row['group'], 'split': row['split'],
                                'mechanism_path': path.relative_to(args.output).as_posix(), 'mechanism_sha256': sha(path),
                                'public_history_and_backbone_equal': True, 'original_branch_fields_replay_equal': True, 'anchor_repeat_equal': True})
            if before != fingerprint(env):
                raise ValueError('True-label replay changed original parent')
        path = args.output / 'trajectories' / f"{scene['episode_index']}.npz"
        outcome, _ = base.run(scene, path, observer)
        if not compare_arrays(arrays(path.read_bytes()), arrays((args.data / 'observed' / path.name).read_bytes())):
            raise ValueError('True-label replay changed original trajectory')
        episodes.append({'episode_index': scene['episode_index'], 'original_arrays_equal': True, 'trajectory_sha256': sha(path), 'safe_capture_success': outcome['safe_capture_success']})
        print(json.dumps({'episodes': len(episodes), 'eligible_labels': len(records), 'public_calls': checked_public}), flush=True)
    if len(records) != sum('arrays_path' in r for r in calls) or checked_public != len(calls):
        raise ValueError('True-label replay population incomplete')
    verify(base.root, base.capsule_manifest)
    if any(sha(ROOT.parent / name) != digest for name, digest in hashes.items()):
        raise ValueError('Run-used true-label sources changed')
    summary = {'status': 'original_public_and_true_label_replay_equal', 'protocol': protocol, 'source_hashes': hashes,
               'data_summary_sha256': sha(args.data / 'summary.json'), 'data_calls_sha256': sha(args.data / 'calls.json'),
               'episodes': episodes, 'eligible_calls': len(records), 'public_calls': checked_public,
               'enhanced_control_enabled': False, 'holdout_used': False, 'private_labels_model_inputs': False,
               'elapsed_seconds': time.perf_counter() - started}
    (args.output / 'records.json').write_text(json.dumps(records, indent=2))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
