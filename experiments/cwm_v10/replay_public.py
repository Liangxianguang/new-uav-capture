"""Independent original replay verifies every public history and supplied GRU."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cwm_v9"))
from local_shadow import BackboneTap, encode_public, decode_public
from geometry_release import arrays, compare_arrays, sha
from freeze_baseline import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    base = BackboneTap(args.capsule, args.output / "restored")
    scenes = [json.loads(line) for line in (args.data / "scenes.jsonl").read_text().splitlines()]
    records = json.loads((args.data / "calls.json").read_text())
    source = Path(__file__)
    sources = [source] + [ROOT.parent / p for p in ("cwm_v9/local_shadow.py", "cwm_v9/public_provider.py", "cwm_v1/baseline.py", "cwm_v1/freeze_baseline.py", "cwm_v2/scout.py")]
    source_hashes = {p.relative_to(ROOT.parent).as_posix(): sha(p) for p in sources}
    for path in sources:
        destination = args.output / "source" / path.relative_to(ROOT.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    episodes = []
    checked = 0
    for scene in scenes:
        selected = [r for r in records if r["episode_index"] == scene["episode_index"]]
        frames, history = [], []
        def observer(env, observation, actions, sequence):
            nonlocal checked
            feature = base.evaluator.policy_observations(env, observation).reshape(-1).copy()
            history.append(feature)
            frames.append(encode_public({"step": int(env.step_count), "observation": observation, "feature": feature, "backbone": base.last_backbone[0]}))
            padded = np.stack([history[0]] * max(0, 8 - len(history)) + history[-8:])
            for row in selected:
                if row["step"] != env.step_count:
                    continue
                context = decode_public(json.loads((args.data / row["context_path"]).read_text()))
                if any(not compare_public(observation[k], context["observation"][k]) for k in observation):
                    raise AssertionError("Collection's public context differs from independent original replay")
                if not np.array_equal(context["backbone"], base.last_backbone[0]):
                    raise AssertionError("Collection's GRU differs from independent original replay")
                if "arrays_path" in row:
                    values = arrays((args.data / row["arrays_path"]).read_bytes())
                    if values["history"].tobytes() != padded.tobytes() or values["backbone"].tobytes() != base.last_backbone[0].tobytes():
                        raise AssertionError("Collection's public history/backbone differs from independent original replay")
                checked += 1
        path = args.output / "trajectories" / f"{scene['episode_index']}.npz"
        result, _ = base.run(scene, path, observer)
        if not compare_arrays(arrays(path.read_bytes()), arrays((args.data / "observed" / path.name).read_bytes())):
            raise AssertionError("Independent public trace changed original arrays")
        frame_path = args.output / "frames" / f"{scene['episode_index']}.json"
        frame_path.parent.mkdir(exist_ok=True)
        frame_path.write_text(json.dumps(frames, indent=2))
        episodes.append({"episode_index": scene["episode_index"], "trajectory_sha256": sha(path), "frames_sha256": sha(frame_path),
                         "frames": len(frames), "original_arrays_byte_equal": True, "safe_capture_success": result["safe_capture_success"]})
        print(json.dumps({"episodes": len(episodes), "checked_calls": checked}), flush=True)
    if checked != len(records):
        raise AssertionError("Incomplete original-call verification")
    verify(base.root, base.capsule_manifest)
    if any(sha(ROOT.parent / n) != h for n, h in source_hashes.items()):
        raise AssertionError("Public replay source changed")
    report = {"status": "public_history_and_backbone_replay_equal", "data_summary_sha256": sha(args.data / "summary.json"),
              "capsule_sha256": sha(args.capsule), "checked_calls": checked, "episodes": episodes, "source_hashes": source_hashes,
              "enhanced_control_enabled": False, "holdout_used": False}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "checked_calls": checked}), flush=True)


def compare_public(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(compare_public(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(compare_public(x, y) for x, y in zip(a, b))
    return a == b


if __name__ == "__main__":
    main()
