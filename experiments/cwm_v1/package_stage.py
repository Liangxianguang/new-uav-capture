"""Publish compact verified milestone artifacts, never active training logs."""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from freeze_baseline import REFERENCE, sha


def require_pass(path: Path):
    value = json.loads(path.read_text())
    if value["status"] != "passed":
        raise ValueError(f"Required audit has not passed: {path}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("baseline", "pilot"), required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--shadow", type=Path)
    args = parser.parse_args()
    root = Path(__file__).parent
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    summary = require_pass(args.replay / "summary.json")
    if summary["completed"] != 256:
        raise ValueError("Full historical baseline replay required")
    # np.array_equal is numerical equality; separately audit byte equality,
    # including dtype and signed zero, before reporting bitwise preservation.
    for row in summary["comparisons"]:
        name = f"level{row['level']}_{row['episode_index']}.npz"
        with np.load(args.replay / "restored" / REFERENCE / "trajectories" / name) as original, np.load(args.replay / "trajectories" / name) as actual:
            for key in ("defender_positions", "target_positions"):
                a, b = original[key], actual[key]
                if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                    raise ValueError("Byte-level reference mismatch")
    shutil.copy2(args.replay / "summary.json", reports / "baseline_replay_all.json")
    release = {"stage": args.stage, "baseline_episodes_verified": 256, "baseline_byte_equality_verified": True,
               "baseline_capsule_sha256": sha(root / "baseline/capsule.zip"), "enhanced_control_enabled": False}
    if args.stage == "pilot":
        if args.collection is None or args.training is None or args.shadow is None:
            raise ValueError("Pilot packaging requires data, training and shadow audit")
        reload = require_pass(args.training / "reload_verification.json")
        shadow = require_pass(args.shadow / "summary.json")
        dataset = json.loads((args.collection / "manifest.json").read_text())
        training = json.loads((args.training / "summary.json").read_text())
        if sha(args.collection / "pairs.npz") != training["dataset_sha256"]:
            raise ValueError("Pilot dataset integrity failure")
        for row in training["results"]:
            checkpoint = args.training / f"{row['kind']}_seed{row['seed']}.pt"
            if sha(checkpoint) != row["checkpoint_sha256"]:
                raise ValueError("Pilot checkpoint hash mismatch")
        members = {}
        for name in ("pairs.npz", "selected_scenes.jsonl", "manifest.json"):
            members[f"dataset/{name}"] = args.collection / name
        for path in args.training.iterdir():
            if path.suffix in (".json", ".pt", ".npz"):
                members[f"training/{path.name}"] = path
        for path in root.glob("*.py"):
            members[f"source/{path.name}"] = path
        members["source/protocol.json"] = root / "protocol.json"
        destination = root / "artifacts/pilot_20261009.zip"
        destination.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in sorted(members.items()):
                archive.write(path, name)
            archive.writestr("ARTIFACT_MANIFEST.json", json.dumps({name: sha(path) for name, path in members.items()}, indent=2))
        for name, path in [("pilot_training_summary.json", args.training / "summary.json"),
                           ("pilot_reload_verification.json", args.training / "reload_verification.json"),
                           ("shadow_invariance.json", args.shadow / "summary.json")]:
            shutil.copy2(path, reports / name)
        release.update(status="pilot_no_go_keep_baseline", trained_models=len(training["results"]),
                       states=dataset["states"], episodes=dataset["episodes"],
                       artifact_sha256=sha(destination), artifact_bytes=destination.stat().st_size,
                       source_hashes={path.name: sha(path) for path in root.glob("*.py")},
                       reason="Both learned pilots are worse than constant velocity; paired intervention effects are too small to support a causal benefit claim.")
    else:
        release["status"] = "baseline_preservation_pass"
    destination = reports / f"{args.stage}_release_manifest.json"
    if destination.exists():
        raise FileExistsError("Do not overwrite a milestone manifest")
    destination.write_text(json.dumps(release, indent=2))
    print(json.dumps(release))


if __name__ == "__main__":
    main()
