"""Recompute ranking metrics and verify two diagnostic replays against V4."""
import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np

from decision_value import ROOT, METHODS, rank_metrics, sha, summarize
from verify_release import check_archive


def audit(read, reference):
    primary = json.loads(read("primary/summary.json"))
    repeated = json.loads(read("repeated/summary.json"))
    protocol = json.loads(read("source/cwm_v5/protocol.json"))
    for stage, summary in (("primary", primary), ("repeated", repeated)):
        if summary["status"] != "offline_diagnostic_complete_not_promoted" or summary["enhanced_control_enabled"] or summary["protocol"] != protocol:
            raise ValueError("Disabled-control protocol mismatch")
        if summary["source_archive_sha256"] != protocol["source_archive_sha256"]:
            raise ValueError("Source experiment mismatch")
        for name, expected in summary["source_hashes"].items():
            if hashlib.sha256(read("source/cwm_v5/" + name)).hexdigest() != expected:
                raise ValueError("Diagnostic source differs from recorded bytes")
        records = json.loads(read(f"{stage}/windows.json"))
        windows = json.loads(reference.read("run/windows.json"))
        expected = {(r["episode_index"], r["step"]): i for i, r in enumerate(windows)}
        actual = {(r["episode_index"], r["step"]): r for r in records}
        if set(actual) != set(expected) or len(actual) != len(records):
            raise ValueError("Actual window coverage mismatch")
        with np.load(io.BytesIO(reference.read("run/pairs.npz")), allow_pickle=False) as data:
            masks = data["valid"]
        for key, record in actual.items():
            original_window = windows[expected[key]]
            if record["group"] != original_window["group"] or not record["parent_integrity"]:
                raise ValueError("Window provenance or isolation mismatch")
            valid = bool(masks[expected[key], :, :8].all())
            if record["all_candidates_full_horizon_valid"] != valid:
                raise ValueError("Actual terminal support mismatch")
            if valid:
                costs = record["costs"]
                if set(costs) != set(METHODS) or not record["reference_score_matches_original_diagnostics"]:
                    raise ValueError("Incomplete oracle comparison")
                for method in METHODS:
                    if np.asarray(costs[method]).shape != (13,):
                        raise ValueError("Incomplete probe population")
                    if rank_metrics(costs[method], costs["action_specific_truth"], protocol["tie_absolute_cost_tolerance"]) != record["metrics"][method]:
                        raise ValueError("Rank metrics differ from actual cost arrays")
            elif "metrics" in record or "costs" in record:
                raise ValueError("Censored window was scored")
        if summarize(records, protocol) != summary["statistics"]:
            raise ValueError("Aggregate ranking statistics mismatch")
        scenes = [json.loads(line) for line in reference.read("run/selected_scenes.jsonl").decode().splitlines()]
        if {r["episode_index"] for r in summary["episodes"]} != {r["episode_index"] for r in scenes} or len(summary["episodes"]) != len(scenes):
            raise ValueError("Replay episode population mismatch")
        old_outcomes = {r["episode_index"]: r for r in json.loads(reference.read("run/episodes.json"))}
        for row in summary["episodes"]:
            raw = read(f"{stage}/trajectories/{row['episode_index']}.npz")
            if hashlib.sha256(raw).hexdigest() != row["trajectory_sha256"] or not row["trajectory_byte_equal"]:
                raise ValueError("Replay trajectory hash mismatch")
            with np.load(io.BytesIO(raw), allow_pickle=False) as trajectory, np.load(io.BytesIO(reference.read(f"run/observed/{row['episode_index']}.npz")), allow_pickle=False) as original:
                if any(trajectory[k].shape != original[k].shape or trajectory[k].dtype != original[k].dtype or trajectory[k].tobytes() != original[k].tobytes() for k in ("target_positions", "defender_positions")):
                    raise ValueError("Actual original trajectory changed")
            if any(row[k] != old_outcomes[row["episode_index"]][k] for k in ("safe_capture_success", "collision", "boundary_violation", "timeout", "termination_reason")):
                raise ValueError("Original physical outcomes changed")
    if read("primary/windows.json") != read("repeated/windows.json"):
        raise ValueError("Independent ranking replay differs")
    for key in ("statistics", "source_hashes", "capsule_sha256", "source_archive_sha256", "episodes"):
        if primary[key] != repeated[key]:
            raise ValueError("Independent replay summary differs")
    return {"status": "passed", "windows": primary["statistics"]["all_windows"], "rankable_windows": primary["statistics"]["rankable_windows"],
            "episodes_per_replay": len(primary["episodes"]), "replay_windows_byte_equal": True,
            "original_trajectory_arrays_byte_equal": True, "enhanced_control_enabled": False,
            "scope": "Two original-TRAIN diagnostic replays, not an enhanced-controller safety or success test."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--repeated", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    reference_path = ROOT.parent / "cwm_v4/artifacts/asymmetric_diagnostic_20261009.zip"
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if sha(reference_path) != protocol["source_archive_sha256"]:
        raise ValueError("Frozen V4 source archive changed")
    capsule = ROOT.parent / "cwm_v1/baseline/capsule.zip"
    if args.verify:
        integrity = check_archive(args.verify, "ARTIFACT_MANIFEST.json", False)
        with zipfile.ZipFile(args.verify) as archive, zipfile.ZipFile(reference_path) as reference:
            result = audit(archive.read, reference)
            if sha(capsule) != json.loads(archive.read("primary/summary.json"))["capsule_sha256"]:
                raise ValueError("Protected baseline changed")
        print(json.dumps({**result, "artifact_sha256": integrity["sha256"]}))
        return
    if args.run is None or args.repeated is None:
        parser.error("--run and --repeated required to package")
    summary = json.loads((args.run / "summary.json").read_text())
    if sha(capsule) != summary["capsule_sha256"]:
        raise ValueError("Protected baseline changed")
    members = {}
    for stage, path in (("primary", args.run), ("repeated", args.repeated)):
        for file in path.rglob("*"):
            if file.is_file() and "restored" not in file.relative_to(path).parts:
                members[f"{stage}/{file.relative_to(path).as_posix()}"] = file.read_bytes()
    for file in ROOT.glob("*.py"):
        members[f"source/cwm_v5/{file.name}"] = file.read_bytes()
    members["source/cwm_v5/protocol.json"] = (ROOT / "protocol.json").read_bytes()
    for name in ("cwm_v4/diagnose.py", "cwm_v2/scout.py", "cwm_v1/baseline.py", "cwm_v1/collect_pairs.py",
                 "cwm_v1/diagnose_interactions.py", "cwm_v1/freeze_baseline.py", "cwm_v1/verify_release.py"):
        members["source/" + name] = (ROOT.parent / name).read_bytes()
    with zipfile.ZipFile(reference_path) as reference:
        result = audit(members.__getitem__, reference)
    destination = ROOT / "artifacts/decision_value_20261009.zip"
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        archive.writestr("ARTIFACT_MANIFEST.json", json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in members.items()}, indent=2))
    check_archive(destination, "ARTIFACT_MANIFEST.json", False)
    with zipfile.ZipFile(destination) as archive, zipfile.ZipFile(reference_path) as reference:
        audit(archive.read, reference)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    if (reports / "decision_summary.json").exists():
        raise FileExistsError("Never overwrite milestone evidence")
    shutil.copy2(args.run / "summary.json", reports / "decision_summary.json")
    with (reports / "release_manifest.json").open("x") as stream:
        json.dump({**result, "artifact_sha256": sha(destination), "artifact_bytes": destination.stat().st_size,
                   "source_archive_sha256": sha(reference_path), "baseline_capsule_sha256": sha(capsule)}, stream, indent=2)
    print(json.dumps({**result, "artifact_bytes": destination.stat().st_size}))


if __name__ == "__main__":
    main()
