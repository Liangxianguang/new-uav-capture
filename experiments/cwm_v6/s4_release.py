"""Package and independently recompute S4 mechanism evidence; never promote control."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from s4_collect import ROOT, sha, s4_statistics
from verify_release import check_archive

BASELINE_SHA = "c83a27a4360ad055b69dbced4c28370cb31a32ae89ea9fbab13523fa6218a329"


def audit(read):
    results = {}
    groups_seen, seeds_seen = set(), set()
    protocols = []
    for stage in ("scout", "confirmation"):
        prefix = stage + "/"
        report = json.loads(read(prefix + "summary.json"))
        protocol = report["protocol"]
        protocols.append(protocol)
        if report["capsule_sha256"] != BASELINE_SHA or report["enhanced_control_enabled"] or report["new_model_trained"]:
            raise ValueError("Preserved-baseline / disabled-enhancement contract changed")
        if protocol["target_branch_rule_overrides"] or protocol["private_labels_model_inputs"] or protocol["original_weights_or_safety_changed"]:
            raise ValueError("Target rules, inputs or safety contract changed")
        for name, expected in report["source_hashes"].items():
            if hashlib.sha256(read(prefix + "source/" + name)).hexdigest() != expected:
                raise ValueError("Recorded experiment source mismatch")
        protocol_sources = [n for n in report["source_hashes"] if n.endswith("protocol.json")]
        if len(protocol_sources) != 1 or json.loads(read(prefix + "source/" + protocol_sources[0])) != protocol:
            raise ValueError("Recorded protocol differs from actual source")
        raw = read(prefix + "pairs.npz")
        if hashlib.sha256(raw).hexdigest() != report["dataset_sha256"]:
            raise ValueError("Paired-data digest mismatch")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            data = {k: archive[k] for k in archive.files}
        windows = json.loads(read(prefix + "windows.json"))
        scenes = [json.loads(line) for line in read(prefix + "selected_scenes.jsonl").decode().splitlines()]
        episodes = json.loads(read(prefix + "episodes.json"))
        n = len(windows)
        shapes = {"target": (n, 8, 16, 3), "defenders": (n, 8, 16, 4, 3),
                  "history": (n, 8, 252), "relative": (n, 4, 6), "backbone": (n, 8, 3),
                  "valid": (n, 8, 16), "branch_sign_label_only": (n, 8, 16),
                  "proposed": (n, 8, 16, 4, 3), "commanded": (n, 8, 16, 4, 3), "executed": (n, 8, 16, 4, 3)}
        for key, shape in shapes.items():
            if data[key].shape != shape or not np.isfinite(data[key]).all():
                raise ValueError("Array shape or finite-data contract mismatch")
        if not np.isin(data["branch_sign_label_only"], [-1, 0, 1]).all():
            raise ValueError("Invalid private label")
        expected_valid = ~np.isin(data["termination"], ["after_terminal", "target_obstacle_violation", "target_boundary_violation", "target_invalid_episode"])
        if not np.array_equal(expected_valid, data["valid"]):
            raise ValueError("Actual termination masks differ from reported validity")
        stats = s4_statistics(data, protocol)
        if stats != report["statistics"]:
            raise ValueError("Statistics differ from actual paired arrays")
        expected_status = "signal_requires_independent_confirmation" if stats["short_horizon_data_gate_passed"] else "s4_data_no_go_keep_baseline"
        if report["status"] != expected_status:
            raise ValueError("Incorrect signal status")
        actual_groups = {r["mirror_group_id"] for r in scenes}
        actual_seeds = {r["episode_seed"] for r in scenes}
        if groups_seen & actual_groups or seeds_seen & actual_seeds:
            raise ValueError("Scout and confirmation are not independent scene groups")
        groups_seen.update(actual_groups)
        seeds_seen.update(actual_seeds)
        if len(scenes) != protocol["groups"] * 2 or len(actual_groups) != protocol["groups"]:
            raise ValueError("Scene/mirror population mismatch")
        if len(episodes) != len(scenes) or episodes != report["episodes_results"]:
            raise ValueError("Incomplete episode results")
        if report["states"] != n or report["episodes"] != len(episodes) or report["groups"] != len(actual_groups):
            raise ValueError("Reported population mismatch")
        scene_by_id = {r["episode_index"]: r for r in scenes}
        expected_windows = set()
        for row in episodes:
            index = row["episode_index"]
            if index not in scene_by_id or not scene_by_id[index]["route_validation"]["branch_route_feasible"]:
                raise ValueError("Unvalidated or unlisted scene")
            arrays = []
            for kind in ("observed", "unobserved"):
                raw = read(f"{prefix}{kind}/{index}.npz")
                if hashlib.sha256(raw).hexdigest() != row[kind + "_sha256"]:
                    raise ValueError("Trajectory digest mismatch")
                with np.load(io.BytesIO(raw), allow_pickle=False) as value:
                    arrays.append({k: value[k] for k in value.files})
            if any(arrays[0][k].shape != arrays[1][k].shape or arrays[0][k].dtype != arrays[1][k].dtype
                   or arrays[0][k].tobytes() != arrays[1][k].tobytes() for k in arrays[0]):
                raise ValueError("Actual baseline trajectory changed")
            if not row["trajectory_byte_equal"] or not row["outcomes_equal"]:
                raise ValueError("Original-controller isolation failed")
            steps = arrays[0]["target_positions"].shape[0] - 1
            expected_windows.update((index, step) for step in protocol["snapshot_steps"] if step < steps)
        if len({(w["episode_index"], w["step"]) for w in windows}) != n or {(w["episode_index"], w["step"]) for w in windows} != expected_windows:
            raise ValueError("Snapshot coverage mismatch")
        for i, window in enumerate(windows):
            if not window["parent_integrity"] or not window["repeat_exact"]:
                raise ValueError("Clone isolation failed")
            if window["group"] != data["group"][i] or window["step"] != data["step"][i] or window["group"] != scene_by_id[window["episode_index"]]["mirror_group_id"]:
                raise ValueError("Actual window provenance mismatch")
        results[stage] = {"states": n, "episodes": len(episodes), "groups": len(actual_groups),
                          "data_gate_passed": stats["short_horizon_data_gate_passed"],
                          "target_invalid_episodes": sum(r["target_invalid_episode"] for r in episodes),
                          "safe_captures": sum(r["safe_capture_success"] for r in episodes)}
    for key in ("scene_variation", "observation_conditions", "target_speed_scales", "branches", "snapshot_steps", "data_gate", "effect_threshold_m"):
        if protocols[0][key] != protocols[1][key]:
            raise ValueError("Confirmation altered the predeclared experiment")
    return {"status": "passed", "stages": results, "enhanced_control_enabled": False,
            "new_model_trained": False, "scope": "Mechanism replication only; target-invalid scenes prevent promotion as a validated benchmark or controller."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scout", type=Path)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    capsule = ROOT.parent / "cwm_v1/baseline/capsule.zip"
    if sha(capsule) != BASELINE_SHA:
        raise ValueError("Original baseline capsule changed")
    if args.verify:
        integrity = check_archive(args.verify, "ARTIFACT_MANIFEST.json", False)
        with zipfile.ZipFile(args.verify) as archive:
            result = audit(archive.read)
        print(json.dumps({**result, "artifact_sha256": integrity["sha256"]}))
        return
    if args.scout is None or args.confirmation is None:
        parser.error("Both experiment runs required")
    members = {}
    for stage, run in (("scout", args.scout), ("confirmation", args.confirmation)):
        for file in run.rglob("*"):
            relative = file.relative_to(run)
            if file.is_file() and "restored" not in relative.parts and file.name != "pairs_partial.npz":
                members[f"{stage}/{relative.as_posix()}"] = file.read_bytes()
    for file in ROOT.glob("*.py"):
        members[f"verification_source/{file.name}"] = file.read_bytes()
    result = audit(members.__getitem__)
    destination = ROOT / "artifacts/s4_mechanism_20261010.zip"
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        archive.writestr("ARTIFACT_MANIFEST.json", json.dumps({n: hashlib.sha256(v).hexdigest() for n, v in members.items()}, indent=2))
    check_archive(destination, "ARTIFACT_MANIFEST.json", False)
    with zipfile.ZipFile(destination) as archive:
        audit(archive.read)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    for stage, run in (("scout", args.scout), ("confirmation", args.confirmation)):
        with (reports / f"{stage}_summary.json").open("xb") as stream:
            stream.write((run / "summary.json").read_bytes())
    with (reports / "release_manifest.json").open("x") as stream:
        json.dump({**result, "artifact_sha256": sha(destination), "artifact_bytes": destination.stat().st_size,
                   "baseline_capsule_sha256": BASELINE_SHA}, stream, indent=2)
    print(json.dumps({**result, "artifact_sha256": sha(destination), "artifact_bytes": destination.stat().st_size}))


if __name__ == "__main__":
    main()
