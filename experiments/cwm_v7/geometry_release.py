"""Audit actual target physics, finite stress support and original replay bytes."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from diagnose_geometry import ROOT, select_geometry, sha
from verify_release import check_archive

BASELINE_SHA = "c83a27a4360ad055b69dbced4c28370cb31a32ae89ea9fbab13523fa6218a329"
V6_SHA = "2d4b0cd57b7e7b016ce7557dab39822e621b831db96d01b3b05e1c7bd247ac3a"


def arrays(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def target_bad(targets, scene):
    targets = np.asarray(targets)
    if not np.isfinite(targets).all():
        raise ValueError("Nonfinite target trajectory")
    # Frozen contract: world +-10m, z=.5..10m, target radius0, boundary margin.3.
    bad = ((targets <= np.array([-9.7, -9.7, .8])) | (targets >= np.array([9.7, 9.7, 9.7]))).any(-1)
    for wall in scene["obstacles"]:
        if wall["shape"] != "wall":
            raise ValueError("Unexpected obstacle contract")
        center = np.array([*wall["center_xy"], wall["height"] * .5])
        half = np.array([*wall["half_extents_xy"], wall["height"] * .5])
        bad |= (np.abs(targets - center) <= half).all(-1)
    return bad


def compare_arrays(a, b):
    return set(a) == set(b) and all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes() for k in a)


def audit(read, v6_read):
    results, stage_protocols, stage_groups = {}, [], []
    for stage in ("development", "qualification"):
        summary = json.loads(read(f"{stage}/summary.json"))
        protocol = summary["protocol"]
        stage_protocols.append(protocol)
        if summary["baseline_capsule_sha256"] != BASELINE_SHA or summary["v6_archive_sha256"] != V6_SHA or summary["enhanced_control_enabled"] or summary["new_model_trained"]:
            raise ValueError("Frozen baseline/diagnostic contract mismatch")
        if protocol["target_branch_rule_overrides"]:
            raise ValueError("Target rules changed")
        for name, expected in summary["source_hashes"].items():
            if hashlib.sha256(read(f"{stage}/source/{name}")).hexdigest() != expected:
                raise ValueError("Recorded source mismatch")
        protocol_names = [n for n in summary["source_hashes"] if n.endswith("protocol.json")]
        if len(protocol_names) != 1 or json.loads(read(f"{stage}/source/{protocol_names[0]}")) != protocol:
            raise ValueError("Actual protocol mismatch")
        traces = json.loads(read(f"{stage}/command_terms.json"))
        old_scenes, old_trajectories = {}, {}
        for old_stage in ("scout", "confirmation"):
            for record in [json.loads(line) for line in v6_read(f"{old_stage}/selected_scenes.jsonl").decode().splitlines()]:
                index = record["episode_index"]
                old_scenes[index] = record["scenario"]
                old_trajectories[index] = arrays(v6_read(f"{old_stage}/observed/{index}.npz"))
        if {r["episode_index"] for r in traces} != set(old_scenes) or len(traces) != 16:
            raise ValueError("Incomplete old-scene command tracing")
        for row in traces:
            index = row["episode_index"]
            actual = arrays(read(f"{stage}/diagnostic_replays/{index}.npz"))
            if not row["trajectory_byte_equal"] or not compare_arrays(actual, old_trajectories[index]):
                raise ValueError("Actual V6 trace replay differs")
            if bool(target_bad(actual["target_positions"], old_scenes[index]).any()) != row["target_invalid_episode"]:
                raise ValueError("V6 trace target-validity flag mismatch")
            if len(row["trace"]) != actual["target_positions"].shape[0] - 1:
                raise ValueError("Incomplete command-term trace")
            for t, term in enumerate(row["trace"]):
                desired = np.array(term["goal_term"]) + np.array(term["defender_terms"]).sum(0) + np.array(term["obstacle_terms"]).sum(0) + np.array(term["boundary_term"])
                if term["step"] != t or not np.array_equal(term["target"], actual["target_positions"][t]) or not np.allclose(desired, term["desired_sum"], atol=1e-12, rtol=0):
                    raise ValueError("Actual command terms mismatch")
        if summary["v6_traces_exact_replays"] != 16 or summary["v6_trace_target_invalid_episodes"] != sum(r["target_invalid_episode"] for r in traces):
            raise ValueError("Trace aggregate mismatch")
        candidates, groups = [], set()
        for center in protocol["wall_center_x_candidates_m"]:
            prefix = f"{stage}/wallx{center:g}"
            records = [json.loads(line) for line in read(prefix + "/scenes.jsonl").decode().splitlines()]
            episodes = json.loads(read(prefix + "/episodes.json"))
            stress = json.loads(read(prefix + "/stress.json"))
            by_id = {r["episode_index"]: r for r in records}
            candidate_groups = {r["mirror_group_id"] for r in records}
            groups |= candidate_groups
            if len(records) != protocol["groups"] * 2 or len(candidate_groups) != protocol["groups"] or len(episodes) != len(records) or {r["episode_index"] for r in episodes} != set(by_id):
                raise ValueError("Incomplete scene/mirror population")
            expected_stress = set()
            for row in episodes:
                index = row["episode_index"]
                record = by_id[index]
                scene = record["scenario"]
                if not record["route_validation"]["branch_route_feasible"] or scene["obstacles"][0]["center_xy"] != [center, 0.]:
                    raise ValueError("Scene geometry/route provenance mismatch")
                a = arrays(read(f"{prefix}/observed/{index}.npz"))
                b = arrays(read(f"{prefix}/plain/{index}.npz"))
                if not row["trajectory_byte_equal"] or not row["outcomes_equal"] or not compare_arrays(a, b):
                    raise ValueError("Actual baseline arrays differ")
                if bool(target_bad(a["target_positions"], scene).any()) != row["target_invalid_episode"]:
                    raise ValueError("Original target-validity flag differs from actual geometry")
                steps = a["target_positions"].shape[0] - 1
                expected_stress.update((index, t, p) for t in protocol["stress_snapshot_steps"] if t < steps for p in protocol["stress_probe_indices"])
            if len(stress) != len(expected_stress) or {(r["episode_index"], r["step"], r["probe"]) for r in stress} != expected_stress:
                raise ValueError("Actual stress snapshot population mismatch")
            for row in stress:
                branch = arrays(read(prefix + "/" + row["file"]))
                observed = branch["termination"] != "after_terminal"
                count = int(observed.sum())
                if not np.array_equal(observed, np.arange(protocol["horizon_steps"]) < count) or count != row["steps_observed"] or str(branch["termination"][count - 1]) != row["termination"]:
                    raise ValueError("Actual stress terminal support mismatch")
                bad = target_bad(branch["target"][:count], by_id[row["episode_index"]]["scenario"])
                expected_valid = np.zeros(protocol["horizon_steps"], bool)
                expected_valid[:count] = ~np.maximum.accumulate(bad)
                if not np.array_equal(branch["valid"], expected_valid) or bool(bad.any()) != row["target_invalid"] or row["horizon_complete"] != (count == protocol["horizon_steps"]):
                    raise ValueError("Actual stress physics/validity mask mismatch")
            candidates.append({"wall_center_x_m": center, "original_target_invalid_episodes": sum(r["target_invalid_episode"] for r in episodes),
                               "original_safe_captures": sum(r["safe_capture_success"] for r in episodes), "stress_target_invalid_branches": sum(r["target_invalid"] for r in stress),
                               "stress_branches": len(stress), "stress_full_horizon_branches": sum(r["horizon_complete"] for r in stress),
                               "stress_termination_counts": {reason: sum(r["termination"] == reason for r in stress) for reason in sorted({r["termination"] for r in stress})}})
        if candidates != summary["candidates"] or candidates != json.loads(read(f"{stage}/candidates.json")) or select_geometry(candidates) != summary["selected_wall_center_x_m"]:
            raise ValueError("Selection/aggregate does not match actual physics")
        stage_groups.append(groups)
        results[stage] = {"selected_wall_center_x_m": select_geometry(candidates), "candidates": candidates, "groups": len(groups)}
    for field in ("scene_variation", "target_speed_scales", "observation_conditions", "horizon_steps", "stress_snapshot_steps", "stress_probe_indices"):
        if stage_protocols[0][field] != stage_protocols[1][field]:
            raise ValueError("Fresh qualification changed frozen non-geometry protocol")
    if stage_groups[0] & stage_groups[1]:
        raise ValueError("Geometry qualification reused development groups")
    selected = results["development"]["selected_wall_center_x_m"]
    qualified = selected is not None and selected == results["qualification"]["selected_wall_center_x_m"]
    return {"status": "passed", "geometry_qualified_for_finite_training_distribution": qualified, "stages": results,
            "enhanced_control_enabled": False, "new_model_trained": False,
            "scope": "Finite scene compatibility only; censored branches and defender failures are not safety/capture guarantees."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    capsule = ROOT.parent / "cwm_v1/baseline/capsule.zip"
    v6_path = ROOT.parent / "cwm_v6/artifacts/s4_mechanism_20261010.zip"
    if sha(capsule) != BASELINE_SHA or sha(v6_path) != V6_SHA:
        raise ValueError("Previous milestone changed")
    if args.verify:
        integrity = check_archive(args.verify, "ARTIFACT_MANIFEST.json", False)
        with zipfile.ZipFile(args.verify) as archive, zipfile.ZipFile(v6_path) as v6:
            result = audit(archive.read, v6.read)
        print(json.dumps({**result, "artifact_sha256": integrity["sha256"]}))
        return
    if args.development is None or args.qualification is None:
        parser.error("Both completed compatibility runs required")
    members = {}
    for stage, run in (("development", args.development), ("qualification", args.qualification)):
        for file in run.rglob("*"):
            relative = file.relative_to(run)
            if file.is_file() and "restored" not in relative.parts:
                members[f"{stage}/{relative.as_posix()}"] = file.read_bytes()
    for file in ROOT.glob("*.py"):
        members["verification_source/" + file.name] = file.read_bytes()
    with zipfile.ZipFile(v6_path) as v6:
        result = audit(members.__getitem__, v6.read)
    path = ROOT / "artifacts/geometry_qualification_20261010.zip"
    path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        archive.writestr("ARTIFACT_MANIFEST.json", json.dumps({n: hashlib.sha256(v).hexdigest() for n, v in members.items()}, indent=2))
    check_archive(path, "ARTIFACT_MANIFEST.json", False)
    with zipfile.ZipFile(path) as archive, zipfile.ZipFile(v6_path) as v6:
        audit(archive.read, v6.read)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    for stage, run in (("development", args.development), ("qualification", args.qualification)):
        with (reports / f"{stage}_summary.json").open("xb") as file:
            file.write((run / "summary.json").read_bytes())
    with (reports / "geometry_release_manifest.json").open("x") as file:
        json.dump({**result, "artifact_sha256": sha(path), "artifact_bytes": path.stat().st_size}, file, indent=2)
    print(json.dumps({**result, "artifact_sha256": sha(path), "artifact_bytes": path.stat().st_size}))


if __name__ == "__main__":
    main()
