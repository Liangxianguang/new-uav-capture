"""Trace frozen S4 command terms and test a predeclared geometry-only grid."""
import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from geometry import ROOT, translated_records, BackboneTap, s4_commands, s4_rollout, fingerprint, sha, verify


def traced_episode(base, record, trajectory):
    from encirclement3d.pursuit_env import _unit
    original = base.env_class._adaptive_branching_target_action
    trace = []
    def traced(env):
        action = original(env)
        sign = env.target_branch_sign
        waypoint_x = float(env.pursuit["target_branch_waypoint_x"])
        if sign is None:
            mode, goal, desired = "before_commit", None, np.array([1., 0., 0.])
        else:
            mode = "approach_exit" if env.target_position[0] < -waypoint_x else "cross_exit" if env.target_position[0] < waypoint_x else "goal"
            x = -waypoint_x if mode == "approach_exit" else waypoint_x if mode == "cross_exit" else float(env.pursuit["target_branch_goal_x"])
            goal = np.array([x, sign * float(env.pursuit["target_branch_exit_offset_y"]), env.target_position[2]])
            desired = _unit(goal - env.target_position, fallback=env.target_escape_direction)
        attraction = desired.copy()
        defender_terms = []
        for position in env.defender_positions:
            delta = env.target_position - position
            distance = float(np.linalg.norm(delta))
            term = (_unit(delta) * (float(env.pursuit["target_defender_avoidance_distance"]) - distance)
                    * float(env.pursuit["target_defender_avoidance_gain"])) if distance < float(env.pursuit["target_defender_avoidance_distance"]) else np.zeros(3)
            defender_terms.append(term)
            desired += term
        obstacle_terms, clearances = [], []
        for obstacle in env.obstacles:
            clearance, normal = env._cylinder_clearance_and_normal(env.target_position, obstacle)
            term = (normal * (float(env.pursuit["target_obstacle_avoidance_distance"]) - clearance)
                    * float(env.pursuit["target_obstacle_avoidance_gain"])) if clearance < float(env.pursuit["target_obstacle_avoidance_distance"]) else np.zeros(3)
            obstacle_terms.append(term)
            clearances.append(clearance)
            desired += term
        boundary_term = np.zeros(3)
        for axis in range(3):
            if env.target_position[axis] < env.lower[axis] + float(env.pursuit["target_boundary_margin"]):
                boundary_term[axis] += float(env.pursuit["target_boundary_gain"])
            if env.target_position[axis] > env.upper[axis] - float(env.pursuit["target_boundary_margin"]):
                boundary_term[axis] -= float(env.pursuit["target_boundary_gain"])
        desired += boundary_term
        rebuilt = _unit(desired, fallback=env.target_escape_direction) * float(env.agents["target_max_speed"]) * float(env.target_speed_scale)
        if not np.allclose(action, rebuilt, atol=1e-12, rtol=0):
            raise AssertionError("Diagnostic terms do not reproduce frozen target command")
        trace.append({"step": int(env.step_count), "target": env.target_position.tolist(), "velocity_before": env.target_velocity.tolist(),
                      "branch_label_only": sign, "waypoint_mode": mode, "goal": None if goal is None else goal.tolist(),
                      "goal_term": attraction.tolist(), "defender_terms": np.asarray(defender_terms).tolist(),
                      "obstacle_terms": np.asarray(obstacle_terms).tolist(), "boundary_term": boundary_term.tolist(),
                      "wall_clearance_m": clearances, "desired_sum": desired.tolist(), "returned_command": action.tolist()})
        return action
    base.env_class._adaptive_branching_target_action = traced
    try:
        row, _ = base.run(record, trajectory)
    finally:
        base.env_class._adaptive_branching_target_action = original
    return row, trace


def select_geometry(candidates):
    passed = [r["wall_center_x_m"] for r in candidates if r["original_target_invalid_episodes"] == 0 and r["stress_target_invalid_branches"] == 0]
    return min(passed) if passed else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--v6-archive", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "geometry_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    if protocol["enhanced_control_enabled"] or protocol["new_model_trained"]:
        raise ValueError("Geometry diagnostic may not enable or train control")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = BackboneTap(args.capsule, args.output / "restored")
    sources = [ROOT / "geometry.py", Path(__file__), args.protocol.resolve()]
    sources += [ROOT.parent / p for p in ("cwm_v6/s4_collect.py", "cwm_v2/scout.py", "cwm_v4/diagnose.py", "cwm_v1/baseline.py", "cwm_v1/freeze_baseline.py", "cwm_v1/collect_pairs.py", "cwm_v1/diagnose_interactions.py")]
    hashes = {}
    for source in sources:
        name = source.relative_to(ROOT.parent).as_posix()
        hashes[name] = sha(source)
        destination = args.output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    started = time.perf_counter()
    replays = []
    with zipfile.ZipFile(args.v6_archive) as archive:
        for stage in ("scout", "confirmation"):
            records = [json.loads(line) for line in archive.read(f"{stage}/selected_scenes.jsonl").decode().splitlines()]
            for record in records:
                index = record["episode_index"]
                trajectory = args.output / "diagnostic_replays" / f"{index}.npz"
                row, trace = traced_episode(base, record, trajectory)
                with np.load(trajectory, allow_pickle=False) as actual, np.load(io.BytesIO(archive.read(f"{stage}/observed/{index}.npz")), allow_pickle=False) as reference:
                    exact = all(actual[k].dtype == reference[k].dtype and actual[k].shape == reference[k].shape and actual[k].tobytes() == reference[k].tobytes() for k in actual.files)
                if not exact:
                    raise AssertionError("Tracing changed original trajectory")
                replays.append({"episode_index": index, "trajectory_byte_equal": exact, "target_invalid_episode": row["target_invalid_episode"], "termination_reason": row["termination_reason"], "trace": trace})
    (args.output / "command_terms.json").write_text(json.dumps(replays, indent=2))
    candidates = []
    for center in protocol["wall_center_x_candidates_m"]:
        records = translated_records(base, protocol, center)
        candidate_path = args.output / f"wallx{center:g}"
        candidate_path.mkdir()
        (candidate_path / "scenes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        stress, episodes = [], []
        for record in records:
            index = record["episode_index"]
            def observer(env, observation, actions, sequence):
                if env.step_count not in protocol["stress_snapshot_steps"]:
                    return
                reference, _ = base.evaluator.belief_reference(observation, base.planner_config)
                proposals = s4_commands(sequence, observation, reference, protocol)
                before = fingerprint(env)
                for probe in protocol["stress_probe_indices"]:
                    key = int.from_bytes(hashlib.sha256(f"{index}/{env.step_count}/{probe}".encode()).digest()[:4], "little")
                    rollout = s4_rollout(base, env, proposals[probe], key)
                    name = f"{index}_step{env.step_count}_probe{probe}.npz"
                    np.savez_compressed(candidate_path / name, **rollout)
                    valid_count = int(np.count_nonzero(rollout["termination"] != "after_terminal"))
                    terminal = str(rollout["termination"][valid_count - 1])
                    stress.append({"episode_index": index, "step": int(env.step_count), "probe": probe, "file": name,
                                   "target_invalid": bool(not rollout["valid"][:valid_count].all()), "steps_observed": valid_count,
                                   "termination": terminal, "horizon_complete": valid_count == protocol["horizon_steps"]})
                if before != fingerprint(env):
                    raise AssertionError("Stress branch changed baseline parent")
            row, _ = base.run(record, candidate_path / "observed" / f"{index}.npz", observer)
            plain, _ = base.run(record, candidate_path / "plain" / f"{index}.npz")
            with np.load(candidate_path / "observed" / f"{index}.npz") as a, np.load(candidate_path / "plain" / f"{index}.npz") as b:
                if any(a[k].tobytes() != b[k].tobytes() for k in a.files):
                    raise AssertionError("Stress observer changed baseline")
            episodes.append({"episode_index": index, "safe_capture_success": row["safe_capture_success"], "target_invalid_episode": row["target_invalid_episode"],
                             "collision": row["collision"], "boundary_violation": row["boundary_violation"], "termination_reason": row["termination_reason"],
                             "trajectory_byte_equal": True, "outcomes_equal": all(row[k] == plain[k] for k in ("safe_capture_success", "target_invalid_episode", "collision", "boundary_violation", "termination_reason"))})
            if not episodes[-1]["outcomes_equal"]:
                raise AssertionError("Observer changed outcomes")
            print(json.dumps({"wallx": center, "episode": index, "target_invalid": row["target_invalid_episode"]}), flush=True)
        (candidate_path / "episodes.json").write_text(json.dumps(episodes, indent=2))
        (candidate_path / "stress.json").write_text(json.dumps(stress, indent=2))
        candidates.append({"wall_center_x_m": center, "original_target_invalid_episodes": sum(r["target_invalid_episode"] for r in episodes),
                           "original_safe_captures": sum(r["safe_capture_success"] for r in episodes), "stress_target_invalid_branches": sum(r["target_invalid"] for r in stress),
                           "stress_branches": len(stress), "stress_full_horizon_branches": sum(r["horizon_complete"] for r in stress),
                           "stress_termination_counts": {reason: sum(r["termination"] == reason for r in stress) for reason in sorted({r["termination"] for r in stress})}})
        (args.output / "candidates.json").write_text(json.dumps(candidates, indent=2))
    verify(base.root, base.capsule_manifest)
    if any(sha(ROOT.parent / name) != expected for name, expected in hashes.items()):
        raise AssertionError("Source changed during geometry testing")
    selected = select_geometry(candidates)
    summary = {"status": "geometry_candidate_requires_fresh_confirmation" if selected is not None else "geometry_grid_no_go_keep_baseline",
               "protocol": protocol, "source_hashes": hashes, "baseline_capsule_sha256": sha(args.capsule), "v6_archive_sha256": sha(args.v6_archive),
               "v6_traces_exact_replays": len(replays), "v6_trace_target_invalid_episodes": sum(r["target_invalid_episode"] for r in replays),
               "candidates": candidates, "selected_wall_center_x_m": selected, "enhanced_control_enabled": False,
               "new_model_trained": False, "elapsed_seconds": time.perf_counter() - started,
               "limitations": ["Geometry-development data; never model validation or test.", "Target validity only assessed for tested finite policies and horizons; no safe-set proof.", "Capture/safety termination censors later target behavior; truncated branches are disclosed.", "Translation changes new scene geometry, not frozen target rules or original Level scenes."]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
