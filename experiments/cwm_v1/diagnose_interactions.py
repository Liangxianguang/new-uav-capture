"""Training-only clone diagnostic; private target variables are logs, not inputs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from baseline import Baseline
from collect_pairs import interventions
from freeze_baseline import sha, verify

BRANCHES = ("baseline_plan", "left_pressure", "right_pressure", "half_speed",
            "ring_clockwise", "ring_counterclockwise", "radial_close", "brake")


def diagnostic_commands(sequence, observation, reference, horizon=16):
    suffix = np.asarray(sequence[:8], dtype=np.float64)
    suffix = np.concatenate([suffix, np.repeat(suffix[-1:], horizon - len(suffix), axis=0)])
    original = list(interventions(suffix))
    radial = reference - observation["defender_positions"]
    radial /= np.maximum(np.linalg.norm(radial, axis=-1, keepdims=True), 1e-9)
    tangent = np.stack([-radial[:, 1], radial[:, 0], np.zeros(len(radial))], axis=-1)
    original.extend([suffix + sign * 2 * tangent[None] for sign in (-1, 1)])
    original.extend([np.repeat((5 * radial)[None], horizon, axis=0), np.zeros_like(suffix)])
    values = np.stack(original)
    values *= np.minimum(1., 5. / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-9))
    return values


def rollout(base, parent, sequence, noise_key):
    env = copy.deepcopy(parent)
    safety = base.filter_class(env)
    observation = env.observe()
    rows = []
    for offset, proposed in enumerate(sequence):
        env.rng = np.random.default_rng(np.random.SeedSequence([noise_key, offset, 0]))
        env.execution_rng = np.random.default_rng(np.random.SeedSequence([noise_key, offset, 1]))
        command, _ = safety.filter(proposed, observation)
        last_replan = env.target_maneuver_last_replan_step
        observation, _, done, truncated, info = base.env_class.step(env, command)
        rows.append({"target": env.target_position.copy(), "defenders": env.defender_positions.copy(),
                     "executed": env.last_executed_actions.copy(),
                     "target_sensor": env.target_maneuver_observed_positions.copy(),
                     "mode": env.target_maneuver_mode, "route": env.target_maneuver_route,
                     "direction": env.target_maneuver_direction.copy(),
                     "replanned": env.target_maneuver_last_replan_step != last_replan,
                     "valid": not any(info.get(k, False) for k in ("target_invalid_episode", "target_boundary_violation", "target_obstacle_violation")),
                     "termination": info.get("termination_reason", "running")})
        if done or truncated or not rows[-1]["valid"]:
            break
    return rows


def summarized(values):
    a = np.asarray(values, dtype=np.float64)
    if not a.size:
        return {"count": 0}
    return {"count": int(a.size), "mean": float(a.mean()), "p50": float(np.quantile(a, .5)),
            "p95": float(np.quantile(a, .95)), "max": float(a.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--selected-scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-per-variant", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Recover saved diagnostic windows; baseline episodes are rerun, saved windows are not resimulated")
    args = parser.parse_args()
    if args.groups_per_variant < 1:
        raise ValueError("Need independent training groups")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = Baseline(args.capsule, args.output / "restored")
    available = [json.loads(line) for line in args.selected_scenes.read_text().splitlines() if line]
    selected = []
    for variant in ("mixed_three_evasive", "dense_mixed"):
        groups = set()
        for record in available:
            if record["model_split"] == "train" and record["variant"] == variant and record["mirror_group_id"] not in groups:
                selected.append(record)
                groups.add(record["mirror_group_id"])
                if len(groups) == args.groups_per_variant:
                    break
        if len(groups) != args.groups_per_variant:
            raise ValueError("Insufficient training groups")
    scene_text = "".join(json.dumps(r) + "\n" for r in selected)
    scene_path = args.output / "selected_scenes.jsonl"
    if args.resume and scene_path.read_text() != scene_text:
        raise ValueError("Resume scene selection mismatch")
    scene_path.write_text(scene_text)
    windows_path = args.output / "windows.json"
    diagnostics = json.loads(windows_path.read_text()) if args.resume and windows_path.exists() else []
    completed = {(row["episode_index"], row["step"]) for row in diagnostics}
    for record in selected:
        def observer(env, observation, actions, sequence):
            if env.step_count not in range(4, 49, 4) or sequence is None:
                return
            if (record["episode_index"], int(env.step_count)) in completed:
                return
            reference, _ = base.evaluator.belief_reference(observation, base.planner_config)
            commands = diagnostic_commands(sequence, observation, reference)
            noise_key = int(np.random.SeedSequence([20261009, record["episode_index"], env.step_count]).generate_state(1)[0])
            before = (env.defender_positions.tobytes(), env.target_position.tobytes(),
                      json.dumps(env.rng.bit_generator.state, sort_keys=True),
                      json.dumps(env.execution_rng.bit_generator.state, sort_keys=True))
            branches = [rollout(base, env, command, noise_key) for command in commands]
            repeated = rollout(base, env, commands[0], noise_key)
            if len(repeated) != len(branches[0]) or any(a["target"].tobytes() != b["target"].tobytes() or a["defenders"].tobytes() != b["defenders"].tobytes() for a, b in zip(repeated, branches[0])):
                raise AssertionError("Clone repetition mismatch")
            after = (env.defender_positions.tobytes(), env.target_position.tobytes(),
                     json.dumps(env.rng.bit_generator.state, sort_keys=True),
                     json.dumps(env.execution_rng.bit_generator.state, sort_keys=True))
            if before != after:
                raise AssertionError("Diagnostic changed parent state")
            row = {"episode_index": record["episode_index"], "level": record["level"],
                   "step": int(env.step_count), "mode_at_snapshot": env.target_maneuver_mode,
                   "steps_since_replan": int(env.step_count - env.target_maneuver_last_replan_step),
                   "private_true_nearest_distance_m": float(np.linalg.norm(env.defender_positions - env.target_position, axis=-1).min()),
                   "target_replan_interval": int(env.pursuit["target_maneuver_replan_interval_steps"]),
                   "target_sensor_delay": int(env.pursuit["target_maneuver_observation_delay_steps"]),
                   "branches": {}}
            for name, branch in zip(BRANCHES[1:], branches[1:]):
                pairs = [(a, b) for a, b in zip(branches[0], branch) if a["valid"] and b["valid"]]
                row["branches"][name] = {
                    "response_m": [float(np.linalg.norm(a["target"] - b["target"])) for a, b in pairs],
                    "geometry_difference_m": [float(np.linalg.norm(a["defenders"] - b["defenders"], axis=-1).mean()) for a, b in pairs],
                    "executed_difference_mps": [float(np.linalg.norm(a["executed"] - b["executed"], axis=-1).mean()) for a, b in pairs],
                    "target_sensor_difference_m": [float(np.linalg.norm(a["target_sensor"] - b["target_sensor"], axis=-1).mean()) for a, b in pairs],
                    "target_direction_difference": [float(np.linalg.norm(a["direction"] - b["direction"])) for a, b in pairs],
                    "mode_differs": [a["mode"] != b["mode"] for a, b in pairs],
                    "route_differs": [a["route"] != b["route"] for a, b in pairs],
                    "baseline_replans": [bool(a["replanned"]) for a, b in pairs],
                    "branch_replans": [bool(b["replanned"]) for a, b in pairs],
                    "terminal": branch[-1]["termination"], "simulated_steps": len(branch)}
            diagnostics.append(row)
            windows_path.write_text(json.dumps(diagnostics, indent=2))
            print(json.dumps({"episode": record["episode_index"], "step": env.step_count,
                              "distance_m": row["private_true_nearest_distance_m"], "windows": len(diagnostics)}), flush=True)
        base.run(record, observer=observer)
    verify(base.root, base.capsule_manifest)
    if not diagnostics:
        raise ValueError("No diagnostic windows")
    summary = {"status": "completed_no_control_promotion", "source_scenes_sha256": sha(args.selected_scenes),
               "capsule_sha256": sha(args.capsule), "episodes": len(selected), "states": len(diagnostics),
               "groups_per_variant": args.groups_per_variant, "branches": list(BRANCHES),
               "target_truth_is_input": False, "baseline_and_parent_integrity_pass": True,
               "horizons": {}, "distance_m": summarized([r["private_true_nearest_distance_m"] for r in diagnostics]),
               "limitations": ["Few original model-train groups only; not a test or a causal benefit claim.",
                               "Steps 9-16 repeat the final MPC command; diagnostic only, not a longer MPC plan.",
                               "Private target decisions and truth are diagnostic logs only, forbidden model inputs.",
                               "Terminated branches are censored; paired horizon populations can differ."]}
    for horizon in (8, 16):
        summary["horizons"][str(horizon)] = {}
        for name in BRANCHES[1:]:
            bucket = {}
            for key in ("response_m", "geometry_difference_m", "executed_difference_mps", "target_sensor_difference_m", "target_direction_difference", "mode_differs", "route_differs", "baseline_replans", "branch_replans"):
                bucket[key] = summarized([v for row in diagnostics for v in row["branches"][name][key][:horizon]])
            effects = [v for row in diagnostics for v in row["branches"][name]["response_m"][:horizon]]
            bucket["response_over_0_1m_fraction"] = float(np.mean(np.asarray(effects) > .1)) if effects else None
            summary["horizons"][str(horizon)][name] = bucket
    windows_path.write_text(json.dumps(diagnostics, indent=2))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": summary["status"], "states": len(diagnostics)}))


if __name__ == "__main__":
    main()
