"""Evaluate the public-belief route expert and filter calibration scenes.

This is a development-only feasibility gate.  The raw scene pool is never
deleted, and expert-success filtering produces a separate calibration file.
The evaluator records target and defender boundary contacts separately from
the environment's historical aggregate safety flag.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_controllers import (  # noqa: E402
    PublicBeliefRouteIntentController,
    PursuitCBFSafetyFilter,
    SafetyFilteredPursuitController,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    capture_contract_metrics,
    crossing_metrics,
    prepare_showcase_episode,
    scenario_from_metadata,
)
from evaluate_s4_closed_loop import config_for_phase73_spec  # noqa: E402


class OracleRouteIntentController(PublicBeliefRouteIntentController):
    """Geometry-aware upper bound that is allowed to read target truth.

    This controller is calibration-only.  It is never used as a deployable
    policy or a positive main-method result; its purpose is to distinguish a
    physically feasible scene from a scene that defeats even a true-state
    route expert.
    """

    def _public_target_estimate(self, observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        del observation
        self.belief_blind = False
        return self.env.target_position.copy(), self.env.target_velocity.copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--difficulty", choices=("all", "easy", "nominal", "hard"), default="all")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--controller", choices=("oracle_route", "public_belief_route"), default="oracle_route")
    parser.add_argument(
        "--experiment-name",
        default="phase77_public_belief_route_expert_calibration",
        help="Experiment label written to the summary artifact.",
    )
    parser.add_argument("--no-local-cbf", action="store_true")
    parser.add_argument(
        "--safety-margin",
        type=float,
        help="Optional calibration-only local-CBF margin override in metres.",
    )
    return parser.parse_args()


def _read_records(path: Path, limit: int | None, difficulty: str) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    records.sort(key=lambda value: int(value["episode_index"]))
    if difficulty != "all":
        records = [record for record in records if str(record.get("difficulty")) == difficulty]
    if limit is not None:
        if limit <= 0:
            raise ValueError("episodes must be positive")
        records = records[:limit]
    if not records:
        raise ValueError("no scene records selected")
    return records


def _guard_not_locked(scenes: Path) -> dict[str, Any]:
    manifest_path = scenes.resolve().parent / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"scene manifest is required next to {scenes}: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("locked_test", False)) or not bool(manifest.get("not_a_locked_test", False)):
        raise ValueError("Phase 77 calibration refuses locked-test or unlabelled scene manifests")
    return manifest


def _boundary_contact_flags(env: CaptureRadiusPursuit3DEnv) -> tuple[bool, bool]:
    lower = np.asarray(env.lower, dtype=np.float64)
    upper = np.asarray(env.upper, dtype=np.float64)
    target = np.asarray(env.target_position, dtype=np.float64)
    defenders = np.asarray(env.defender_positions, dtype=np.float64)
    target_contact = bool(
        np.any(np.isclose(target, lower, atol=1.0e-9, rtol=0.0))
        or np.any(np.isclose(target, upper, atol=1.0e-9, rtol=0.0))
    )
    defender_contact = bool(
        np.any(np.isclose(defenders, lower[None, :], atol=1.0e-9, rtol=0.0))
        or np.any(np.isclose(defenders, upper[None, :], atol=1.0e-9, rtol=0.0))
    )
    return target_contact, defender_contact


def _config_for_record(
    environment_config: Path,
    record: dict[str, Any],
    safety_margin: float | None,
) -> dict[str, Any]:
    config = config_for_phase73_spec(environment_config.resolve(), record, None)
    execution = config.setdefault("dynamics", {}).setdefault("execution", {})
    execution_profile = record.get("execution", {})
    if not isinstance(execution_profile, dict):
        raise ValueError("scene execution profile must be a mapping")
    execution.update(
        {
            "enabled": True,
            "action_delay_steps": int(execution_profile.get("action_delay_steps", 0)),
            "command_noise_std": float(execution_profile.get("command_noise_std_mps", 0.0)),
        }
    )
    if safety_margin is not None:
        if safety_margin < 0.0:
            raise ValueError("safety-margin must be non-negative")
        config.setdefault("task", {}).setdefault("pursuit", {})["safety_margin"] = float(safety_margin)
    return config


def _rollout(
    record: dict[str, Any],
    environment_config: Path,
    use_local_cbf: bool,
    controller_name: str,
    safety_margin: float | None,
) -> dict[str, Any]:
    config = _config_for_record(environment_config, record, safety_margin)
    scenario = scenario_from_metadata(record["scenario"])
    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=int(record["obstacle_count"]),
        target_speed_scale=float(record["target_speed_scale"]),
    )
    observation = prepare_showcase_episode(
        env,
        scenario,
        int(record["episode_seed"]),
        record_history=True,
        validate_scenario=False,
    )
    controller_class = (
        OracleRouteIntentController
        if controller_name == "oracle_route"
        else PublicBeliefRouteIntentController
    )
    base_controller = controller_class(
        env,
        horizon_seconds=0.75,
        replan_interval_steps=8,
        min_hold_steps=6,
        grid_step=0.75,
        route_margin=0.85,
    )
    controller: Any = SafetyFilteredPursuitController(base_controller) if use_local_cbf else base_controller
    target_boundary = False
    defender_boundary = False
    target_obstacle_collision = False
    route_latencies: list[float] = []
    safety_corrections: list[float] = []
    final_info: dict[str, Any] = {}
    while True:
        start = time.perf_counter()
        action = controller.act(observation)
        route_latencies.append((time.perf_counter() - start) * 1000.0)
        diagnostics = getattr(controller, "last_diagnostics", None)
        if diagnostics is not None:
            safety_corrections.append(float(diagnostics.action_correction_norm))
        observation, _reward, terminated, truncated, final_info = env.step(action, record_history=True)
        target_contact, defender_contact = _boundary_contact_flags(env)
        target_boundary |= target_contact
        defender_boundary |= defender_contact
        if any(float(env._obstacle_clearance(env.target_position, obstacle)) < 0.0 for obstacle in env.obstacles):
            target_obstacle_collision = True
        if terminated or truncated:
            break

    crossing = crossing_metrics(env, scenario.obstacle_zone_x)
    contract = capture_contract_metrics(
        final_info,
        crossing,
        target_collision=target_obstacle_collision,
        target_crossing_required=bool(record.get("target_crossing_required", False)),
        required_defender_zone_entries=int(record["scenario"].get("required_defender_zone_entries", 1)),
        require_target_zone_entry=bool(record.get("target_crossing_required", False)),
    )
    physical_collision = bool(int(final_info.get("collision_steps", 0)) > 0)
    physical_feasible = bool(
        final_info.get("safe_capture_success", False)
        and not target_boundary
        and not defender_boundary
        and not physical_collision
        and not target_obstacle_collision
    )
    route_exercise_accepted = bool(physical_feasible and crossing["target_zone_entered"])
    return {
        "episode_index": int(record["episode_index"]),
        "episode_seed": int(record["episode_seed"]),
        "mirror_group_id": str(record["mirror_group_id"]),
        "difficulty": str(record["difficulty"]),
        "target_motion_mode": str(record["target_motion_mode"]),
        "target_speed_scale": float(record["target_speed_scale"]),
        "observation_condition": str(record["observation_condition"]),
        "execution": copy.deepcopy(record["execution"]),
        "safe_capture_success": bool(final_info.get("safe_capture_success", False)),
        "safe_capture_in_pursuit": bool(contract["safe_capture_in_pursuit"]),
        "capture_event": bool(final_info.get("capture_event", False)),
        "collision": bool(final_info.get("collision", False)),
        "physical_collision": physical_collision,
        "physical_collision_steps": int(final_info.get("collision_steps", 0)),
        "target_obstacle_collision": target_obstacle_collision,
        "target_boundary_violation": target_boundary,
        "defender_boundary_violation": defender_boundary,
        "world_violation_steps": int(final_info.get("world_violation_steps", 0)),
        "timeout": bool(final_info.get("termination_reason") == "timeout"),
        "termination_reason": str(final_info.get("termination_reason", "unknown")),
        "task_termination_reason": str(contract["task_termination_reason"]),
        "target_crossed": bool(crossing["target_crossed"]),
        "target_zone_entered": bool(crossing["target_zone_entered"]),
        "defender_zone_entry_count": int(crossing["defender_zone_entry_count"]),
        "steps": int(env.step_count),
        "capture_time_seconds": (
            float(final_info["capture_time_seconds"])
            if final_info.get("capture_time_seconds") is not None
            else float(env.max_steps * env.dt)
        ),
        "min_clearance_m": float(final_info.get("min_clearance_so_far", float("inf"))),
        "route_intent": str(base_controller.route_name),
        "target_maneuver_fallback_count": int(final_info.get("target_maneuver_fallback_count", 0)),
        "target_maneuver_feasible_candidate_count": int(
            final_info.get("target_maneuver_feasible_candidate_count", 0)
        ),
        "target_maneuver_rejected_candidate_count": int(
            final_info.get("target_maneuver_rejected_candidate_count", 0)
        ),
        "expert_calibration_accepted": physical_feasible,
        "expert_route_exercise_accepted": route_exercise_accepted,
        "local_cbf_empirical_filter": bool(use_local_cbf),
        "latency_ms": {
            "route_and_safety_p50": float(np.percentile(route_latencies, 50)),
            "route_and_safety_p95": float(np.percentile(route_latencies, 95)),
            "route_and_safety_p99": float(np.percentile(route_latencies, 99)),
        },
        "latency_samples_ms": [float(value) for value in route_latencies],
        "mean_safety_correction_norm": float(np.mean(safety_corrections)) if safety_corrections else 0.0,
    }


def _percentile(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {f"p{p}": float(np.percentile(array, p)) for p in (50, 95, 99)}


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(key: str) -> float:
        return float(np.mean([bool(row[key]) for row in rows])) if rows else 0.0

    latency_samples = [
        float(value)
        for row in rows
        for value in row.get("latency_samples_ms", [])
    ]
    return {
        "episodes": len(rows),
        "mirror_groups": len({str(row["mirror_group_id"]) for row in rows}),
        "difficulty_counts": dict(sorted(Counter(str(row["difficulty"]) for row in rows).items())),
        "expert_calibration_accepted_rate": rate("expert_calibration_accepted"),
        "expert_route_exercise_accepted_rate": rate("expert_route_exercise_accepted"),
        "safe_capture_in_pursuit_rate": rate("safe_capture_in_pursuit"),
        "capture_event_rate": rate("capture_event"),
        "physical_collision_rate": rate("physical_collision"),
        "target_obstacle_collision_rate": rate("target_obstacle_collision"),
        "target_boundary_violation_rate": rate("target_boundary_violation"),
        "defender_boundary_violation_rate": rate("defender_boundary_violation"),
        "timeout_rate": rate("timeout"),
        "target_zone_entry_rate": rate("target_zone_entered"),
        "target_crossing_rate": rate("target_crossed"),
        "mean_min_clearance_m": float(np.mean([float(row["min_clearance_m"]) for row in rows])) if rows else 0.0,
        "mean_capture_time_seconds": float(np.mean([float(row["capture_time_seconds"]) for row in rows])) if rows else 0.0,
        "route_intent_counts": dict(sorted(Counter(str(row["route_intent"]) for row in rows).items())),
        "termination_reason_counts": dict(sorted(Counter(str(row["termination_reason"]) for row in rows).items())),
        "target_maneuver_fallback_rate": float(
            np.mean([int(row["target_maneuver_fallback_count"]) > 0 for row in rows])
        )
        if rows
        else 0.0,
        "latency_ms": {"route_and_safety": _percentile(latency_samples)},
    }


def main() -> None:
    args = parse_args()
    scenes = args.scenes.resolve()
    manifest = _guard_not_locked(scenes)
    records = _read_records(scenes, args.episodes, args.difficulty)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        rows.append(
            _rollout(
                record,
                args.environment_config.resolve(),
                not args.no_local_cbf,
                args.controller,
                args.safety_margin,
            )
        )
        if index % 10 == 0 or index == len(records):
            print(f"evaluated {index}/{len(records)}", flush=True)

    rows_by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_profile[str(row["difficulty"])].append(row)
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    accepted_indices = {int(row["episode_index"]) for row in rows if bool(row["expert_calibration_accepted"])}
    accepted_records = [record for record in records if int(record["episode_index"]) in accepted_indices]
    route_exercise_indices = {
        int(row["episode_index"])
        for row in rows
        if bool(row["expert_route_exercise_accepted"])
    }
    route_exercise_records = [record for record in records if int(record["episode_index"]) in route_exercise_indices]
    (output / "accepted_calibration_scenes.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in accepted_records),
        encoding="utf-8",
    )
    (output / "route_exercise_calibration_scenes.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in route_exercise_records),
        encoding="utf-8",
    )
    profile_summaries = {name: _summarize(profile_rows) for name, profile_rows in sorted(rows_by_profile.items())}
    summary = {
        "experiment_name": str(args.experiment_name),
        "evaluation_split": "development_calibration_only",
        "locked_test_used": False,
        "source_scene_manifest_sha256": hashlib.sha256((scenes.parent / "manifest.json").read_bytes()).hexdigest(),
        "source_scene_file_sha256": str(manifest.get("scene_file_sha256", "unknown")),
        "controller": f"{args.controller}_route_intent_v1",
        "controller_mode": str(args.controller),
        "safety_margin_override_m": args.safety_margin,
        "local_cbf_is_empirical_filter_only": not args.no_local_cbf,
        "formal_robust_cbf_qp_claim": False,
        "expert_acceptance_contract": {
            "safe_capture_success": True,
            "no_target_boundary_violation": True,
            "no_defender_boundary_violation": True,
            "no_defender_physical_collision": True,
        },
        "route_exercise_is_reported_separately": True,
        "raw_pool_episodes": len(rows),
        "accepted_calibration_episodes": len(accepted_records),
        "accepted_route_exercise_episodes": int(
            sum(bool(row["expert_route_exercise_accepted"]) for row in rows)
        ),
        "raw_pool_summary": _summarize(rows),
        "by_difficulty": profile_summaries,
        "accepted_scene_file": str((output / "accepted_calibration_scenes.jsonl").resolve()),
        "route_exercise_scene_file": str((output / "route_exercise_calibration_scenes.jsonl").resolve()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "scenes": str(scenes),
                "environment_config": str(args.environment_config.resolve()),
                "difficulty": args.difficulty,
                "episodes": len(records),
                "local_cbf": not args.no_local_cbf,
                "locked_test_used": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
