"""Evaluate a route-aware recurrent actor on a frozen development scene pool.

The evaluator is intentionally separate from the S4/MPC evaluator: a trained
low-level actor is rolled out directly, while route-intent features are
generated from delayed public beliefs and observed geometry.  It never reads
the simulator target state for control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.learning import RecurrentCentralizedSharedActorCritic  # noqa: E402
from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.pursuit_controllers import (  # noqa: E402
    PublicBeliefRouteIntentController,
    PursuitCBFSafetyFilter,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    capture_contract_metrics,
    crossing_metrics,
    prepare_showcase_episode,
    scenario_from_metadata,
)
from evaluate_s4_closed_loop import config_for_phase73_spec, read_scenes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--delay-steps", type=int, default=2)
    parser.add_argument("--noise-std-mps", type=float, default=0.04)
    parser.add_argument("--target-motion-mode", choices=("scene", "flee_persistence", "s_curve", "adaptive_maneuvering"), default="scene")
    parser.add_argument("--use-local-cbf", action="store_true")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else name)


def percentile(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {f"p{int(p)}": float(np.percentile(array, p)) for p in (50, 95, 99)}


def main() -> None:
    args = parse_args()
    if args.delay_steps < 0 or args.noise_std_mps < 0.0:
        raise ValueError("delay-steps and noise-std-mps must be non-negative.")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not bool(checkpoint.get("actor_recurrent", False)):
        raise ValueError("The checkpoint must contain a recurrent actor.")
    device = select_device(args.device)
    records = read_scenes(args.scenes, args.episodes)
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict[str, Any]] = []
    total_latencies: list[float] = []
    actor_latencies: list[float] = []
    route_latencies: list[float] = []
    safety_latencies: list[float] = []
    policy: RecurrentCentralizedSharedActorCritic | None = None
    with output.joinpath("episodes.jsonl").open("w", encoding="utf-8") as episode_file:
        for record in records:
            spec = dict(record)
            config = config_for_phase73_spec(args.environment_config, spec, None)
            if args.target_motion_mode != "scene":
                config["task"]["pursuit"]["target_motion_mode"] = args.target_motion_mode
            execution = config.setdefault("dynamics", {}).setdefault("execution", {})
            execution.update(
                {
                    "enabled": True,
                    "action_delay_steps": int(args.delay_steps),
                    "command_noise_std": float(args.noise_std_mps),
                }
            )
            scenario = scenario_from_metadata(spec["scenario"])
            env = CaptureRadiusPursuit3DEnv(
                config,
                obstacle_count=int(spec.get("obstacle_count", len(scenario.obstacles))),
                target_speed_scale=float(spec["target_speed_scale"]),
            )
            observation = prepare_showcase_episode(env, scenario, int(spec["episode_seed"]), record_history=True, validate_scenario=False)
            route_helper = PublicBeliefRouteIntentController(env)
            route_hint = route_helper.route_features(observation)
            local = np.concatenate([policy_observations(env, observation), route_hint], axis=1).astype(np.float32)
            if policy is None:
                policy = RecurrentCentralizedSharedActorCritic(
                    local_observation_dim=int(local.shape[-1]),
                    centralized_state_dim=int(checkpoint["centralized_state_dim"]),
                    hidden_dim=int(checkpoint["recurrent_hidden_dim"]),
                ).to(device)
                if int(checkpoint["local_observation_dim"]) != int(local.shape[-1]):
                    raise ValueError("Checkpoint and route-aware evaluation observation dimensions differ.")
                policy.load_state_dict(checkpoint["state_dict"], strict=True)
                policy.eval()
            hidden = policy.initial_actor_hidden(env.n_defenders, device=device)
            safety_filter = PursuitCBFSafetyFilter(env) if args.use_local_cbf else None
            while True:
                route_start = time.perf_counter()
                route_hint = route_helper.route_features(observation)
                route_ms = (time.perf_counter() - route_start) * 1000.0
                local = np.concatenate([policy_observations(env, observation), route_hint], axis=1).astype(np.float32)
                actor_start = time.perf_counter()
                with torch.no_grad():
                    distribution, hidden = policy.distribution_step(torch.as_tensor(local, device=device), hidden)
                    action = torch.tanh(distribution.mean).cpu().numpy() * float(checkpoint["action_scale"])
                actor_ms = (time.perf_counter() - actor_start) * 1000.0
                safety_ms = 0.0
                if safety_filter is not None:
                    safety_start = time.perf_counter()
                    action, _diagnostics = safety_filter.filter(action, observation)
                    safety_ms = (time.perf_counter() - safety_start) * 1000.0
                total_ms = route_ms + actor_ms + safety_ms
                route_latencies.append(route_ms)
                actor_latencies.append(actor_ms)
                safety_latencies.append(safety_ms)
                total_latencies.append(total_ms)
                observation, _reward, terminated, truncated, info = env.step(action, record_history=True)
                if terminated or truncated:
                    crossing = crossing_metrics(env, scenario.obstacle_zone_x)
                    contract = capture_contract_metrics(
                        info,
                        crossing,
                        target_crossing_required=bool(spec.get("target_crossing_required", False)),
                        required_defender_zone_entries=int(spec.get("scenario", {}).get("required_defender_zone_entries", 1)),
                        require_target_zone_entry=bool(spec.get("target_crossing_required", False)),
                    )
                    row = {
                        "episode_index": int(spec["episode_index"]),
                        "episode_seed": int(spec["episode_seed"]),
                        "mirror_group_id": str(spec.get("mirror_group_id", "unknown")),
                        "observation_condition": str(spec.get("observation_condition", "unknown")),
                        "safe_capture_success": bool(contract["safe_capture_in_pursuit"]),
                        "capture_event": bool(info["capture_event"]),
                        "collision": bool(info["collision"]),
                        "boundary_violation": bool(info["world_violation_steps"] > 0),
                        "timeout": bool(info["termination_reason"] == "timeout"),
                        "termination_reason": str(info["termination_reason"]),
                        "steps": int(env.step_count),
                        "capture_time_seconds": float(info["capture_time_seconds"])
                        if info["capture_time_seconds"] is not None
                        else float(env.max_steps * env.dt),
                        "min_clearance_m": float(info["min_clearance_so_far"]),
                        "target_crossed": bool(crossing["target_crossed"]),
                        "defender_zone_entry_count": int(crossing["defender_zone_entry_count"]),
                        "route_intent": str(route_helper.route_name),
                        "action_delay_steps": int(args.delay_steps),
                        "command_noise_std_mps": float(args.noise_std_mps),
                        "local_cbf_empirical_filter": bool(args.use_local_cbf),
                        **contract,
                    }
                    episode_rows.append(row)
                    episode_file.write(json.dumps(row, allow_nan=False) + "\n")
                    break

    def rate(key: str) -> float:
        return float(np.mean([bool(row[key]) for row in episode_rows])) if episode_rows else 0.0

    mirror_groups = sorted({str(row["mirror_group_id"]) for row in episode_rows})
    group_safe: list[float] = []
    for group in mirror_groups:
        members = [row for row in episode_rows if str(row["mirror_group_id"]) == group]
        group_safe.append(float(np.mean([bool(row["safe_capture_success"]) for row in members])))
    summary = {
        "experiment_name": "phase76_route_aware_policy_development_validation",
        "evaluation_split": "development_validation_only",
        "locked_test_used": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "episodes": len(episode_rows),
        "mirror_groups": len(mirror_groups),
        "safe_capture_rate": rate("safe_capture_success"),
        "capture_event_rate": rate("capture_event"),
        "collision_rate": rate("collision"),
        "boundary_violation_rate": rate("boundary_violation"),
        "timeout_rate": rate("timeout"),
        "target_crossing_rate": rate("target_crossed"),
        "mean_capture_time_seconds": float(np.mean([row["capture_time_seconds"] for row in episode_rows])) if episode_rows else 0.0,
        "mean_min_clearance_m": float(np.mean([row["min_clearance_m"] for row in episode_rows])) if episode_rows else 0.0,
        "paired_mirror_group_safe_capture_mean": float(np.mean(group_safe)) if group_safe else 0.0,
        "latency_ms": {
            "route_intent": percentile(route_latencies),
            "actor": percentile(actor_latencies),
            "safety": percentile(safety_latencies),
            "total": percentile(total_latencies),
        },
        "local_cbf_is_empirical_filter_only": bool(args.use_local_cbf),
        "formal_robust_cbf_qp_claim": False,
    }
    output.joinpath("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    output.joinpath("config.yaml").write_text(
        yaml.safe_dump(
            {
                "scenes": str(args.scenes.resolve()),
                "environment_config": str(args.environment_config.resolve()),
                "checkpoint": str(checkpoint_path),
                "episodes": len(episode_rows),
                "delay_steps": int(args.delay_steps),
                "noise_std_mps": float(args.noise_std_mps),
                "use_local_cbf": bool(args.use_local_cbf),
                "locked_test_used": False,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
