"""Evaluate the rule teacher on immutable Phase87-style records.

This development-only evaluator intentionally reuses the same environment
construction and metric schema as ``evaluate_mappo_ippo_phase87.py`` so a
teacher reference can distinguish imitation or PPO failure from an infeasible
validation scene.  ``--use-cbf`` denotes the same empirical safety filter
available to baseline policies; it is not a formal safety guarantee.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_controllers import DynamicEncirclementController, SafetyFilteredPursuitController  # noqa: E402
from train_phase79_dagger_residual import build_environment, load_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--use-cbf", action="store_true")
    return parser.parse_args()


def load_environment(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    env_path = Path(str(document["environment_config"]))
    if not env_path.is_absolute():
        env_path = path.parent / env_path
    environment = yaml.safe_load(env_path.resolve().read_text(encoding="utf-8"))

    def merge(base: dict[str, Any], update: dict[str, Any]) -> None:
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    merge(environment, document.get("environment_overrides", {}))
    return environment


def evaluate_episode(env: Any, observation: dict[str, Any], use_cbf: bool, max_steps: int) -> dict[str, Any]:
    controller: Any = DynamicEncirclementController(env)
    if use_cbf:
        controller = SafetyFilteredPursuitController(controller)
    final_info: dict[str, Any] = {}
    latencies: list[float] = []
    for _ in range(max_steps):
        start = time.perf_counter()
        action = controller.act(observation)
        latencies.append((time.perf_counter() - start) * 1000.0)
        observation, _reward, terminated, truncated, final_info = env.step(action)
        if terminated or truncated:
            break
    return {
        "safe_capture_success": bool(final_info.get("safe_capture_success", False)),
        "capture_event": bool(final_info.get("capture_event", False)),
        "collision": bool(final_info.get("collision", False)),
        "boundary_violation": bool(final_info.get("defender_boundary_violation", False)),
        "target_invalid_episode": bool(final_info.get("target_invalid_episode", False)),
        "target_boundary_violation": bool(final_info.get("target_boundary_violation", False)),
        "target_obstacle_violation": bool(final_info.get("target_obstacle_violation", False)),
        "timeout": bool(final_info.get("termination_reason") == "timeout"),
        "steps": int(env.step_count),
        "min_clearance_m": float(final_info.get("min_clearance_so_far", 0.0)),
        "termination_reason": str(final_info.get("termination_reason", "unknown")),
        "mean_policy_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    if args.start_index < 0 or args.episodes <= 0:
        raise ValueError("start-index must be non-negative and episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    records = load_records(args.scenes.resolve(), args.start_index + args.episodes, allow_target_crossing=True)[args.start_index : args.start_index + args.episodes]
    environment = load_environment(args.config)
    rows = []
    for record in records:
        env, observation, _scenario = build_environment(environment, record, max_steps=args.max_steps)
        rows.append({"episode_index": int(record["episode_index"]), "mirror_group_id": str(record["mirror_group_id"]), **evaluate_episode(env, observation, args.use_cbf, args.max_steps)})
    rate = lambda key: float(np.mean([bool(row[key]) for row in rows])) if rows else 0.0
    result = {
        "algorithm": "dynamic_encirclement_rule_teacher",
        "episodes": len(rows),
        "episode_start_index": args.start_index,
        "episode_end_index_exclusive": args.start_index + len(rows),
        "locked_test_used": False,
        "use_cbf": bool(args.use_cbf),
        "empirical_safety_filter_only": bool(args.use_cbf),
        "safe_capture_rate": rate("safe_capture_success"),
        "capture_event_rate": rate("capture_event"),
        "collision_rate": rate("collision"),
        "boundary_violation_rate": rate("boundary_violation"),
        "target_invalid_episode_rate": rate("target_invalid_episode"),
        "target_boundary_violation_rate": rate("target_boundary_violation"),
        "target_obstacle_violation_rate": rate("target_obstacle_violation"),
        "timeout_rate": rate("timeout"),
        "mean_min_clearance_m": float(np.mean([row["min_clearance_m"] for row in rows])),
        "mean_policy_latency_ms": float(np.mean([row["mean_policy_latency_ms"] for row in rows])),
        "rows": rows,
    }
    (args.output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
