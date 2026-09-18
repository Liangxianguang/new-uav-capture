"""Evaluate a MAPPO/IPPO checkpoint on immutable Phase87 scene records."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.learning import CentralizedSharedActorCritic, SharedActorCritic  # noqa: E402
from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.pursuit_controllers import PursuitCBFSafetyFilter  # noqa: E402
from encirclement3d.showcase import prepare_showcase_episode, scenario_from_metadata  # noqa: E402
from train_phase79_dagger_residual import build_environment, load_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scenes", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--algorithm", choices=("mappo", "ippo"), required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--start-index", type=int, default=0, help="Zero-based scene-record offset for reproducible validation chunks.")
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--use-cbf", action="store_true")
    return p.parse_args()


def load_environment(config_path: Path) -> dict[str, Any]:
    document = yaml.safe_load(config_path.resolve().read_text(encoding="utf-8"))
    env_path = Path(str(document["environment_config"]))
    if not env_path.is_absolute():
        env_path = config_path.parent / env_path
    environment = yaml.safe_load(env_path.resolve().read_text(encoding="utf-8"))

    def merge(base: dict[str, Any], update: dict[str, Any]) -> None:
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    merge(environment, document.get("environment_overrides", {}))
    return environment


def load_policy(path: Path, env: Any, observation: dict[str, Any], algorithm: str, dev: torch.device) -> tuple[torch.nn.Module, float]:
    checkpoint = torch.load(path.resolve(), map_location=dev, weights_only=True)
    local_dim = int(policy_observations(env, observation).shape[-1])
    state_dim = int(env.centralized_state().shape[-1])
    if int(checkpoint["local_observation_dim"]) != local_dim:
        raise ValueError(f"local dimension mismatch: checkpoint={checkpoint['local_observation_dim']} env={local_dim}")
    if int(checkpoint["centralized_state_dim"]) != state_dim:
        raise ValueError(f"centralized dimension mismatch: checkpoint={checkpoint['centralized_state_dim']} env={state_dim}")
    hidden = int(checkpoint.get("hidden_dim", 128))
    if algorithm == "mappo":
        policy = CentralizedSharedActorCritic(local_dim, state_dim, hidden_dim=hidden).to(dev)
    else:
        policy = SharedActorCritic(local_dim, hidden_dim=hidden).to(dev)
    policy.load_state_dict(checkpoint["state_dict"], strict=True)
    return policy.eval(), float(checkpoint["action_scale"])


def evaluate_episode(policy: torch.nn.Module, env: Any, observation: dict[str, Any], algorithm: str, dev: torch.device, action_scale: float, use_cbf: bool, max_steps: int) -> dict[str, Any]:
    safety = PursuitCBFSafetyFilter(env) if use_cbf else None
    final_info: dict[str, Any] = {}
    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(max_steps):
            local = torch.as_tensor(policy_observations(env, observation), device=dev)
            start = torch.cuda.Event(enable_timing=True) if dev.type == "cuda" else None
            if start is not None:
                end = torch.cuda.Event(enable_timing=True)
                start.record()
            else:
                import time
                clock = time.perf_counter()
            if algorithm == "mappo":
                distribution = policy.distribution(local)  # type: ignore[union-attr]
            else:
                distribution, _value = policy.distribution_and_value(local)  # type: ignore[union-attr]
            action = torch.tanh(distribution.mean).cpu().numpy() * action_scale
            if start is not None:
                end.record(); torch.cuda.synchronize(); latencies.append(float(start.elapsed_time(end)))
            else:
                latencies.append((time.perf_counter() - clock) * 1000.0)
            if safety is not None:
                action, _diagnostics = safety.filter(action, observation)
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
    a = parse_args()
    if a.output.exists() and any(a.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {a.output}")
    a.output.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    environment = load_environment(a.config)
    if a.start_index < 0 or a.episodes <= 0:
        raise ValueError("start-index must be non-negative and episodes must be positive")
    records = load_records(a.scenes.resolve(), a.start_index + a.episodes, allow_target_crossing=True)[a.start_index : a.start_index + a.episodes]
    first_env, first_observation, _ = build_environment(environment, records[0], max_steps=a.max_steps)
    policy, action_scale = load_policy(a.checkpoint, first_env, first_observation, a.algorithm, dev)
    rows = []
    for record in records:
        env, observation, _scenario = build_environment(environment, record, max_steps=a.max_steps)
        rows.append({"episode_index": int(record["episode_index"]), "mirror_group_id": str(record["mirror_group_id"]), **evaluate_episode(policy, env, observation, a.algorithm, dev, action_scale, a.use_cbf, a.max_steps)})
    rate = lambda key: float(np.mean([bool(row[key]) for row in rows])) if rows else 0.0
    result = {
        "algorithm": a.algorithm,
        "episodes": len(rows),
        "episode_start_index": int(a.start_index),
        "episode_end_index_exclusive": int(a.start_index + len(rows)),
        "locked_test_used": False,
        "use_cbf": bool(a.use_cbf),
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
    (a.output / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
