"""Audit robust safety under delayed, noisy and imperfect execution dynamics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if os.name == "nt":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - environment-dependent optional package
    SummaryWriter = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_controllers import DynamicEncirclementController, PursuitCBFSafetyFilter  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.safety_certificate import (  # noqa: E402
    check_execution_continuous_segment_safety,
    check_execution_rollout_safety,
    check_execution_swept_volume_safety,
    check_one_step_safety,
)
from encirclement3d.safety_qp import RobustCBFQPConfig, RobustCBFQPFilter  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "innovation_safety_execution.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=("nominal", "local_cbf", "robust_cbf_qp"))
    parser.add_argument(
        "--linearization-backend",
        choices=("analytic", "finite_difference"),
        help="Override the execution-aware safety linearization backend.",
    )
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return value


def require_summary_writer() -> Any:
    if SummaryWriter is None:
        raise RuntimeError("TensorBoard logging requires the 'tensorboard' package.")
    return SummaryWriter


def source_hashes(config_path: Path) -> dict[str, str]:
    paths = (
        config_path.resolve(),
        PROJECT_ROOT / "scripts" / "evaluate_safety_execution.py",
        PROJECT_ROOT / "scripts" / "evaluate_safety_qp.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_env.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "execution_dynamics.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "safety_qp.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "safety_certificate.py",
    )
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def certificate_observation(observation: dict[str, Any], env: CaptureRadiusPursuit3DEnv) -> dict[str, Any]:
    value = dict(observation)
    value["world_lower_bounds"] = env.lower.copy()
    value["world_upper_bounds"] = env.upper.copy()
    value["obstacles"] = list(getattr(env, "obstacles", ()))
    return value


def apply_execution_variant(config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    updated.setdefault("dynamics", {})["execution"] = copy.deepcopy(variant)
    return updated


def audit_initial_seeds(
    config: dict[str, Any],
    *,
    seeds: list[int],
    obstacle_count: int,
    target_speed_scale: float,
    qp_config: RobustCBFQPConfig,
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for seed in seeds:
        env = CaptureRadiusPursuit3DEnv(copy.deepcopy(config), obstacle_count=obstacle_count, target_speed_scale=target_speed_scale)
        observation = env.reset(seed=int(seed))
        certificate = check_one_step_safety(
            certificate_observation(observation, env),
            np.zeros((env.n_defenders, 3), dtype=np.float64),
            dt=float(env.dt),
            drone_radius=float(env.agents["drone_radius"]),
            max_speed_mps=float(env.agents["defender_max_speed"]),
            max_acceleration_mps2=float(env.agents["defender_max_acceleration"]),
            safety_margin_m=float(env.pursuit["safety_margin"]),
            robust_margin_m=float(qp_config.robust_margin_m),
            action_change_limit_mps=qp_config.action_change_limit_mps,
            enforce_action_change=True,
        )
        audits.append(
            {
                "seed": int(seed),
                "initial_min_barrier_m": float(certificate.current_min_barrier_m),
                "current_state_safe": bool(certificate.current_state_safe),
                "violations": list(certificate.violations),
            }
        )
    invalid = [item for item in audits if not item["current_state_safe"]]
    if invalid:
        raise RuntimeError(f"Fixed robust-safe seeds are not valid under the reset contract: {invalid}")
    return audits


def _finite_mean(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.mean(finite)) if finite.size else float("nan")


def _percentile(values: list[float], quantile: float) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.percentile(finite, quantile)) if finite.size else float("nan")


def run_episode(
    config: dict[str, Any],
    *,
    seed: int,
    method: str,
    obstacle_count: int,
    target_speed_scale: float,
    max_steps: int,
    swept_volume_subdivisions_per_step: int,
    qp_config: RobustCBFQPConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=obstacle_count, target_speed_scale=target_speed_scale)
    observation = env.reset(seed=int(seed))
    controller = DynamicEncirclementController(env)
    local_filter = PursuitCBFSafetyFilter(env)
    robust_filter = RobustCBFQPFilter(env, qp_config)
    command_valid = 0
    command_execution_valid = 0
    swept_volume_valid = 0
    continuous_segment_valid = 0
    executed_valid = 0
    command_next_safe = 0
    executed_next_safe = 0
    current_safe = 0
    actual_robust_current_safe = 0
    command_barriers: list[float] = []
    command_execution_barriers: list[float] = []
    swept_volume_barriers: list[float] = []
    continuous_segment_barriers: list[float] = []
    executed_barriers: list[float] = []
    execution_errors: list[float] = []
    safety_latencies: list[float] = []
    step_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    while True:
        desired = np.asarray(controller.act(observation), dtype=np.float64)
        filter_started = time.perf_counter()
        diagnostics: Any = None
        if method == "nominal":
            action = desired
        elif method == "local_cbf":
            action, diagnostics = local_filter.filter(desired, observation)
        else:
            action, diagnostics = robust_filter.filter(desired, observation)
        filter_latency_ms = (time.perf_counter() - filter_started) * 1000.0
        command_authority = (
            None
            if diagnostics is None
            else dict(getattr(diagnostics, "command_authority", None) or {})
        )
        pre_step = observation
        command_certificate = check_one_step_safety(
            certificate_observation(pre_step, env),
            action,
            dt=float(env.dt),
            drone_radius=float(env.agents["drone_radius"]),
            max_speed_mps=float(env.agents["defender_max_speed"]),
            max_acceleration_mps2=float(env.agents["defender_max_acceleration"]),
            safety_margin_m=float(env.pursuit["safety_margin"]),
            robust_margin_m=float(qp_config.robust_margin_m),
            action_change_limit_mps=qp_config.action_change_limit_mps,
            enforce_action_change=True,
        )
        execution_certificate = check_execution_rollout_safety(
            certificate_observation(pre_step, env),
            action,
            dt=float(env.dt),
            drone_radius=float(env.agents["drone_radius"]),
            max_speed_mps=float(env.agents["defender_max_speed"]),
            max_acceleration_mps2=float(env.agents["defender_max_acceleration"]),
            safety_margin_m=float(env.pursuit["safety_margin"]),
            robust_margin_m=float(qp_config.robust_margin_m),
            action_change_limit_mps=qp_config.action_change_limit_mps,
            horizon_steps=qp_config.execution_preview_horizon_steps,
            command_authority=command_authority,
        )
        swept_certificate = check_execution_swept_volume_safety(
            certificate_observation(pre_step, env),
            action,
            dt=float(env.dt),
            drone_radius=float(env.agents["drone_radius"]),
            safety_margin_m=float(env.pursuit["safety_margin"]),
            robust_margin_m=float(qp_config.robust_margin_m),
            horizon_steps=qp_config.execution_preview_horizon_steps,
            subdivisions_per_step=int(swept_volume_subdivisions_per_step),
            command_authority=command_authority,
        )
        continuous_segment_certificate = check_execution_continuous_segment_safety(
            certificate_observation(pre_step, env),
            action,
            dt=float(env.dt),
            drone_radius=float(env.agents["drone_radius"]),
            safety_margin_m=float(env.pursuit["safety_margin"]),
            robust_margin_m=float(qp_config.robust_margin_m),
            horizon_steps=qp_config.execution_preview_horizon_steps,
            command_authority=command_authority,
        )
        observation, _reward, terminated, truncated, info = env.step(
            action,
            command_authority=command_authority,
        )
        executed_action = np.asarray(env.last_executed_actions, dtype=np.float64).copy()
        executed_certificate = check_one_step_safety(
            certificate_observation(pre_step, env),
            executed_action,
            dt=float(env.dt),
            drone_radius=float(env.agents["drone_radius"]),
            max_speed_mps=float(env.agents["defender_max_speed"]),
            max_acceleration_mps2=float(env.agents["defender_max_acceleration"]),
            safety_margin_m=float(env.pursuit["safety_margin"]),
            robust_margin_m=float(qp_config.robust_margin_m),
            action_change_limit_mps=qp_config.action_change_limit_mps,
            enforce_action_change=True,
        )
        # The pre-step certificate predicts the nominal velocity-level next
        # state.  Check the actual post-step state independently as well,
        # because delay, noise, tracking, clipping, and randomized dynamics
        # can make it differ from that prediction.
        actual_post_certificate = check_one_step_safety(
            certificate_observation(observation, env),
            np.asarray(observation["defender_velocities"], dtype=np.float64),
            dt=float(env.dt),
            drone_radius=float(env.agents["drone_radius"]),
            max_speed_mps=float(env.agents["defender_max_speed"]),
            max_acceleration_mps2=float(env.agents["defender_max_acceleration"]),
            safety_margin_m=float(env.pursuit["safety_margin"]),
            robust_margin_m=float(qp_config.robust_margin_m),
            action_change_limit_mps=qp_config.action_change_limit_mps,
            enforce_action_change=False,
        )
        actual_post_robust_certificate = check_execution_rollout_safety(
            certificate_observation(observation, env),
            np.zeros_like(executed_action),
            dt=float(env.dt),
            drone_radius=float(env.agents["drone_radius"]),
            max_speed_mps=float(env.agents["defender_max_speed"]),
            max_acceleration_mps2=float(env.agents["defender_max_acceleration"]),
            safety_margin_m=float(env.pursuit["safety_margin"]),
            robust_margin_m=float(qp_config.robust_margin_m),
            action_change_limit_mps=qp_config.action_change_limit_mps,
            horizon_steps=1,
        )
        command_valid += int(command_certificate.valid)
        command_execution_valid += int(execution_certificate.valid)
        swept_volume_valid += int(swept_certificate.valid)
        continuous_segment_valid += int(continuous_segment_certificate.valid)
        executed_valid += int(executed_certificate.valid)
        command_next_safe += int(command_certificate.next_state_safe)
        executed_next_safe += int(executed_certificate.next_state_safe)
        current_safe += int(actual_post_certificate.current_state_safe)
        actual_robust_current_safe += int(actual_post_robust_certificate.current_state_safe)
        command_barriers.append(float(command_certificate.next_min_barrier_m))
        command_execution_barriers.append(float(execution_certificate.minimum_robust_barrier_m))
        swept_volume_barriers.append(float(swept_certificate.minimum_robust_barrier_m))
        continuous_segment_barriers.append(float(continuous_segment_certificate.minimum_robust_barrier_m))
        executed_barriers.append(float(executed_certificate.next_min_barrier_m))
        error_norm = float(np.mean(np.linalg.norm(executed_action - action, axis=1)))
        execution_errors.append(error_norm)
        safety_latencies.append(filter_latency_ms)
        step_rows.append(
            {
                "step": int(env.step_count),
                "command_certificate_valid": bool(command_certificate.valid),
                "command_next_state_safe": bool(command_certificate.next_state_safe),
                "command_next_min_barrier_m": float(command_certificate.next_min_barrier_m),
                "execution_rollout_certificate_valid": bool(execution_certificate.valid),
                "execution_rollout_state_safe": bool(execution_certificate.rollout_state_safe),
                "execution_rollout_min_robust_barrier_m": float(execution_certificate.minimum_robust_barrier_m),
                "execution_rollout_min_nominal_barrier_m": float(execution_certificate.minimum_nominal_barrier_m),
                "execution_rollout_horizon_steps": int(execution_certificate.horizon_steps),
                "execution_rollout_violations": list(execution_certificate.violations),
                "swept_volume_certificate_valid": bool(swept_certificate.valid),
                "swept_volume_safe": bool(swept_certificate.swept_volume_safe),
                "swept_volume_min_robust_barrier_m": float(swept_certificate.minimum_robust_barrier_m),
                "swept_volume_min_nominal_barrier_m": float(swept_certificate.minimum_nominal_barrier_m),
                "swept_volume_sample_count": int(swept_certificate.sample_count),
                "swept_volume_violations": list(swept_certificate.violations),
                "continuous_segment_certificate_valid": bool(continuous_segment_certificate.valid),
                "continuous_segment_safe": bool(continuous_segment_certificate.continuous_segment_safe),
                "continuous_segment_min_robust_barrier_m": float(
                    continuous_segment_certificate.minimum_robust_barrier_m
                ),
                "continuous_segment_min_nominal_barrier_m": float(
                    continuous_segment_certificate.minimum_nominal_barrier_m
                ),
                "continuous_segment_violations": list(continuous_segment_certificate.violations),
                "executed_certificate_valid": bool(executed_certificate.valid),
                "executed_current_state_safe": bool(executed_certificate.current_state_safe),
                "executed_next_state_safe": bool(executed_certificate.next_state_safe),
                "executed_next_min_barrier_m": float(executed_certificate.next_min_barrier_m),
                "executed_certificate_violations": list(executed_certificate.violations),
                "actual_post_state_safe": bool(actual_post_certificate.current_state_safe),
                "actual_post_min_barrier_m": float(actual_post_certificate.current_min_barrier_m),
                "actual_post_state_violations": list(actual_post_certificate.violations),
                "actual_post_robust_state_safe": bool(actual_post_robust_certificate.current_state_safe),
                "actual_post_robust_min_barrier_m": float(actual_post_robust_certificate.minimum_robust_barrier_m),
                "actual_post_robust_violations": list(actual_post_robust_certificate.violations),
                "command_action_norm_mps": float(np.max(np.linalg.norm(action, axis=1))),
                "executed_action_norm_mps": float(np.max(np.linalg.norm(executed_action, axis=1))),
                "action_execution_error_norm_mps": error_norm,
                "safety_latency_ms": filter_latency_ms,
                "solver_status": None if diagnostics is None else str(getattr(diagnostics, "status", "unknown")),
                "solver_success": None if diagnostics is None else bool(getattr(diagnostics, "solver_success", False)),
                "fallback_used": None if diagnostics is None else bool(getattr(diagnostics, "fallback_used", False)),
                "certificate_valid": None if diagnostics is None else bool(getattr(diagnostics, "certificate_valid", False)),
                "fallback_certificate_valid": None
                if diagnostics is None
                else bool(
                    getattr(diagnostics, "fallback_used", False)
                    and getattr(diagnostics, "certificate_valid", False)
                ),
                "failure_category": None if diagnostics is None else str(getattr(diagnostics, "failure_category", "none")),
                "solver_backend": None if diagnostics is None else str(getattr(diagnostics, "solver_backend", "none")),
                "linearization_iterations": None if diagnostics is None else int(getattr(diagnostics, "linearization_iterations", 0)),
                "linearization_evaluations": None if diagnostics is None else int(getattr(diagnostics, "linearization_evaluations", 0)),
                "linearization_active_constraints": None
                if diagnostics is None
                else int(getattr(diagnostics, "linearization_active_constraints", 0)),
                "emergency_brake_requested": bool(info.get("emergency_brake_requested", False)),
                "queue_override_slots": int(info.get("queue_override_slots", 0)),
                "command_authority_mode": str(info.get("command_authority_mode", "immutable")),
            }
        )
        if terminated or truncated or env.step_count >= max_steps:
            break
    return (
        {
            "seed": int(seed),
            "method": method,
            "steps": len(step_rows),
            "safe_capture_success": bool(info.get("safe_capture_success", False)),
            "capture_event": bool(info.get("capture_event", False)),
            "collision": bool(info.get("collision", False)),
            "boundary_violation": bool(info.get("world_violation_steps", 0) > 0),
            "timeout": str(info.get("termination_reason")) == "timeout",
            "termination_reason": str(info.get("termination_reason")),
            "capture_time_seconds": info.get("capture_time_seconds"),
            "min_clearance_m": float(info.get("min_clearance_so_far", info.get("min_clearance", np.nan))),
            "command_certificate_valid_rate": float(command_valid / max(len(step_rows), 1)),
            "execution_rollout_certificate_valid_rate": float(command_execution_valid / max(len(step_rows), 1)),
            "swept_volume_certificate_valid_rate": float(swept_volume_valid / max(len(step_rows), 1)),
            "continuous_segment_certificate_valid_rate": float(
                continuous_segment_valid / max(len(step_rows), 1)
            ),
            "executed_certificate_valid_rate": float(executed_valid / max(len(step_rows), 1)),
            "command_next_state_safe_rate": float(command_next_safe / max(len(step_rows), 1)),
            "executed_next_state_safe_rate": float(executed_next_safe / max(len(step_rows), 1)),
            "executed_current_state_safe_rate": float(current_safe / max(len(step_rows), 1)),
            "actual_post_state_safe_rate": float(current_safe / max(len(step_rows), 1)),
            "actual_post_robust_state_safe_rate": float(actual_robust_current_safe / max(len(step_rows), 1)),
            "mean_action_execution_error_norm_mps": _finite_mean(execution_errors),
            "safety_latency_p95_ms": _percentile(safety_latencies, 95.0),
            "minimum_command_next_barrier_m": float(np.min(command_barriers, initial=np.inf)),
            "minimum_execution_rollout_robust_barrier_m": float(np.min(command_execution_barriers, initial=np.inf)),
            "minimum_swept_volume_robust_barrier_m": float(np.min(swept_volume_barriers, initial=np.inf)),
            "minimum_continuous_segment_robust_barrier_m": float(
                np.min(continuous_segment_barriers, initial=np.inf)
            ),
            "minimum_executed_next_barrier_m": float(np.min(executed_barriers, initial=np.inf)),
            "minimum_actual_post_barrier_m": float(
                min((float(row["actual_post_min_barrier_m"]) for row in step_rows), default=float("inf"))
            ),
            "solver_fallback_count": int(sum(bool(row["fallback_used"]) for row in step_rows if row["fallback_used"] is not None)),
            "elapsed_seconds": float(time.perf_counter() - started),
        },
        step_rows,
    )


def summarize(rows: list[dict[str, Any]], steps: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(key: str) -> float:
        return float(np.mean([bool(row[key]) for row in rows]))

    backend_counts: dict[str, int] = {}
    for row in steps:
        backend = str(row.get("solver_backend", "none"))
        backend_counts[backend] = backend_counts.get(backend, 0) + 1

    return {
        "episodes": len(rows),
        "safe_capture_rate": rate("safe_capture_success"),
        "capture_rate": rate("capture_event"),
        "collision_rate": rate("collision"),
        "boundary_violation_rate": rate("boundary_violation"),
        "timeout_rate": rate("timeout"),
        "mean_capture_time_seconds": _finite_mean([float(row["capture_time_seconds"]) for row in rows if row["capture_time_seconds"] is not None]),
        "mean_min_clearance_m": _finite_mean([float(row["min_clearance_m"]) for row in rows]),
        "command_certificate_valid_rate": _finite_mean([float(row["command_certificate_valid_rate"]) for row in rows]),
        "execution_rollout_certificate_valid_rate": _finite_mean(
            [float(row["execution_rollout_certificate_valid_rate"]) for row in rows]
        ),
        "swept_volume_certificate_valid_rate": _finite_mean(
            [float(row["swept_volume_certificate_valid_rate"]) for row in rows]
        ),
        "continuous_segment_certificate_valid_rate": _finite_mean(
            [float(row["continuous_segment_certificate_valid_rate"]) for row in rows]
        ),
        "executed_certificate_valid_rate": _finite_mean([float(row["executed_certificate_valid_rate"]) for row in rows]),
        "command_next_state_safe_rate": _finite_mean([float(row["command_next_state_safe_rate"]) for row in rows]),
        "executed_next_state_safe_rate": _finite_mean([float(row["executed_next_state_safe_rate"]) for row in rows]),
        "executed_current_state_safe_rate": _finite_mean([float(row["executed_current_state_safe_rate"]) for row in rows]),
        "actual_post_state_safe_rate": _finite_mean([float(row["actual_post_state_safe_rate"]) for row in rows]),
        "actual_post_robust_state_safe_rate": _finite_mean(
            [float(row["actual_post_robust_state_safe_rate"]) for row in rows]
        ),
        "mean_action_execution_error_norm_mps": _finite_mean([float(row["mean_action_execution_error_norm_mps"]) for row in rows]),
        "safety_latency_ms": {
            "p50": _percentile([float(row["safety_latency_ms"]) for row in steps], 50),
            "p95": _percentile([float(row["safety_latency_ms"]) for row in steps], 95),
        },
        "minimum_command_next_barrier_m": float(np.min([row["minimum_command_next_barrier_m"] for row in rows])),
        "minimum_execution_rollout_robust_barrier_m": float(
            np.min([row["minimum_execution_rollout_robust_barrier_m"] for row in rows])
        ),
        "minimum_swept_volume_robust_barrier_m": float(
            np.min([row["minimum_swept_volume_robust_barrier_m"] for row in rows])
        ),
        "minimum_continuous_segment_robust_barrier_m": float(
            np.min([row["minimum_continuous_segment_robust_barrier_m"] for row in rows])
        ),
        "minimum_executed_next_barrier_m": float(np.min([row["minimum_executed_next_barrier_m"] for row in rows])),
        "minimum_actual_post_barrier_m": float(np.min([row["minimum_actual_post_barrier_m"] for row in rows])),
        "solver_fallback_count": int(sum(row["solver_fallback_count"] for row in rows)),
        "fallback_certificate_valid_rate": float(
            np.mean(
                [
                    bool(row["fallback_certificate_valid"])
                    for row in steps
                    if row.get("fallback_used") is True
                ]
            )
            if any(row.get("fallback_used") is True for row in steps)
            else 1.0
        ),
        "emergency_brake_count": int(sum(bool(row.get("emergency_brake_requested", False)) for row in steps)),
        "queue_override_slots": int(sum(int(row.get("queue_override_slots", 0)) for row in steps)),
        "mean_linearization_iterations": _finite_mean(
            [float(row["linearization_iterations"]) for row in steps if row.get("linearization_iterations") is not None]
        ),
        "mean_linearization_evaluations": _finite_mean(
            [float(row["linearization_evaluations"]) for row in steps if row.get("linearization_evaluations") is not None]
        ),
        "mean_linearization_active_constraints": _finite_mean(
            [
                float(row["linearization_active_constraints"])
                for row in steps
                if row.get("linearization_active_constraints") is not None
            ]
        ),
        "solver_backend_counts": backend_counts,
        "analytic_backend_step_rate": float(
            backend_counts.get("analytic_rollout_jacobian", 0) / max(len(steps), 1)
        ),
        "multi_step_step_count": len(steps),
    }


def log_tensorboard(
    output: Path,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    with require_summary_writer()(log_dir=str(output / "tensorboard"), flush_secs=5) as writer:
        writer.add_text("Evaluation/config", yaml.safe_dump(config, sort_keys=False), 0)
        writer.add_text("Evaluation/source_hashes", json.dumps(config["source_hashes"], indent=2), 0)
        for index, row in enumerate(rows):
            for key, value in row.items():
                if isinstance(value, (int, float, bool)) and np.isfinite(float(value)):
                    writer.add_scalar(f"Episode/{key}", float(value), index)
        for index, row in enumerate(steps):
            for key, value in row.items():
                if isinstance(value, (int, float, bool)) and np.isfinite(float(value)):
                    writer.add_scalar(f"Step/{key}", float(value), index)
        for key, value in summary.items():
            if isinstance(value, (int, float, bool)) and np.isfinite(float(value)):
                writer.add_scalar(f"Summary/{key}", float(value), 0)
        writer.add_hparams(
            {
                "method": str(config["evaluation"]["method"]),
                "execution_variant": str(config["execution_variant"]),
                "action_delay_steps": int(config["execution"]["action_delay_steps"]),
                "pending_command_authority": str(config["execution"].get("pending_command_authority", "immutable")),
                "command_noise_std": float(config["execution"]["command_noise_std"]),
                "velocity_time_constant_seconds": float(config["execution"]["velocity_time_constant_seconds"]),
                "execution_linearization_backend": str(
                    config["safety"].get("execution_linearization_backend", "analytic")
                ),
                "execution_linearization_active_margin_m": float(
                    config["safety"].get("execution_linearization_active_margin_m", 0.0)
                ),
                "execution_linearization_iterations": int(
                    config["safety"].get("execution_linearization_iterations", 0)
                ),
                "execution_projection_iterations": int(
                    config["safety"].get("execution_projection_iterations", 0)
                ),
                "execution_emergency_brake_enabled": bool(
                    config["safety"].get("execution_emergency_brake_enabled", False)
                ),
            },
            {
                "hparam/safe_capture_rate": float(summary["safe_capture_rate"]),
                "hparam/executed_next_state_safe_rate": float(summary["executed_next_state_safe_rate"]),
                "hparam/executed_certificate_valid_rate": float(summary["executed_certificate_valid_rate"]),
                "hparam/execution_rollout_certificate_valid_rate": float(
                    summary["execution_rollout_certificate_valid_rate"]
                ),
                "hparam/swept_volume_certificate_valid_rate": float(
                    summary["swept_volume_certificate_valid_rate"]
                ),
                "hparam/actual_post_robust_state_safe_rate": float(
                    summary["actual_post_robust_state_safe_rate"]
                ),
                "hparam/analytic_backend_step_rate": float(summary["analytic_backend_step_rate"]),
                "hparam/mean_linearization_active_constraints": float(
                    summary["mean_linearization_active_constraints"]
                ),
                "hparam/fallback_certificate_valid_rate": float(summary["fallback_certificate_valid_rate"]),
            },
        )


def main() -> None:
    args = parse_args()
    document = load_yaml(args.config)
    environment_path = (args.config.parent / str(document["environment_config"])).resolve()
    base_config = load_yaml(environment_path)
    evaluation = dict(document["evaluation"])
    safety_mapping = dict(document["safety"])
    if args.linearization_backend is not None:
        safety_mapping["execution_linearization_backend"] = str(args.linearization_backend)
    env_probe = CaptureRadiusPursuit3DEnv(copy.deepcopy(base_config), obstacle_count=int(evaluation["obstacle_count"]), target_speed_scale=float(evaluation["target_speed_scale"]))
    safety_mapping.setdefault("max_speed_mps", float(env_probe.agents["defender_max_speed"]))
    safety_mapping.setdefault("max_acceleration_mps2", float(env_probe.agents["defender_max_acceleration"]))
    safety_mapping.setdefault("safety_margin_m", float(env_probe.pursuit["safety_margin"]))
    qp_config = RobustCBFQPConfig.from_mapping(safety_mapping)
    seed_protocol = dict(evaluation["robust_safe_seed_protocol"])
    seeds = [int(seed) for seed in seed_protocol["episode_seeds"]]
    if int(evaluation["episodes"]) != len(seeds):
        raise ValueError("evaluation.episodes must equal robust_safe_seed_protocol.episode_seeds length")
    variants = dict(document["variants"])
    selected_variants = list(args.variants or variants)
    methods = list(args.methods or evaluation["methods"])
    max_steps = int(args.max_steps or evaluation["max_steps"])
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    root_config = {
        "experiment_name": document.get("experiment_name"),
        "config": str(args.config.resolve()),
        "environment_config": str(environment_path),
        "safety": safety_mapping,
        "evaluation": evaluation,
        "selected_variants": selected_variants,
        "methods": methods,
        "source_hashes": source_hashes(args.config),
    }
    output.joinpath("config.yaml").write_text(yaml.safe_dump(root_config, sort_keys=False), encoding="utf-8")
    all_summaries: dict[str, Any] = {}
    for variant_name in selected_variants:
        if variant_name not in variants:
            raise ValueError(f"Unknown execution variant: {variant_name}")
        variant = dict(variants[variant_name])
        variant_config = apply_execution_variant(base_config, variant)
        variant_config.setdefault("world", {})["max_steps"] = max_steps
        audits = audit_initial_seeds(
            variant_config,
            seeds=seeds,
            obstacle_count=int(evaluation["obstacle_count"]),
            target_speed_scale=float(evaluation["target_speed_scale"]),
            qp_config=qp_config,
        )
        variant_output = output / variant_name
        variant_output.mkdir(parents=True, exist_ok=True)
        variant_summaries: dict[str, Any] = {}
        for method in methods:
            method_output = variant_output / method
            method_output.mkdir(parents=True, exist_ok=True)
            rows: list[dict[str, Any]] = []
            steps: list[dict[str, Any]] = []
            for episode_index, seed in enumerate(seeds):
                row, episode_steps = run_episode(
                    variant_config,
                    seed=seed,
                    method=method,
                    obstacle_count=int(evaluation["obstacle_count"]),
                    target_speed_scale=float(evaluation["target_speed_scale"]),
                    max_steps=max_steps,
                    swept_volume_subdivisions_per_step=int(
                        evaluation.get("swept_volume_subdivisions_per_step", 4)
                    ),
                    qp_config=qp_config,
                )
                row["episode_index"] = episode_index
                rows.append(row)
                steps.extend({"episode_index": episode_index, **step} for step in episode_steps)
            run_config = {
                **root_config,
                "execution_variant": variant_name,
                "execution": variant,
                "evaluation": {**evaluation, "method": method, "episode_seeds": seeds, "max_steps": max_steps},
                "initial_state_audit": audits,
            }
            summary = summarize(rows, steps)
            method_output.joinpath("config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
            method_output.joinpath("episodes.jsonl").write_text("".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8")
            method_output.joinpath("steps.jsonl").write_text("".join(json.dumps(row, allow_nan=True) + "\n" for row in steps), encoding="utf-8")
            method_output.joinpath("summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
            log_tensorboard(method_output, run_config, rows, steps, summary)
            variant_summaries[method] = summary
        all_summaries[variant_name] = variant_summaries
    result = {
        **root_config,
        "decision": "execution_audit_only",
        "decision_reason": "Execution perturbation and multi-step results do not constitute a closed-loop formal proof.",
        "variants": all_summaries,
    }
    output.joinpath("summary.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
