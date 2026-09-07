"""Evaluate nominal, local-CBF, and robust CBF-QP execution filters."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections import Counter
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

from encirclement3d.pursuit_controllers import (  # noqa: E402
    DynamicEncirclementController,
    PursuitCBFSafetyFilter,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.safety_certificate import check_one_step_safety  # noqa: E402
from encirclement3d.safety_qp import (  # noqa: E402
    RobustCBFQPConfig,
    RobustCBFQPFilter,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "innovation_safety.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--methods", nargs="+", choices=("nominal", "local_cbf", "robust_cbf_qp"))
    parser.add_argument("--device", choices=("cpu",), default="cpu")
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
        PROJECT_ROOT / "scripts" / "evaluate_safety_qp.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "safety_qp.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "safety_certificate.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_controllers.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_env.py",
    )
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else float("nan")


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")


def _probe_initial_state(
    base_config: dict[str, Any],
    *,
    seed: int,
    obstacle_count: int,
    target_speed_scale: float,
    qp_config: RobustCBFQPConfig,
) -> dict[str, Any]:
    """Audit reset membership using only public geometry and a zero action."""

    env = CaptureRadiusPursuit3DEnv(
        copy.deepcopy(base_config),
        obstacle_count=obstacle_count,
        target_speed_scale=target_speed_scale,
    )
    observation = env.reset(seed=seed)
    certificate = check_one_step_safety(
        _certificate_observation(observation, env),
        np.zeros((env.n_defenders, 3), dtype=np.float64),
        dt=float(env.dt),
        drone_radius=float(env.agents["drone_radius"]),
        max_speed_mps=float(env.agents["defender_max_speed"]),
        max_acceleration_mps2=float(env.agents["defender_max_acceleration"]),
        safety_margin_m=float(env.pursuit["safety_margin"]),
        robust_margin_m=float(qp_config.robust_margin_m),
        action_change_limit_mps=(
            float(qp_config.action_change_limit_mps)
            if qp_config.action_change_limit_mps is not None
            else None
        ),
        enforce_action_change=True,
    )
    return {
        "seed": int(seed),
        "initial_min_barrier_m": float(certificate.current_min_barrier_m),
        "current_state_safe": bool(certificate.current_state_safe),
        "next_state_safe_under_zero_action": bool(certificate.next_state_safe),
        "violations": list(certificate.violations),
    }


def _classify_initial_stratum(minimum_barrier: float, strata: list[dict[str, Any]]) -> str | None:
    for stratum in strata:
        lower = float(stratum["min_barrier_m"])
        upper = float(stratum.get("max_barrier_m", float("inf")))
        if lower <= minimum_barrier < upper:
            return str(stratum["name"])
    return None


def select_episode_seeds(
    base_config: dict[str, Any],
    *,
    evaluation: dict[str, Any],
    requested_episodes: int,
    seed_start: int,
    obstacle_count: int,
    target_speed_scale: float,
    qp_config: RobustCBFQPConfig,
) -> tuple[list[int], dict[str, Any]]:
    """Select fixed episode seeds under an auditable reset protocol."""

    protocol = dict(evaluation.get("initial_state_protocol", {}))
    mode = str(protocol.get("mode", "report_all"))
    if mode == "report_all":
        seeds = [int(seed_start + index) for index in range(requested_episodes)]
        return seeds, {
            "mode": mode,
            "requested_episodes": int(requested_episodes),
            "candidate_seed_start": int(seed_start),
            "candidate_seed_count": int(requested_episodes),
            "accepted_seeds": seeds,
            "rejected_seeds": [],
            "strata": [],
        }
    if mode != "stratified_robust_safe":
        raise ValueError(f"Unsupported initial_state_protocol.mode: {mode}")

    candidate_count = int(protocol.get("candidate_seed_count", max(4 * requested_episodes, requested_episodes)))
    strata = [dict(item) for item in protocol.get("strata", ())]
    if not strata:
        first_count = requested_episodes // 2
        strata = [
            {"name": "tight", "min_barrier_m": 0.0, "max_barrier_m": 0.25, "count": first_count},
            {"name": "nominal", "min_barrier_m": 0.25, "max_barrier_m": float("inf"), "count": requested_episodes - first_count},
        ]
    expected_count = sum(int(item.get("count", 0)) for item in strata)
    if expected_count != requested_episodes:
        raise ValueError("initial_state_protocol.strata counts must equal requested episodes")
    if candidate_count < requested_episodes:
        raise ValueError("candidate_seed_count must be at least requested episodes")

    audits: list[dict[str, Any]] = []
    selected_by_stratum: dict[str, list[int]] = {str(item["name"]): [] for item in strata}
    for seed in range(seed_start, seed_start + candidate_count):
        audit = _probe_initial_state(
            base_config,
            seed=seed,
            obstacle_count=obstacle_count,
            target_speed_scale=target_speed_scale,
            qp_config=qp_config,
        )
        stratum = (
            _classify_initial_stratum(audit["initial_min_barrier_m"], strata)
            if audit["current_state_safe"]
            else None
        )
        audit["stratum"] = stratum
        audit["accepted"] = False
        if stratum is not None:
            stratum_config = next(item for item in strata if str(item["name"]) == stratum)
            selected = selected_by_stratum[stratum]
            if len(selected) < int(stratum_config["count"]):
                selected.append(int(seed))
                audit["accepted"] = True
        audits.append(audit)
    missing = {
        name: int(next(item for item in strata if str(item["name"]) == name)["count"]) - len(seeds)
        for name, seeds in selected_by_stratum.items()
        if len(seeds) < int(next(item for item in strata if str(item["name"]) == name)["count"])
    }
    if missing:
        raise RuntimeError(
            "Unable to satisfy initial_state_protocol strata; "
            f"missing={missing}, candidate_seed_count={candidate_count}"
        )
    accepted_seeds = [seed for item in strata for seed in selected_by_stratum[str(item["name"])] ]
    return accepted_seeds, {
        "mode": mode,
        "requested_episodes": int(requested_episodes),
        "candidate_seed_start": int(seed_start),
        "candidate_seed_count": int(candidate_count),
        "accepted_seeds": accepted_seeds,
        "rejected_seeds": [audit for audit in audits if not bool(audit["accepted"])],
        "accepted_seed_audits": [audit for audit in audits if bool(audit["accepted"])],
        "strata": strata,
    }


def _certificate_observation(
    observation: dict[str, Any],
    env: CaptureRadiusPursuit3DEnv,
) -> dict[str, Any]:
    """Add public static geometry required by the independent checker."""

    value = dict(observation)
    value["world_lower_bounds"] = env.lower.copy()
    value["world_upper_bounds"] = env.upper.copy()
    value["obstacles"] = list(getattr(env, "obstacles", ()))
    return value


def run_episode(
    base_config: dict[str, Any],
    *,
    seed: int,
    method: str,
    obstacle_count: int,
    target_speed_scale: float,
    target_motion_mode: str,
    max_steps: int,
    qp_config: RobustCBFQPConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = copy.deepcopy(base_config)
    config.setdefault("task", {}).setdefault("pursuit", {})["target_motion_mode"] = target_motion_mode
    config["world"]["max_steps"] = int(max_steps)
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=obstacle_count, target_speed_scale=target_speed_scale)
    observation = env.reset(seed=seed)
    controller = DynamicEncirclementController(env)
    local_filter = PursuitCBFSafetyFilter(env)
    robust_filter = RobustCBFQPFilter(env, qp_config)
    step_rows: list[dict[str, Any]] = []
    correction_norms: list[float] = []
    safety_latencies: list[float] = []
    certificate_next_barriers: list[float] = []
    certificate_valid_count = 0
    certificate_invalid_count = 0
    certificate_precondition_invalid_count = 0
    next_state_safe_count = 0
    qp_certificate_invalid_count = 0
    solver_success_count = 0
    solver_fallback_count = 0
    failure_categories: Counter[str] = Counter()
    maximum_violation = 0.0
    maximum_slack = 0.0
    minimum_barrier = float("inf")
    initial_precondition_valid: bool | None = None
    initial_min_barrier = float("nan")
    started_episode = time.perf_counter()

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
        safety_latency_ms = (time.perf_counter() - filter_started) * 1000.0

        robust_margin = qp_config.robust_margin_m if method == "robust_cbf_qp" else 0.0
        enforce_action_change = method == "robust_cbf_qp"
        certificate = check_one_step_safety(
            _certificate_observation(observation, env),
            action,
            dt=float(env.dt),
            drone_radius=float(env.agents["drone_radius"]),
            max_speed_mps=float(env.agents["defender_max_speed"]),
            max_acceleration_mps2=float(env.agents["defender_max_acceleration"]),
            safety_margin_m=float(env.pursuit["safety_margin"]),
            robust_margin_m=float(robust_margin),
            action_change_limit_mps=(
                float(qp_config.action_change_limit_mps)
                if qp_config.action_change_limit_mps is not None
                else None
            ),
            enforce_action_change=enforce_action_change,
        )
        if initial_precondition_valid is None:
            initial_precondition_valid = bool(certificate.current_state_safe)
            initial_min_barrier = float(certificate.current_min_barrier_m)
        certificate_valid_count += int(certificate.valid)
        certificate_invalid_count += int(not certificate.valid)
        certificate_precondition_invalid_count += int(not certificate.current_state_safe)
        next_state_safe_count += int(certificate.next_state_safe)
        qp_certificate_valid = None if diagnostics is None else bool(getattr(diagnostics, "certificate_valid", True))
        qp_certificate_invalid_count += int(qp_certificate_valid is False)
        certificate_next_barriers.append(float(certificate.next_min_barrier_m))
        minimum_barrier = min(minimum_barrier, float(certificate.next_min_barrier_m))
        correction = float(np.mean(np.linalg.norm(action - desired, axis=1)))
        correction_norms.append(correction)
        safety_latencies.append(float(safety_latency_ms))
        if diagnostics is not None:
            solver_success_count += int(bool(getattr(diagnostics, "solver_success", False)))
            solver_fallback_count += int(bool(getattr(diagnostics, "fallback_used", False)))
            failure_category = str(getattr(diagnostics, "failure_category", "none"))
            failure_categories[failure_category] += 1
            maximum_violation = max(maximum_violation, float(getattr(diagnostics, "maximum_constraint_violation", 0.0)))
            maximum_slack = max(maximum_slack, float(getattr(diagnostics, "maximum_safety_slack_m", 0.0)))
        else:
            failure_category = "none"

        observation, _reward, terminated, truncated, info = env.step(action)
        fallback_used = None if diagnostics is None else bool(getattr(diagnostics, "fallback_used", False))
        step_rows.append(
            {
                "step": int(env.step_count),
                "certificate_valid": bool(certificate.valid and qp_certificate_valid is not False),
                "certificate_status": certificate.status,
                "certificate_violations": list(certificate.violations),
                "current_state_safe": bool(certificate.current_state_safe),
                "next_state_safe": bool(certificate.next_state_safe),
                "qp_certificate_valid": qp_certificate_valid,
                "solver_status": None if diagnostics is None else str(getattr(diagnostics, "status", "unknown")),
                "solver_message": None if diagnostics is None else str(getattr(diagnostics, "solver_message", "")),
                "failure_category": failure_category,
                "precondition_phase": (
                    "initial" if int(env.step_count) == 0 else "rollout"
                ),
                "fallback_reason": None if diagnostics is None else getattr(diagnostics, "fallback_reason", None),
                "precondition_valid": None if diagnostics is None else bool(getattr(diagnostics, "precondition_valid", True)),
                "recovery_action_used": None if diagnostics is None else bool(getattr(diagnostics, "recovery_action_used", False)),
                "current_min_barrier_m": float(certificate.current_min_barrier_m),
                "next_min_barrier_m": float(certificate.next_min_barrier_m),
                "action_correction_norm_mps": correction,
                "safety_latency_ms": float(safety_latency_ms),
                "solver_success": None if diagnostics is None else bool(getattr(diagnostics, "solver_success", False)),
                "fallback_used": fallback_used,
                "fallback_current_state_safe": (
                    None if not fallback_used else bool(certificate.current_state_safe)
                ),
                "fallback_next_state_safe": None if not fallback_used else bool(certificate.next_state_safe),
                "fallback_certificate_valid": None if not fallback_used else bool(certificate.valid),
                "active_constraint_count": (
                    None if diagnostics is None else int(getattr(diagnostics, "active_constraint_count", 0))
                ),
                "constraint_count": (
                    None if diagnostics is None else int(getattr(diagnostics, "constraint_count", 0))
                ),
                "minimum_constraint_residual": (
                    None if diagnostics is None else float(getattr(diagnostics, "minimum_constraint_residual", float("nan")))
                ),
                "maximum_constraint_violation": (
                    None if diagnostics is None else float(getattr(diagnostics, "maximum_constraint_violation", 0.0))
                ),
                "maximum_safety_slack_m": (
                    None if diagnostics is None else float(getattr(diagnostics, "maximum_safety_slack_m", 0.0))
                ),
                "barrier_values_m": None if diagnostics is None else dict(getattr(diagnostics, "barrier_values_m", {}) or {}),
                "constraint_residuals_m": (
                    None if diagnostics is None else dict(getattr(diagnostics, "constraint_residuals_m", {}) or {})
                ),
            }
        )
        if terminated or truncated:
            break

    termination_reason = str(info.get("termination_reason", "unknown"))
    row = {
        "seed": int(seed),
        "method": method,
        "initial_precondition_valid": bool(initial_precondition_valid),
        "initial_min_barrier_m": float(initial_min_barrier),
        "steps": int(env.step_count),
        "safe_capture_success": bool(info.get("safe_capture_success", False)),
        "capture_event": bool(info.get("capture_event", False)),
        "collision": bool(info.get("collision", False)),
        "boundary_violation": bool(info.get("world_violation_steps", 0) > 0),
        "timeout": termination_reason == "timeout",
        "termination_reason": termination_reason,
        "capture_time_seconds": info.get("capture_time_seconds"),
        "min_clearance_m": float(info.get("min_clearance_so_far", info.get("min_clearance", float("nan")))),
        "mean_action_correction_norm_mps": _mean(correction_norms),
        "mean_safety_latency_ms": _mean(safety_latencies),
        "safety_latency_p95_ms": _percentile(safety_latencies, 95.0),
        "certificate_valid_rate": float(certificate_valid_count / max(len(step_rows), 1)),
        "certificate_invalid_count": int(certificate_invalid_count),
        "certificate_precondition_invalid_count": int(certificate_precondition_invalid_count),
        "rollout_precondition_invalid_count": int(
            sum(not bool(step["current_state_safe"]) for step in step_rows[1:])
        ),
        "next_state_safe_rate": float(next_state_safe_count / max(len(step_rows), 1)),
        "qp_certificate_invalid_count": int(qp_certificate_invalid_count),
        "solver_success_rate": float(solver_success_count / max(len(step_rows), 1)) if diagnostics is not None else 1.0,
        "solver_fallback_count": int(solver_fallback_count),
        "failure_category_counts": dict(failure_categories),
        "precondition_failure_count": int(failure_categories.get("precondition_invalid", 0)),
        "qp_infeasible_count": int(failure_categories.get("qp_infeasible", 0)),
        "solver_failure_count": int(failure_categories.get("solver_failure", 0)),
        "inconsistent_action_bounds_count": int(failure_categories.get("inconsistent_action_bounds", 0)),
        "fallback_next_state_safe_rate": float(
            np.mean([
                bool(step["fallback_next_state_safe"])
                for step in step_rows
                if step["fallback_next_state_safe"] is not None
            ])
            if any(step["fallback_next_state_safe"] is not None for step in step_rows)
            else float("nan")
        ),
        "maximum_constraint_violation": float(maximum_violation),
        "maximum_safety_slack_m": float(maximum_slack),
        "minimum_certificate_barrier_m": float(minimum_barrier),
        "episode_wall_time_seconds": float(time.perf_counter() - started_episode),
    }
    return row, step_rows


def summarize(rows: list[dict[str, Any]], step_rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["mean_safety_latency_ms"]) for row in rows]
    failure_categories = Counter(
        str(step.get("failure_category", "none"))
        for step in step_rows
        if step.get("failure_category") is not None
    )
    initial_invalid_episodes = int(sum(not bool(row["initial_precondition_valid"]) for row in rows))
    initial_valid_rows = [row for row in rows if bool(row["initial_precondition_valid"])]
    conditional_steps = [
        step for step in step_rows
        if bool(step.get("precondition_valid", True))
    ]
    conditional_solver_success = [
        step for step in conditional_steps
        if step.get("solver_success") is not None
    ]
    return {
        "episodes": len(rows),
        "safe_capture_rate": float(np.mean([bool(row["safe_capture_success"]) for row in rows])),
        "capture_rate": float(np.mean([bool(row["capture_event"]) for row in rows])),
        "collision_rate": float(np.mean([bool(row["collision"]) for row in rows])),
        "boundary_violation_rate": float(np.mean([bool(row["boundary_violation"]) for row in rows])),
        "timeout_rate": float(np.mean([bool(row["timeout"]) for row in rows])),
        "mean_capture_time_seconds": _mean(
            [float(row["capture_time_seconds"]) for row in rows if row["capture_time_seconds"] is not None]
        ),
        "mean_min_clearance_m": _mean([float(row["min_clearance_m"]) for row in rows]),
        "certificate_valid_rate": float(np.mean([float(row["certificate_valid_rate"]) for row in rows])),
        "certificate_invalid_episode_rate": float(
            np.mean([int(row["certificate_invalid_count"] > 0) for row in rows])
        ),
        "certificate_precondition_invalid_count": int(
            sum(int(row["certificate_precondition_invalid_count"]) for row in rows)
        ),
        "initial_precondition_valid_rate": float(np.mean([bool(row["initial_precondition_valid"]) for row in rows])),
        "initial_precondition_invalid_episode_count": initial_invalid_episodes,
        "initial_precondition_min_barrier_m": float(
            min(float(row["initial_min_barrier_m"]) for row in rows)
        ),
        "initial_valid_episode_count": len(initial_valid_rows),
        "conditional_solver_success_rate": float(
            np.mean([bool(step["solver_success"]) for step in conditional_solver_success])
        ) if conditional_solver_success else float("nan"),
        "conditional_next_state_safe_rate": float(
            np.mean([bool(step["next_state_safe"]) for step in conditional_steps])
        ) if conditional_steps else float("nan"),
        "next_state_safe_rate": float(np.mean([float(row["next_state_safe_rate"]) for row in rows])),
        "qp_certificate_invalid_count": int(sum(int(row["qp_certificate_invalid_count"]) for row in rows)),
        "solver_success_rate": float(np.mean([float(row["solver_success_rate"]) for row in rows])),
        "solver_fallback_count": int(sum(int(row["solver_fallback_count"]) for row in rows)),
        "failure_category_counts": dict(failure_categories),
        "precondition_failure_count": int(failure_categories.get("precondition_invalid", 0)),
        "qp_infeasible_count": int(failure_categories.get("qp_infeasible", 0)),
        "solver_failure_count": int(failure_categories.get("solver_failure", 0)),
        "inconsistent_action_bounds_count": int(failure_categories.get("inconsistent_action_bounds", 0)),
        "fallback_next_state_safe_rate": float(
            np.mean([
                bool(step["fallback_next_state_safe"])
                for step in step_rows
                if step.get("fallback_next_state_safe") is not None
            ])
            if any(step.get("fallback_next_state_safe") is not None for step in step_rows)
            else float("nan")
        ),
        "maximum_constraint_violation_m": float(max(float(row["maximum_constraint_violation"]) for row in rows)),
        "maximum_safety_slack_m": float(max(float(row["maximum_safety_slack_m"]) for row in rows)),
        "minimum_certificate_barrier_m": float(min(float(row["minimum_certificate_barrier_m"]) for row in rows)),
        "safety_latency_ms": {
            "p50": _percentile(latencies, 50.0),
            "p95": _percentile(latencies, 95.0),
            "p99": _percentile(latencies, 99.0),
        },
        "step_certificate_invalid_count": int(sum(not bool(row["certificate_valid"]) for row in step_rows)),
        "steps": len(step_rows),
    }


def log_tensorboard(output: Path, run_config: dict[str, Any], rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    writer_type = require_summary_writer()
    with writer_type(log_dir=str(output / "tensorboard"), flush_secs=5) as writer:
        writer.add_text("Evaluation/config", yaml.safe_dump(run_config, sort_keys=False), 0)
        writer.add_text("Evaluation/source_hashes", json.dumps(run_config["source_hashes"], indent=2), 0)
        writer.add_text(
            "Evaluation/initial_state_protocol",
            json.dumps(run_config["initial_state_protocol"], indent=2),
            0,
        )
        for index, row in enumerate(rows):
            for key in (
                "safe_capture_success",
                "capture_event",
                "collision",
                "boundary_violation",
                "timeout",
                "min_clearance_m",
                "mean_action_correction_norm_mps",
                "mean_safety_latency_ms",
                "safety_latency_p95_ms",
                "certificate_valid_rate",
                "certificate_invalid_count",
                "certificate_precondition_invalid_count",
                "initial_precondition_valid",
                "initial_min_barrier_m",
                "rollout_precondition_invalid_count",
                "next_state_safe_rate",
                "qp_certificate_invalid_count",
                "solver_success_rate",
                "solver_fallback_count",
                "precondition_failure_count",
                "qp_infeasible_count",
                "solver_failure_count",
                "inconsistent_action_bounds_count",
                "fallback_next_state_safe_rate",
                "initial_precondition_valid_rate",
                "initial_precondition_invalid_episode_count",
                "initial_precondition_min_barrier_m",
                "initial_valid_episode_count",
                "conditional_solver_success_rate",
                "conditional_next_state_safe_rate",
                "maximum_constraint_violation",
                "maximum_safety_slack_m",
                "minimum_certificate_barrier_m",
            ):
                value = row.get(key)
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    writer.add_scalar(f"Episode/{key}", float(value), index)
        for key, value in summary.items():
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                writer.add_scalar(f"Summary/{key}", float(value), 0)
        for percentile, value in summary["safety_latency_ms"].items():
            writer.add_scalar(f"Summary/SafetyLatency/{percentile}_ms", float(value), 0)
        writer.add_text("Evaluation/failure_category_counts", json.dumps(summary["failure_category_counts"], indent=2), 0)
        writer.add_hparams(
            {
                "method": run_config["evaluation"]["method"],
                "episodes": run_config["evaluation"]["episodes"],
                "obstacle_count": run_config["evaluation"]["obstacle_count"],
                "target_speed_scale": run_config["evaluation"]["target_speed_scale"],
                "gamma": run_config["safety"]["gamma"],
                "safety_margin_m": run_config["safety"]["safety_margin_m"],
                "robust_margin_m": run_config["safety"]["robust_margin_m"],
                "initial_state_mode": run_config["initial_state_protocol"]["mode"],
                "candidate_seed_count": run_config["initial_state_protocol"]["candidate_seed_count"],
            },
            {
                "hparam/safe_capture_rate": float(summary["safe_capture_rate"]),
                "hparam/certificate_valid_rate": float(summary["certificate_valid_rate"]),
                "hparam/latency_p95_ms": float(summary["safety_latency_ms"]["p95"]),
            },
        )


def main() -> None:
    args = parse_args()
    document = load_yaml(args.config)
    evaluation = dict(document.get("evaluation", {}))
    safety_mapping = dict(document.get("safety", {}))
    environment_path = (args.config.parent / str(document["environment_config"])).resolve()
    base_config = load_yaml(environment_path)
    episodes = int(args.episodes if args.episodes is not None else evaluation.get("episodes", 8))
    seed_start = int(args.seed_start if args.seed_start is not None else evaluation.get("seed_start", 649101))
    methods = list(args.methods or evaluation.get("methods", ["local_cbf", "robust_cbf_qp"]))
    obstacle_count = int(evaluation.get("obstacle_count", 3))
    target_speed_scale = float(evaluation.get("target_speed_scale", 0.45))
    target_motion_mode = str(evaluation.get("target_motion_mode", "flee_persistence"))
    max_steps = int(evaluation.get("max_steps", base_config["world"]["max_steps"]))
    if episodes <= 0 or max_steps <= 0 or obstacle_count < 0 or target_speed_scale <= 0.0:
        raise ValueError("episodes/max_steps must be positive, obstacle_count non-negative, and target speed positive")

    env_probe = CaptureRadiusPursuit3DEnv(copy.deepcopy(base_config), obstacle_count=obstacle_count, target_speed_scale=target_speed_scale)
    safety_mapping.setdefault("max_speed_mps", float(env_probe.agents["defender_max_speed"]))
    safety_mapping.setdefault("max_acceleration_mps2", float(env_probe.agents["defender_max_acceleration"]))
    safety_mapping.setdefault("safety_margin_m", float(env_probe.pursuit["safety_margin"]))
    qp_config = RobustCBFQPConfig.from_mapping(safety_mapping)
    episode_seeds, protocol_audit = select_episode_seeds(
        base_config,
        evaluation=evaluation,
        requested_episodes=episodes,
        seed_start=seed_start,
        obstacle_count=obstacle_count,
        target_speed_scale=target_speed_scale,
        qp_config=qp_config,
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    run_hashes = source_hashes(args.config)

    for method in methods:
        method_output = output / method
        method_output.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        all_steps: list[dict[str, Any]] = []
        for episode_index, episode_seed in enumerate(episode_seeds):
            row, step_rows = run_episode(
                base_config,
                seed=episode_seed,
                method=method,
                obstacle_count=obstacle_count,
                target_speed_scale=target_speed_scale,
                target_motion_mode=target_motion_mode,
                max_steps=max_steps,
                qp_config=qp_config,
            )
            rows.append(row)
            all_steps.extend({"method": method, "episode_index": episode_index, **step} for step in step_rows)
        summary = summarize(rows, all_steps)
        run_config = {
            "experiment_name": document.get("experiment_name", "phase5_robust_cbf_qp_validation"),
            "config": str(args.config.resolve()),
            "environment_config": str(environment_path),
            "safety": {**qp_config.__dict__, "robust_margin_m": qp_config.robust_margin_m},
            "evaluation": {
                "method": method,
                "episodes": episodes,
                "seed_start": seed_start,
                "episode_seeds": episode_seeds,
                "obstacle_count": obstacle_count,
                "target_speed_scale": target_speed_scale,
                "target_motion_mode": target_motion_mode,
                "max_steps": max_steps,
                "control_cycle_budget_ms": evaluation.get("control_cycle_budget_ms"),
            },
            "initial_state_protocol": protocol_audit,
            "source_hashes": run_hashes,
        }
        method_output.joinpath("config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
        method_output.joinpath("episodes.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8"
        )
        method_output.joinpath("steps.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=True) + "\n" for row in all_steps), encoding="utf-8"
        )
        method_output.joinpath("summary.json").write_text(
            json.dumps({"config": run_config, "summary": summary}, indent=2, allow_nan=True), encoding="utf-8"
        )
        log_tensorboard(method_output, run_config, rows, summary)
        summaries[method] = summary

    robust_summary = summaries.get("robust_cbf_qp", {})
    budget_ms = float(evaluation.get("control_cycle_budget_ms", float("inf")))
    conditional_gate_pass = bool(
        protocol_audit["mode"] == "stratified_robust_safe"
        and float(robust_summary.get("initial_precondition_valid_rate", 0.0)) >= 1.0
        and float(robust_summary.get("certificate_valid_rate", 0.0)) >= 1.0
        and float(robust_summary.get("next_state_safe_rate", 0.0)) >= 1.0
        and int(robust_summary.get("qp_infeasible_count", 1)) == 0
        and int(robust_summary.get("solver_failure_count", 1)) == 0
        and float(robust_summary.get("collision_rate", 1.0)) == 0.0
        and float(robust_summary.get("boundary_violation_rate", 1.0)) == 0.0
        and float(robust_summary.get("safety_latency_ms", {}).get("p95", float("inf"))) <= budget_ms
    )
    decision = "conditional_p5_pass" if conditional_gate_pass else "pending_gate"
    decision_reason = (
        "Stratified robust-safe reset passed the velocity-level P5 gate; hard-case scan and formal proof remain separate."
        if conditional_gate_pass
        else "P5 requires independent certificate, local-CBF comparison, and hard-case review."
    )
    result = {
        "experiment_name": document.get("experiment_name", "phase5_robust_cbf_qp_validation"),
        "config": str(args.config.resolve()),
        "methods": summaries,
        "initial_state_protocol": protocol_audit,
        "source_hashes": run_hashes,
        "decision": decision,
        "decision_reason": decision_reason,
    }
    output.joinpath("summary.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
