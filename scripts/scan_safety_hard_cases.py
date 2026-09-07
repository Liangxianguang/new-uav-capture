"""Run deterministic one-step pressure tests for the safety filters."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.pursuit_controllers import PursuitCBFSafetyFilter  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv, CylinderObstacle  # noqa: E402
from encirclement3d.safety_certificate import check_one_step_safety  # noqa: E402
from encirclement3d.safety_qp import RobustCBFQPConfig, RobustCBFQPFilter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "innovation_safety.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return value


def source_hashes(paths: list[Path]) -> dict[str, str]:
    return {
        str(path.resolve().relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def obstacle(
    center_xy: tuple[float, float],
    *,
    height: float,
    radius: float = 0.0,
    shape: str = "box",
    half_extents_xy: tuple[float, float] | None = None,
) -> CylinderObstacle:
    return CylinderObstacle(
        center_xy=np.asarray(center_xy, dtype=np.float64),
        radius=float(radius),
        height=float(height),
        shape=shape,
        half_extents_xy=(
            None if half_extents_xy is None else np.asarray(half_extents_xy, dtype=np.float64)
        ),
    )


def cases() -> list[dict[str, Any]]:
    far = np.array([[5.0, 5.0, 4.0], [-5.0, 5.0, 4.0], [5.0, -5.0, 4.0], [-5.0, -5.0, 4.0]])
    return [
        {
            "name": "narrow_channel",
            "positions": np.array([[0.0, 0.0, 4.0], [0.0, 4.5, 4.0], [0.0, -4.5, 4.0], [0.0, 7.0, 4.0]]),
            "desired": np.array([[-5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "obstacles": [
                obstacle((-1.5, 0.0), height=8.0, half_extents_xy=(0.25, 3.0)),
                obstacle((1.5, 0.0), height=8.0, half_extents_xy=(0.25, 3.0)),
            ],
        },
        {
            "name": "box_corner",
            "positions": np.array([[2.1, 2.1, 4.0], [5.0, 5.0, 4.0], [-5.0, 5.0, 4.0], [-5.0, -5.0, 4.0]]),
            "desired": np.array([[-5.0, -5.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "obstacles": [obstacle((0.0, 0.0), height=8.0, half_extents_xy=(1.0, 1.0))],
        },
        {
            "name": "upper_boundary",
            "positions": np.array([[8.8, 0.0, 4.0], *far[1:]]),
            "desired": np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "obstacles": [],
        },
        {
            "name": "close_inter_agent",
            "positions": np.array([[0.0, 0.0, 4.0], [1.4, 0.0, 4.0], [5.0, 5.0, 4.0], [-5.0, -5.0, 4.0]]),
            "desired": np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            "obstacles": [],
        },
        {
            "name": "unsafe_reset_altitude",
            "positions": np.array([[0.0, 0.0, 1.1], [4.0, 4.0, 4.0], [-4.0, 4.0, 4.0], [-4.0, -4.0, 4.0]]),
            "desired": np.zeros((4, 3), dtype=np.float64),
            "obstacles": [],
        },
    ]


def observation(env: CaptureRadiusPursuit3DEnv, case: dict[str, Any]) -> dict[str, Any]:
    return {
        "defender_positions": np.asarray(case["positions"], dtype=np.float64),
        "defender_velocities": np.zeros((4, 3), dtype=np.float64),
        "world_lower_bounds": env.lower.copy(),
        "world_upper_bounds": env.upper.copy(),
        "obstacles": list(case["obstacles"]),
    }


def run_case(
    env: CaptureRadiusPursuit3DEnv,
    case: dict[str, Any],
    base_config: RobustCBFQPConfig,
) -> list[dict[str, Any]]:
    current = observation(env, case)
    env.obstacles = list(case["obstacles"])
    methods: list[tuple[str, np.ndarray, Any]] = []
    for name, config in (
        ("hard_barrier", base_config),
        ("soft_slack_diagnostic", replace(base_config, slack_enabled=True)),
    ):
        actions, diagnostics = RobustCBFQPFilter(env, config).filter(case["desired"], current)
        methods.append((name, actions, diagnostics))
    local_actions, local_diagnostics = PursuitCBFSafetyFilter(env).filter(case["desired"], current)
    methods.append(("local_cbf", local_actions, local_diagnostics))
    methods.append(("zero_fallback", np.zeros((4, 3), dtype=np.float64), None))

    rows: list[dict[str, Any]] = []
    for method, action, diagnostics in methods:
        certificate = check_one_step_safety(
            current,
            action,
            dt=float(env.dt),
            drone_radius=float(env.agents["drone_radius"]),
            max_speed_mps=float(env.agents["defender_max_speed"]),
            max_acceleration_mps2=float(env.agents["defender_max_acceleration"]),
            safety_margin_m=float(env.pursuit["safety_margin"]),
            robust_margin_m=(base_config.robust_margin_m if method != "local_cbf" else 0.0),
            enforce_action_change=method in {"hard_barrier", "soft_slack_diagnostic"},
        )
        row: dict[str, Any] = {
            "case": case["name"],
            "method": method,
            "current_state_safe": bool(certificate.current_state_safe),
            "next_state_safe": bool(certificate.next_state_safe),
            "certificate_valid": bool(certificate.valid),
            "current_min_barrier_m": float(certificate.current_min_barrier_m),
            "next_min_barrier_m": float(certificate.next_min_barrier_m),
            "action_norm_mean_mps": float(np.mean(np.linalg.norm(action, axis=1))),
            "action_correction_norm_mps": float(np.mean(np.linalg.norm(action - case["desired"], axis=1))),
            "certificate_violations": list(certificate.violations),
            "obstacle_count": len(case["obstacles"]),
        }
        if diagnostics is not None:
            payload = diagnostics.as_dict() if hasattr(diagnostics, "as_dict") else diagnostics.__dict__
            row.update(
                {
                    "solver_status": payload.get("status"),
                    "solver_message": payload.get("solver_message", ""),
                    "failure_category": payload.get("failure_category", "none"),
                    "fallback_reason": payload.get("fallback_reason"),
                    "precondition_valid": payload.get("precondition_valid", True),
                    "fallback_used": payload.get("fallback_used", False),
                    "recovery_action_used": payload.get("recovery_action_used", False),
                    "constraint_count": payload.get("constraint_count", 0),
                    "active_constraint_count": payload.get("active_constraint_count", 0),
                    "minimum_constraint_residual": payload.get("minimum_constraint_residual"),
                    "maximum_constraint_violation": payload.get("maximum_constraint_violation"),
                    "maximum_safety_slack_m": payload.get("maximum_safety_slack_m"),
                    "latency_ms": payload.get("latency_ms"),
                }
            )
        else:
            row.update(
                {
                    "solver_status": "direct_action",
                    "solver_message": "",
                    "failure_category": "none",
                    "fallback_reason": None,
                    "precondition_valid": None,
                    "fallback_used": method == "zero_fallback",
                    "recovery_action_used": False,
                    "constraint_count": 0,
                    "active_constraint_count": 0,
                    "minimum_constraint_residual": None,
                    "maximum_constraint_violation": None,
                    "maximum_safety_slack_m": 0.0,
                    "latency_ms": 0.0,
                }
            )
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    document = load_yaml(args.config)
    environment_path = (args.config.parent / str(document["environment_config"])).resolve()
    base_environment = load_yaml(environment_path)
    probe = CaptureRadiusPursuit3DEnv(copy.deepcopy(base_environment), obstacle_count=0, target_speed_scale=0.45)
    probe.reset(seed=1)
    safety = dict(document.get("safety", {}))
    safety.setdefault("max_speed_mps", float(probe.agents["defender_max_speed"]))
    safety.setdefault("max_acceleration_mps2", float(probe.agents["defender_max_acceleration"]))
    safety.setdefault("safety_margin_m", float(probe.pursuit["safety_margin"]))
    base_config = RobustCBFQPConfig.from_mapping(safety)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = CaptureRadiusPursuit3DEnv(copy.deepcopy(base_environment), obstacle_count=0, target_speed_scale=0.45)
    rows = [row for case in cases() for row in run_case(env, case, base_config)]
    hashes = source_hashes(
        [
            args.config,
            environment_path,
            Path(__file__),
            PROJECT_ROOT / "src" / "encirclement3d" / "safety_qp.py",
            PROJECT_ROOT / "src" / "encirclement3d" / "safety_certificate.py",
            PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_controllers.py",
        ]
    )
    config_snapshot = {
        "config": str(args.config.resolve()),
        "environment_config": str(environment_path),
        "safety": {**base_config.__dict__, "robust_margin_m": base_config.robust_margin_m},
        "cases": [case["name"] for case in cases()],
        "source_hashes": hashes,
    }
    output.joinpath("config.yaml").write_text(yaml.safe_dump(config_snapshot, sort_keys=False), encoding="utf-8")
    output.joinpath("steps.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8"
    )
    summary = {
        "cases": len(cases()),
        "rows": len(rows),
        "valid_certificate_count": int(sum(bool(row["certificate_valid"]) for row in rows)),
        "failure_category_counts": {
            key: sum(row["failure_category"] == key for row in rows)
            for key in sorted({str(row["failure_category"]) for row in rows})
        },
        "source_hashes": hashes,
    }
    output.joinpath("summary.json").write_text(
        json.dumps({"config": config_snapshot, "summary": summary}, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    with SummaryWriter(log_dir=str(output / "tensorboard"), flush_secs=5) as writer:
        writer.add_text("Evaluation/config", yaml.safe_dump(config_snapshot, sort_keys=False), 0)
        writer.add_text("Evaluation/summary", json.dumps(summary, indent=2), 0)
        for index, row in enumerate(rows):
            for key in (
                "current_state_safe",
                "next_state_safe",
                "certificate_valid",
                "current_min_barrier_m",
                "next_min_barrier_m",
                "action_correction_norm_mps",
                "maximum_constraint_violation",
                "maximum_safety_slack_m",
                "latency_ms",
            ):
                value = row.get(key)
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    writer.add_scalar(f"Case/{row['case']}/{row['method']}/{key}", float(value), index)
    print(json.dumps(summary, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
