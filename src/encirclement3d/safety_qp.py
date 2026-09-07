"""Auditable robust CBF-QP safety filter for the velocity-level benchmark.

This module deliberately implements a one-step robust CBF-QP contract.  It
does not claim a learned CLBF or a closed-loop flight-dynamics certificate.
Safety constraints are linearized at the observed positions, while the
certificate module independently checks the resulting next positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm > 1e-12:
        return value / norm
    if fallback is not None:
        return np.asarray(fallback, dtype=np.float64).copy()
    return np.zeros_like(value)


def _clip_rows(values: np.ndarray, max_norm: float) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    return rows * np.minimum(1.0, float(max_norm) / np.maximum(norms, 1e-12))


def _obstacle_clearance_and_normal(
    position: np.ndarray,
    obstacle: Any,
) -> tuple[float, np.ndarray]:
    """Return signed clearance and an outward normal for an observation obstacle."""

    point = np.asarray(position, dtype=np.float64)
    if isinstance(obstacle, dict):
        shape = str(obstacle.get("shape", "cylinder"))
        center_xy = np.asarray(obstacle["center_xy"], dtype=np.float64)
        radius = float(obstacle.get("radius", 0.0))
        height = float(obstacle["height"])
        half_xy = obstacle.get("half_extents_xy")
    else:
        shape = str(obstacle.shape)
        center_xy = np.asarray(obstacle.center_xy, dtype=np.float64)
        radius = float(obstacle.radius)
        height = float(obstacle.height)
        half_xy = obstacle.half_extents_xy

    if shape == "cylinder":
        delta_xy = point[:2] - center_xy
        radial_distance = float(np.linalg.norm(delta_xy))
        radial_gap = radial_distance - radius
        radial_normal = _unit(
            np.array([delta_xy[0], delta_xy[1], 0.0]),
            fallback=np.array([1.0, 0.0, 0.0]),
        )
        if 0.0 <= point[2] <= height:
            return radial_gap, radial_normal
        nearest_z = 0.0 if point[2] < 0.0 else height
        vertical_gap = abs(point[2] - nearest_z)
        if radial_gap <= 0.0:
            return vertical_gap, np.array([0.0, 0.0, -1.0 if point[2] < 0.0 else 1.0])
        normal = _unit(
            radial_normal * radial_gap + np.array([0.0, 0.0, point[2] - nearest_z]),
            fallback=radial_normal,
        )
        return float(np.hypot(radial_gap, vertical_gap)), normal

    if half_xy is None:
        half_xy = [radius, radius]
    center = np.array([center_xy[0], center_xy[1], height * 0.5], dtype=np.float64)
    half = np.array([float(half_xy[0]), float(half_xy[1]), height * 0.5], dtype=np.float64)
    delta = point - center
    signed = np.abs(delta) - half
    outside = np.maximum(signed, 0.0)
    outside_norm = float(np.linalg.norm(outside))
    if outside_norm > 1e-12:
        return outside_norm, _unit(outside * np.sign(delta), fallback=np.array([1.0, 0.0, 0.0]))
    penetration = float(np.min(-signed))
    axis = int(np.argmin(-signed))
    normal = np.zeros(3, dtype=np.float64)
    normal[axis] = 1.0 if delta[axis] >= 0.0 else -1.0
    return -penetration, normal


@dataclass(frozen=True)
class RobustCBFQPConfig:
    """Numerical and robustness assumptions for the one-step safety QP."""

    gamma: float = 0.25
    safety_margin_m: float = 0.35
    disturbance_margin_m: float = 0.10
    observation_error_margin_m: float = 0.06
    delay_margin_m: float = 0.20
    execution_margin_m: float = 0.10
    max_speed_mps: float = 5.0
    max_acceleration_mps2: float = 6.0
    action_change_limit_mps: float | None = None
    slack_enabled: bool = True
    action_weight: float = 1.0
    slack_weight: float = 1.0e6
    slack_tolerance_m: float = 1.0e-6
    solver_tolerance: float = 1.0e-8
    solver_max_iterations: int = 100
    fallback_policy: str = "zero_action"

    def __post_init__(self) -> None:
        if not 0.0 < float(self.gamma) <= 1.0:
            raise ValueError("gamma must lie in (0, 1].")
        positive = {
            "safety_margin_m": self.safety_margin_m,
            "max_speed_mps": self.max_speed_mps,
            "max_acceleration_mps2": self.max_acceleration_mps2,
            "action_weight": self.action_weight,
            "slack_weight": self.slack_weight,
            "solver_tolerance": self.solver_tolerance,
        }
        for name, value in positive.items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        nonnegative = {
            "disturbance_margin_m": self.disturbance_margin_m,
            "observation_error_margin_m": self.observation_error_margin_m,
            "delay_margin_m": self.delay_margin_m,
            "execution_margin_m": self.execution_margin_m,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.action_change_limit_mps is not None and float(self.action_change_limit_mps) <= 0.0:
            raise ValueError("action_change_limit_mps must be positive when provided.")
        if int(self.solver_max_iterations) <= 0:
            raise ValueError("solver_max_iterations must be positive.")
        if str(self.fallback_policy) not in {"zero_action", "nominal_clipped", "barrier_recovery"}:
            raise ValueError("Unsupported fallback_policy.")

    @property
    def robust_margin_m(self) -> float:
        return float(
            self.disturbance_margin_m
            + self.observation_error_margin_m
            + self.delay_margin_m
            + self.execution_margin_m
        )

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "RobustCBFQPConfig":
        return cls(**dict(mapping))


@dataclass(frozen=True)
class RobustCBFQPDiagnostics:
    status: str
    solver_success: bool
    solver_message: str
    solver_iterations: int
    action_correction_norm: float
    minimum_barrier_value_m: float
    minimum_constraint_residual: float
    maximum_constraint_violation: float
    maximum_safety_slack_m: float
    active_constraint_count: int
    constraint_count: int
    fallback_used: bool
    certificate_valid: bool
    latency_ms: float
    assumptions: dict[str, float]
    failure_category: str = "none"
    fallback_reason: str | None = None
    precondition_valid: bool = True
    recovery_action_used: bool = False
    barrier_values_m: dict[str, float] | None = None
    constraint_residuals_m: dict[str, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "solver_success": self.solver_success,
            "solver_message": self.solver_message,
            "solver_iterations": self.solver_iterations,
            "action_correction_norm": self.action_correction_norm,
            "minimum_barrier_value_m": self.minimum_barrier_value_m,
            "minimum_constraint_residual": self.minimum_constraint_residual,
            "maximum_constraint_violation": self.maximum_constraint_violation,
            "maximum_safety_slack_m": self.maximum_safety_slack_m,
            "active_constraint_count": self.active_constraint_count,
            "constraint_count": self.constraint_count,
            "fallback_used": self.fallback_used,
            "certificate_valid": self.certificate_valid,
            "latency_ms": self.latency_ms,
            "assumptions": dict(self.assumptions),
            "failure_category": self.failure_category,
            "fallback_reason": self.fallback_reason,
            "precondition_valid": self.precondition_valid,
            "recovery_action_used": self.recovery_action_used,
            "barrier_values_m": dict(self.barrier_values_m or {}),
            "constraint_residuals_m": dict(self.constraint_residuals_m or {}),
        }


class RobustCBFQPFilter:
    """Solve a small convex QP that tracks nominal actions under robust CBFs."""

    def __init__(self, env: Any, config: RobustCBFQPConfig | None = None) -> None:
        self.env = env
        self.config = config or RobustCBFQPConfig(
            safety_margin_m=float(env.pursuit["safety_margin"]),
            max_speed_mps=float(env.agents["defender_max_speed"]),
            max_acceleration_mps2=float(env.agents["defender_max_acceleration"]),
        )

    def filter(
        self,
        desired_actions: np.ndarray,
        observation: dict[str, Any],
    ) -> tuple[np.ndarray, RobustCBFQPDiagnostics]:
        started = perf_counter()
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        velocities = np.asarray(observation.get("defender_velocities", np.zeros_like(positions)), dtype=np.float64)
        if positions.ndim != 2 or positions.shape[-1] != 3 or velocities.shape != positions.shape:
            raise ValueError("defender positions and velocities must have shape [defenders, 3]")
        desired = np.asarray(desired_actions, dtype=np.float64)
        if desired.shape != positions.shape or not np.isfinite(desired).all():
            raise ValueError("desired_actions must be finite with shape [defenders, 3]")
        desired = _clip_rows(desired, self.config.max_speed_mps)
        action_dimension = int(desired.size)

        rows: list[np.ndarray] = []
        rhs: list[float] = []
        barrier_values: list[float] = []
        names: list[str] = []
        effective_margin = float(self.config.safety_margin_m + self.config.robust_margin_m)
        dt = float(self.env.dt)
        gamma = float(self.config.gamma)
        radius = float(self.env.agents["drone_radius"])
        obstacles = observation.get("obstacles", getattr(self.env, "obstacles", []))
        lower = np.asarray(observation.get("world_lower_bounds", self.env.lower), dtype=np.float64)
        upper = np.asarray(observation.get("world_upper_bounds", self.env.upper), dtype=np.float64)

        def add_barrier(name: str, barrier: float, control_row: np.ndarray) -> None:
            rows.append(control_row)
            rhs.append(float(-gamma * barrier))
            barrier_values.append(float(barrier))
            names.append(name)

        for index, position in enumerate(positions):
            for obstacle_index, obstacle in enumerate(obstacles):
                clearance, normal = _obstacle_clearance_and_normal(position, obstacle)
                row = np.zeros(action_dimension, dtype=np.float64)
                row[index * 3 : index * 3 + 3] = normal * dt
                add_barrier(
                    f"obstacle[{obstacle_index}]/{index}",
                    clearance - radius - effective_margin,
                    row,
                )
            for axis in range(3):
                lower_row = np.zeros(action_dimension, dtype=np.float64)
                lower_row[index * 3 + axis] = dt
                add_barrier(
                    f"boundary_lower[{axis}]/{index}",
                    float(position[axis] - lower[axis] - radius - effective_margin),
                    lower_row,
                )
                upper_row = np.zeros(action_dimension, dtype=np.float64)
                upper_row[index * 3 + axis] = -dt
                add_barrier(
                    f"boundary_upper[{axis}]/{index}",
                    float(upper[axis] - position[axis] - radius - effective_margin),
                    upper_row,
                )

        minimum_distance = float(2.0 * radius + effective_margin)
        for first in range(positions.shape[0]):
            for second in range(first + 1, positions.shape[0]):
                delta = positions[first] - positions[second]
                normal = _unit(delta, fallback=np.array([1.0, 0.0, 0.0]))
                row = np.zeros(action_dimension, dtype=np.float64)
                row[first * 3 : first * 3 + 3] = normal * dt
                row[second * 3 : second * 3 + 3] = -normal * dt
                add_barrier(
                    f"inter_agent[{first},{second}]",
                    float(np.linalg.norm(delta) - minimum_distance),
                    row,
                )

        barrier_count = len(rows)
        slack_count = barrier_count if bool(self.config.slack_enabled) else 0
        variable_count = action_dimension + slack_count
        matrix = np.zeros((barrier_count, variable_count), dtype=np.float64)
        if barrier_count:
            matrix[:, :action_dimension] = np.stack(rows, axis=0)
            if slack_count:
                matrix[:, action_dimension:] = np.eye(barrier_count, dtype=np.float64)
        lower_action = np.full(action_dimension, -self.config.max_speed_mps / np.sqrt(3.0), dtype=np.float64)
        upper_action = np.full(action_dimension, self.config.max_speed_mps / np.sqrt(3.0), dtype=np.float64)
        change_limit = self.config.action_change_limit_mps
        if change_limit is None:
            change_limit = float(self.config.max_acceleration_mps2 * dt)
        lower_action = np.maximum(lower_action, (velocities.reshape(-1) - float(change_limit)))
        upper_action = np.minimum(upper_action, (velocities.reshape(-1) + float(change_limit)))
        if np.any(lower_action > upper_action + 1e-12):
            return self._fallback(
                desired,
                observation,
                barrier_values,
                names,
                rows,
                rhs,
                started,
                "inconsistent action bounds",
                failure_category="inconsistent_action_bounds",
                precondition_valid=bool(min(barrier_values, default=float("inf")) >= -self.config.solver_tolerance),
            )

        barrier_map = {name: float(value) for name, value in zip(names, barrier_values)}
        precondition_valid = bool(min(barrier_values, default=float("inf")) >= -self.config.solver_tolerance)
        if not precondition_valid:
            worst_name = min(barrier_map, key=barrier_map.get)
            return self._fallback(
                desired,
                observation,
                barrier_values,
                names,
                rows,
                rhs,
                started,
                f"current_state_outside_robust_safe_set:{worst_name}",
                failure_category="precondition_invalid",
                precondition_valid=False,
            )

        initial = np.concatenate([desired.reshape(-1), np.zeros(slack_count, dtype=np.float64)])
        initial[:action_dimension] = np.clip(initial[:action_dimension], lower_action, upper_action)
        if barrier_count and slack_count:
            initial[action_dimension:] = np.maximum(
                rhs if isinstance(rhs, np.ndarray) else np.asarray(rhs, dtype=np.float64)
                - matrix[:, :action_dimension] @ initial[:action_dimension],
                0.0,
            )
        nominal_flat = desired.reshape(-1)
        action_weight = float(self.config.action_weight)
        slack_weight = float(self.config.slack_weight)

        def objective(value: np.ndarray) -> float:
            action_delta = value[:action_dimension] - nominal_flat
            slack = value[action_dimension:] if slack_count else np.empty(0, dtype=np.float64)
            return float(0.5 * action_weight * np.dot(action_delta, action_delta) + 0.5 * slack_weight * np.dot(slack, slack))

        def gradient(value: np.ndarray) -> np.ndarray:
            result = np.zeros(variable_count, dtype=np.float64)
            result[:action_dimension] = action_weight * (value[:action_dimension] - nominal_flat)
            if slack_count:
                result[action_dimension:] = slack_weight * value[action_dimension:]
            return result

        constraints = () if not barrier_count else (
            LinearConstraint(matrix, np.asarray(rhs, dtype=np.float64), np.full(barrier_count, np.inf)),
        )
        if slack_count:
            lower_bounds = np.concatenate([lower_action, np.zeros(slack_count, dtype=np.float64)])
            upper_bounds = np.concatenate([upper_action, np.full(slack_count, np.inf, dtype=np.float64)])
        else:
            lower_bounds = lower_action
            upper_bounds = upper_action
        bounds = Bounds(lower_bounds, upper_bounds)
        try:
            result = minimize(
                objective,
                initial,
                jac=gradient,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={
                    "ftol": float(self.config.solver_tolerance),
                    "maxiter": int(self.config.solver_max_iterations),
                    "disp": False,
                },
            )
        except (FloatingPointError, ValueError, RuntimeError) as error:
            return self._fallback(
                desired,
                observation,
                barrier_values,
                names,
                rows,
                rhs,
                started,
                str(error),
                failure_category="solver_failure",
                precondition_valid=precondition_valid,
            )

        if not bool(result.success) or not np.isfinite(result.x).all():
            failure_category = "solver_failure"
            if not np.isfinite(result.x).all():
                failure_category = "solver_failure"
            elif not slack_count:
                feasibility = linprog(
                    np.zeros(action_dimension, dtype=np.float64),
                    A_ub=-matrix if barrier_count else None,
                    b_ub=-np.asarray(rhs, dtype=np.float64) if barrier_count else None,
                    bounds=list(zip(lower_action, upper_action)),
                    method="highs",
                )
                if not bool(feasibility.success):
                    failure_category = "qp_infeasible"
            return self._fallback(
                desired,
                observation,
                barrier_values,
                names,
                rows,
                rhs,
                started,
                str(result.message),
                failure_category=failure_category,
                precondition_valid=precondition_valid,
                solver_iterations=int(getattr(result, "nit", 0)),
            )
        solution = np.asarray(result.x, dtype=np.float64)
        actions = solution[:action_dimension].reshape(desired.shape)
        slack = solution[action_dimension:] if slack_count else np.empty(0, dtype=np.float64)
        residuals = matrix @ solution - np.asarray(rhs, dtype=np.float64) if barrier_count else np.empty(0)
        minimum_residual = float(np.min(residuals)) if residuals.size else float("inf")
        maximum_violation = float(max(0.0, -minimum_residual))
        maximum_slack = float(np.max(slack)) if slack.size else 0.0
        certificate_valid = bool(maximum_violation <= self.config.solver_tolerance and maximum_slack <= self.config.slack_tolerance_m)
        active = int(np.count_nonzero(residuals <= max(self.config.solver_tolerance * 10.0, 1e-7))) if residuals.size else 0
        diagnostics = RobustCBFQPDiagnostics(
            status="optimal" if certificate_valid else "optimal_with_slack",
            solver_success=True,
            solver_message=str(result.message),
            solver_iterations=int(getattr(result, "nit", 0)),
            action_correction_norm=float(np.mean(np.linalg.norm(actions - desired, axis=1))),
            minimum_barrier_value_m=float(min(barrier_values)) if barrier_values else float("inf"),
            minimum_constraint_residual=minimum_residual,
            maximum_constraint_violation=maximum_violation,
            maximum_safety_slack_m=maximum_slack,
            active_constraint_count=active,
            constraint_count=barrier_count,
            fallback_used=False,
            certificate_valid=certificate_valid,
            latency_ms=float((perf_counter() - started) * 1000.0),
            assumptions=self._assumption_dict(),
            failure_category="none" if certificate_valid else "slack_violation",
            fallback_reason=None,
            precondition_valid=precondition_valid,
            recovery_action_used=False,
            barrier_values_m={name: float(value) for name, value in zip(names, barrier_values)},
            constraint_residuals_m={name: float(value) for name, value in zip(names, residuals)},
        )
        return actions, diagnostics

    def _assumption_dict(self) -> dict[str, float]:
        return {
            "gamma": float(self.config.gamma),
            "base_safety_margin_m": float(self.config.safety_margin_m),
            "robust_margin_m": float(self.config.robust_margin_m),
            "max_speed_mps": float(self.config.max_speed_mps),
            "max_acceleration_mps2": float(self.config.max_acceleration_mps2),
        }

    def _fallback(
        self,
        desired: np.ndarray,
        observation: dict[str, Any],
        barrier_values: list[float],
        names: list[str],
        rows: list[np.ndarray],
        rhs: list[float],
        started: float,
        reason: str,
        *,
        failure_category: str,
        precondition_valid: bool,
        solver_iterations: int = 0,
    ) -> tuple[np.ndarray, RobustCBFQPDiagnostics]:
        if self.config.fallback_policy == "zero_action":
            actions = np.zeros_like(desired)
            recovery_action_used = False
        elif self.config.fallback_policy == "nominal_clipped":
            actions = _clip_rows(desired, self.config.max_speed_mps)
            recovery_action_used = False
        else:
            actions = self._barrier_recovery_action(desired, observation, barrier_values, rows, rhs)
            recovery_action_used = True
        flat_actions = actions.reshape(-1)
        if rows:
            residuals = np.asarray([float(np.dot(row, flat_actions) - bound) for row, bound in zip(rows, rhs)])
            minimum_residual = float(np.min(residuals))
            maximum_violation = float(max(0.0, -minimum_residual))
            residual_map = {name: float(value) for name, value in zip(names, residuals)}
        else:
            minimum_residual = float("inf")
            maximum_violation = 0.0
            residual_map = {}
        diagnostics = RobustCBFQPDiagnostics(
            status="fallback",
            solver_success=False,
            solver_message=str(reason),
            solver_iterations=int(solver_iterations),
            action_correction_norm=float(np.mean(np.linalg.norm(actions - desired, axis=1))),
            minimum_barrier_value_m=float(min(barrier_values)) if barrier_values else float("inf"),
            minimum_constraint_residual=minimum_residual,
            maximum_constraint_violation=maximum_violation,
            maximum_safety_slack_m=0.0,
            active_constraint_count=0,
            constraint_count=len(names),
            fallback_used=True,
            certificate_valid=False,
            latency_ms=float((perf_counter() - started) * 1000.0),
            assumptions=self._assumption_dict(),
            failure_category=failure_category,
            fallback_reason=str(reason),
            precondition_valid=bool(precondition_valid),
            recovery_action_used=recovery_action_used,
            barrier_values_m={name: float(value) for name, value in zip(names, barrier_values)},
            constraint_residuals_m=residual_map,
        )
        return actions, diagnostics

    def _barrier_recovery_action(
        self,
        desired: np.ndarray,
        observation: dict[str, Any],
        barrier_values: list[float],
        rows: list[np.ndarray],
        rhs: list[float],
    ) -> np.ndarray:
        """Move toward violated barrier normals within the one-step actuator limit."""

        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        velocities = np.asarray(observation.get("defender_velocities", np.zeros_like(positions)), dtype=np.float64)
        action = _clip_rows(velocities, self.config.max_speed_mps)
        if not rows:
            return action
        dt = float(self.env.dt)
        change_limit = float(self.config.action_change_limit_mps or self.config.max_acceleration_mps2 * dt)
        direction = np.zeros_like(positions)
        flat_size = int(positions.size)
        for barrier, row, bound in zip(barrier_values, rows, rhs):
            if float(barrier) >= 0.0:
                continue
            flat_row = np.asarray(row, dtype=np.float64)
            for index in range(positions.shape[0]):
                segment = flat_row[index * 3 : index * 3 + 3]
                norm = float(np.linalg.norm(segment))
                if norm <= 1.0e-12:
                    continue
                # ``row`` already contains dt, so bound / dt is a velocity
                # projection.  The extra term accelerates recovery while the
                # independent checker still decides whether the step is safe.
                direction[index] += segment / norm * max(float(bound) / max(dt, 1.0e-12), 0.0)
        target = action + direction
        delta = np.clip(target - velocities, -change_limit, change_limit)
        action = velocities + delta
        action = _clip_rows(action, self.config.max_speed_mps)
        if not np.isfinite(action).all() or action.size != flat_size:
            return np.zeros_like(desired)
        return action


__all__ = ["RobustCBFQPConfig", "RobustCBFQPDiagnostics", "RobustCBFQPFilter"]
