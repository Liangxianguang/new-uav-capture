"""Auditable robust CBF-QP safety filter for the velocity-level benchmark.

This module deliberately implements a one-step robust CBF-QP contract.  It
does not claim a learned CLBF or a closed-loop flight-dynamics certificate.
Safety constraints are linearized at the observed positions, while the
certificate module independently checks the resulting next positions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, linprog, minimize

from encirclement3d.execution_dynamics import (
    CommandAuthorityDirective,
    apply_command_authority,
    command_authority_from_observation,
    parameters_from_observation,
    queue_from_observation,
    rollout_execution,
)
from encirclement3d.safety_certificate import (
    assess_execution_recoverability,
    check_execution_rollout_safety,
    execution_barrier_values,
    execution_barrier_values_with_action_jacobian,
)


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
    enforce_action_change: bool = True
    slack_enabled: bool = True
    action_weight: float = 1.0
    slack_weight: float = 1.0e6
    slack_tolerance_m: float = 1.0e-6
    solver_tolerance: float = 1.0e-8
    solver_max_iterations: int = 100
    fallback_policy: str = "zero_action"
    execution_preview_horizon_steps: int = 5
    execution_linearization_backend: str = "analytic"
    execution_linearization_active_margin_m: float = 0.5
    execution_linearization_iterations: int = 3
    execution_linearization_fd_step_mps: float = 1.0e-3
    execution_projection_iterations: int = 160
    execution_projection_tolerance: float = 1.0e-7
    execution_emergency_brake_enabled: bool = True
    execution_reachable_tube_multiplier: float = 1.0
    execution_continuous_segment_constraints: bool = False
    execution_continuous_segment_subdivisions: int = 4
    fallback_progress_weight: float = 1.0
    fallback_barrier_weight: float = 0.05
    execution_fallback_receding_step_enabled: bool = True

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
            "fallback_progress_weight": self.fallback_progress_weight,
        }
        for name, value in positive.items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        nonnegative = {
            "disturbance_margin_m": self.disturbance_margin_m,
            "observation_error_margin_m": self.observation_error_margin_m,
            "delay_margin_m": self.delay_margin_m,
            "execution_margin_m": self.execution_margin_m,
            "fallback_barrier_weight": self.fallback_barrier_weight,
        }
        for name, value in nonnegative.items():
            if not np.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if self.action_change_limit_mps is not None and float(self.action_change_limit_mps) <= 0.0:
            raise ValueError("action_change_limit_mps must be positive when provided.")
        if int(self.solver_max_iterations) <= 0:
            raise ValueError("solver_max_iterations must be positive.")
        if int(self.execution_preview_horizon_steps) <= 0:
            raise ValueError("execution_preview_horizon_steps must be positive.")
        if str(self.execution_linearization_backend) not in {"analytic", "finite_difference"}:
            raise ValueError("execution_linearization_backend must be analytic or finite_difference.")
        if not np.isfinite(float(self.execution_linearization_active_margin_m)) or float(
            self.execution_linearization_active_margin_m
        ) < 0.0:
            raise ValueError("execution_linearization_active_margin_m must be finite and non-negative.")
        if int(self.execution_linearization_iterations) <= 0:
            raise ValueError("execution_linearization_iterations must be positive.")
        if int(self.execution_projection_iterations) <= 0:
            raise ValueError("execution_projection_iterations must be positive.")
        if int(self.execution_continuous_segment_subdivisions) <= 0:
            raise ValueError("execution_continuous_segment_subdivisions must be positive.")
        if float(self.execution_linearization_fd_step_mps) <= 0.0:
            raise ValueError("execution_linearization_fd_step_mps must be positive.")
        if float(self.execution_projection_tolerance) <= 0.0:
            raise ValueError("execution_projection_tolerance must be positive.")
        if (
            not np.isfinite(float(self.execution_reachable_tube_multiplier))
            or float(self.execution_reachable_tube_multiplier) < 1.0
        ):
            raise ValueError("execution_reachable_tube_multiplier must be finite and at least one.")
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
    solver_backend: str = "slsqp"
    linearization_iterations: int = 0
    linearization_evaluations: int = 0
    linearization_active_constraints: int = 0
    command_authority: dict[str, Any] | None = None
    emergency_brake_requested: bool = False
    queue_override_slots: int = 0
    recoverability_status: str = "not_checked"
    abort_required: bool = False
    prefix_admissible: bool = True
    immutable_prefix_horizon_steps: int = 0
    immutable_prefix_min_robust_barrier_m: float = float("inf")
    fallback_candidate_type: str = "none"
    fallback_target_distance_before_m: float = float("nan")
    fallback_target_distance_after_m: float = float("nan")
    fallback_goal_progress_m: float = 0.0
    fallback_certificate_horizon_steps: int = 0
    fallback_certificate_scope: str = "none"

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
            "solver_backend": self.solver_backend,
            "linearization_iterations": self.linearization_iterations,
            "linearization_evaluations": self.linearization_evaluations,
            "linearization_active_constraints": self.linearization_active_constraints,
            "command_authority": dict(self.command_authority or {}),
            "emergency_brake_requested": self.emergency_brake_requested,
            "queue_override_slots": self.queue_override_slots,
            "recoverability_status": self.recoverability_status,
            "abort_required": self.abort_required,
            "prefix_admissible": self.prefix_admissible,
            "immutable_prefix_horizon_steps": self.immutable_prefix_horizon_steps,
            "immutable_prefix_min_robust_barrier_m": self.immutable_prefix_min_robust_barrier_m,
            "fallback_candidate_type": self.fallback_candidate_type,
            "fallback_target_distance_before_m": self.fallback_target_distance_before_m,
            "fallback_target_distance_after_m": self.fallback_target_distance_after_m,
            "fallback_goal_progress_m": self.fallback_goal_progress_m,
            "fallback_certificate_horizon_steps": self.fallback_certificate_horizon_steps,
            "fallback_certificate_scope": self.fallback_certificate_scope,
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

        execution = observation.get("execution", {})
        if isinstance(execution, dict) and bool(execution.get("enabled", False)):
            return self._filter_with_execution_model(desired, observation, started)

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
        # The simulator and independent certificate constrain each defender by
        # its Euclidean speed norm. A +/- v_max/sqrt(3) component box is only
        # an inscribed approximation and can make legal axis-aligned commands
        # appear incompatible with the action-change bounds.
        lower_action = np.full(action_dimension, -self.config.max_speed_mps, dtype=np.float64)
        upper_action = np.full(action_dimension, self.config.max_speed_mps, dtype=np.float64)
        change_limit = self.config.action_change_limit_mps
        if change_limit is None:
            change_limit = float(self.config.max_acceleration_mps2 * dt)
        if self.config.enforce_action_change:
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

        constraints: tuple[Any, ...] = () if not barrier_count else (
            LinearConstraint(matrix, np.asarray(rhs, dtype=np.float64), np.full(barrier_count, np.inf)),
        )

        def speed_budget(value: np.ndarray) -> np.ndarray:
            actions = np.asarray(value[:action_dimension], dtype=np.float64).reshape(-1, 3)
            return float(self.config.max_speed_mps) ** 2 - np.sum(actions * actions, axis=1)

        def speed_budget_jacobian(value: np.ndarray) -> np.ndarray:
            action = np.asarray(value[:action_dimension], dtype=np.float64).reshape(-1, 3)
            jacobian = np.zeros((action.shape[0], variable_count), dtype=np.float64)
            for index, row in enumerate(action):
                jacobian[index, index * 3 : index * 3 + 3] = -2.0 * row
            return jacobian

        constraints += (
            NonlinearConstraint(
                speed_budget,
                np.zeros(positions.shape[0], dtype=np.float64),
                np.full(positions.shape[0], np.inf, dtype=np.float64),
                jac=speed_budget_jacobian,
            ),
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
            "action_change_constraint_enabled": float(bool(self.config.enforce_action_change)),
            "execution_reachable_tube_multiplier": float(self.config.execution_reachable_tube_multiplier),
            "execution_continuous_segment_constraints": float(
                bool(self.config.execution_continuous_segment_constraints)
            ),
            "execution_continuous_segment_subdivisions": float(
                self.config.execution_continuous_segment_subdivisions
            ),
            "fallback_progress_weight": float(self.config.fallback_progress_weight),
            "fallback_barrier_weight": float(self.config.fallback_barrier_weight),
            "execution_fallback_receding_step_enabled": float(
                bool(self.config.execution_fallback_receding_step_enabled)
            ),
        }

    def _filter_with_execution_model(
        self,
        desired: np.ndarray,
        observation: dict[str, Any],
        started: float,
    ) -> tuple[np.ndarray, RobustCBFQPDiagnostics]:
        """Use a bounded sequential linearization for the execution rollout.

        The previous nonlinear SLSQP formulation made a finite-difference
        nested rollout the online solver.  This path explicitly linearizes the
        queued execution barrier at a small number of operating points and
        projects onto the resulting bounded halfspaces.  Every accepted result
        is still checked against the original nonlinear rollout independently.
        """

        safety_observation = dict(observation)
        safety_observation.setdefault("world_lower_bounds", np.asarray(self.env.lower, dtype=np.float64))
        safety_observation.setdefault("world_upper_bounds", np.asarray(self.env.upper, dtype=np.float64))
        safety_observation.setdefault("obstacles", list(getattr(self.env, "obstacles", ())))
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        velocities = np.asarray(observation.get("defender_velocities", np.zeros_like(positions)), dtype=np.float64)
        queue = queue_from_observation(observation, positions.shape[0])
        preview_steps = max(int(self.config.execution_preview_horizon_steps), len(queue) + 1)
        authority_mode = command_authority_from_observation(safety_observation)
        nominal_directive = CommandAuthorityDirective(mode=authority_mode, emergency_brake=False)
        last_emergency_brake = bool(
            isinstance(safety_observation.get("execution", {}), dict)
            and safety_observation.get("execution", {}).get("last_emergency_brake_requested", False)
        )
        lower_action, upper_action = self._execution_action_bounds(velocities)

        recoverability = assess_execution_recoverability(
            safety_observation,
            dt=float(self.env.dt),
            drone_radius=float(self.env.agents["drone_radius"]),
            safety_margin_m=float(self.config.safety_margin_m),
            robust_margin_m=float(self.config.robust_margin_m),
            horizon_steps=preview_steps,
            reachable_tube_multiplier=float(self.config.execution_reachable_tube_multiplier),
            tolerance=float(self.config.solver_tolerance),
            command_authority=nominal_directive,
        )
        selected_recoverability = recoverability
        selected_directive = nominal_directive
        if (
            "immutable_execution_prefix_outside_robust_safe_set" in recoverability.violations
            and bool(self.config.execution_emergency_brake_enabled)
            and authority_mode != "immutable"
        ):
            selected_directive = CommandAuthorityDirective(mode=authority_mode, emergency_brake=True)
            selected_recoverability = assess_execution_recoverability(
                safety_observation,
                dt=float(self.env.dt),
                drone_radius=float(self.env.agents["drone_radius"]),
                safety_margin_m=float(self.config.safety_margin_m),
                robust_margin_m=float(self.config.robust_margin_m),
                horizon_steps=preview_steps,
                reachable_tube_multiplier=float(self.config.execution_reachable_tube_multiplier),
                tolerance=float(self.config.solver_tolerance),
                command_authority=selected_directive,
            )
        if "immutable_execution_prefix_outside_robust_safe_set" in selected_recoverability.violations:
            return self._execution_fallback(
                desired,
                observation,
                started,
                reason="abort_required:" + ",".join(selected_recoverability.violations),
                category="unrecoverable_pending_execution",
                precondition_valid=False,
                directive=selected_directive,
                force_zero_action=True,
            )

        def rollout_certificate(
            actions: np.ndarray,
            directive: CommandAuthorityDirective,
        ) -> Any:
            return check_execution_rollout_safety(
                safety_observation,
                actions,
                dt=float(self.env.dt),
                drone_radius=float(self.env.agents["drone_radius"]),
                max_speed_mps=float(self.config.max_speed_mps),
                max_acceleration_mps2=float(self.config.max_acceleration_mps2),
                safety_margin_m=float(self.config.safety_margin_m),
                robust_margin_m=float(self.config.robust_margin_m),
                tolerance=float(self.config.solver_tolerance),
                action_change_limit_mps=self.config.action_change_limit_mps,
                horizon_steps=preview_steps,
                reachable_tube_multiplier=float(self.config.execution_reachable_tube_multiplier),
                continuous_segment_constraints=bool(self.config.execution_continuous_segment_constraints),
                continuous_segment_subdivisions=int(self.config.execution_continuous_segment_subdivisions),
                command_authority=directive,
            )

        probe_directive = selected_directive if selected_directive.emergency_brake else nominal_directive
        current_probe = rollout_certificate(np.zeros_like(desired), probe_directive)
        if not current_probe.current_state_safe:
            return self._execution_fallback(
                desired,
                observation,
                started,
                reason="current_state_outside_robust_safe_set_before_projection",
                category="execution_precondition_invalid",
                precondition_valid=False,
                directive=nominal_directive,
            )
        if np.any(lower_action > upper_action + 1.0e-12):
            return self._execution_fallback(
                desired,
                observation,
                started,
                reason="inconsistent action bounds",
                category="inconsistent_action_bounds",
                precondition_valid=True,
                directive=nominal_directive,
            )

        attempts = [(desired, nominal_directive, "nominal_command")]
        if (
            bool(self.config.execution_emergency_brake_enabled)
            and authority_mode != "immutable"
            and not last_emergency_brake
        ):
            attempts.append(
                (
                    desired,
                    CommandAuthorityDirective(mode=authority_mode, emergency_brake=True),
                    "authorized_queue_clear_resume",
                )
            )

        last_reason = "no bounded execution-safe command found"
        last_directive = nominal_directive
        for reference, directive, label in attempts:
            result = self._solve_execution_linearized_projection(
                safety_observation,
                reference,
                lower_action,
                upper_action,
                preview_steps=preview_steps,
                directive=directive,
            )
            last_reason = str(result["message"])
            last_directive = directive
            actions = np.asarray(result["actions"], dtype=np.float64)
            certificate = rollout_certificate(actions, directive)
            if bool(result["success"]) and certificate.valid:
                return actions, self._execution_diagnostics(
                    desired,
                    actions,
                    certificate,
                    result,
                    started,
                    directive,
                    safety_observation,
                    status="optimal" if not directive.emergency_brake else "optimal_emergency_brake",
                    solver_message=label,
                )
            if not bool(result["success"]):
                last_reason = f"{label}:{last_reason}"
            elif not certificate.valid:
                last_reason = f"{label}:independent_rollout_check_failed:{','.join(certificate.violations)}"

        return self._execution_fallback(
            desired,
            observation,
            started,
            reason=last_reason,
            category="execution_rollout_infeasible",
            precondition_valid=True,
            directive=last_directive,
        )

    def _execution_action_bounds(self, velocities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        action_dimension = int(velocities.size)
        lower = np.full(action_dimension, -self.config.max_speed_mps / np.sqrt(3.0), dtype=np.float64)
        upper = np.full(action_dimension, self.config.max_speed_mps / np.sqrt(3.0), dtype=np.float64)
        change_limit = self.config.action_change_limit_mps
        if change_limit is None:
            change_limit = float(self.config.max_acceleration_mps2 * self.env.dt)
        return (
            np.maximum(lower, velocities.reshape(-1) - float(change_limit)),
            np.minimum(upper, velocities.reshape(-1) + float(change_limit)),
        )

    def _solve_execution_linearized_projection(
        self,
        observation: dict[str, Any],
        reference_actions: np.ndarray,
        lower_action: np.ndarray,
        upper_action: np.ndarray,
        *,
        preview_steps: int,
        directive: CommandAuthorityDirective,
    ) -> dict[str, Any]:
        """Solve successive local linearizations with Dykstra projections."""

        reference = np.clip(np.asarray(reference_actions, dtype=np.float64).reshape(-1), lower_action, upper_action)
        evaluations = 0
        last_values: dict[str, float] = {}
        last_residuals = np.empty(0, dtype=np.float64)
        last_message = "linearized projection did not converge"
        last_active_constraints = 0
        action_shape = reference_actions.shape

        def values(flat_actions: np.ndarray) -> dict[str, float]:
            nonlocal evaluations
            _nominal, robust, _assumptions = execution_barrier_values(
                observation,
                np.asarray(flat_actions, dtype=np.float64).reshape(action_shape),
                dt=float(self.env.dt),
                drone_radius=float(self.env.agents["drone_radius"]),
                safety_margin_m=float(self.config.safety_margin_m),
                robust_margin_m=float(self.config.robust_margin_m),
                horizon_steps=preview_steps,
                reachable_tube_multiplier=float(self.config.execution_reachable_tube_multiplier),
                continuous_segment_constraints=bool(self.config.execution_continuous_segment_constraints),
                continuous_segment_subdivisions=int(self.config.execution_continuous_segment_subdivisions),
                command_authority=directive,
            )
            evaluations += 1
            return robust

        def linearize(flat_actions: np.ndarray) -> tuple[dict[str, float], np.ndarray | None]:
            nonlocal evaluations
            if str(self.config.execution_linearization_backend) == "analytic":
                _nominal, robust, jacobian, _assumptions = execution_barrier_values_with_action_jacobian(
                    observation,
                    np.asarray(flat_actions, dtype=np.float64).reshape(action_shape),
                    dt=float(self.env.dt),
                    drone_radius=float(self.env.agents["drone_radius"]),
                    safety_margin_m=float(self.config.safety_margin_m),
                    robust_margin_m=float(self.config.robust_margin_m),
                    horizon_steps=preview_steps,
                    jacobian_active_margin_m=float(self.config.execution_linearization_active_margin_m),
                    reachable_tube_multiplier=float(self.config.execution_reachable_tube_multiplier),
                    continuous_segment_constraints=bool(self.config.execution_continuous_segment_constraints),
                    continuous_segment_subdivisions=int(self.config.execution_continuous_segment_subdivisions),
                    command_authority=directive,
                )
                evaluations += 1
                return robust, np.asarray(jacobian, dtype=np.float64)
            return values(flat_actions), None

        for linearization_index in range(1, int(self.config.execution_linearization_iterations) + 1):
            base_map, analytic_jacobian = linearize(reference)
            names = list(base_map)
            base = np.asarray([float(base_map[name]) for name in names], dtype=np.float64)
            active_mask = base <= (
                float(self.config.execution_linearization_active_margin_m)
                + float(self.config.solver_tolerance)
            )
            if not bool(np.any(active_mask)):
                active_mask = np.ones_like(base, dtype=bool)
            active_count = int(np.count_nonzero(active_mask))
            last_active_constraints = active_count
            if not np.isfinite(base).all():
                return {
                    "success": False,
                    "actions": reference.reshape(action_shape),
                    "message": "non_finite_execution_barrier",
                    "projection_iterations": 0,
                    "linearization_iterations": linearization_index,
                    "evaluations": evaluations,
                    "barrier_values": base_map,
                        "residuals": base,
                        "solver_backend": "analytic_rollout_jacobian" if analytic_jacobian is not None else "finite_difference_dykstra",
                        "linearization_active_constraints": active_count,
                }
            if float(np.min(base, initial=np.inf)) >= -float(self.config.solver_tolerance):
                return {
                    "success": True,
                    "actions": reference.reshape(action_shape),
                    "message": "reference_is_execution_safe",
                    "projection_iterations": 0,
                    "linearization_iterations": linearization_index,
                    "evaluations": evaluations,
                    "barrier_values": base_map,
                    "residuals": base,
                    "solver_backend": "analytic_rollout_jacobian" if analytic_jacobian is not None else "finite_difference_dykstra",
                    "linearization_active_constraints": active_count,
                }

            if analytic_jacobian is not None:
                jacobian = analytic_jacobian
                if jacobian.shape != (len(names), reference.size) or not np.isfinite(jacobian).all():
                    return {
                        "success": False,
                        "actions": reference.reshape(action_shape),
                        "message": "non_finite_execution_jacobian",
                        "projection_iterations": 0,
                        "linearization_iterations": linearization_index,
                        "evaluations": evaluations,
                        "barrier_values": base_map,
                        "residuals": base,
                        "solver_backend": "analytic_rollout_jacobian",
                        "linearization_active_constraints": active_count,
                    }
            else:
                jacobian = np.zeros((len(names), reference.size), dtype=np.float64)
                finite_difference = float(self.config.execution_linearization_fd_step_mps)
                for index in range(reference.size):
                    perturbed = reference.copy()
                    perturbed[index] = min(upper_action[index], perturbed[index] + finite_difference)
                    delta = float(perturbed[index] - reference[index])
                    if delta <= 1.0e-12:
                        perturbed[index] = max(lower_action[index], reference[index] - finite_difference)
                        delta = float(perturbed[index] - reference[index])
                    if abs(delta) <= 1.0e-12:
                        continue
                    perturbed_map = values(perturbed)
                    jacobian[:, index] = (
                        np.asarray([float(perturbed_map[name]) for name in names], dtype=np.float64) - base
                    ) / delta

            jacobian = jacobian[active_mask]
            active_base = base[active_mask]
            lower_rhs = -active_base + jacobian @ reference
            candidate, projection_success, projection_iterations, message = self._project_linearized_halfspaces(
                reference,
                jacobian,
                lower_rhs,
                lower_action,
                upper_action,
            )
            candidate_map = values(candidate)
            candidate_values = np.asarray([float(candidate_map[name]) for name in names], dtype=np.float64)
            last_values = candidate_map
            last_residuals = candidate_values
            last_message = message
            if projection_success and float(np.min(candidate_values, initial=np.inf)) >= -float(self.config.solver_tolerance):
                return {
                    "success": True,
                    "actions": candidate.reshape(action_shape),
                    "message": message,
                    "projection_iterations": projection_iterations,
                    "linearization_iterations": linearization_index,
                    "evaluations": evaluations,
                    "barrier_values": candidate_map,
                    "residuals": candidate_values,
                    "solver_backend": "analytic_rollout_jacobian" if analytic_jacobian is not None else "finite_difference_dykstra",
                    "linearization_active_constraints": active_count,
                }
            reference = candidate

        return {
            "success": False,
            "actions": reference.reshape(action_shape),
            "message": last_message,
            "projection_iterations": int(self.config.execution_projection_iterations),
            "linearization_iterations": int(self.config.execution_linearization_iterations),
            "evaluations": evaluations,
            "barrier_values": last_values,
            "residuals": last_residuals,
            "solver_backend": "analytic_rollout_jacobian" if str(self.config.execution_linearization_backend) == "analytic" else "finite_difference_dykstra",
            "linearization_active_constraints": last_active_constraints,
        }

    def _project_linearized_halfspaces(
        self,
        target: np.ndarray,
        matrix: np.ndarray,
        lower_rhs: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> tuple[np.ndarray, bool, int, str]:
        """Dykstra projection for the Euclidean bounded linearized subproblem."""

        action = np.clip(np.asarray(target, dtype=np.float64), lower_bounds, upper_bounds)
        if matrix.size == 0:
            return action, True, 0, "no_execution_constraints"
        norms = np.einsum("ij,ij->i", matrix, matrix)
        static_violations = (norms <= 1.0e-18) & (lower_rhs > float(self.config.execution_projection_tolerance))
        if bool(np.any(static_violations)):
            return action, False, 0, "linearized_static_constraint_infeasible"
        corrections = np.zeros((matrix.shape[0] + 1, action.size), dtype=np.float64)
        tolerance = float(self.config.execution_projection_tolerance)
        for iteration in range(1, int(self.config.execution_projection_iterations) + 1):
            previous = action.copy()
            for index, (row, norm_squared) in enumerate(zip(matrix, norms)):
                shifted = action + corrections[index]
                deficit = float(lower_rhs[index] - np.dot(row, shifted))
                if norm_squared > 1.0e-18 and deficit > 0.0:
                    projected = shifted + deficit / norm_squared * row
                else:
                    projected = shifted
                corrections[index] = shifted - projected
                action = projected
            shifted = action + corrections[-1]
            action = np.clip(shifted, lower_bounds, upper_bounds)
            corrections[-1] = shifted - action
            residuals = matrix @ action - lower_rhs
            if float(np.min(residuals, initial=np.inf)) >= -tolerance and float(
                np.max(np.abs(action - previous), initial=0.0)
            ) <= tolerance:
                return action, True, iteration, "sequential_linearized_dykstra"
        residuals = matrix @ action - lower_rhs
        return (
            action,
            bool(float(np.min(residuals, initial=np.inf)) >= -tolerance),
            int(self.config.execution_projection_iterations),
            "sequential_linearized_projection_max_iterations",
        )

    def _execution_diagnostics(
        self,
        desired: np.ndarray,
        actions: np.ndarray,
        certificate: Any,
        result: dict[str, Any],
        started: float,
        directive: CommandAuthorityDirective,
        observation: dict[str, Any],
        *,
        status: str,
        solver_message: str,
    ) -> RobustCBFQPDiagnostics:
        barrier_values = {name: float(value) for name, value in result["barrier_values"].items()}
        residuals = np.asarray(result["residuals"], dtype=np.float64)
        minimum_residual = float(np.min(residuals, initial=np.inf))
        residual_map = {
            name: float(value)
            for name, value in zip(barrier_values, residuals)
        }
        return RobustCBFQPDiagnostics(
            status=status,
            solver_success=True,
            solver_message=solver_message,
            solver_iterations=int(result["projection_iterations"]),
            action_correction_norm=float(np.mean(np.linalg.norm(actions - desired, axis=1))),
            minimum_barrier_value_m=float(min(barrier_values.values(), default=float("inf"))),
            minimum_constraint_residual=minimum_residual,
            maximum_constraint_violation=float(max(0.0, -minimum_residual)),
            maximum_safety_slack_m=0.0,
            active_constraint_count=int(
                np.count_nonzero(residuals <= max(float(self.config.solver_tolerance) * 10.0, 1.0e-7))
            ),
            constraint_count=len(barrier_values),
            fallback_used=False,
            certificate_valid=bool(certificate.valid),
            latency_ms=float((perf_counter() - started) * 1000.0),
            assumptions={**self._assumption_dict(), **certificate.assumptions},
            failure_category="none",
            fallback_reason=None,
            precondition_valid=True,
            recovery_action_used=bool(directive.emergency_brake),
            barrier_values_m=barrier_values,
            constraint_residuals_m=residual_map,
            solver_backend=str(result.get("solver_backend", "sequential_linearized_dykstra")),
            linearization_iterations=int(result["linearization_iterations"]),
            linearization_evaluations=int(result["evaluations"]),
            linearization_active_constraints=int(result.get("linearization_active_constraints", 0)),
            command_authority=directive.as_dict(),
            emergency_brake_requested=bool(directive.emergency_brake),
            queue_override_slots=int(certificate.assumptions.get("queue_override_slots", 0.0)),
            **self._recoverability_fields(observation, directive),
        )

    def _recoverability_fields(
        self,
        observation: dict[str, Any],
        directive: CommandAuthorityDirective,
    ) -> dict[str, Any]:
        try:
            result = assess_execution_recoverability(
                observation,
                dt=float(self.env.dt),
                drone_radius=float(self.env.agents["drone_radius"]),
                safety_margin_m=float(self.config.safety_margin_m),
                robust_margin_m=float(self.config.robust_margin_m),
                horizon_steps=max(
                    int(self.config.execution_preview_horizon_steps),
                    len(queue_from_observation(observation, len(observation["defender_positions"])) ) + 1,
                ),
                reachable_tube_multiplier=float(self.config.execution_reachable_tube_multiplier),
                tolerance=float(self.config.solver_tolerance),
                command_authority=directive,
            )
        except (FloatingPointError, ValueError, RuntimeError):
            return {
                "recoverability_status": "not_checked",
                "abort_required": False,
                "prefix_admissible": False,
                "immutable_prefix_horizon_steps": 0,
                "immutable_prefix_min_robust_barrier_m": float("nan"),
            }
        return {
            "recoverability_status": result.status,
            "abort_required": bool(result.abort_required),
            "prefix_admissible": bool(result.prefix_admissible),
            "immutable_prefix_horizon_steps": int(result.immutable_prefix_horizon_steps),
            "immutable_prefix_min_robust_barrier_m": float(result.minimum_prefix_robust_barrier_m),
        }

    @staticmethod
    def _goal_reference(observation: dict[str, Any]) -> np.ndarray | None:
        """Build the same belief-only team target used by pursuit control."""

        predictions = observation.get("target_prediction_positions")
        if predictions is None:
            beliefs = observation.get("target_belief_positions")
            velocities = observation.get("target_belief_velocities")
            if beliefs is None or velocities is None:
                return None
            predictions = np.asarray(beliefs, dtype=np.float64) + 0.55 * np.asarray(velocities, dtype=np.float64)
        predictions = np.asarray(predictions, dtype=np.float64)
        if predictions.ndim != 2 or predictions.shape[1:] != (3,) or not np.isfinite(predictions).all():
            return None
        ages = np.asarray(observation.get("message_age_steps", np.zeros(predictions.shape[0])), dtype=np.float64)
        if ages.shape != (predictions.shape[0],) or not np.isfinite(ages).all():
            ages = np.zeros(predictions.shape[0], dtype=np.float64)
        weights = 1.0 / (1.0 + np.maximum(ages, 0.0))
        weights /= max(float(np.sum(weights)), 1.0e-12)
        return np.sum(predictions * weights[:, None], axis=0)

    def _goal_progress(
        self,
        observation: dict[str, Any],
        action: np.ndarray,
        directive: CommandAuthorityDirective,
    ) -> tuple[float, float, float] | None:
        """Estimate one-step target-distance progress without target ground truth."""

        target = self._goal_reference(observation)
        if target is None:
            return None
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        velocities = np.asarray(observation.get("defender_velocities", np.zeros_like(positions)), dtype=np.float64)
        before = float(np.min(np.linalg.norm(positions - target[None, :], axis=1)))
        queue = queue_from_observation(observation, positions.shape[0])
        queue, _resolved, _overridden = apply_command_authority(
            queue,
            directive,
            allowed_mode=command_authority_from_observation(observation),
        )
        parameters = parameters_from_observation(observation, float(self.env.dt))
        future_positions, _future_velocities, _steps = rollout_execution(
            positions,
            velocities,
            queue,
            np.asarray(action, dtype=np.float64),
            parameters,
            horizon_steps=1,
        )
        after = float(np.min(np.linalg.norm(future_positions[0] - target[None, :], axis=1)))
        return before, after, before - after

    def _bounded_execution_action(self, action: np.ndarray, observation: dict[str, Any]) -> np.ndarray:
        """Keep a nominal fallback inside actuator and one-step change bounds."""

        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        velocities = np.asarray(observation.get("defender_velocities", np.zeros_like(positions)), dtype=np.float64)
        lower, upper = self._execution_action_bounds(velocities)
        bounded = np.clip(
            _clip_rows(np.asarray(action, dtype=np.float64), self.config.max_speed_mps).reshape(-1),
            lower,
            upper,
        ).reshape(positions.shape)
        return _clip_rows(bounded, self.config.max_speed_mps)

    def _execution_fallback(
        self,
        desired: np.ndarray,
        observation: dict[str, Any],
        started: float,
        *,
        reason: str,
        category: str,
        precondition_valid: bool,
        directive: CommandAuthorityDirective,
        force_zero_action: bool = False,
    ) -> tuple[np.ndarray, RobustCBFQPDiagnostics]:
        # Queue authority cancels pending commands; it does not force the new
        # command to zero.  Only an explicit force-zero path does that.
        action = np.zeros_like(desired) if force_zero_action else desired
        safety_observation = dict(observation)
        safety_observation.setdefault("world_lower_bounds", np.asarray(self.env.lower, dtype=np.float64))
        safety_observation.setdefault("world_upper_bounds", np.asarray(self.env.upper, dtype=np.float64))
        safety_observation.setdefault("obstacles", list(getattr(self.env, "obstacles", ())))
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        queue = queue_from_observation(observation, positions.shape[0])
        _queue, _resolved, override_slots = apply_command_authority(
            queue,
            directive,
            allowed_mode=command_authority_from_observation(safety_observation),
        )
        preview_steps = max(int(self.config.execution_preview_horizon_steps), len(queue) + 1)
        try:
            _nominal, robust, _assumptions = execution_barrier_values(
                safety_observation,
                action,
                dt=float(self.env.dt),
                drone_radius=float(self.env.agents["drone_radius"]),
                safety_margin_m=float(self.config.safety_margin_m),
                robust_margin_m=float(self.config.robust_margin_m),
                horizon_steps=preview_steps,
                reachable_tube_multiplier=float(self.config.execution_reachable_tube_multiplier),
                continuous_segment_constraints=bool(self.config.execution_continuous_segment_constraints),
                continuous_segment_subdivisions=int(self.config.execution_continuous_segment_subdivisions),
                command_authority=directive,
            )
            names = list(robust)
            barrier_values = [float(robust[name]) for name in names]
            if self.config.fallback_policy == "barrier_recovery" and names:
                _nominal, robust_with_jacobian, jacobian, _assumptions = (
                    execution_barrier_values_with_action_jacobian(
                        safety_observation,
                        action,
                        dt=float(self.env.dt),
                        drone_radius=float(self.env.agents["drone_radius"]),
                        safety_margin_m=float(self.config.safety_margin_m),
                        robust_margin_m=float(self.config.robust_margin_m),
                        horizon_steps=preview_steps,
                        jacobian_active_margin_m=None,
                        reachable_tube_multiplier=float(self.config.execution_reachable_tube_multiplier),
                        continuous_segment_constraints=bool(self.config.execution_continuous_segment_constraints),
                        continuous_segment_subdivisions=int(self.config.execution_continuous_segment_subdivisions),
                        command_authority=directive,
                    )
                )
                names = list(robust_with_jacobian)
                barrier_values = [float(robust_with_jacobian[name]) for name in names]
                rows = [np.asarray(row, dtype=np.float64) for row in jacobian]
            else:
                rows = [np.zeros(action.size, dtype=np.float64) for _ in names]
            rhs = [-value for value in barrier_values]
        except (FloatingPointError, ValueError, RuntimeError):
            names = []
            barrier_values = []
            rows = []
            rhs = []
        recoverability = self._recoverability_fields(safety_observation, directive)
        actions, diagnostics = self._fallback(
            action,
            observation,
            barrier_values,
            names,
            rows,
            rhs,
            started,
            reason,
            failure_category=category,
            precondition_valid=precondition_valid,
            command_authority=directive,
            solver_backend=(
                str(self.config.execution_linearization_backend) + "_rollout_jacobian"
                if str(self.config.execution_linearization_backend) == "analytic"
                else "finite_difference_dykstra"
            ),
            force_zero_action=bool(force_zero_action),
            queue_override_slots=int(override_slots),
            recoverability=recoverability,
        )
        # A fallback is only a command proposal until the same independent
        # execution certificate accepts it. Try a small, deterministic set of
        # conservative candidates so the diagnostic distinguishes a certified
        # fallback from a best-effort recovery command.
        candidate_specs: list[tuple[str, np.ndarray]] = [("barrier_recovery", actions)]
        if not force_zero_action:
            candidate_specs.extend(
                [
                    ("zero_action", np.zeros_like(actions)),
                    ("nominal_clipped", self._bounded_execution_action(desired, observation)),
                    (
                        "recovery_nominal_blend",
                        self._bounded_execution_action(0.5 * (actions + desired), observation),
                    ),
                ]
            )
        last_emergency_brake = bool(
            isinstance(safety_observation.get("execution", {}), dict)
            and safety_observation.get("execution", {}).get("last_emergency_brake_requested", False)
        )
        directive_specs: list[tuple[CommandAuthorityDirective, str]] = [(directive, "selected")]
        if directive.emergency_brake and directive.mode != "immutable" and last_emergency_brake:
            # An emergency brake is a queue-clearing event. Repeating it every
            # tick would keep replacing newly appended commands with zeros.
            directive_specs = [(CommandAuthorityDirective(mode=directive.mode, emergency_brake=False), "resume")]
        certified_candidates: list[
            tuple[str, np.ndarray, Any, tuple[float, float, float] | None, int, str, CommandAuthorityDirective]
        ] = []
        for candidate_directive, directive_label in directive_specs:
            candidate_horizons = [preview_steps]
            if (
                bool(self.config.execution_fallback_receding_step_enabled)
                and candidate_directive.mode != "immutable"
                and bool(recoverability.get("prefix_admissible", False))
            ):
                candidate_horizons.append(1)
            for horizon in candidate_horizons:
                scope = "full_horizon" if horizon == preview_steps else "one_step_receding"
                for candidate_label, candidate in candidate_specs:
                    try:
                        candidate_certificate = check_execution_rollout_safety(
                            safety_observation,
                            candidate,
                            dt=float(self.env.dt),
                            drone_radius=float(self.env.agents["drone_radius"]),
                            max_speed_mps=float(self.config.max_speed_mps),
                            max_acceleration_mps2=float(self.config.max_acceleration_mps2),
                            safety_margin_m=float(self.config.safety_margin_m),
                            robust_margin_m=float(self.config.robust_margin_m),
                            tolerance=float(self.config.solver_tolerance),
                            action_change_limit_mps=self.config.action_change_limit_mps,
                            horizon_steps=horizon,
                            reachable_tube_multiplier=float(self.config.execution_reachable_tube_multiplier),
                            continuous_segment_constraints=bool(self.config.execution_continuous_segment_constraints),
                            continuous_segment_subdivisions=int(self.config.execution_continuous_segment_subdivisions),
                            command_authority=candidate_directive,
                        )
                    except (FloatingPointError, ValueError, RuntimeError):
                        continue
                    if bool(candidate_certificate.valid):
                        try:
                            progress = self._goal_progress(observation, candidate, candidate_directive)
                        except (FloatingPointError, ValueError, RuntimeError):
                            progress = None
                        certified_candidates.append(
                            (
                                candidate_label if directive_label == "selected" else f"{candidate_label}_{directive_label}",
                                np.asarray(candidate, dtype=np.float64),
                                candidate_certificate,
                                progress,
                                horizon,
                                scope,
                                candidate_directive,
                            )
                        )
        if certified_candidates:
            if any(item[3] is not None for item in certified_candidates):
                def candidate_score(
                    item: tuple[
                        str,
                        np.ndarray,
                        Any,
                        tuple[float, float, float] | None,
                        int,
                        str,
                        CommandAuthorityDirective,
                    ],
                ) -> float:
                    progress = item[3]
                    progress_value = 0.0 if progress is None else float(progress[2])
                    return float(
                        self.config.fallback_progress_weight * progress_value
                        + self.config.fallback_barrier_weight * item[2].minimum_robust_barrier_m
                        - 1.0e-6 * float(item[6].emergency_brake)
                    )

                selected = max(certified_candidates, key=candidate_score)
            else:
                selected = certified_candidates[0]
            (
                selected_label,
                certified_action,
                certified_certificate,
                progress,
                selected_horizon,
                selected_scope,
                selected_directive,
            ) = selected
            actions = certified_action
            diagnostics = replace(
                diagnostics,
                status="fallback_certified",
                certificate_valid=True,
                barrier_values_m=dict(certified_certificate.barrier_values_m),
                minimum_barrier_value_m=float(certified_certificate.minimum_robust_barrier_m),
                assumptions={**diagnostics.assumptions, **certified_certificate.assumptions},
                fallback_candidate_type=selected_label,
                fallback_target_distance_before_m=(float(progress[0]) if progress is not None else float("nan")),
                fallback_target_distance_after_m=(float(progress[1]) if progress is not None else float("nan")),
                fallback_goal_progress_m=(float(progress[2]) if progress is not None else 0.0),
                fallback_certificate_horizon_steps=int(selected_horizon),
                fallback_certificate_scope=selected_scope,
                command_authority=selected_directive.as_dict(),
                emergency_brake_requested=bool(selected_directive.emergency_brake),
                queue_override_slots=int(certified_certificate.assumptions.get("queue_override_slots", 0.0)),
                **self._recoverability_fields(safety_observation, selected_directive),
            )
        return actions, diagnostics

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
        command_authority: CommandAuthorityDirective | None = None,
        solver_backend: str = "slsqp",
        force_zero_action: bool = False,
        queue_override_slots: int = 0,
        recoverability: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, RobustCBFQPDiagnostics]:
        if force_zero_action or self.config.fallback_policy == "zero_action":
            actions = np.zeros_like(desired)
            recovery_action_used = bool(force_zero_action)
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
            solver_backend=solver_backend,
            command_authority=None if command_authority is None else command_authority.as_dict(),
            emergency_brake_requested=False if command_authority is None else bool(command_authority.emergency_brake),
            queue_override_slots=int(queue_override_slots),
            **dict(recoverability or {}),
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
