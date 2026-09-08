"""Independent one-step safety certificate checks for the velocity benchmark.

The checker intentionally does not call the QP implementation.  It recomputes
the next state and all geometric margins from the public observation, so a
solver diagnostic cannot certify an action that violates the actual contract.
This is a one-step certificate under the supplied velocity-level assumptions,
not a learned CLBF or a real-flight guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from encirclement3d.execution_dynamics import (
    CommandAuthorityDirective,
    apply_command_authority,
    command_authority_from_observation,
    parameters_from_observation,
    position_uncertainty_radii,
    queue_from_observation,
    rollout_execution,
    rollout_execution_with_action_jacobian,
)


def _unit(value: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm > 1.0e-12:
        return vector / norm
    if fallback is not None:
        return np.asarray(fallback, dtype=np.float64).copy()
    return np.zeros_like(vector)


def _obstacle_geometry(obstacle: Any) -> tuple[str, np.ndarray, float, float, np.ndarray | None]:
    if isinstance(obstacle, Mapping):
        return (
            str(obstacle.get("shape", "cylinder")),
            np.asarray(obstacle["center_xy"], dtype=np.float64),
            float(obstacle.get("radius", 0.0)),
            float(obstacle["height"]),
            None if obstacle.get("half_extents_xy") is None else np.asarray(obstacle["half_extents_xy"], dtype=np.float64),
        )
    return (
        str(obstacle.shape),
        np.asarray(obstacle.center_xy, dtype=np.float64),
        float(obstacle.radius),
        float(obstacle.height),
        None if obstacle.half_extents_xy is None else np.asarray(obstacle.half_extents_xy, dtype=np.float64),
    )


def _signed_obstacle_clearance(position: np.ndarray, obstacle: Any) -> float:
    """Return signed distance to a finite cylinder or axis-aligned box."""

    point = np.asarray(position, dtype=np.float64)
    shape, center_xy, radius, height, half_extents_xy = _obstacle_geometry(obstacle)
    if shape == "cylinder":
        delta_xy = point[:2] - center_xy
        radial_gap = float(np.linalg.norm(delta_xy) - radius)
        if 0.0 <= point[2] <= height:
            return radial_gap
        nearest_z = 0.0 if point[2] < 0.0 else height
        vertical_gap = abs(float(point[2]) - nearest_z)
        if radial_gap <= 0.0:
            return vertical_gap
        return float(np.hypot(radial_gap, vertical_gap))

    if half_extents_xy is None:
        half_extents_xy = np.array([radius, radius], dtype=np.float64)
    center = np.array([center_xy[0], center_xy[1], 0.5 * height], dtype=np.float64)
    half = np.array([half_extents_xy[0], half_extents_xy[1], 0.5 * height], dtype=np.float64)
    signed_axes = np.abs(point - center) - half
    outside = np.maximum(signed_axes, 0.0)
    outside_norm = float(np.linalg.norm(outside))
    if outside_norm > 1.0e-12:
        return outside_norm
    return -float(np.min(-signed_axes))


def _signed_obstacle_clearance_and_gradient(position: np.ndarray, obstacle: Any) -> tuple[float, np.ndarray]:
    """Return signed obstacle clearance and one deterministic local gradient."""

    point = np.asarray(position, dtype=np.float64)
    shape, center_xy, radius, height, half_extents_xy = _obstacle_geometry(obstacle)
    if shape == "cylinder":
        delta_xy = point[:2] - center_xy
        radial_norm = float(np.linalg.norm(delta_xy))
        radial_gap = radial_norm - radius
        radial_gradient = _unit(
            np.array([delta_xy[0], delta_xy[1], 0.0], dtype=np.float64),
            fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        if 0.0 <= point[2] <= height:
            return radial_gap, radial_gradient
        nearest_z = 0.0 if point[2] < 0.0 else height
        vertical_delta = float(point[2] - nearest_z)
        vertical_gap = abs(vertical_delta)
        vertical_gradient = np.array([0.0, 0.0, -1.0 if vertical_delta < 0.0 else 1.0], dtype=np.float64)
        if radial_gap <= 0.0:
            return vertical_gap, vertical_gradient
        distance = float(np.hypot(radial_gap, vertical_gap))
        if distance <= 1.0e-12:
            return 0.0, radial_gradient
        return distance, (radial_gap * radial_gradient + vertical_gap * vertical_gradient) / distance

    if half_extents_xy is None:
        half_extents_xy = np.array([radius, radius], dtype=np.float64)
    center = np.array([center_xy[0], center_xy[1], 0.5 * height], dtype=np.float64)
    half = np.array([half_extents_xy[0], half_extents_xy[1], 0.5 * height], dtype=np.float64)
    delta = point - center
    signed_axes = np.abs(delta) - half
    outside = np.maximum(signed_axes, 0.0)
    outside_norm = float(np.linalg.norm(outside))
    if outside_norm > 1.0e-12:
        gradient = (outside * np.sign(delta)) / outside_norm
        return outside_norm, gradient
    axis = int(np.argmax(signed_axes))
    gradient = np.zeros(3, dtype=np.float64)
    gradient[axis] = 1.0 if delta[axis] >= 0.0 else -1.0
    return -float(np.min(-signed_axes)), gradient


@dataclass(frozen=True)
class SafetyCertificateResult:
    """Auditable result of an independent one-step safety check."""

    valid: bool
    status: str
    current_state_safe: bool
    next_state_safe: bool
    current_min_barrier_m: float
    next_min_barrier_m: float
    maximum_action_norm_mps: float
    maximum_action_change_mps: float
    barrier_values_m: dict[str, float]
    violations: tuple[str, ...]
    assumptions: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status,
            "current_state_safe": self.current_state_safe,
            "next_state_safe": self.next_state_safe,
            "current_min_barrier_m": self.current_min_barrier_m,
            "next_min_barrier_m": self.next_min_barrier_m,
            "maximum_action_norm_mps": self.maximum_action_norm_mps,
            "maximum_action_change_mps": self.maximum_action_change_mps,
            "barrier_values_m": dict(self.barrier_values_m),
            "violations": list(self.violations),
            "assumptions": dict(self.assumptions),
        }


@dataclass(frozen=True)
class ExecutionRolloutCertificateResult:
    """Independent multi-step certificate for the shared execution contract."""

    valid: bool
    status: str
    current_state_safe: bool
    rollout_state_safe: bool
    horizon_steps: int
    minimum_robust_barrier_m: float
    minimum_nominal_barrier_m: float
    barrier_values_m: dict[str, float]
    nominal_barrier_values_m: dict[str, float]
    violations: tuple[str, ...]
    assumptions: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status,
            "current_state_safe": self.current_state_safe,
            "rollout_state_safe": self.rollout_state_safe,
            "horizon_steps": self.horizon_steps,
            "minimum_robust_barrier_m": self.minimum_robust_barrier_m,
            "minimum_nominal_barrier_m": self.minimum_nominal_barrier_m,
            "barrier_values_m": dict(self.barrier_values_m),
            "nominal_barrier_values_m": dict(self.nominal_barrier_values_m),
            "violations": list(self.violations),
            "assumptions": dict(self.assumptions),
        }


@dataclass(frozen=True)
class SweptVolumeCertificateResult:
    """Continuous-time sampled swept-volume safety certificate."""

    valid: bool
    status: str
    current_state_safe: bool
    swept_volume_safe: bool
    horizon_steps: int
    subdivisions_per_step: int
    sample_count: int
    minimum_robust_barrier_m: float
    minimum_nominal_barrier_m: float
    violations: tuple[str, ...]
    assumptions: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status,
            "current_state_safe": self.current_state_safe,
            "swept_volume_safe": self.swept_volume_safe,
            "horizon_steps": self.horizon_steps,
            "subdivisions_per_step": self.subdivisions_per_step,
            "sample_count": self.sample_count,
            "minimum_robust_barrier_m": self.minimum_robust_barrier_m,
            "minimum_nominal_barrier_m": self.minimum_nominal_barrier_m,
            "violations": list(self.violations),
            "assumptions": dict(self.assumptions),
        }


@dataclass(frozen=True)
class ContinuousSegmentCertificateResult:
    """Conservative all-points check for piecewise-linear benchmark segments.

    The lower bound follows from the 1-Lipschitz signed-distance and pairwise
    separation barriers.  It certifies the interpolation implied by the
    benchmark position update, not an unmodeled real-airframe trajectory.
    """

    valid: bool
    status: str
    current_state_safe: bool
    continuous_segment_safe: bool
    horizon_steps: int
    segment_count: int
    minimum_robust_barrier_m: float
    minimum_nominal_barrier_m: float
    violations: tuple[str, ...]
    assumptions: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "status": self.status,
            "current_state_safe": self.current_state_safe,
            "continuous_segment_safe": self.continuous_segment_safe,
            "horizon_steps": self.horizon_steps,
            "segment_count": self.segment_count,
            "minimum_robust_barrier_m": self.minimum_robust_barrier_m,
            "minimum_nominal_barrier_m": self.minimum_nominal_barrier_m,
            "violations": list(self.violations),
            "assumptions": dict(self.assumptions),
        }


def _barriers(
    positions: np.ndarray,
    obstacles: list[Any] | tuple[Any, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    radius: float,
    effective_margin: float,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for defender_index, position in enumerate(positions):
        for obstacle_index, obstacle in enumerate(obstacles):
            values[f"obstacle[{obstacle_index}]/{defender_index}"] = (
                _signed_obstacle_clearance(position, obstacle) - radius - effective_margin
            )
        for axis in range(3):
            values[f"boundary_lower[{axis}]/{defender_index}"] = (
                position[axis] - lower[axis] - radius - effective_margin
            )
            values[f"boundary_upper[{axis}]/{defender_index}"] = (
                upper[axis] - position[axis] - radius - effective_margin
            )
    minimum_distance = 2.0 * radius + effective_margin
    for first in range(len(positions)):
        for second in range(first + 1, len(positions)):
            values[f"inter_agent[{first},{second}]"] = (
                float(np.linalg.norm(positions[first] - positions[second])) - minimum_distance
            )
    return values


def check_one_step_safety(
    observation: Mapping[str, Any],
    safe_actions: np.ndarray,
    *,
    dt: float,
    drone_radius: float,
    max_speed_mps: float,
    max_acceleration_mps2: float,
    safety_margin_m: float,
    robust_margin_m: float,
    tolerance: float = 1.0e-6,
    action_change_limit_mps: float | None = None,
    enforce_action_change: bool = True,
) -> SafetyCertificateResult:
    """Check current and next-step geometry plus velocity/action limits."""

    positions = np.asarray(observation["defender_positions"], dtype=np.float64)
    velocities = np.asarray(observation.get("defender_velocities", np.zeros_like(positions)), dtype=np.float64)
    actions = np.asarray(safe_actions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[-1] != 3 or velocities.shape != positions.shape:
        raise ValueError("defender positions and velocities must have shape [defenders, 3]")
    if actions.shape != positions.shape:
        raise ValueError("safe_actions must have shape [defenders, 3]")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("observation state must be finite")

    lower = np.asarray(observation["world_lower_bounds"], dtype=np.float64)
    upper = np.asarray(observation["world_upper_bounds"], dtype=np.float64)
    obstacles = list(observation.get("obstacles", ()))
    effective_margin = float(safety_margin_m) + float(robust_margin_m)
    current = _barriers(positions, obstacles, lower, upper, float(drone_radius), effective_margin)

    assumptions = {
        "dt_seconds": float(dt),
        "drone_radius_m": float(drone_radius),
        "max_speed_mps": float(max_speed_mps),
        "max_acceleration_mps2": float(max_acceleration_mps2),
        "safety_margin_m": float(safety_margin_m),
        "robust_margin_m": float(robust_margin_m),
        "tolerance_m": float(tolerance),
        "action_change_check_enabled": float(bool(enforce_action_change)),
        "action_change_limit_mps": float(
            max_acceleration_mps2 * dt if action_change_limit_mps is None else action_change_limit_mps
        ),
    }
    if not np.isfinite(actions).all():
        return SafetyCertificateResult(
            valid=False,
            status="invalid_non_finite_action",
            current_state_safe=bool(min(current.values(), default=float("inf")) >= -float(tolerance)),
            next_state_safe=False,
            current_min_barrier_m=float(min(current.values(), default=float("inf"))),
            next_min_barrier_m=float("-inf"),
            maximum_action_norm_mps=float("inf"),
            maximum_action_change_mps=float("inf"),
            barrier_values_m=current,
            violations=("non_finite_action",),
            assumptions=assumptions,
        )

    next_positions = positions + float(dt) * actions
    next_barriers = _barriers(next_positions, obstacles, lower, upper, float(drone_radius), effective_margin)
    barrier_values = {f"current/{key}": value for key, value in current.items()}
    barrier_values.update({f"next/{key}": value for key, value in next_barriers.items()})
    current_minimum = float(min(current.values(), default=float("inf")))
    next_minimum = float(min(next_barriers.values(), default=float("inf")))
    action_norms = np.linalg.norm(actions, axis=1)
    action_changes = np.abs(actions - velocities)
    maximum_action_norm = float(np.max(action_norms, initial=0.0))
    maximum_action_change = float(np.max(action_changes, initial=0.0))
    violations: list[str] = []
    if current_minimum < -float(tolerance):
        violations.append("current_state_outside_safe_set")
    if next_minimum < -float(tolerance):
        violations.append("next_state_outside_safe_set")
    if maximum_action_norm > float(max_speed_mps) + float(tolerance):
        violations.append("speed_limit")
    change_limit = float(max_acceleration_mps2) * float(dt) if action_change_limit_mps is None else float(action_change_limit_mps)
    if enforce_action_change and maximum_action_change > change_limit + float(tolerance):
        violations.append("action_change_limit")
    current_state_safe = current_minimum >= -float(tolerance)
    next_state_safe = next_minimum >= -float(tolerance)
    valid = not violations
    return SafetyCertificateResult(
        valid=valid,
        status="valid" if valid else "invalid",
        current_state_safe=current_state_safe,
        next_state_safe=next_state_safe,
        current_min_barrier_m=current_minimum,
        next_min_barrier_m=next_minimum,
        maximum_action_norm_mps=maximum_action_norm,
        maximum_action_change_mps=maximum_action_change,
        barrier_values_m=barrier_values,
        violations=tuple(violations),
        assumptions=assumptions,
    )


def _robust_barriers(
    positions: np.ndarray,
    obstacles: list[Any] | tuple[Any, ...],
    lower: np.ndarray,
    upper: np.ndarray,
    radius: float,
    effective_margin: float,
    uncertainty_radii: np.ndarray,
) -> tuple[dict[str, float], dict[str, float]]:
    nominal = _barriers(positions, obstacles, lower, upper, radius, effective_margin)
    robust: dict[str, float] = {}
    for name, value in nominal.items():
        if name.startswith("inter_agent["):
            indices = name.removeprefix("inter_agent[").removesuffix("]").split(",")
            uncertainty = float(uncertainty_radii[int(indices[0])] + uncertainty_radii[int(indices[1])])
        else:
            uncertainty = float(uncertainty_radii[int(name.rsplit("/", 1)[-1])])
        robust[name] = float(value - uncertainty)
    return nominal, robust


def execution_barrier_values(
    observation: Mapping[str, Any],
    action: np.ndarray,
    *,
    dt: float,
    drone_radius: float,
    safety_margin_m: float,
    robust_margin_m: float,
    horizon_steps: int | None = None,
    swept_substeps: int = 4,
    command_authority: CommandAuthorityDirective | Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return nominal and uncertainty-robust barriers along executed rollout."""

    positions = np.asarray(observation["defender_positions"], dtype=np.float64)
    velocities = np.asarray(observation.get("defender_velocities", np.zeros_like(positions)), dtype=np.float64)
    actions = np.asarray(action, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[-1] != 3 or velocities.shape != positions.shape:
        raise ValueError("defender positions and velocities must have shape [defenders, 3]")
    if actions.shape != positions.shape:
        raise ValueError("action must have shape [defenders, 3]")
    lower = np.asarray(observation["world_lower_bounds"], dtype=np.float64)
    upper = np.asarray(observation["world_upper_bounds"], dtype=np.float64)
    obstacles = list(observation.get("obstacles", ()))
    parameters = parameters_from_observation(observation, dt)
    queue = queue_from_observation(observation, positions.shape[0])
    queue, directive, overridden_slots = apply_command_authority(
        queue,
        command_authority,
        allowed_mode=command_authority_from_observation(observation),
    )
    preview_steps = max(1, len(queue) + 1) if horizon_steps is None else int(horizon_steps)
    if preview_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    if int(swept_substeps) <= 0:
        raise ValueError("swept_substeps must be positive")
    rollout_positions, _rollout_velocities, _steps = rollout_execution(
        positions,
        velocities,
        queue,
        actions,
        parameters,
        horizon_steps=preview_steps,
    )
    uncertainty = position_uncertainty_radii(parameters, positions.shape[0], preview_steps)
    nominal_values: dict[str, float] = {}
    robust_values: dict[str, float] = {}
    for step_index, (future_positions, radii) in enumerate(zip(rollout_positions, uncertainty), start=1):
        previous_positions = positions if step_index == 1 else rollout_positions[step_index - 2]
        individual_gaps = np.linalg.norm(future_positions - previous_positions, axis=1) / float(swept_substeps)
        for subdivision in range(1, int(swept_substeps) + 1):
            fraction = float(subdivision) / float(swept_substeps)
            sample_positions = previous_positions + fraction * (future_positions - previous_positions)
            nominal, robust = _robust_barriers(
                sample_positions,
                obstacles,
                lower,
                upper,
                float(drone_radius),
                float(safety_margin_m) + float(robust_margin_m),
                radii,
            )
            for name, value in nominal.items():
                if name.startswith("inter_agent["):
                    first, second = name.removeprefix("inter_agent[").removesuffix("]").split(",")
                    gap = float(individual_gaps[int(first)] + individual_gaps[int(second)])
                else:
                    gap = float(individual_gaps[int(name.rsplit("/", 1)[-1])])
                key = f"step[{step_index}]/sweep[{subdivision}]/{name}"
                nominal_values[key] = float(value - gap)
                robust_values[key] = float(robust[name] - gap)
    assumptions = {
        "dt_seconds": float(dt),
        "horizon_steps": float(preview_steps),
        "action_delay_steps": float(parameters.action_delay_steps),
        "command_noise_bound_mps": float(parameters.command_noise_bound_mps),
        "tracking_alpha": float(parameters.tracking_alpha),
        "drag_gain": float(parameters.drag_gain),
        "max_speed_mps": float(parameters.max_speed_mps),
        "max_acceleration_mps2": float(parameters.max_acceleration_mps2),
        "mass_scale": float(parameters.mass_scale),
        "swept_substeps": float(swept_substeps),
        "command_authority_mode": float(
            {"immutable": 0, "replace_nonexecuting": 1, "flush_pending": 2}[directive.mode]
        ),
        "emergency_brake_requested": float(directive.emergency_brake),
        "queue_override_slots": float(overridden_slots),
    }
    return nominal_values, robust_values, assumptions


def execution_barrier_values_with_action_jacobian(
    observation: Mapping[str, Any],
    action: np.ndarray,
    *,
    dt: float,
    drone_radius: float,
    safety_margin_m: float,
    robust_margin_m: float,
    horizon_steps: int | None = None,
    swept_substeps: int = 4,
    command_authority: CommandAuthorityDirective | Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, float], np.ndarray, dict[str, float]]:
    """Evaluate execution barriers with local action derivatives.

    This shares the rollout convention of :func:`execution_barrier_values`.
    The returned Jacobian is ordered by the robust-barrier dictionary and is
    used only for sequential linearization; independent nonlinear certificates
    remain the acceptance criterion.
    """

    positions = np.asarray(observation["defender_positions"], dtype=np.float64)
    velocities = np.asarray(observation.get("defender_velocities", np.zeros_like(positions)), dtype=np.float64)
    actions = np.asarray(action, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[-1] != 3 or velocities.shape != positions.shape:
        raise ValueError("defender positions and velocities must have shape [defenders, 3]")
    if actions.shape != positions.shape:
        raise ValueError("action must have shape [defenders, 3]")
    lower = np.asarray(observation["world_lower_bounds"], dtype=np.float64)
    upper = np.asarray(observation["world_upper_bounds"], dtype=np.float64)
    obstacles = list(observation.get("obstacles", ()))
    parameters = parameters_from_observation(observation, dt)
    queue = queue_from_observation(observation, positions.shape[0])
    queue, directive, overridden_slots = apply_command_authority(
        queue,
        command_authority,
        allowed_mode=command_authority_from_observation(observation),
    )
    preview_steps = max(1, len(queue) + 1) if horizon_steps is None else int(horizon_steps)
    if preview_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    if int(swept_substeps) <= 0:
        raise ValueError("swept_substeps must be positive")
    rollout_positions, _rollout_velocities, _steps, position_jacobians, _velocity_jacobians = (
        rollout_execution_with_action_jacobian(
            positions,
            velocities,
            queue,
            actions,
            parameters,
            horizon_steps=preview_steps,
        )
    )
    uncertainty = position_uncertainty_radii(parameters, positions.shape[0], preview_steps)
    nominal_values: dict[str, float] = {}
    robust_values: dict[str, float] = {}
    gradients: list[np.ndarray] = []
    action_dimension = int(actions.size)
    effective_margin = float(safety_margin_m) + float(robust_margin_m)

    def append_value(key: str, nominal: float, robust: float, gradient: np.ndarray) -> None:
        nominal_values[key] = float(nominal)
        robust_values[key] = float(robust)
        gradients.append(np.asarray(gradient, dtype=np.float64).reshape(action_dimension))

    for step_index, (future_positions, radii, endpoint_jacobian) in enumerate(
        zip(rollout_positions, uncertainty, position_jacobians), start=1
    ):
        previous_positions = positions if step_index == 1 else rollout_positions[step_index - 2]
        previous_jacobian = (
            np.zeros_like(endpoint_jacobian) if step_index == 1 else position_jacobians[step_index - 2]
        )
        segment_jacobian = endpoint_jacobian - previous_jacobian
        displacement = future_positions - previous_positions
        for subdivision in range(1, int(swept_substeps) + 1):
            fraction = float(subdivision) / float(swept_substeps)
            sample_positions = previous_positions + fraction * displacement
            sample_jacobian = previous_jacobian + fraction * segment_jacobian
            for defender_index, position in enumerate(sample_positions):
                position_gradient = sample_jacobian[
                    defender_index * 3 : defender_index * 3 + 3
                ]
                motion = displacement[defender_index]
                motion_norm = float(np.linalg.norm(motion))
                if motion_norm <= 1.0e-12:
                    motion_gradient = np.zeros(action_dimension, dtype=np.float64)
                else:
                    motion_gradient = (motion / motion_norm) @ segment_jacobian[
                        defender_index * 3 : defender_index * 3 + 3
                    ]
                for obstacle_index, obstacle in enumerate(obstacles):
                    clearance, clearance_gradient = _signed_obstacle_clearance_and_gradient(position, obstacle)
                    nominal = clearance - float(drone_radius) - effective_margin - motion_norm / float(swept_substeps)
                    robust = nominal - float(radii[defender_index])
                    key = f"step[{step_index}]/sweep[{subdivision}]/obstacle[{obstacle_index}]/{defender_index}"
                    append_value(key, nominal, robust, clearance_gradient @ position_gradient - motion_gradient / float(swept_substeps))
                for axis in range(3):
                    lower_nominal = (
                        position[axis] - lower[axis] - float(drone_radius) - effective_margin - motion_norm / float(swept_substeps)
                    )
                    lower_key = f"step[{step_index}]/sweep[{subdivision}]/boundary_lower[{axis}]/{defender_index}"
                    append_value(
                        lower_key,
                        lower_nominal,
                        lower_nominal - float(radii[defender_index]),
                        position_gradient[axis] - motion_gradient / float(swept_substeps),
                    )
                    upper_nominal = (
                        upper[axis] - position[axis] - float(drone_radius) - effective_margin - motion_norm / float(swept_substeps)
                    )
                    upper_key = f"step[{step_index}]/sweep[{subdivision}]/boundary_upper[{axis}]/{defender_index}"
                    append_value(
                        upper_key,
                        upper_nominal,
                        upper_nominal - float(radii[defender_index]),
                        -position_gradient[axis] - motion_gradient / float(swept_substeps),
                    )
            minimum_distance = 2.0 * float(drone_radius) + effective_margin
            for first in range(len(sample_positions)):
                for second in range(first + 1, len(sample_positions)):
                    relative = sample_positions[first] - sample_positions[second]
                    relative_norm = float(np.linalg.norm(relative))
                    normal = _unit(relative, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
                    relative_gradient = normal @ (
                        sample_jacobian[first * 3 : first * 3 + 3]
                        - sample_jacobian[second * 3 : second * 3 + 3]
                    )
                    first_motion = displacement[first]
                    second_motion = displacement[second]
                    first_norm = float(np.linalg.norm(first_motion))
                    second_norm = float(np.linalg.norm(second_motion))
                    first_gradient = (
                        np.zeros(action_dimension, dtype=np.float64)
                        if first_norm <= 1.0e-12
                        else (first_motion / first_norm) @ segment_jacobian[first * 3 : first * 3 + 3]
                    )
                    second_gradient = (
                        np.zeros(action_dimension, dtype=np.float64)
                        if second_norm <= 1.0e-12
                        else (second_motion / second_norm) @ segment_jacobian[second * 3 : second * 3 + 3]
                    )
                    motion_penalty = (first_norm + second_norm) / float(swept_substeps)
                    nominal = relative_norm - minimum_distance - motion_penalty
                    robust = nominal - float(radii[first] + radii[second])
                    key = f"step[{step_index}]/sweep[{subdivision}]/inter_agent[{first},{second}]"
                    append_value(
                        key,
                        nominal,
                        robust,
                        relative_gradient - (first_gradient + second_gradient) / float(swept_substeps),
                    )
    assumptions = {
        "dt_seconds": float(dt),
        "horizon_steps": float(preview_steps),
        "action_delay_steps": float(parameters.action_delay_steps),
        "command_noise_bound_mps": float(parameters.command_noise_bound_mps),
        "tracking_alpha": float(parameters.tracking_alpha),
        "drag_gain": float(parameters.drag_gain),
        "max_speed_mps": float(parameters.max_speed_mps),
        "max_acceleration_mps2": float(parameters.max_acceleration_mps2),
        "mass_scale": float(parameters.mass_scale),
        "swept_substeps": float(swept_substeps),
        "command_authority_mode": float(
            {"immutable": 0, "replace_nonexecuting": 1, "flush_pending": 2}[directive.mode]
        ),
        "emergency_brake_requested": float(directive.emergency_brake),
        "queue_override_slots": float(overridden_slots),
    }
    return nominal_values, robust_values, np.stack(gradients, axis=0), assumptions


def check_execution_rollout_safety(
    observation: Mapping[str, Any],
    action: np.ndarray,
    *,
    dt: float,
    drone_radius: float,
    max_speed_mps: float,
    max_acceleration_mps2: float,
    safety_margin_m: float,
    robust_margin_m: float,
    tolerance: float = 1.0e-6,
    action_change_limit_mps: float | None = None,
    horizon_steps: int | None = None,
    command_authority: CommandAuthorityDirective | Mapping[str, Any] | None = None,
) -> ExecutionRolloutCertificateResult:
    """Check the actual queued execution model over a finite preview horizon."""

    positions = np.asarray(observation["defender_positions"], dtype=np.float64)
    velocities = np.asarray(observation.get("defender_velocities", np.zeros_like(positions)), dtype=np.float64)
    actions = np.asarray(action, dtype=np.float64)
    current = _barriers(
        positions,
        list(observation.get("obstacles", ())),
        np.asarray(observation["world_lower_bounds"], dtype=np.float64),
        np.asarray(observation["world_upper_bounds"], dtype=np.float64),
        float(drone_radius),
        float(safety_margin_m) + float(robust_margin_m),
    )
    current_minimum = float(min(current.values(), default=float("inf")))
    nominal_values, robust_values, assumptions = execution_barrier_values(
        observation,
        actions,
        dt=dt,
        drone_radius=drone_radius,
        safety_margin_m=safety_margin_m,
        robust_margin_m=robust_margin_m,
        horizon_steps=horizon_steps,
        command_authority=command_authority,
    )
    violations: list[str] = []
    tolerance_value = float(tolerance)
    if current_minimum < -tolerance_value:
        violations.append("current_state_outside_safe_set")
    robust_minimum = float(min(robust_values.values(), default=float("inf")))
    nominal_minimum = float(min(nominal_values.values(), default=float("inf")))
    if robust_minimum < -tolerance_value:
        violations.append("execution_rollout_outside_robust_safe_set")
    if not np.isfinite(actions).all():
        violations.append("non_finite_action")
    action_norm = float(np.max(np.linalg.norm(actions, axis=1), initial=0.0))
    if action_norm > float(max_speed_mps) + tolerance_value:
        violations.append("speed_limit")
    change_limit = float(max_acceleration_mps2) * float(dt) if action_change_limit_mps is None else float(action_change_limit_mps)
    action_change = float(np.max(np.abs(actions - velocities), initial=0.0))
    if action_change > change_limit + tolerance_value:
        violations.append("action_change_limit")
    valid = not violations
    return ExecutionRolloutCertificateResult(
        valid=valid,
        status="valid" if valid else "invalid",
        current_state_safe=bool(current_minimum >= -tolerance_value),
        rollout_state_safe=bool(robust_minimum >= -tolerance_value),
        horizon_steps=int(assumptions["horizon_steps"]),
        minimum_robust_barrier_m=robust_minimum,
        minimum_nominal_barrier_m=nominal_minimum,
        barrier_values_m=robust_values,
        nominal_barrier_values_m=nominal_values,
        violations=tuple(violations),
        assumptions=assumptions,
    )


def check_execution_swept_volume_safety(
    observation: Mapping[str, Any],
    action: np.ndarray,
    *,
    dt: float,
    drone_radius: float,
    safety_margin_m: float,
    robust_margin_m: float,
    horizon_steps: int | None = None,
    subdivisions_per_step: int = 4,
    tolerance: float = 1.0e-6,
    command_authority: CommandAuthorityDirective | Mapping[str, Any] | None = None,
) -> SweptVolumeCertificateResult:
    """Sample each executed motion segment, including obstacle volume crossing."""

    if int(subdivisions_per_step) <= 0:
        raise ValueError("subdivisions_per_step must be positive")
    positions = np.asarray(observation["defender_positions"], dtype=np.float64)
    velocities = np.asarray(observation.get("defender_velocities", np.zeros_like(positions)), dtype=np.float64)
    lower = np.asarray(observation["world_lower_bounds"], dtype=np.float64)
    upper = np.asarray(observation["world_upper_bounds"], dtype=np.float64)
    obstacles = list(observation.get("obstacles", ()))
    parameters = parameters_from_observation(observation, dt)
    queue = queue_from_observation(observation, positions.shape[0])
    queue, directive, overridden_slots = apply_command_authority(
        queue,
        command_authority,
        allowed_mode=command_authority_from_observation(observation),
    )
    preview_steps = max(1, len(queue) + 1) if horizon_steps is None else int(horizon_steps)
    if preview_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    rollout_positions, _rollout_velocities, _steps = rollout_execution(
        positions,
        velocities,
        queue,
        np.asarray(action, dtype=np.float64),
        parameters,
        horizon_steps=preview_steps,
    )
    uncertainty = position_uncertainty_radii(parameters, positions.shape[0], preview_steps)
    initial_barriers = _barriers(
        positions,
        obstacles,
        lower,
        upper,
        float(drone_radius),
        float(safety_margin_m) + float(robust_margin_m),
    )
    nominal_minimum = float(min(initial_barriers.values(), default=float("inf")))
    robust_minimum = nominal_minimum
    sample_count = 1
    for step_index, endpoint in enumerate(rollout_positions):
        start = positions if step_index == 0 else rollout_positions[step_index - 1]
        individual_gaps = np.linalg.norm(endpoint - start, axis=1) / float(subdivisions_per_step)
        for subdivision in range(1, int(subdivisions_per_step) + 1):
            fraction = float(subdivision) / float(subdivisions_per_step)
            sample = start + fraction * (endpoint - start)
            nominal, robust = _robust_barriers(
                sample,
                obstacles,
                lower,
                upper,
                float(drone_radius),
                float(safety_margin_m) + float(robust_margin_m),
                uncertainty[step_index],
            )
            nominal_adjusted: list[float] = []
            robust_adjusted: list[float] = []
            for name, value in nominal.items():
                if name.startswith("inter_agent["):
                    first, second = name.removeprefix("inter_agent[").removesuffix("]").split(",")
                    gap = float(individual_gaps[int(first)] + individual_gaps[int(second)])
                else:
                    gap = float(individual_gaps[int(name.rsplit("/", 1)[-1])])
                nominal_adjusted.append(float(value - gap))
                robust_adjusted.append(float(robust[name] - gap))
            nominal_minimum = min(nominal_minimum, float(min(nominal_adjusted, default=float("inf"))))
            robust_minimum = min(robust_minimum, float(min(robust_adjusted, default=float("inf"))))
            sample_count += 1
    tolerance_value = float(tolerance)
    current_safe = bool(min(initial_barriers.values(), default=float("inf")) >= -tolerance_value)
    swept_safe = bool(robust_minimum >= -tolerance_value)
    violations: list[str] = []
    if not current_safe:
        violations.append("current_state_outside_safe_set")
    if not swept_safe:
        violations.append("swept_volume_outside_robust_safe_set")
    return SweptVolumeCertificateResult(
        valid=not violations,
        status="valid" if not violations else "invalid",
        current_state_safe=current_safe,
        swept_volume_safe=swept_safe,
        horizon_steps=preview_steps,
        subdivisions_per_step=int(subdivisions_per_step),
        sample_count=sample_count,
        minimum_robust_barrier_m=float(robust_minimum),
        minimum_nominal_barrier_m=float(nominal_minimum),
        violations=tuple(violations),
        assumptions={
            "dt_seconds": float(dt),
            "horizon_steps": float(preview_steps),
            "subdivisions_per_step": float(subdivisions_per_step),
            "command_noise_bound_mps": float(parameters.command_noise_bound_mps),
            "tracking_alpha": float(parameters.tracking_alpha),
            "drag_gain": float(parameters.drag_gain),
            "max_speed_mps": float(parameters.max_speed_mps),
                "max_acceleration_mps2": float(parameters.max_acceleration_mps2),
                "command_authority_mode": float(
                    {"immutable": 0, "replace_nonexecuting": 1, "flush_pending": 2}[directive.mode]
                ),
                "emergency_brake_requested": float(directive.emergency_brake),
                "queue_override_slots": float(overridden_slots),
        },
    )


def check_execution_continuous_segment_safety(
    observation: Mapping[str, Any],
    action: np.ndarray,
    *,
    dt: float,
    drone_radius: float,
    safety_margin_m: float,
    robust_margin_m: float,
    horizon_steps: int | None = None,
    tolerance: float = 1.0e-6,
    command_authority: CommandAuthorityDirective | Mapping[str, Any] | None = None,
) -> ContinuousSegmentCertificateResult:
    """Certify every point on each linear segment of the benchmark rollout.

    Each obstacle/boundary signed clearance is 1-Lipschitz in position.  The
    pairwise separation barrier is 1-Lipschitz in relative position.  For a
    segment, the midpoint can be at most half its path length from an endpoint;
    subtracting that distance yields a lower bound over the entire segment.
    """

    positions = np.asarray(observation["defender_positions"], dtype=np.float64)
    velocities = np.asarray(observation.get("defender_velocities", np.zeros_like(positions)), dtype=np.float64)
    lower = np.asarray(observation["world_lower_bounds"], dtype=np.float64)
    upper = np.asarray(observation["world_upper_bounds"], dtype=np.float64)
    obstacles = list(observation.get("obstacles", ()))
    parameters = parameters_from_observation(observation, dt)
    queue = queue_from_observation(observation, positions.shape[0])
    queue, directive, overridden_slots = apply_command_authority(
        queue,
        command_authority,
        allowed_mode=command_authority_from_observation(observation),
    )
    preview_steps = max(1, len(queue) + 1) if horizon_steps is None else int(horizon_steps)
    if preview_steps <= 0:
        raise ValueError("horizon_steps must be positive")
    rollout_positions, _rollout_velocities, _steps = rollout_execution(
        positions,
        velocities,
        queue,
        np.asarray(action, dtype=np.float64),
        parameters,
        horizon_steps=preview_steps,
    )
    uncertainty = position_uncertainty_radii(parameters, positions.shape[0], preview_steps)
    effective_margin = float(safety_margin_m) + float(robust_margin_m)
    initial = _barriers(positions, obstacles, lower, upper, float(drone_radius), effective_margin)
    current_minimum = float(min(initial.values(), default=float("inf")))
    robust_minimum = current_minimum
    nominal_minimum = current_minimum
    for step_index, endpoint in enumerate(rollout_positions):
        start = positions if step_index == 0 else rollout_positions[step_index - 1]
        # Use the end-of-step radius at both endpoints so uncertainty is valid
        # everywhere on a segment under the nondecreasing tube contract.
        nominal_start, robust_start = _robust_barriers(
            start,
            obstacles,
            lower,
            upper,
            float(drone_radius),
            effective_margin,
            uncertainty[step_index],
        )
        nominal_end, robust_end = _robust_barriers(
            endpoint,
            obstacles,
            lower,
            upper,
            float(drone_radius),
            effective_margin,
            uncertainty[step_index],
        )
        individual_lengths = np.linalg.norm(endpoint - start, axis=1)
        for name, start_value in nominal_start.items():
            if name.startswith("inter_agent["):
                first, second = name.removeprefix("inter_agent[").removesuffix("]").split(",")
                half_path_bound = 0.5 * (individual_lengths[int(first)] + individual_lengths[int(second)])
            else:
                defender_index = int(name.rsplit("/", 1)[-1])
                half_path_bound = 0.5 * individual_lengths[defender_index]
            nominal_lower = min(float(start_value), float(nominal_end[name])) - float(half_path_bound)
            robust_lower = min(float(robust_start[name]), float(robust_end[name])) - float(half_path_bound)
            nominal_minimum = min(nominal_minimum, nominal_lower)
            robust_minimum = min(robust_minimum, robust_lower)
    tolerance_value = float(tolerance)
    current_safe = bool(current_minimum >= -tolerance_value)
    continuous_safe = bool(robust_minimum >= -tolerance_value)
    violations: list[str] = []
    if not current_safe:
        violations.append("current_state_outside_safe_set")
    if not continuous_safe:
        violations.append("continuous_segment_outside_robust_safe_set")
    return ContinuousSegmentCertificateResult(
        valid=not violations,
        status="valid" if not violations else "invalid",
        current_state_safe=current_safe,
        continuous_segment_safe=continuous_safe,
        horizon_steps=preview_steps,
        segment_count=preview_steps,
        minimum_robust_barrier_m=float(robust_minimum),
        minimum_nominal_barrier_m=float(nominal_minimum),
        violations=tuple(violations),
        assumptions={
            "dt_seconds": float(dt),
            "horizon_steps": float(preview_steps),
            "command_noise_bound_mps": float(parameters.command_noise_bound_mps),
            "tracking_alpha": float(parameters.tracking_alpha),
            "drag_gain": float(parameters.drag_gain),
            "max_speed_mps": float(parameters.max_speed_mps),
            "max_acceleration_mps2": float(parameters.max_acceleration_mps2),
            "continuous_interpolation_contract": 1.0,
            "signed_distance_lipschitz_bound": 1.0,
            "command_authority_mode": float(
                {"immutable": 0, "replace_nonexecuting": 1, "flush_pending": 2}[directive.mode]
            ),
            "emergency_brake_requested": float(directive.emergency_brake),
            "queue_override_slots": float(overridden_slots),
        },
    )


__all__ = [
    "ContinuousSegmentCertificateResult",
    "ExecutionRolloutCertificateResult",
    "SafetyCertificateResult",
    "SweptVolumeCertificateResult",
    "check_execution_rollout_safety",
    "check_execution_continuous_segment_safety",
    "check_execution_swept_volume_safety",
    "check_one_step_safety",
    "execution_barrier_values",
    "execution_barrier_values_with_action_jacobian",
]
