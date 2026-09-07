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


__all__ = ["SafetyCertificateResult", "check_one_step_safety"]
