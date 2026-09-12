"""Reachability-normalized interception costs for finite-horizon planning.

The helper uses a one-dimensional, acceleration- and speed-limited arrival-time
surrogate along the line of sight.  It is deliberately exposed as a heuristic
cost, not as a complete 3-D reachable-set proof.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any

import numpy as np


_DEFAULT_FORMATION_SLOT_DIRECTIONS = np.asarray(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ],
    dtype=np.float64,
)


def minimum_arrival_time(
    distance_m: np.ndarray | float,
    velocity_projection_mps: np.ndarray | float,
    *,
    max_speed_mps: float,
    max_acceleration_mps2: float,
) -> np.ndarray:
    """Compute a bounded 1-D minimum arrival-time surrogate.

    Only velocity directed toward the point receives credit.  A defender
    moving away from the point therefore starts with zero useful projection,
    which avoids optimistic negative-time or braking assumptions.
    """

    distance = np.asarray(distance_m, dtype=np.float64)
    projection = np.asarray(velocity_projection_mps, dtype=np.float64)
    if distance.shape != projection.shape:
        raise ValueError("distance and velocity projection must have matching shapes")
    speed = float(max_speed_mps)
    acceleration = float(max_acceleration_mps2)
    if not np.isfinite([speed, acceleration]).all() or speed <= 0.0 or acceleration <= 0.0:
        raise ValueError("max speed and acceleration must be finite and positive")
    if not np.isfinite(distance).all() or not np.isfinite(projection).all():
        raise ValueError("distance and velocity projection must be finite")
    distance = np.maximum(distance, 0.0)
    useful_speed = np.clip(projection, 0.0, speed)
    accelerate_time = np.maximum((speed - useful_speed) / acceleration, 0.0)
    accelerate_distance = useful_speed * accelerate_time + 0.5 * acceleration * accelerate_time**2
    accelerating_time = (
        np.sqrt(np.maximum(useful_speed**2 + 2.0 * acceleration * distance, 0.0)) - useful_speed
    ) / acceleration
    cruise_time = accelerate_time + (distance - accelerate_distance) / speed
    return np.where(distance <= accelerate_distance, accelerating_time, cruise_time)


def reachability_normalized_interception_cost(
    defender_position_paths: np.ndarray,
    defender_velocity_paths: np.ndarray,
    target_paths: np.ndarray,
    *,
    dt_seconds: float,
    max_speed_mps: float,
    max_acceleration_mps2: float,
    time_margin_s: float = 0.15,
    time_scale_s: float = 0.50,
    target_tube_radius_m: np.ndarray | tuple[float, ...] | None = None,
    activation_slack_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return RNIC cost, best slack and per-defender arrival times.

    Parameters use shapes ``[sequence, horizon, defender, 3]`` for defender
    paths and ``[scenario, horizon, 3]`` for target paths.  The cost has shape
    ``[sequence, scenario]``.  At each horizon step, the best defender's
    arrival slack is compared with the available time.  Positive shortfall is
    normalized by ``time_scale_s`` before squaring.
    """

    positions = np.asarray(defender_position_paths, dtype=np.float64)
    velocities = np.asarray(defender_velocity_paths, dtype=np.float64)
    targets = np.asarray(target_paths, dtype=np.float64)
    if positions.ndim != 4 or positions.shape[-1] != 3:
        raise ValueError("defender_position_paths must have shape [sequence, horizon, defender, 3]")
    if velocities.shape != positions.shape:
        raise ValueError("defender_velocity_paths must match defender_position_paths")
    if targets.ndim != 3 or targets.shape[1] != positions.shape[1] or targets.shape[-1] != 3:
        raise ValueError("target_paths must have shape [scenario, horizon, 3]")
    if not np.isfinite([dt_seconds, time_margin_s, time_scale_s]).all() or dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be finite and positive")
    if time_margin_s < 0.0 or time_scale_s <= 0.0:
        raise ValueError("time margin must be non-negative and time scale positive")
    if activation_slack_s is not None and not np.isfinite(float(activation_slack_s)):
        raise ValueError("activation_slack_s must be finite when provided")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all() or not np.isfinite(targets).all():
        raise ValueError("reachability inputs must be finite")

    sequence_count, horizon, defender_count, _ = positions.shape
    scenario_count = targets.shape[0]
    if target_tube_radius_m is None:
        tube_radius = np.zeros(horizon, dtype=np.float64)
    else:
        tube_radius = np.asarray(target_tube_radius_m, dtype=np.float64)
        if (
            tube_radius.ndim != 1
            or tube_radius.shape[0] != horizon
            or not np.isfinite(tube_radius).all()
            or np.any(tube_radius < 0.0)
        ):
            raise ValueError("target_tube_radius_m must be finite, non-negative and match the horizon")
    cost = np.zeros((sequence_count, scenario_count), dtype=np.float64)
    best_slack = np.empty((sequence_count, scenario_count, horizon), dtype=np.float64)
    arrival_times = np.empty((sequence_count, scenario_count, horizon, defender_count), dtype=np.float64)
    for timestep in range(horizon):
        position = positions[:, None, timestep, :, :]
        velocity = velocities[:, None, timestep, :, :]
        target = targets[None, :, timestep, None, :]
        delta = target - position
        # A target tube enlarges the center-line distance by its radius.  This
        # is a conservative planning surrogate, not a claim that every point
        # in the tube is dynamically reachable.
        distance = np.linalg.norm(delta, axis=-1) + tube_radius[timestep]
        direction = delta / np.maximum(distance[..., None], 1.0e-12)
        projection = np.sum(velocity * direction, axis=-1)
        arrival = minimum_arrival_time(
            distance,
            projection,
            max_speed_mps=max_speed_mps,
            max_acceleration_mps2=max_acceleration_mps2,
        )
        available = (float(timestep) + 1.0) * float(dt_seconds)
        slack = available - arrival
        arrival_times[:, :, timestep, :] = arrival
        best = np.max(slack, axis=-1)
        best_slack[:, :, timestep] = best
        normalized_shortfall = np.maximum(float(time_margin_s) - best, 0.0)
        if activation_slack_s is not None:
            # Keep mild deficits visible in diagnostics, but do not let them
            # distort the nominal distance/formation objective.  The gate is
            # activated only for severe negative reachability slack.
            normalized_shortfall = np.where(
                best < float(activation_slack_s), normalized_shortfall, 0.0
            )
        normalized_shortfall /= float(time_scale_s)
        cost += normalized_shortfall * normalized_shortfall
    if not np.isfinite(cost).all() or not np.isfinite(best_slack).all() or not np.isfinite(arrival_times).all():
        raise FloatingPointError("RNIC emitted non-finite cost or diagnostics")
    return cost, best_slack, arrival_times


def formation_slot_reachability_cost(
    defender_position_paths: np.ndarray,
    defender_velocity_paths: np.ndarray,
    target_paths: np.ndarray,
    *,
    slot_radius_m: float,
    dt_seconds: float,
    max_speed_mps: float,
    max_acceleration_mps2: float,
    time_margin_s: float = 0.15,
    time_scale_s: float = 0.50,
    target_tube_radius_m: np.ndarray | tuple[float, ...] | None = None,
    activation_slack_s: float | None = None,
    slot_directions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a cooperative reachability cost for assigned formation slots.

    The inputs use shapes ``[sequence, horizon, defender, 3]`` and
    ``[scenario, horizon, 3]``.  Every defender must be assigned to one slot;
    the assignment minimizing the sum of normalized arrival shortfalls is
    selected by exact permutation enumeration.  This is deliberately bounded
    to at most six defenders so the result remains deterministic and easy to
    audit without an external assignment solver.

    Returns ``(cost, selected_min_slack, assignments, arrival_times)`` where
    ``cost`` is ``[sequence, scenario]``, ``selected_min_slack`` is
    ``[sequence, scenario, horizon]``, ``assignments`` is
    ``[sequence, scenario, horizon, defender]`` and ``arrival_times`` is
    ``[sequence, scenario, horizon, defender, slot]``.  This is a planning
    heuristic, not a reachable-set proof.
    """

    positions = np.asarray(defender_position_paths, dtype=np.float64)
    velocities = np.asarray(defender_velocity_paths, dtype=np.float64)
    targets = np.asarray(target_paths, dtype=np.float64)
    if positions.ndim != 4 or positions.shape[-1] != 3 or positions.shape[2] <= 0:
        raise ValueError("defender_position_paths must have shape [sequence, horizon, defender, 3]")
    if velocities.shape != positions.shape:
        raise ValueError("defender_velocity_paths must match defender_position_paths")
    if targets.ndim != 3 or targets.shape[1] != positions.shape[1] or targets.shape[-1] != 3:
        raise ValueError("target_paths must have shape [scenario, horizon, 3]")
    defender_count = int(positions.shape[2])
    if defender_count > 6:
        raise ValueError("formation slot assignment is limited to at most six defenders")
    if not np.isfinite([slot_radius_m, dt_seconds, time_margin_s, time_scale_s]).all():
        raise ValueError("slot radius, time step, margin and scale must be finite")
    if slot_radius_m <= 0.0 or dt_seconds <= 0.0 or time_margin_s < 0.0 or time_scale_s <= 0.0:
        raise ValueError("slot radius and time step must be positive; margin non-negative; scale positive")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all() or not np.isfinite(targets).all():
        raise ValueError("formation reachability inputs must be finite")
    if activation_slack_s is not None and not np.isfinite(float(activation_slack_s)):
        raise ValueError("activation_slack_s must be finite when provided")

    if slot_directions is None:
        if defender_count == 4:
            directions = _DEFAULT_FORMATION_SLOT_DIRECTIONS.copy()
        elif defender_count == 1:
            directions = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
        else:
            raise ValueError("slot_directions are required unless defender count is one or four")
    else:
        directions = np.asarray(slot_directions, dtype=np.float64)
        if directions.shape != (defender_count, 3):
            raise ValueError("slot_directions must have shape [defender, 3]")
        if not np.isfinite(directions).all():
            raise ValueError("slot_directions must be finite")
    direction_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(direction_norms <= 1.0e-12):
        raise ValueError("slot_directions must contain non-zero vectors")
    directions = directions / direction_norms

    horizon = int(positions.shape[1])
    if target_tube_radius_m is None:
        tube_radius = np.zeros(horizon, dtype=np.float64)
    else:
        tube_radius = np.asarray(target_tube_radius_m, dtype=np.float64)
        if (
            tube_radius.ndim != 1
            or tube_radius.shape[0] != horizon
            or not np.isfinite(tube_radius).all()
            or np.any(tube_radius < 0.0)
        ):
            raise ValueError("target_tube_radius_m must be finite, non-negative and match the horizon")

    sequence_count = int(positions.shape[0])
    scenario_count = int(targets.shape[0])
    cost = np.zeros((sequence_count, scenario_count), dtype=np.float64)
    selected_min_slack = np.empty((sequence_count, scenario_count, horizon), dtype=np.float64)
    assignments = np.empty(
        (sequence_count, scenario_count, horizon, defender_count),
        dtype=np.int64,
    )
    arrival_times = np.empty(
        (sequence_count, scenario_count, horizon, defender_count, defender_count),
        dtype=np.float64,
    )
    defender_indices = np.arange(defender_count, dtype=np.int64)
    permutation_array = np.asarray(list(permutations(range(defender_count))), dtype=np.int64)
    for timestep in range(horizon):
        position = positions[:, None, timestep, :, None, :]
        velocity = velocities[:, None, timestep, :, None, :]
        target = targets[None, :, timestep, None, None, :]
        slot_points = target + float(slot_radius_m) * directions[None, None, None, :, :]
        delta = slot_points - position
        distance = np.linalg.norm(delta, axis=-1) + tube_radius[timestep]
        direction = delta / np.maximum(distance[..., None], 1.0e-12)
        projection = np.sum(velocity * direction, axis=-1)
        arrival = minimum_arrival_time(
            distance,
            projection,
            max_speed_mps=max_speed_mps,
            max_acceleration_mps2=max_acceleration_mps2,
        )
        available = (float(timestep) + 1.0) * float(dt_seconds)
        slack = available - arrival
        arrival_times[:, :, timestep] = arrival

        assignment_costs: list[np.ndarray] = []
        assignment_slacks: list[np.ndarray] = []
        for permutation in permutation_array:
            assigned = slack[:, :, defender_indices, permutation]
            shortfall = np.maximum(float(time_margin_s) - assigned, 0.0)
            if activation_slack_s is not None:
                shortfall = np.where(
                    np.min(assigned, axis=-1, keepdims=True) < float(activation_slack_s),
                    shortfall,
                    0.0,
                )
            shortfall /= float(time_scale_s)
            assignment_costs.append(np.sum(shortfall * shortfall, axis=-1))
            assignment_slacks.append(assigned)
        stacked_costs = np.stack(assignment_costs, axis=-1)
        best_permutation = np.argmin(stacked_costs, axis=-1)
        cost += np.take_along_axis(stacked_costs, best_permutation[..., None], axis=-1)[..., 0]
        stacked_slacks = np.stack(assignment_slacks, axis=-1)
        selected = np.take_along_axis(
            stacked_slacks,
            best_permutation[:, :, None, None],
            axis=-1,
        )[..., 0]
        selected_min_slack[:, :, timestep] = np.min(selected, axis=-1)
        assignments[:, :, timestep, :] = permutation_array[best_permutation]

    if not np.isfinite(cost).all() or not np.isfinite(selected_min_slack).all() or not np.isfinite(arrival_times).all():
        raise FloatingPointError("formation RNIC emitted non-finite values")
    return cost, selected_min_slack, assignments, arrival_times


def rnic_summary(best_slack: np.ndarray) -> dict[str, float]:
    """Summarize best arrival-time slack for step-level diagnostics."""

    values = np.asarray(best_slack, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("best_slack must be finite and non-empty")
    return {
        "minimum_best_slack_s": float(np.min(values)),
        "mean_best_slack_s": float(np.mean(values)),
        "maximum_best_slack_s": float(np.max(values)),
    }


def planned_rnic_diagnostics(
    defender_positions: np.ndarray,
    action_sequence: np.ndarray,
    target_paths: np.ndarray,
    *,
    dt_seconds: float,
    max_speed_mps: float,
    max_acceleration_mps2: float,
    time_margin_s: float = 0.15,
    time_scale_s: float = 0.50,
    target_tube_radius_m: np.ndarray | tuple[float, ...] | None = None,
    activation_slack_s: float | None = None,
    cost_mode: str = "interceptor",
    slot_radius_m: float | None = None,
) -> dict[str, Any]:
    """Return auditable RNIC diagnostics for one selected team plan.

    ``action_sequence`` follows the simulator contract and is interpreted as
    the commanded velocity at each future step. This helper reconstructs the
    same nominal position rollout used by the planner and deliberately accepts
    no simulator truth. It is intended for step/episode logging, not for a
    safety certificate or for changing the selected plan.
    """

    positions = np.asarray(defender_positions, dtype=np.float64)
    actions = np.asarray(action_sequence, dtype=np.float64)
    targets = np.asarray(target_paths, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("defender_positions must have shape [defenders, 3]")
    if actions.ndim != 3 or actions.shape[-1] != 3 or actions.shape[1] != positions.shape[0]:
        raise ValueError("action_sequence must have shape [horizon, defenders, 3]")
    if targets.ndim != 3 or targets.shape[1] != actions.shape[0] or targets.shape[-1] != 3:
        raise ValueError("target_paths must have shape [scenarios, horizon, 3]")
    if not np.isfinite(positions).all() or not np.isfinite(actions).all() or not np.isfinite(targets).all():
        raise ValueError("planned RNIC diagnostic inputs must be finite")

    position_paths = positions[None, :, :] + np.cumsum(actions * float(dt_seconds), axis=0)
    if cost_mode == "interceptor":
        _cost, best_slack, arrival_times = reachability_normalized_interception_cost(
            position_paths[None, :, :, :],
            actions[None, :, :, :],
            targets,
            dt_seconds=dt_seconds,
            max_speed_mps=max_speed_mps,
            max_acceleration_mps2=max_acceleration_mps2,
            time_margin_s=time_margin_s,
            time_scale_s=time_scale_s,
            target_tube_radius_m=target_tube_radius_m,
            activation_slack_s=activation_slack_s,
        )
        best = best_slack[0]
        arrival = arrival_times[0]
        assignments = None
    elif cost_mode == "formation_slot":
        _cost, best, assignments, arrival_times = formation_slot_reachability_cost(
            position_paths[None, :, :, :],
            actions[None, :, :, :],
            targets,
            slot_radius_m=float(1.0 if slot_radius_m is None else slot_radius_m),
            dt_seconds=dt_seconds,
            max_speed_mps=max_speed_mps,
            max_acceleration_mps2=max_acceleration_mps2,
            time_margin_s=time_margin_s,
            time_scale_s=time_scale_s,
            target_tube_radius_m=target_tube_radius_m,
            activation_slack_s=activation_slack_s,
        )
        best = best[0]
        arrival = arrival_times[0]
        assignments = assignments[0]
    else:
        raise ValueError("cost_mode must be interceptor or formation_slot")
    summary = rnic_summary(best)
    feasible_by_step = np.all(best >= float(time_margin_s), axis=0)
    feasible_steps = np.flatnonzero(feasible_by_step)
    result: dict[str, Any] = {
        **summary,
        "cost_mode": cost_mode,
        "unreachable_slot_ratio": float(np.mean(best < 0.0)),
        "margin_violation_ratio": float(np.mean(best < float(time_margin_s))),
        "earliest_feasible_intercept_step": (
            float(feasible_steps[0] + 1) if feasible_steps.size else float(best.shape[1] + 1)
        ),
        "mean_arrival_time_s": float(np.mean(arrival)),
        "maximum_arrival_time_s": float(np.max(arrival)),
    }
    if assignments is not None:
        result["assignment_switch_rate"] = float(
            np.mean(np.any(assignments[:, 1:, :] != assignments[:, :-1, :], axis=-1))
            if assignments.shape[1] > 1
            else 0.0
        )
    else:
        result["assignment_switch_rate"] = 0.0
    return result


__all__ = [
    "minimum_arrival_time",
    "reachability_normalized_interception_cost",
    "formation_slot_reachability_cost",
    "rnic_summary",
    "planned_rnic_diagnostics",
]
