"""Reachability-normalized interception costs for finite-horizon planning.

The helper uses a one-dimensional, acceleration- and speed-limited arrival-time
surrogate along the line of sight.  It is deliberately exposed as a heuristic
cost, not as a complete 3-D reachable-set proof.
"""

from __future__ import annotations

from typing import Any

import numpy as np


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
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all() or not np.isfinite(targets).all():
        raise ValueError("reachability inputs must be finite")

    sequence_count, horizon, defender_count, _ = positions.shape
    scenario_count = targets.shape[0]
    cost = np.zeros((sequence_count, scenario_count), dtype=np.float64)
    best_slack = np.empty((sequence_count, scenario_count, horizon), dtype=np.float64)
    arrival_times = np.empty((sequence_count, scenario_count, horizon, defender_count), dtype=np.float64)
    for timestep in range(horizon):
        position = positions[:, None, timestep, :, :]
        velocity = velocities[:, None, timestep, :, :]
        target = targets[None, :, timestep, None, :]
        delta = target - position
        distance = np.linalg.norm(delta, axis=-1)
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
        normalized_shortfall = np.maximum(float(time_margin_s) - best, 0.0) / float(time_scale_s)
        cost += normalized_shortfall * normalized_shortfall
    if not np.isfinite(cost).all() or not np.isfinite(best_slack).all() or not np.isfinite(arrival_times).all():
        raise FloatingPointError("RNIC emitted non-finite cost or diagnostics")
    return cost, best_slack, arrival_times


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


__all__ = [
    "minimum_arrival_time",
    "reachability_normalized_interception_cost",
    "rnic_summary",
]
