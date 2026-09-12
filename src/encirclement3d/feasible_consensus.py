"""Finite feasible-consensus formation gates for distributed encirclement.

The helper is intentionally small and deterministic.  It evaluates a finite
set of formation slots against already generated defender rollouts and target
belief candidates.  A candidate is locally feasible when its selected slot
assignment makes non-negative progress (within a declared tolerance), has a
bounded arrival-time deficit over a declared gate horizon, and does not switch
assignments too aggressively.  Assignment changes are reported and optionally
penalized so a delayed team can hold its previous topology.

This is a planning gate, not a safety certificate.  Collision, obstacle and
boundary safety remain the responsibility of the existing safety layer.
The function accepts no simulator truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .reachability_interception import formation_slot_reachability_cost, minimum_arrival_time


_DEFAULT_SLOT_DIRECTIONS = np.asarray(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ],
    dtype=np.float64,
)


def _validated_directions(defender_count: int, slot_directions: np.ndarray | None) -> np.ndarray:
    if slot_directions is None:
        if defender_count != 4:
            raise ValueError("slot_directions are required unless defender count is four")
        directions = _DEFAULT_SLOT_DIRECTIONS.copy()
    else:
        directions = np.asarray(slot_directions, dtype=np.float64)
        if directions.shape != (defender_count, 3):
            raise ValueError("slot_directions must have shape [defenders, 3]")
        if not np.isfinite(directions).all():
            raise ValueError("slot_directions must be finite")
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if np.any(norms <= 1.0e-12):
        raise ValueError("slot_directions must contain non-zero vectors")
    return directions / norms


def feasible_consensus_slot_gate(
    defender_position_paths: np.ndarray,
    defender_velocity_paths: np.ndarray,
    target_paths: np.ndarray,
    *,
    slot_radius_m: float,
    dt_seconds: float,
    max_speed_mps: float,
    max_acceleration_mps2: float,
    slot_tolerance_m: float = 3.0,
    min_slot_slack_s: float = -2.0,
    min_progress_m: float = -0.10,
    gate_horizon_steps: int | None = None,
    max_assignment_switch_rate: float = 0.5,
    switch_penalty: float = 0.10,
    time_scale_s: float = 0.50,
    slot_directions: np.ndarray | None = None,
    previous_assignment: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | bool]:
    """Evaluate finite formation-slot feasibility and consensus stability.

    Shapes are ``[sequence, horizon, defender, 3]`` for defender paths and
    ``[scenario, horizon, 3]`` for target paths.  Returned ``cost`` and
    ``feasible`` have shape ``[sequence, scenario]``.  The selected assignment
    is indexed by ``[sequence, scenario, defender]`` at the first gate step.

    ``min_slot_slack_s`` and ``min_progress_m`` may be negative because the current finite-shooting
    horizon is shorter than some physical arrivals.  The threshold is an
    explicit development contract, not a hidden feasibility claim.
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
    scalar_values = np.asarray(
        [
            slot_radius_m,
            dt_seconds,
            max_speed_mps,
            max_acceleration_mps2,
            slot_tolerance_m,
            min_slot_slack_s,
            min_progress_m,
            max_assignment_switch_rate,
            switch_penalty,
            time_scale_s,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(scalar_values).all():
        raise ValueError("FC-DBF scalar parameters must be finite")
    if slot_radius_m <= 0.0 or dt_seconds <= 0.0 or max_speed_mps <= 0.0 or max_acceleration_mps2 <= 0.0:
        raise ValueError("slot radius, time step, speed and acceleration must be positive")
    if slot_tolerance_m <= 0.0 or time_scale_s <= 0.0:
        raise ValueError("slot_tolerance_m and time_scale_s must be positive")
    if not 0.0 <= max_assignment_switch_rate <= 1.0:
        raise ValueError("max_assignment_switch_rate must lie in [0, 1]")
    if switch_penalty < 0.0:
        raise ValueError("switch_penalty must be non-negative")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all() or not np.isfinite(targets).all():
        raise ValueError("FC-DBF paths must be finite")

    sequence_count, horizon, defender_count, _ = positions.shape
    scenario_count = targets.shape[0]
    directions = _validated_directions(defender_count, slot_directions)
    if gate_horizon_steps is None:
        gate_steps = horizon
    else:
        gate_steps = int(gate_horizon_steps)
        if gate_steps <= 0 or gate_steps > horizon:
            raise ValueError("gate_horizon_steps must lie in [1, horizon]")
    gate_slice = slice(0, gate_steps)

    # Use the same bounded arrival surrogate as the existing formation-slot
    # RNIC implementation, but with zero nominal time margin.  Assignment is
    # therefore the least-deficit finite permutation, not simulator truth.
    _assignment_cost, selected_slack, assignments, _arrival = formation_slot_reachability_cost(
        positions,
        velocities,
        targets,
        slot_radius_m=float(slot_radius_m),
        dt_seconds=float(dt_seconds),
        max_speed_mps=float(max_speed_mps),
        max_acceleration_mps2=float(max_acceleration_mps2),
        time_margin_s=0.0,
        time_scale_s=float(time_scale_s),
        slot_directions=directions,
    )

    selected_assignments = assignments[:, :, 0, :].copy()
    if previous_assignment is not None:
        previous = np.asarray(previous_assignment, dtype=np.int64)
        if (
            previous.shape != (defender_count,)
            or np.any(previous < 0)
            or np.any(previous >= defender_count)
            or np.unique(previous).size != defender_count
        ):
            raise ValueError("previous_assignment must be a valid [defender] permutation")
    else:
        previous = None

    # Gather the slot point selected for each defender and each trajectory.
    selected_directions = directions[assignments]
    slot_points = targets[None, :, :, None, :] + float(slot_radius_m) * selected_directions
    tracking_error = np.linalg.norm(positions[:, None, :, :, :] - slot_points, axis=-1)
    max_slot_error = np.max(tracking_error[:, :, gate_slice, :], axis=(2, 3))
    mean_slot_error = np.mean(tracking_error[:, :, gate_slice, :], axis=(2, 3))
    min_slack = np.min(selected_slack[:, :, gate_slice], axis=2)
    initial_slot_error = np.mean(tracking_error[:, :, 0, :], axis=2)
    terminal_gate_slot_error = np.mean(tracking_error[:, :, gate_steps - 1, :], axis=2)
    mean_slot_progress = initial_slot_error - terminal_gate_slot_error

    if gate_steps > 1:
        assignment_switches = np.sum(
            assignments[:, :, 1:gate_steps, :] != assignments[:, :, : gate_steps - 1, :],
            axis=(2, 3),
        )
        switch_denominator = float(max((gate_steps - 1) * defender_count, 1))
        switch_rate = assignment_switches / switch_denominator
    else:
        assignment_switches = np.zeros((sequence_count, scenario_count), dtype=np.float64)
        switch_rate = np.zeros((sequence_count, scenario_count), dtype=np.float64)
    if previous is not None:
        initial_switches = np.sum(selected_assignments != previous[None, None, :], axis=2)
        initial_switch_rate = initial_switches / float(max(defender_count, 1))
        switch_rate = np.minimum(1.0, switch_rate + initial_switch_rate / float(max(gate_steps, 1)))
        assignment_switches = assignment_switches + initial_switches

    feasible = (
        (mean_slot_progress >= float(min_progress_m))
        & (min_slack >= float(min_slot_slack_s))
        & (switch_rate <= float(max_assignment_switch_rate))
    )
    normalized_error = (mean_slot_error / float(slot_tolerance_m)) ** 2
    progress_deficit = np.maximum(-mean_slot_progress, 0.0) / float(slot_tolerance_m)
    slack_deficit = np.maximum(float(min_slot_slack_s) - min_slack, 0.0) / float(time_scale_s)
    cost = normalized_error + progress_deficit * progress_deficit + slack_deficit * slack_deficit
    cost += float(switch_penalty) * switch_rate
    if not np.isfinite(cost).all() or not np.isfinite(tracking_error).all():
        raise FloatingPointError("FC-DBF emitted non-finite diagnostics")

    return {
        "cost": cost,
        "feasible": feasible,
        "assignments": selected_assignments,
        "assignment_paths": assignments,
        "min_slot_slack_s": min_slack,
        "max_slot_error_m": max_slot_error,
        "mean_slot_error_m": mean_slot_error,
        "mean_slot_progress_m": mean_slot_progress,
        "assignment_switches": assignment_switches.astype(np.float64),
        "assignment_switch_rate": switch_rate,
        "feasible_rate": np.mean(feasible, axis=1),
        "gate_exhausted": bool(not np.any(feasible)),
    }


def evaluate_fixed_consensus_slots(
    defender_position_paths: np.ndarray,
    defender_velocity_paths: np.ndarray,
    target_paths: np.ndarray,
    slot_assignments: np.ndarray,
    *,
    slot_radius_m: float,
    dt_seconds: float,
    max_speed_mps: float,
    max_acceleration_mps2: float,
    slot_tolerance_m: float = 3.0,
    min_slot_slack_s: float = -2.0,
    min_progress_m: float = -0.10,
    gate_horizon_steps: int | None = None,
    max_assignment_switch_rate: float = 0.5,
    switch_penalty: float = 0.10,
    time_scale_s: float = 0.50,
    slot_directions: np.ndarray | None = None,
    previous_assignment: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | bool]:
    """Evaluate many local candidates against one consensus assignment token.

    ``slot_assignments`` has shape ``[scenario, horizon, defender]`` and is
    normally generated from one nominal local candidate.  Reusing this token
    avoids repeating exact permutation enumeration for every local candidate
    while retaining candidate-specific tracking, progress and arrival checks.
    This is the delayed distributed approximation used by FC-DBF.
    """

    positions = np.asarray(defender_position_paths, dtype=np.float64)
    velocities = np.asarray(defender_velocity_paths, dtype=np.float64)
    targets = np.asarray(target_paths, dtype=np.float64)
    assignments = np.asarray(slot_assignments, dtype=np.int64)
    if positions.ndim != 4 or positions.shape[-1] != 3 or positions.shape[2] <= 0:
        raise ValueError("defender_position_paths must have shape [sequence, horizon, defender, 3]")
    if velocities.shape != positions.shape:
        raise ValueError("defender_velocity_paths must match defender_position_paths")
    if targets.ndim != 3 or targets.shape[1] != positions.shape[1] or targets.shape[-1] != 3:
        raise ValueError("target_paths must have shape [scenario, horizon, 3]")
    if assignments.shape != (targets.shape[0], positions.shape[1], positions.shape[2]):
        raise ValueError("slot_assignments must have shape [scenario, horizon, defender]")
    defender_count = int(positions.shape[2])
    directions = _validated_directions(defender_count, slot_directions)
    if np.any(assignments < 0) or np.any(assignments >= defender_count):
        raise ValueError("slot_assignments contain an invalid slot index")
    if any(
        np.unique(assignments[:, timestep, :], axis=1).shape[1] != defender_count
        for timestep in range(assignments.shape[1])
    ):
        raise ValueError("slot_assignments must be permutations at every horizon step")
    scalar_values = np.asarray(
        [
            slot_radius_m,
            dt_seconds,
            max_speed_mps,
            max_acceleration_mps2,
            slot_tolerance_m,
            min_slot_slack_s,
            min_progress_m,
            max_assignment_switch_rate,
            switch_penalty,
            time_scale_s,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(scalar_values).all():
        raise ValueError("FC-DBF scalar parameters must be finite")
    if slot_radius_m <= 0.0 or dt_seconds <= 0.0 or max_speed_mps <= 0.0 or max_acceleration_mps2 <= 0.0:
        raise ValueError("slot radius, time step, speed and acceleration must be positive")
    if slot_tolerance_m <= 0.0 or time_scale_s <= 0.0:
        raise ValueError("slot_tolerance_m and time_scale_s must be positive")
    if not 0.0 <= max_assignment_switch_rate <= 1.0:
        raise ValueError("max_assignment_switch_rate must lie in [0, 1]")
    if switch_penalty < 0.0:
        raise ValueError("switch_penalty must be non-negative")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all() or not np.isfinite(targets).all():
        raise ValueError("FC-DBF paths must be finite")
    previous = None
    if previous_assignment is not None:
        previous = np.asarray(previous_assignment, dtype=np.int64)
        if (
            previous.shape != (defender_count,)
            or np.any(previous < 0)
            or np.any(previous >= defender_count)
            or np.unique(previous).size != defender_count
        ):
            raise ValueError("previous_assignment must be a valid [defender] permutation")

    sequence_count, horizon, _defender_count, _ = positions.shape
    scenario_count = targets.shape[0]
    if gate_horizon_steps is None:
        gate_steps = horizon
    else:
        gate_steps = int(gate_horizon_steps)
        if gate_steps <= 0 or gate_steps > horizon:
            raise ValueError("gate_horizon_steps must lie in [1, horizon]")
    selected_directions = directions[assignments]
    slot_points = targets[None, :, :, None, :] + float(slot_radius_m) * selected_directions[None, ...]
    tracking_error = np.linalg.norm(positions[:, None, :, :, :] - slot_points, axis=-1)
    gate_slice = slice(0, gate_steps)
    max_slot_error = np.max(tracking_error[:, :, gate_slice, :], axis=(2, 3))
    mean_slot_error = np.mean(tracking_error[:, :, gate_slice, :], axis=(2, 3))
    initial_slot_error = np.mean(tracking_error[:, :, 0, :], axis=2)
    terminal_gate_slot_error = np.mean(tracking_error[:, :, gate_steps - 1, :], axis=2)
    mean_slot_progress = initial_slot_error - terminal_gate_slot_error

    delta = slot_points - positions[:, None, :, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    direction = delta / np.maximum(distance[..., None], 1.0e-12)
    projection = np.sum(velocities[:, None, :, :, :] * direction, axis=-1)
    arrival = minimum_arrival_time(
        distance,
        projection,
        max_speed_mps=float(max_speed_mps),
        max_acceleration_mps2=float(max_acceleration_mps2),
    )
    available = (np.arange(horizon, dtype=np.float64) + 1.0)[None, None, :, None] * float(dt_seconds)
    slack = available - arrival
    min_slack = np.min(slack[:, :, gate_slice, :], axis=(2, 3))

    if gate_steps > 1:
        assignment_switches = np.sum(
            assignments[None, :, 1:gate_steps, :] != assignments[None, :, : gate_steps - 1, :],
            axis=(2, 3),
        )
        switch_rate = assignment_switches / float(max((gate_steps - 1) * defender_count, 1))
    else:
        assignment_switches = np.zeros((sequence_count, scenario_count), dtype=np.float64)
        switch_rate = np.zeros((sequence_count, scenario_count), dtype=np.float64)
    if previous is not None:
        initial_switches = np.sum(assignments[:, 0, :][None, ...] != previous[None, None, :], axis=2)
        initial_switch_rate = initial_switches / float(max(defender_count, 1))
        switch_rate = np.minimum(1.0, switch_rate + initial_switch_rate / float(max(gate_steps, 1)))
        assignment_switches = assignment_switches + initial_switches

    feasible = (
        (mean_slot_progress >= float(min_progress_m))
        & (min_slack >= float(min_slot_slack_s))
        & (switch_rate <= float(max_assignment_switch_rate))
    )
    normalized_error = (mean_slot_error / float(slot_tolerance_m)) ** 2
    progress_deficit = np.maximum(-mean_slot_progress, 0.0) / float(slot_tolerance_m)
    slack_deficit = np.maximum(float(min_slot_slack_s) - min_slack, 0.0) / float(time_scale_s)
    cost = normalized_error + progress_deficit * progress_deficit + slack_deficit * slack_deficit
    cost += float(switch_penalty) * switch_rate
    if not np.isfinite(cost).all():
        raise FloatingPointError("fixed FC-DBF evaluation emitted non-finite values")
    return {
        "cost": cost,
        "feasible": feasible,
        "assignments": assignments[:, 0, :][None, ...].repeat(sequence_count, axis=0),
        "assignment_paths": assignments[None, ...].repeat(sequence_count, axis=0),
        "min_slot_slack_s": min_slack,
        "max_slot_error_m": max_slot_error,
        "mean_slot_error_m": mean_slot_error,
        "mean_slot_progress_m": mean_slot_progress,
        "assignment_switches": assignment_switches.astype(np.float64),
        "assignment_switch_rate": switch_rate,
        "feasible_rate": np.mean(feasible, axis=1),
        "gate_exhausted": bool(not np.any(feasible)),
    }


def fc_dbf_summary(
    metrics: dict[str, np.ndarray | float | bool],
    sequence_index: int,
    scenario_index: int,
) -> dict[str, Any]:
    """Extract scalar diagnostics for one selected sequence/scenario."""

    feasible = np.asarray(metrics["feasible"], dtype=bool)
    cost = np.asarray(metrics["cost"], dtype=np.float64)
    min_slack = np.asarray(metrics["min_slot_slack_s"], dtype=np.float64)
    max_error = np.asarray(metrics["max_slot_error_m"], dtype=np.float64)
    mean_error = np.asarray(metrics["mean_slot_error_m"], dtype=np.float64)
    switch_rate = np.asarray(metrics["assignment_switch_rate"], dtype=np.float64)
    return {
        "fc_dbf_cost": float(cost[sequence_index, scenario_index]),
        "fc_dbf_feasible": bool(feasible[sequence_index, scenario_index]),
        "fc_dbf_min_slot_slack_s": float(min_slack[sequence_index, scenario_index]),
        "fc_dbf_max_slot_error_m": float(max_error[sequence_index, scenario_index]),
        "fc_dbf_mean_slot_error_m": float(mean_error[sequence_index, scenario_index]),
        "fc_dbf_mean_slot_progress_m": float(
            np.asarray(metrics["mean_slot_progress_m"], dtype=np.float64)[sequence_index, scenario_index]
        ),
        "fc_dbf_assignment_switch_rate": float(switch_rate[sequence_index, scenario_index]),
        "fc_dbf_feasible_rate": float(np.mean(feasible[sequence_index])),
        "fc_dbf_gate_exhausted": bool(metrics.get("gate_exhausted", False)),
    }
