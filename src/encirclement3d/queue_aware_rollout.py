"""Queue-aware planning-state adapters for delayed command execution.

The benchmark exposes commands that are already in flight through the public
observation.  This module rolls those commands forward with the shared
execution model before a planner evaluates a newly selected action.  It is a
state-alignment adapter: it does not cancel immutable commands and it is not a
robust safety certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .execution_dynamics import (
    ExecutionParameters,
    advance_execution,
    parameters_from_observation,
    queue_from_observation,
)
from .minimax_mpc import ScenarioTrajectorySet


@dataclass(frozen=True)
class DelayedPlanningState:
    """State reached after executing the currently queued commands."""

    prefix_positions: np.ndarray
    prefix_velocities: np.ndarray
    prefix_actions: np.ndarray
    delayed_positions: np.ndarray
    delayed_velocities: np.ndarray
    queue_length: int
    first_controllable_step: int
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        positions = np.asarray(self.delayed_positions, dtype=np.float64)
        velocities = np.asarray(self.delayed_velocities, dtype=np.float64)
        prefix_positions = np.asarray(self.prefix_positions, dtype=np.float64)
        prefix_velocities = np.asarray(self.prefix_velocities, dtype=np.float64)
        prefix_actions = np.asarray(self.prefix_actions, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[-1] != 3 or velocities.shape != positions.shape:
            raise ValueError("delayed positions and velocities must have shape [defenders, 3]")
        if prefix_positions.shape != prefix_velocities.shape:
            raise ValueError("prefix positions and velocities must have matching shapes")
        if prefix_positions.ndim != 3 or prefix_positions.shape[1:] != positions.shape:
            raise ValueError("prefix positions must have shape [queue, defenders, 3]")
        if prefix_actions.shape != prefix_positions.shape:
            raise ValueError("prefix actions must have shape [queue, defenders, 3]")
        if prefix_positions.shape[0] != int(self.queue_length):
            raise ValueError("prefix length must equal queue_length")
        if int(self.queue_length) < 0 or int(self.first_controllable_step) < 0:
            raise ValueError("queue lengths and controllable-step indices must be non-negative")
        if int(self.first_controllable_step) != int(self.queue_length):
            raise ValueError("first_controllable_step must equal queue_length")
        arrays = (prefix_positions, prefix_velocities, prefix_actions, positions, velocities)
        if not all(np.isfinite(value).all() for value in arrays):
            raise ValueError("delayed planning state must be finite")

    def as_dict(self) -> dict[str, Any]:
        return {
            "queue_length": int(self.queue_length),
            "first_controllable_step": int(self.first_controllable_step),
            "prefix_positions": self.prefix_positions.tolist(),
            "prefix_velocities": self.prefix_velocities.tolist(),
            "prefix_actions": self.prefix_actions.tolist(),
            "delayed_positions": self.delayed_positions.tolist(),
            "delayed_velocities": self.delayed_velocities.tolist(),
            "latency_ms": float(self.latency_ms),
        }


def rollout_queue_prefix(
    positions: np.ndarray,
    velocities: np.ndarray,
    action_queue: list[np.ndarray] | tuple[np.ndarray, ...],
    parameters: ExecutionParameters,
) -> DelayedPlanningState:
    """Execute the immutable queue with deterministic zero execution noise.

    The benchmark environment samples command noise when it advances the
    actual system.  A planner cannot know those future samples, so the
    planning-state rollout uses the nominal shared dynamics and records the
    prefix explicitly for auditability.
    """

    position = np.asarray(positions, dtype=np.float64).copy()
    velocity = np.asarray(velocities, dtype=np.float64).copy()
    if position.ndim != 2 or position.shape[-1] != 3 or velocity.shape != position.shape:
        raise ValueError("positions and velocities must have shape [defenders, 3]")
    queue = [np.asarray(item, dtype=np.float64).copy() for item in action_queue]
    if any(item.shape != position.shape for item in queue):
        raise ValueError("every queued action must have shape [defenders, 3]")
    if any(not np.isfinite(item).all() for item in queue):
        raise ValueError("queued actions must be finite")

    prefix_positions: list[np.ndarray] = []
    prefix_velocities: list[np.ndarray] = []
    prefix_actions: list[np.ndarray] = []
    for queued_action in queue:
        step = advance_execution(velocity, queued_action, parameters, noise=np.zeros_like(queued_action))
        velocity = step.executed.copy()
        position = position + float(parameters.dt_seconds) * velocity
        prefix_actions.append(queued_action.copy())
        prefix_positions.append(position.copy())
        prefix_velocities.append(velocity.copy())

    defenders = position.shape[0]
    empty = np.empty((0, defenders, 3), dtype=np.float64)
    return DelayedPlanningState(
        prefix_positions=(np.stack(prefix_positions, axis=0) if prefix_positions else empty),
        prefix_velocities=(np.stack(prefix_velocities, axis=0) if prefix_velocities else empty.copy()),
        prefix_actions=(np.stack(prefix_actions, axis=0) if prefix_actions else empty.copy()),
        delayed_positions=position,
        delayed_velocities=velocity,
        queue_length=len(queue),
        first_controllable_step=len(queue),
    )


def prepare_queue_aware_observation(
    observation: Mapping[str, Any],
    *,
    dt_seconds: float,
) -> tuple[dict[str, Any], DelayedPlanningState]:
    """Return a planner observation aligned to the first controllable state."""

    positions = np.asarray(observation["defender_positions"], dtype=np.float64)
    velocities = np.asarray(observation["defender_velocities"], dtype=np.float64)
    queue = queue_from_observation(observation, positions.shape[0])
    parameters = parameters_from_observation(observation, float(dt_seconds))
    state = rollout_queue_prefix(positions, velocities, queue, parameters)
    value = dict(observation)
    value["defender_positions"] = state.delayed_positions.copy()
    value["defender_velocities"] = state.delayed_velocities.copy()
    value["execution_action_queue"] = []
    execution = dict(value.get("execution", {})) if isinstance(value.get("execution", {}), Mapping) else {}
    execution["action_queue"] = []
    value["execution"] = execution
    value["qdr"] = state.as_dict()
    return value, state


def _extend_trajectory_tail(
    trajectory: np.ndarray,
    *,
    required_steps: int,
    dt_seconds: float,
    max_speed_mps: float,
) -> np.ndarray:
    """Extend one absolute target trajectory with a bounded constant velocity."""

    source = np.asarray(trajectory, dtype=np.float64)
    if source.ndim != 2 or source.shape[-1] != 3 or source.shape[0] <= 0:
        raise ValueError("trajectory must have shape [horizon, 3]")
    if required_steps <= source.shape[0]:
        return source[:required_steps].copy()
    if source.shape[0] >= 2:
        velocity = (source[-1] - source[-2]) / max(float(dt_seconds), 1.0e-12)
        norm = float(np.linalg.norm(velocity))
        if norm > float(max_speed_mps):
            velocity = velocity * (float(max_speed_mps) / max(norm, 1.0e-12))
    else:
        velocity = np.zeros(3, dtype=np.float64)
    tail = [source[-1].copy()]
    for _ in range(required_steps - source.shape[0]):
        tail.append(tail[-1] + float(dt_seconds) * velocity)
    return np.concatenate([source[:-1], np.stack(tail, axis=0)], axis=0)


def shift_scenario_trajectory_set(
    scenarios: ScenarioTrajectorySet,
    *,
    offset_steps: int,
    horizon_steps: int,
    dt_seconds: float,
    max_speed_mps: float,
) -> ScenarioTrajectorySet:
    """Align predicted target paths with the delayed planning state.

    ``ScenarioTrajectorySet`` stores positions at t+1, t+2, ... relative to
    the prediction origin.  Once a queue of length ``q`` is rolled out, the
    planner must consume positions beginning at t+q+1.  The tail extension is
    deterministic and bounded so this adapter never fabricates an unbounded
    target speed.
    """

    offset = int(offset_steps)
    horizon = int(horizon_steps)
    if offset < 0 or horizon <= 0:
        raise ValueError("offset_steps must be non-negative and horizon_steps must be positive")
    source = np.asarray(scenarios.trajectories, dtype=np.float64)
    required = offset + horizon
    aligned = np.stack(
        [
            _extend_trajectory_tail(
                candidate,
                required_steps=required,
                dt_seconds=dt_seconds,
                max_speed_mps=max_speed_mps,
            )[offset:required]
            for candidate in source
        ],
        axis=0,
    )
    return ScenarioTrajectorySet(
        trajectories=aligned,
        weights=np.asarray(scenarios.weights, dtype=np.float64).copy(),
        score_kind=scenarios.score_kind,
        dynamics_status=scenarios.dynamics_status,
        conformal_radius_m=scenarios.conformal_radius_m,
        source_model_hash=scenarios.source_model_hash,
        timestamp_step=scenarios.timestamp_step,
    )


__all__ = [
    "DelayedPlanningState",
    "prepare_queue_aware_observation",
    "rollout_queue_prefix",
    "shift_scenario_trajectory_set",
]
