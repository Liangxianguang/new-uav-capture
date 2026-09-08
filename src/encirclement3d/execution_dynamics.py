"""Shared bounded command-execution dynamics for safety audits.

The pursuit environment, safety filter, and independent certificate use this
module so that delayed commands and the executed velocity have one explicit
contract.  The model is still a benchmark execution model, not a flight
controller or a real-airframe model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def clip_rows(values: np.ndarray, max_norm: float) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    return rows * np.minimum(1.0, float(max_norm) / np.maximum(norms, 1.0e-12))


def move_toward_velocity(current: np.ndarray, desired: np.ndarray, max_delta: float) -> np.ndarray:
    current_array = np.asarray(current, dtype=np.float64)
    desired_array = np.asarray(desired, dtype=np.float64)
    delta = desired_array - current_array
    delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
    return current_array + delta * np.minimum(1.0, float(max_delta) / np.maximum(delta_norm, 1.0e-12))


@dataclass(frozen=True)
class ExecutionParameters:
    enabled: bool
    dt_seconds: float
    action_delay_steps: int
    command_noise_std_mps: float
    command_noise_bound_mps: float
    clip_command_noise: bool
    velocity_time_constant_seconds: float
    drag_coefficient: float
    max_speed_mps: float
    max_acceleration_mps2: float
    mass_scale: float

    @property
    def tracking_alpha(self) -> float:
        if self.velocity_time_constant_seconds <= 0.0:
            return 1.0
        return min(1.0, self.dt_seconds / self.velocity_time_constant_seconds)

    @property
    def drag_gain(self) -> float:
        return float(np.exp(-max(self.drag_coefficient, 0.0) * self.dt_seconds))


@dataclass(frozen=True)
class ExecutionStep:
    command: np.ndarray
    tracked: np.ndarray
    executed: np.ndarray
    noise: np.ndarray
    noise_was_clipped: bool


def parameters_from_observation(observation: Mapping[str, Any], dt_seconds: float) -> ExecutionParameters:
    execution = observation.get("execution", {})
    if not isinstance(execution, Mapping):
        execution = {}
    command_noise_std = float(execution.get("command_noise_std_mps", execution.get("command_noise_std", 0.0)))
    configured_bound = execution.get("command_noise_bound_mps")
    if configured_bound is None:
        sigma = float(execution.get("command_noise_bound_sigma", 3.0))
        configured_bound = sigma * command_noise_std
    return ExecutionParameters(
        enabled=bool(execution.get("enabled", False)),
        dt_seconds=float(dt_seconds),
        action_delay_steps=int(execution.get("action_delay_steps", 0)),
        command_noise_std_mps=command_noise_std,
        command_noise_bound_mps=max(0.0, float(configured_bound)),
        clip_command_noise=bool(execution.get("clip_command_noise", True)),
        velocity_time_constant_seconds=float(execution.get("velocity_time_constant_seconds", 0.0)),
        drag_coefficient=float(execution.get("drag_coefficient", 0.0)),
        max_speed_mps=float(execution.get("max_speed_mps", 5.0)),
        max_acceleration_mps2=float(execution.get("max_acceleration_mps2", 6.0)),
        mass_scale=max(float(execution.get("mass_scale", 1.0)), 1.0e-9),
    )


def advance_execution(
    current_velocity: np.ndarray,
    delayed_action: np.ndarray,
    parameters: ExecutionParameters,
    noise: np.ndarray | None = None,
) -> ExecutionStep:
    current = np.asarray(current_velocity, dtype=np.float64)
    delayed = np.asarray(delayed_action, dtype=np.float64)
    if current.shape != delayed.shape or current.ndim != 2 or current.shape[-1] != 3:
        raise ValueError("current_velocity and delayed_action must have shape [defenders, 3]")
    sampled_noise = np.zeros_like(delayed) if noise is None else np.asarray(noise, dtype=np.float64)
    if sampled_noise.shape != delayed.shape:
        raise ValueError("execution noise must have shape [defenders, 3]")
    if not parameters.enabled:
        zero = np.zeros_like(delayed)
        return ExecutionStep(
            command=delayed.copy(),
            tracked=delayed.copy(),
            executed=delayed.copy(),
            noise=zero,
            noise_was_clipped=False,
        )

    if parameters.clip_command_noise:
        bound = float(parameters.command_noise_bound_mps)
        clipped_noise = np.clip(sampled_noise, -bound, bound)
    else:
        clipped_noise = sampled_noise.copy()
    was_clipped = bool(np.any(np.abs(clipped_noise - sampled_noise) > 1.0e-12))
    command = clip_rows(delayed + clipped_noise, parameters.max_speed_mps)
    tracked = current + parameters.tracking_alpha * (command - current)
    tracked *= parameters.drag_gain
    max_delta = parameters.max_acceleration_mps2 * parameters.dt_seconds / parameters.mass_scale
    executed = move_toward_velocity(current, tracked, max_delta)
    executed = clip_rows(executed, parameters.max_speed_mps)
    return ExecutionStep(
        command=command,
        tracked=tracked,
        executed=executed,
        noise=clipped_noise,
        noise_was_clipped=was_clipped,
    )


def queue_from_observation(observation: Mapping[str, Any], defenders: int) -> list[np.ndarray]:
    queue_value = observation.get("execution_action_queue")
    if queue_value is None:
        execution = observation.get("execution", {})
        queue_value = execution.get("action_queue", []) if isinstance(execution, Mapping) else []
    queue = np.asarray(queue_value, dtype=np.float64)
    if queue.size == 0:
        return []
    if queue.ndim != 3 or queue.shape[1:] != (int(defenders), 3):
        raise ValueError("execution_action_queue must have shape [delay, defenders, 3]")
    return [item.copy() for item in queue]


def rollout_execution(
    positions: np.ndarray,
    velocities: np.ndarray,
    action_queue: list[np.ndarray] | tuple[np.ndarray, ...],
    new_action: np.ndarray,
    parameters: ExecutionParameters,
    *,
    horizon_steps: int | None = None,
    noises: list[np.ndarray] | tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[ExecutionStep]]:
    position = np.asarray(positions, dtype=np.float64).copy()
    velocity = np.asarray(velocities, dtype=np.float64).copy()
    action = np.asarray(new_action, dtype=np.float64)
    if position.shape != velocity.shape or position.shape != action.shape:
        raise ValueError("positions, velocities, and new_action must have the same shape")
    sequence = [np.asarray(item, dtype=np.float64).copy() for item in action_queue]
    sequence.append(action.copy())
    steps = len(sequence) if horizon_steps is None else int(horizon_steps)
    if steps <= 0:
        raise ValueError("horizon_steps must be positive")
    results: list[ExecutionStep] = []
    positions_out: list[np.ndarray] = []
    velocities_out: list[np.ndarray] = []
    for index in range(steps):
        delayed = sequence[index] if index < len(sequence) else action
        noise = None if noises is None or index >= len(noises) else noises[index]
        result = advance_execution(velocity, delayed, parameters, noise=noise)
        velocity = result.executed.copy()
        position = position + parameters.dt_seconds * velocity
        results.append(result)
        positions_out.append(position.copy())
        velocities_out.append(velocity.copy())
    return np.stack(positions_out, axis=0), np.stack(velocities_out, axis=0), results


def position_uncertainty_radii(parameters: ExecutionParameters, defenders: int, horizon_steps: int) -> np.ndarray:
    """Conservative position tube from bounded command noise only."""

    noise_norm = np.sqrt(3.0) * float(parameters.command_noise_bound_mps)
    velocity_error = 0.0
    position_error = 0.0
    radii: list[np.ndarray] = []
    alpha = float(parameters.tracking_alpha)
    drag = float(parameters.drag_gain)
    for _ in range(int(horizon_steps)):
        velocity_error = min(
            2.0 * float(parameters.max_speed_mps),
            velocity_error + drag * alpha * (velocity_error + noise_norm),
        )
        position_error += float(parameters.dt_seconds) * velocity_error
        radii.append(np.full(int(defenders), position_error, dtype=np.float64))
    return np.stack(radii, axis=0)


__all__ = [
    "ExecutionParameters",
    "ExecutionStep",
    "advance_execution",
    "clip_rows",
    "move_toward_velocity",
    "parameters_from_observation",
    "position_uncertainty_radii",
    "queue_from_observation",
    "rollout_execution",
]
