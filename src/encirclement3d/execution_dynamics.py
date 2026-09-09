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


_COMMAND_AUTHORITY_MODES = frozenset({"immutable", "replace_nonexecuting", "flush_pending"})


@dataclass(frozen=True)
class CommandAuthorityDirective:
    """Supervisor authority over commands that are already in the delay queue.

    ``immutable`` can only append the newly selected command.  Under
    ``replace_nonexecuting``, the command due this tick remains immutable while
    later queued commands can be replaced with a braking command.  Under
    ``flush_pending``, the benchmark grants the supervisor authority to cancel
    every queued command before execution.  The latter is an explicit simulated
    actuator contract, not an assumption about a physical flight controller.
    """

    mode: str
    emergency_brake: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "emergency_brake": bool(self.emergency_brake)}


def validate_command_authority_mode(value: Any) -> str:
    mode = str(value)
    if mode not in _COMMAND_AUTHORITY_MODES:
        allowed = ", ".join(sorted(_COMMAND_AUTHORITY_MODES))
        raise ValueError(f"pending command authority must be one of: {allowed}")
    return mode


def command_authority_from_observation(observation: Mapping[str, Any]) -> str:
    execution = observation.get("execution", {})
    if not isinstance(execution, Mapping):
        execution = {}
    return validate_command_authority_mode(execution.get("pending_command_authority", "immutable"))


def command_authority_directive(
    value: CommandAuthorityDirective | Mapping[str, Any] | None,
    *,
    allowed_mode: str,
) -> CommandAuthorityDirective:
    """Resolve a request while preventing callers from escalating authority."""

    permitted = validate_command_authority_mode(allowed_mode)
    if value is None:
        return CommandAuthorityDirective(mode=permitted, emergency_brake=False)
    if isinstance(value, CommandAuthorityDirective):
        requested_mode = value.mode
        emergency_brake = value.emergency_brake
    elif isinstance(value, Mapping):
        requested_mode = value.get("mode", permitted)
        emergency_brake = value.get("emergency_brake", False)
    else:
        raise ValueError("command authority directive must be a mapping or CommandAuthorityDirective")
    requested = validate_command_authority_mode(requested_mode)
    if requested != permitted:
        raise ValueError(
            f"command authority escalation is not permitted: requested={requested}, allowed={permitted}"
        )
    return CommandAuthorityDirective(mode=permitted, emergency_brake=bool(emergency_brake))


def apply_command_authority(
    action_queue: list[np.ndarray] | tuple[np.ndarray, ...],
    directive: CommandAuthorityDirective | Mapping[str, Any] | None,
    *,
    allowed_mode: str,
) -> tuple[list[np.ndarray], CommandAuthorityDirective, int]:
    """Apply an authorized emergency brake to delayed commands.

    The returned queue keeps the original shape and contains zero velocity
    commands wherever authority permits cancellation.  The caller remains
    responsible for appending the new command and advancing the actuator model.
    """

    queue = [np.asarray(item, dtype=np.float64).copy() for item in action_queue]
    resolved = command_authority_directive(directive, allowed_mode=allowed_mode)
    if not resolved.emergency_brake or not queue or resolved.mode == "immutable":
        return queue, resolved, 0
    start = 0 if resolved.mode == "flush_pending" else 1
    overridden = 0
    for index in range(start, len(queue)):
        queue[index].fill(0.0)
        overridden += 1
    return queue, resolved, overridden


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


def _clip_row_with_jacobian(
    value: np.ndarray,
    max_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip one velocity row and return its local Jacobian."""

    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    identity = np.eye(vector.size, dtype=np.float64)
    if norm <= float(max_norm) + 1.0e-12:
        return vector.copy(), identity
    if norm <= 1.0e-12:
        return np.zeros_like(vector), identity
    unit = vector / norm
    jacobian = (float(max_norm) / norm) * (identity - np.outer(unit, unit))
    return vector * (float(max_norm) / norm), jacobian


def _advance_execution_with_jacobian(
    current_velocity: np.ndarray,
    delayed_action: np.ndarray,
    parameters: ExecutionParameters,
) -> tuple[ExecutionStep, np.ndarray, np.ndarray]:
    """Advance deterministic execution and return local velocity Jacobians.

    The derivatives follow the same clipped, tracked and acceleration-limited
    branches as :func:`advance_execution`.  They are local derivatives; the
    independent rollout certificate remains authoritative at branch changes.
    """

    current = np.asarray(current_velocity, dtype=np.float64)
    delayed = np.asarray(delayed_action, dtype=np.float64)
    if current.shape != delayed.shape or current.ndim != 2 or current.shape[-1] != 3:
        raise ValueError("current_velocity and delayed_action must have shape [defenders, 3]")
    defenders = int(current.shape[0])
    current_jacobian = np.zeros((defenders * 3, defenders * 3), dtype=np.float64)
    delayed_jacobian = np.zeros_like(current_jacobian)
    for index in range(defenders):
        start = index * 3
        delayed_jacobian[start : start + 3, start : start + 3] = np.eye(3, dtype=np.float64)

    if not parameters.enabled:
        zero = np.zeros_like(delayed)
        step = ExecutionStep(
            command=delayed.copy(),
            tracked=delayed.copy(),
            executed=delayed.copy(),
            noise=zero,
            noise_was_clipped=False,
        )
        return step, current_jacobian, delayed_jacobian

    commands = np.zeros_like(delayed)
    tracked = np.zeros_like(delayed)
    executed = np.zeros_like(delayed)
    command_jacobian = np.zeros_like(delayed_jacobian)
    tracked_from_current = np.zeros_like(current_jacobian)
    tracked_from_delayed = np.zeros_like(delayed_jacobian)
    execution_from_current = np.zeros_like(current_jacobian)
    execution_from_delayed = np.zeros_like(delayed_jacobian)
    alpha = float(parameters.tracking_alpha)
    drag = float(parameters.drag_gain)
    max_delta = float(parameters.max_acceleration_mps2) * float(parameters.dt_seconds) / float(parameters.mass_scale)

    for index in range(defenders):
        start = index * 3
        stop = start + 3
        command, command_local_jacobian = _clip_row_with_jacobian(
            delayed[index], parameters.max_speed_mps
        )
        commands[index] = command
        command_jacobian[start:stop, start:stop] = command_local_jacobian

        current_block = current[index]
        tracked_block = drag * ((1.0 - alpha) * current_block + alpha * command)
        tracked[index] = tracked_block
        tracked_from_current[start:stop, start:stop] = drag * (1.0 - alpha) * np.eye(3)
        tracked_from_delayed[start:stop, start:stop] = (
            drag * alpha * command_local_jacobian
        )

        delta = tracked_block - current_block
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm <= max_delta + 1.0e-12:
            executed_block = tracked_block
            executed_from_current_local = drag * (1.0 - alpha) * np.eye(3)
            executed_from_delayed_local = drag * alpha * command_local_jacobian
        elif delta_norm <= 1.0e-12:
            executed_block = current_block.copy()
            executed_from_current_local = np.eye(3)
            executed_from_delayed_local = np.zeros((3, 3), dtype=np.float64)
        else:
            unit = delta / delta_norm
            projection = (float(max_delta) / delta_norm) * (
                np.eye(3) - np.outer(unit, unit)
            )
            executed_block = current_block + float(max_delta) * unit
            executed_from_current_local = np.eye(3) + projection @ (
                drag * (1.0 - alpha) * np.eye(3) - np.eye(3)
            )
            executed_from_delayed_local = projection @ (
                drag * alpha * command_local_jacobian
            )

        executed_block, final_clip_jacobian = _clip_row_with_jacobian(
            executed_block, parameters.max_speed_mps
        )
        executed[index] = executed_block
        execution_from_current[start:stop, start:stop] = (
            final_clip_jacobian @ executed_from_current_local
        )
        execution_from_delayed[start:stop, start:stop] = (
            final_clip_jacobian @ executed_from_delayed_local
        )

    step = ExecutionStep(
        command=commands,
        tracked=tracked,
        executed=executed,
        noise=np.zeros_like(delayed),
        noise_was_clipped=False,
    )
    return step, execution_from_current, execution_from_delayed


def rollout_execution_with_action_jacobian(
    positions: np.ndarray,
    velocities: np.ndarray,
    action_queue: list[np.ndarray] | tuple[np.ndarray, ...],
    new_action: np.ndarray,
    parameters: ExecutionParameters,
    *,
    horizon_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray, list[ExecutionStep], np.ndarray, np.ndarray]:
    """Roll out deterministic execution and differentiate w.r.t. ``new_action``.

    Returns position and velocity Jacobians with shape
    ``[steps, defenders * 3, defenders * 3]``.  Queued commands are constants;
    the newly selected action is reused after the queue is exhausted, exactly
    as in :func:`rollout_execution`.
    """

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
    action_dimension = int(action.size)
    position_jacobian = np.zeros((action_dimension, action_dimension), dtype=np.float64)
    velocity_jacobian = np.zeros_like(position_jacobian)
    results: list[ExecutionStep] = []
    positions_out: list[np.ndarray] = []
    velocities_out: list[np.ndarray] = []
    position_jacobians: list[np.ndarray] = []
    velocity_jacobians: list[np.ndarray] = []
    for index in range(steps):
        delayed = sequence[index] if index < len(sequence) else action
        delayed_jacobian = np.eye(action_dimension, dtype=np.float64) if index >= len(action_queue) else np.zeros_like(position_jacobian)
        result, execution_from_current, execution_from_delayed = _advance_execution_with_jacobian(
            velocity, delayed, parameters
        )
        velocity_jacobian = (
            execution_from_current @ velocity_jacobian
            + execution_from_delayed @ delayed_jacobian
        )
        velocity = result.executed.copy()
        position = position + parameters.dt_seconds * velocity
        position_jacobian = position_jacobian + parameters.dt_seconds * velocity_jacobian
        results.append(result)
        positions_out.append(position.copy())
        velocities_out.append(velocity.copy())
        position_jacobians.append(position_jacobian.copy())
        velocity_jacobians.append(velocity_jacobian.copy())
    return (
        np.stack(positions_out, axis=0),
        np.stack(velocities_out, axis=0),
        results,
        np.stack(position_jacobians, axis=0),
        np.stack(velocity_jacobians, axis=0),
    )


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


def reachable_tube_radii(
    parameters: ExecutionParameters,
    defenders: int,
    horizon_steps: int,
    *,
    multiplier: float = 1.0,
) -> np.ndarray:
    """Return a preview-indexed position tube with an audited calibration factor.

    ``position_uncertainty_radii`` is the analytic noise-only tube shared by
    the execution model and certificates.  The multiplier permits a separate
    calibration experiment to cover bounded tracking and parameter mismatch.
    It is explicitly an empirical calibration factor unless a reachability
    proof for the execution model supplies it.
    """

    if not np.isfinite(float(multiplier)) or float(multiplier) < 1.0:
        raise ValueError("reachable-tube multiplier must be finite and at least one.")
    return float(multiplier) * position_uncertainty_radii(parameters, defenders, horizon_steps)


__all__ = [
    "CommandAuthorityDirective",
    "ExecutionParameters",
    "ExecutionStep",
    "apply_command_authority",
    "advance_execution",
    "clip_rows",
    "command_authority_directive",
    "command_authority_from_observation",
    "move_toward_velocity",
    "parameters_from_observation",
    "position_uncertainty_radii",
    "reachable_tube_radii",
    "queue_from_observation",
    "rollout_execution",
    "rollout_execution_with_action_jacobian",
    "validate_command_authority_mode",
]
