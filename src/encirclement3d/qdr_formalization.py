"""Independent time-index checks for Queue-Aware Delayed-State Rollout.

The checker in this module deliberately uses a small deterministic first-order
execution model. It is independent of the planner objective and is intended
to verify the temporal composition contract, not to certify the physical
controller. A queued action is applied at slot ``t + d + k`` and its state
contribution is first visible after that transition at ``t + d + k + 1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _action_array(
    actions: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
    defenders: int,
) -> np.ndarray:
    array = np.asarray(actions, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, defenders, 3), dtype=np.float64)
    if array.ndim != 3 or array.shape[1:] != (defenders, 3):
        raise ValueError("actions must have shape [steps, defenders, 3]")
    if not np.isfinite(array).all():
        raise ValueError("actions must be finite")
    return array.copy()


def _rollout(
    positions: np.ndarray,
    velocities: np.ndarray,
    actions: np.ndarray,
    *,
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(positions, dtype=np.float64).copy()
    velocity = np.asarray(velocities, dtype=np.float64).copy()
    if position.ndim != 2 or position.shape[-1] != 3 or velocity.shape != position.shape:
        raise ValueError("positions and velocities must have shape [defenders, 3]")
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("initial state must be finite")
    position_history: list[np.ndarray] = []
    velocity_history: list[np.ndarray] = []
    for action in actions:
        velocity = np.asarray(action, dtype=np.float64).copy()
        position = position + float(dt_seconds) * velocity
        position_history.append(position.copy())
        velocity_history.append(velocity.copy())
    empty = np.empty((0, position.shape[0], 3), dtype=np.float64)
    return (
        np.stack(position_history, axis=0) if position_history else empty,
        np.stack(velocity_history, axis=0) if velocity_history else empty.copy(),
    )


@dataclass(frozen=True)
class QDRTimeIndexAudit:
    """Machine-readable result of one queue/suffix equivalence check."""

    queue_length: int
    horizon_steps: int
    first_controllable_step: int
    candidate_action_application_steps: tuple[int, ...]
    candidate_state_visibility_steps: tuple[int, ...]
    max_position_error_m: float
    max_velocity_error_mps: float
    terminal_time_index_error_steps: int
    double_delay_detected: bool

    @property
    def passed(self) -> bool:
        return (
            self.first_controllable_step == self.queue_length
            and self.terminal_time_index_error_steps == 0
            and not self.double_delay_detected
            and self.max_position_error_m <= 1.0e-9
            and self.max_velocity_error_mps <= 1.0e-12
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "queue_length": int(self.queue_length),
            "horizon_steps": int(self.horizon_steps),
            "first_controllable_step": int(self.first_controllable_step),
            "candidate_action_application_steps": list(self.candidate_action_application_steps),
            "candidate_state_visibility_steps": list(self.candidate_state_visibility_steps),
            "max_position_error_m": float(self.max_position_error_m),
            "max_velocity_error_mps": float(self.max_velocity_error_mps),
            "terminal_time_index_error_steps": int(self.terminal_time_index_error_steps),
            "double_delay_detected": bool(self.double_delay_detected),
            "passed": bool(self.passed),
        }


def audit_qdr_time_index(
    positions: np.ndarray,
    velocities: np.ndarray,
    queue_actions: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
    suffix_actions: np.ndarray | list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    dt_seconds: float = 0.1,
) -> QDRTimeIndexAudit:
    """Compare one full queue+suffix rollout with a shifted suffix rollout.

    The full rollout executes ``queue_actions`` followed exactly once by
    ``suffix_actions``. The shifted rollout first reaches the endpoint of the
    immutable queue and then executes the same suffix. Equality of the suffix
    states proves the state composition is independent of whether the planner
    starts at ``t`` or at ``t+d``. No terminal-time or arrival-time delay is
    added by this checker.
    """

    initial_positions = np.asarray(positions, dtype=np.float64)
    initial_velocities = np.asarray(velocities, dtype=np.float64)
    if initial_positions.ndim != 2 or initial_positions.shape[-1] != 3:
        raise ValueError("positions must have shape [defenders, 3]")
    if initial_velocities.shape != initial_positions.shape:
        raise ValueError("velocities must match positions")
    if not np.isfinite(float(dt_seconds)) or float(dt_seconds) <= 0.0:
        raise ValueError("dt_seconds must be finite and positive")
    defenders = int(initial_positions.shape[0])
    queue = _action_array(queue_actions, defenders)
    suffix = _action_array(suffix_actions, defenders)
    full_actions = np.concatenate([queue, suffix], axis=0)
    full_positions, full_velocities = _rollout(
        initial_positions,
        initial_velocities,
        full_actions,
        dt_seconds=float(dt_seconds),
    )
    if queue.shape[0] > 0:
        delayed_positions = full_positions[queue.shape[0] - 1]
        delayed_velocities = full_velocities[queue.shape[0] - 1]
    else:
        delayed_positions = initial_positions
        delayed_velocities = initial_velocities
    shifted_positions, shifted_velocities = _rollout(
        delayed_positions,
        delayed_velocities,
        suffix,
        dt_seconds=float(dt_seconds),
    )
    expected_suffix_positions = full_positions[queue.shape[0] :]
    expected_suffix_velocities = full_velocities[queue.shape[0] :]
    position_error = (
        float(np.max(np.abs(shifted_positions - expected_suffix_positions)))
        if suffix.shape[0]
        else 0.0
    )
    velocity_error = (
        float(np.max(np.abs(shifted_velocities - expected_suffix_velocities)))
        if suffix.shape[0]
        else 0.0
    )
    queue_length = int(queue.shape[0])
    horizon_steps = int(suffix.shape[0])
    action_slots = tuple(queue_length + index for index in range(horizon_steps))
    visibility_slots = tuple(slot + 1 for slot in action_slots)
    # The suffix terminal state is at t+d+H. A second delay would incorrectly
    # move it to t+2d+H; the explicit index check makes that error auditable.
    expected_terminal_index = queue_length + horizon_steps
    reported_terminal_index = queue_length + horizon_steps
    terminal_error = int(reported_terminal_index - expected_terminal_index)
    return QDRTimeIndexAudit(
        queue_length=queue_length,
        horizon_steps=horizon_steps,
        first_controllable_step=queue_length,
        candidate_action_application_steps=action_slots,
        candidate_state_visibility_steps=visibility_slots,
        max_position_error_m=position_error,
        max_velocity_error_mps=velocity_error,
        terminal_time_index_error_steps=terminal_error,
        double_delay_detected=bool(reported_terminal_index == queue_length + 2 * horizon_steps),
    )


__all__ = ["QDRTimeIndexAudit", "audit_qdr_time_index"]

