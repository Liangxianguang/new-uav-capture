"""Precondition auditing for queue-aware delayed-state rollouts.

The queue itself is part of the plant contract.  A newly planned suffix cannot
repair an already unsafe immutable prefix, so the evaluator should distinguish
that case from a safe prefix followed by an unsafe planned suffix.  This module
is deliberately diagnostic: it never escalates command authority and it does
not provide a reachable-set or safety proof.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np

from .execution_dynamics import ExecutionParameters, advance_execution
from .queue_aware_rollout import DelayedPlanningState, prefix_geometry_diagnostics


QDRPreconditionStatus = Literal[
    "prefix_safe_suffix_safe",
    "prefix_safe_suffix_unsafe",
    "prefix_unsafe_recoverable",
    "prefix_unsafe_unrecoverable",
]


@dataclass(frozen=True)
class QDRPreconditionAssessment:
    """Classification of the queued prefix and newly planned suffix."""

    status: QDRPreconditionStatus
    prefix_minimum_barrier_m: float
    suffix_minimum_barrier_m: float
    prefix_admissible: bool
    suffix_admissible: bool
    authority_mode: str
    recovery_allowed: bool
    recovery_recommended: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "prefix_minimum_barrier_m": float(self.prefix_minimum_barrier_m),
            "suffix_minimum_barrier_m": float(self.suffix_minimum_barrier_m),
            "prefix_admissible": bool(self.prefix_admissible),
            "suffix_admissible": bool(self.suffix_admissible),
            "authority_mode": str(self.authority_mode),
            "recovery_allowed": bool(self.recovery_allowed),
            "recovery_recommended": bool(self.recovery_recommended),
            "reason": str(self.reason),
        }


def _barrier(diagnostics: Mapping[str, Any], key: str) -> float:
    value = float(diagnostics.get(key, float("nan")))
    if not np.isfinite(value) and not np.isinf(value):
        raise ValueError(f"{key} must be finite or positive infinity")
    return value


def classify_qdr_precondition(
    prefix_diagnostics: Mapping[str, Any],
    suffix_diagnostics: Mapping[str, Any],
    *,
    authority_mode: str,
    tolerance_m: float = 0.0,
) -> QDRPreconditionAssessment:
    """Classify prefix/suffix feasibility without changing execution."""

    tolerance = float(tolerance_m)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_m must be finite and non-negative")
    authority = str(authority_mode)
    if not authority:
        raise ValueError("authority_mode must be non-empty")
    prefix_barrier = _barrier(prefix_diagnostics, "minimum_prefix_barrier_m")
    suffix_barrier = _barrier(suffix_diagnostics, "minimum_prefix_barrier_m")
    prefix_admissible = prefix_barrier >= -tolerance
    suffix_admissible = suffix_barrier >= -tolerance
    if prefix_admissible:
        if suffix_admissible:
            status: QDRPreconditionStatus = "prefix_safe_suffix_safe"
            reason = "queued prefix and planned suffix satisfy the nominal geometry precondition"
        else:
            status = "prefix_safe_suffix_unsafe"
            reason = "queued prefix is admissible but the newly planned suffix is not"
        return QDRPreconditionAssessment(
            status=status,
            prefix_minimum_barrier_m=prefix_barrier,
            suffix_minimum_barrier_m=suffix_barrier,
            prefix_admissible=True,
            suffix_admissible=suffix_admissible,
            authority_mode=authority,
            recovery_allowed=authority != "immutable",
            recovery_recommended=False,
            reason=reason,
        )
    recoverable = authority != "immutable"
    status = "prefix_unsafe_recoverable" if recoverable else "prefix_unsafe_unrecoverable"
    reason = (
        "queued prefix is already unsafe and the declared authority permits a recovery request"
        if recoverable
        else "queued prefix is already unsafe and immutable authority cannot repair it"
    )
    return QDRPreconditionAssessment(
        status=status,
        prefix_minimum_barrier_m=prefix_barrier,
        suffix_minimum_barrier_m=suffix_barrier,
        prefix_admissible=False,
        suffix_admissible=suffix_admissible,
        authority_mode=authority,
        recovery_allowed=recoverable,
        recovery_recommended=recoverable,
        reason=reason,
    )


def rollout_suffix_state(
    positions: np.ndarray,
    velocities: np.ndarray,
    action_sequence: np.ndarray,
    parameters: ExecutionParameters,
    *,
    authority_mode: str = "suffix",
) -> DelayedPlanningState:
    """Roll out a planned command suffix from the delayed planning state."""

    current_positions = np.asarray(positions, dtype=np.float64).copy()
    current_velocities = np.asarray(velocities, dtype=np.float64).copy()
    actions = np.asarray(action_sequence, dtype=np.float64)
    if (
        current_positions.ndim != 2
        or current_positions.shape[-1] != 3
        or current_velocities.shape != current_positions.shape
    ):
        raise ValueError("positions and velocities must have shape [defenders, 3]")
    if actions.ndim != 3 or actions.shape[1:] != current_positions.shape:
        raise ValueError("action_sequence must have shape [horizon, defenders, 3]")
    if actions.shape[0] <= 0 or not np.isfinite(actions).all():
        raise ValueError("action_sequence must be non-empty and finite")
    if not np.isfinite(current_positions).all() or not np.isfinite(current_velocities).all():
        raise ValueError("positions and velocities must be finite")

    prefix_positions: list[np.ndarray] = []
    prefix_velocities: list[np.ndarray] = []
    prefix_actions: list[np.ndarray] = []
    for action in actions:
        executed = advance_execution(
            current_velocities,
            action,
            parameters,
            noise=np.zeros_like(action),
        ).executed
        current_velocities = executed.copy()
        current_positions = current_positions + float(parameters.dt_seconds) * current_velocities
        prefix_positions.append(current_positions.copy())
        prefix_velocities.append(current_velocities.copy())
        prefix_actions.append(action.copy())
    return DelayedPlanningState(
        prefix_positions=np.stack(prefix_positions, axis=0),
        prefix_velocities=np.stack(prefix_velocities, axis=0),
        prefix_actions=np.stack(prefix_actions, axis=0),
        delayed_positions=current_positions,
        delayed_velocities=current_velocities,
        queue_length=int(actions.shape[0]),
        first_controllable_step=int(actions.shape[0]),
        authority_mode=str(authority_mode),
    )


def audit_qdr_precondition(
    *,
    prefix_diagnostics: Mapping[str, Any],
    delayed_positions: np.ndarray,
    delayed_velocities: np.ndarray,
    action_sequence: np.ndarray,
    observation: Mapping[str, Any],
    parameters: ExecutionParameters,
    drone_radius_m: float,
    safety_margin_m: float,
    authority_mode: str,
    tolerance_m: float = 0.0,
) -> dict[str, Any]:
    """Return prefix/suffix geometry diagnostics and a causal classification."""

    suffix_state = rollout_suffix_state(
        delayed_positions,
        delayed_velocities,
        action_sequence,
        parameters,
    )
    suffix_diagnostics = prefix_geometry_diagnostics(
        suffix_state,
        observation,
        drone_radius_m=float(drone_radius_m),
        safety_margin_m=float(safety_margin_m),
    )
    assessment = classify_qdr_precondition(
        prefix_diagnostics,
        suffix_diagnostics,
        authority_mode=authority_mode,
        tolerance_m=float(tolerance_m),
    )
    return {
        "assessment": assessment.as_dict(),
        "suffix": suffix_diagnostics,
    }


__all__ = [
    "QDRPreconditionAssessment",
    "QDRPreconditionStatus",
    "audit_qdr_precondition",
    "classify_qdr_precondition",
    "rollout_suffix_state",
]
