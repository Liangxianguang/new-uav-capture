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


QDRSegmentStatus = Literal[
    "prefix_infeasible",
    "suffix_infeasible",
    "terminal_all_candidates_feasible",
    "terminal_candidate_feasible",
    "terminal_progress_only",
    "terminal_not_feasible",
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


@dataclass(frozen=True)
class QDRSegmentLivenessAssessment:
    """Public-belief prefix/suffix/terminal liveness diagnostics.

    The terminal quantities are computed against the projected candidate set,
    not the simulator target.  ``terminal_candidate_feasible`` is therefore an
    optimistic candidate-space diagnostic; it is not a robust reachability
    guarantee and it does not change command authority.
    """

    status: QDRSegmentStatus
    prefix_feasible: bool
    suffix_feasible: bool
    terminal_any_candidate_feasible: bool
    terminal_all_candidate_feasible: bool
    earliest_any_candidate_capture_step: int
    earliest_all_candidate_capture_step: int
    best_terminal_distance_m: float
    worst_terminal_distance_m: float
    best_progress_m: float
    worst_progress_m: float
    finite_progress_available: bool
    horizon_steps: int
    candidate_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "prefix_feasible": bool(self.prefix_feasible),
            "suffix_feasible": bool(self.suffix_feasible),
            "terminal_any_candidate_feasible": bool(self.terminal_any_candidate_feasible),
            "terminal_all_candidate_feasible": bool(self.terminal_all_candidate_feasible),
            "earliest_any_candidate_capture_step": int(self.earliest_any_candidate_capture_step),
            "earliest_all_candidate_capture_step": int(self.earliest_all_candidate_capture_step),
            "best_terminal_distance_m": float(self.best_terminal_distance_m),
            "worst_terminal_distance_m": float(self.worst_terminal_distance_m),
            "best_progress_m": float(self.best_progress_m),
            "worst_progress_m": float(self.worst_progress_m),
            "finite_progress_available": bool(self.finite_progress_available),
            "horizon_steps": int(self.horizon_steps),
            "candidate_count": int(self.candidate_count),
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


def queue_prefix_risk_score(
    diagnostics: Mapping[str, Any],
    *,
    safety_margin_m: float,
    risk_buffer_m: float = 0.15,
    scale_m: float = 0.50,
) -> float:
    """Convert public queued-prefix geometry into a bounded UAKR risk feature.

    The score is intentionally a transparent, non-learned feature.  It uses
    only the minimum public clearance along the immutable queue prefix and the
    reported safety-margin violation.  An empty queue has infinite clearance
    and therefore zero risk.  This is a budget trigger, not a safety
    certificate and it does not change command authority.
    """

    margin = float(safety_margin_m)
    buffer = float(risk_buffer_m)
    scale = float(scale_m)
    if not np.isfinite([margin, buffer, scale]).all() or margin < 0.0 or buffer < 0.0 or scale <= 0.0:
        raise ValueError("safety_margin_m and risk_buffer_m must be non-negative; scale_m must be positive")
    if "minimum_clearance_m" not in diagnostics:
        raise ValueError("diagnostics must contain minimum_clearance_m")
    clearance = float(diagnostics["minimum_clearance_m"])
    violation = float(diagnostics.get("maximum_safety_margin_violation_m", 0.0))
    if not np.isfinite(clearance) and not np.isinf(clearance):
        raise ValueError("minimum_clearance_m must be finite or infinite")
    if not np.isfinite(violation) or violation < 0.0:
        raise ValueError("maximum_safety_margin_violation_m must be finite and non-negative")
    if np.isinf(clearance):
        return 0.0
    near_margin_gap = max(margin + buffer - clearance, 0.0)
    return float(np.clip(max(near_margin_gap, violation) / scale, 0.0, 1.0))


def audit_qdr_segment_liveness(
    *,
    prefix_diagnostics: Mapping[str, Any],
    suffix_diagnostics: Mapping[str, Any],
    defender_positions_path: np.ndarray,
    candidate_target_paths: np.ndarray,
    capture_radius_m: float,
    progress_tolerance_m: float = 1.0e-9,
) -> dict[str, Any]:
    """Audit prefix, suffix, and terminal progress in public candidate space.

    ``defender_positions_path`` has shape ``[H, defenders, 3]`` and
    ``candidate_target_paths`` has shape ``[K, H, 3]``.  The first candidate
    capture step is the earliest step for which *any* projected target
    candidate is within ``capture_radius_m`` of one defender.  The all-candidate
    step uses the maximum over candidates and is a stricter diagnostic.  The
    function deliberately does not read simulator truth or mutate a queue.
    """

    defenders = np.asarray(defender_positions_path, dtype=np.float64)
    targets = np.asarray(candidate_target_paths, dtype=np.float64)
    radius = float(capture_radius_m)
    tolerance = float(progress_tolerance_m)
    if (
        defenders.ndim != 3
        or defenders.shape[-1] != 3
        or defenders.shape[0] <= 0
        or defenders.shape[1] <= 0
    ):
        raise ValueError("defender_positions_path must have shape [horizon, defenders, 3]")
    if (
        targets.ndim != 3
        or targets.shape[-1] != 3
        or targets.shape[0] <= 0
        or targets.shape[1] != defenders.shape[0]
    ):
        raise ValueError("candidate_target_paths must have shape [candidates, horizon, 3]")
    if not np.isfinite(defenders).all() or not np.isfinite(targets).all():
        raise ValueError("segment audit paths must be finite")
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("capture_radius_m must be finite and non-negative")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("progress_tolerance_m must be finite and non-negative")

    # [H, K, defenders] nearest-defender distance for every public target
    # candidate.  The selected plan and candidate set are the only inputs.
    distances = np.linalg.norm(
        defenders[:, None, :, :] - targets.transpose(1, 0, 2)[:, :, None, :],
        axis=-1,
    )
    nearest = np.min(distances, axis=2)
    any_capture = np.min(nearest, axis=1) <= radius
    all_capture = np.max(nearest, axis=1) <= radius
    any_steps = np.flatnonzero(any_capture)
    all_steps = np.flatnonzero(all_capture)
    earliest_any = int(any_steps[0] + 1) if any_steps.size else -1
    earliest_all = int(all_steps[0] + 1) if all_steps.size else -1
    initial_by_candidate = nearest[0]
    terminal_by_candidate = nearest[-1]
    candidate_progress = initial_by_candidate - terminal_by_candidate
    best_progress = float(np.max(candidate_progress))
    worst_progress = float(np.min(candidate_progress))
    prefix_barrier = float(prefix_diagnostics.get("minimum_prefix_barrier_m", float("nan")))
    suffix_barrier = float(suffix_diagnostics.get("minimum_prefix_barrier_m", float("nan")))
    if not np.isfinite(prefix_barrier) and not np.isinf(prefix_barrier):
        raise ValueError("prefix_diagnostics minimum_prefix_barrier_m must be finite or infinity")
    if not np.isfinite(suffix_barrier) and not np.isinf(suffix_barrier):
        raise ValueError("suffix_diagnostics minimum_prefix_barrier_m must be finite or infinity")
    prefix_feasible = prefix_barrier >= -tolerance
    suffix_feasible = suffix_barrier >= -tolerance
    terminal_any = bool(np.any(terminal_by_candidate <= radius))
    terminal_all = bool(np.all(terminal_by_candidate <= radius))
    if not prefix_feasible:
        status: QDRSegmentStatus = "prefix_infeasible"
    elif not suffix_feasible:
        status = "suffix_infeasible"
    elif terminal_all:
        status = "terminal_all_candidates_feasible"
    elif terminal_any:
        status = "terminal_candidate_feasible"
    elif best_progress > tolerance:
        status = "terminal_progress_only"
    else:
        status = "terminal_not_feasible"
    assessment = QDRSegmentLivenessAssessment(
        status=status,
        prefix_feasible=bool(prefix_feasible),
        suffix_feasible=bool(suffix_feasible),
        terminal_any_candidate_feasible=terminal_any,
        terminal_all_candidate_feasible=terminal_all,
        earliest_any_candidate_capture_step=earliest_any,
        earliest_all_candidate_capture_step=earliest_all,
        best_terminal_distance_m=float(np.min(terminal_by_candidate)),
        worst_terminal_distance_m=float(np.max(terminal_by_candidate)),
        best_progress_m=best_progress,
        worst_progress_m=worst_progress,
        finite_progress_available=bool(earliest_any > 0 or best_progress > tolerance),
        horizon_steps=int(defenders.shape[0]),
        candidate_count=int(targets.shape[0]),
    )
    return {"assessment": assessment.as_dict()}


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
    "QDRSegmentLivenessAssessment",
    "QDRSegmentStatus",
    "audit_qdr_precondition",
    "audit_qdr_segment_liveness",
    "classify_qdr_precondition",
    "queue_prefix_risk_score",
    "rollout_suffix_state",
]
