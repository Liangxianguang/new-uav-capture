"""Deterministic escape-gap geometry for multi-UAV encirclement.

The metric in this module is deliberately planner-facing and observation-only.
It does not infer target truth: the target paths supplied by the caller are
the same public, dynamics-projected candidates already consumed by the MPC.
The result is a finite soft cost, not a hard safety constraint or a
reachability proof.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_TWO_PI = 2.0 * np.pi


def _validate_paths(defender_position_paths: np.ndarray, target_paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    defenders = np.asarray(defender_position_paths, dtype=np.float64)
    targets = np.asarray(target_paths, dtype=np.float64)
    if defenders.ndim != 4 or defenders.shape[-1] != 3:
        raise ValueError("defender_position_paths must have shape [sequences, horizon, defenders, 3].")
    if targets.ndim != 3 or targets.shape[-1] != 3:
        raise ValueError("target_paths must have shape [candidates, horizon, 3].")
    if defenders.shape[1] != targets.shape[1]:
        raise ValueError("defender and target horizons must match.")
    if defenders.shape[1] <= 0 or defenders.shape[2] <= 0 or targets.shape[0] <= 0:
        raise ValueError("escape-gap paths must contain at least one sequence, timestep, defender and candidate.")
    if not np.isfinite(defenders).all() or not np.isfinite(targets).all():
        raise ValueError("escape-gap paths must be finite.")
    return defenders, targets


def _target_velocity_directions(target_paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return target horizontal direction angles and speeds for each future step."""

    targets = np.asarray(target_paths, dtype=np.float64)
    candidate_count, horizon, _ = targets.shape
    deltas = np.zeros_like(targets)
    if horizon >= 2:
        # The first future direction is estimated from the first two public
        # candidate points; later directions use the preceding displacement.
        deltas[:, 0] = targets[:, 1] - targets[:, 0]
        deltas[:, 1:] = targets[:, 1:] - targets[:, :-1]
    horizontal = deltas[..., :2]
    speed = np.linalg.norm(horizontal, axis=-1)
    direction = np.arctan2(horizontal[..., 1], horizontal[..., 0])
    return direction, speed


def escape_gap_metrics(
    defender_position_paths: np.ndarray,
    target_paths: np.ndarray,
    *,
    gap_safe_rad: float = 1.80,
    escape_gap_safe_rad: float = 1.80,
    max_gap_weight: float = 1.0,
    escape_gap_weight: float = 1.0,
    horizon_discount: float = 1.0,
    min_target_speed_mps: float = 1.0e-6,
) -> dict[str, np.ndarray]:
    """Compute soft maximum-gap and escape-gap costs.

    Parameters
    ----------
    defender_position_paths:
        Array with shape ``[sequence, horizon, defender, 3]``.
    target_paths:
        Array with shape ``[candidate, horizon, 3]``.
    gap_safe_rad / escape_gap_safe_rad:
        Soft thresholds.  Values above the thresholds are penalized; no
        threshold is a hard feasibility constraint.
    max_gap_weight / escape_gap_weight:
        Relative contributions of the two hinge-squared terms.
    horizon_discount:
        Geometric discount in ``(0, 1]``.  The returned cost is normalized by
        the sum of discounts, so it remains comparable across horizons.
    min_target_speed_mps:
        When the public candidate has no horizontal direction, its escape gap
        is conservatively set to the maximum gap.

    Returns
    -------
    dict[str, numpy.ndarray]
        ``cost`` has shape ``[sequence, candidate]``.  The remaining metrics
        have the same shape and summarize the horizon: ``max_gap_rad`` and
        ``escape_gap_rad`` are maxima, ``coverage_ratio`` is the mean circular
        coverage fraction, and ``max_gap_violation_rate`` is the fraction of
        steps above ``gap_safe_rad``.
    """

    defenders, targets = _validate_paths(defender_position_paths, target_paths)
    scalar_values = {
        "gap_safe_rad": gap_safe_rad,
        "escape_gap_safe_rad": escape_gap_safe_rad,
        "max_gap_weight": max_gap_weight,
        "escape_gap_weight": escape_gap_weight,
        "horizon_discount": horizon_discount,
        "min_target_speed_mps": min_target_speed_mps,
    }
    for name, value in scalar_values.items():
        if not np.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
    if float(gap_safe_rad) < 0.0 or float(gap_safe_rad) > _TWO_PI:
        raise ValueError("gap_safe_rad must lie in [0, 2*pi].")
    if float(escape_gap_safe_rad) < 0.0 or float(escape_gap_safe_rad) > _TWO_PI:
        raise ValueError("escape_gap_safe_rad must lie in [0, 2*pi].")
    if float(max_gap_weight) < 0.0 or float(escape_gap_weight) < 0.0:
        raise ValueError("escape-gap weights must be non-negative.")
    if not 0.0 < float(horizon_discount) <= 1.0:
        raise ValueError("horizon_discount must lie in (0, 1].")
    if float(min_target_speed_mps) < 0.0:
        raise ValueError("min_target_speed_mps must be non-negative.")

    # Relative horizontal bearings have shape [sequence, candidate, horizon,
    # defender].  A defender exactly at the target is assigned angle zero;
    # such a state is already handled by the existing capture/safety terms.
    relative = defenders[:, None, :, :, :2] - targets[None, :, :, None, :2]
    angles = np.mod(np.arctan2(relative[..., 1], relative[..., 0]), _TWO_PI)
    sorted_angles = np.sort(angles, axis=-1)
    circular = np.concatenate(
        [sorted_angles, sorted_angles[..., :1] + _TWO_PI],
        axis=-1,
    )
    gaps = np.diff(circular, axis=-1)
    max_gap_by_step = np.max(gaps, axis=-1)

    direction, speed = _target_velocity_directions(targets)
    direction = direction[None, :, :, None]
    relative_direction = np.mod(direction - sorted_angles, _TWO_PI)
    contains_direction = relative_direction <= gaps + 1.0e-10
    contained_gap = np.where(contains_direction, gaps, -np.inf)
    escape_gap_by_step = np.max(contained_gap, axis=-1)

    # A floating-point direction can fall exactly outside every interval by a
    # tiny amount.  Use the nearest gap midpoint as a deterministic fallback.
    midpoints = sorted_angles + 0.5 * gaps
    midpoint_delta = np.abs(
        np.mod(direction - midpoints + np.pi, _TWO_PI) - np.pi
    )
    nearest_gap_index = np.argmin(midpoint_delta, axis=-1)
    nearest_gap = np.take_along_axis(gaps, nearest_gap_index[..., None], axis=-1)[..., 0]
    escape_gap_by_step = np.where(
        np.isfinite(escape_gap_by_step),
        escape_gap_by_step,
        nearest_gap,
    )
    # A stationary/vertical candidate has no trustworthy horizontal escape
    # heading.  Treat its largest gap as the relevant risk.
    escape_gap_by_step = np.where(
        speed[None, ...] > float(min_target_speed_mps),
        escape_gap_by_step,
        max_gap_by_step,
    )

    horizon = defenders.shape[1]
    discounts = np.power(float(horizon_discount), np.arange(horizon, dtype=np.float64))
    discounts /= max(float(discounts.sum()), 1.0e-12)
    max_gap_excess = np.maximum(max_gap_by_step - float(gap_safe_rad), 0.0)
    escape_gap_excess = np.maximum(escape_gap_by_step - float(escape_gap_safe_rad), 0.0)
    per_step_cost = (
        float(max_gap_weight) * max_gap_excess * max_gap_excess
        + float(escape_gap_weight) * escape_gap_excess * escape_gap_excess
    )
    cost = np.sum(per_step_cost * discounts[None, None, :], axis=-1)
    result: dict[str, np.ndarray] = {
        "cost": cost,
        "max_gap_rad": np.max(max_gap_by_step, axis=-1),
        "escape_gap_rad": np.max(escape_gap_by_step, axis=-1),
        "coverage_ratio": np.mean(1.0 - max_gap_by_step / _TWO_PI, axis=-1),
        "max_gap_violation_rate": np.mean(max_gap_excess > 0.0, axis=-1),
        "escape_gap_violation_rate": np.mean(escape_gap_excess > 0.0, axis=-1),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("escape-gap metrics contain non-finite values.")
    return result


__all__ = ["escape_gap_metrics"]
