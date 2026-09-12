"""Delay-aware, split-conformal reachable tubes for target candidates.

The tube is calibrated offline from public-belief prediction candidates and
future labels.  Runtime code only consumes the frozen radius schedule together
with public queue/cache metadata; it never reads target truth.  The artifact
is an empirical calibration, not a continuous-time safety proof.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def upper_conformal_quantile(values: np.ndarray, coverage: float) -> float:
    """Return a finite-sample upper order statistic for split conformal use."""

    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("values must be a non-empty finite vector")
    if not 0.0 < float(coverage) < 1.0:
        raise ValueError("coverage must lie in (0, 1)")
    index = min(scores.size - 1, max(0, int(math.ceil((scores.size + 1) * float(coverage))) - 1))
    return float(np.sort(scores)[index])


def candidate_minimum_errors(candidates: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return the nearest-candidate error for every sample and horizon step."""

    values = np.asarray(candidates, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] != 3:
        raise ValueError("candidates must have shape [samples, candidates, horizon, 3]")
    if truth.ndim != 3 or truth.shape != (values.shape[0], values.shape[2], 3):
        raise ValueError("targets must have shape [samples, horizon, 3]")
    if not np.isfinite(values).all() or not np.isfinite(truth).all():
        raise ValueError("candidates and targets must be finite")
    distances = np.linalg.norm(values - truth[:, None, :, :], axis=-1)
    result = np.min(distances, axis=1)
    if not np.isfinite(result).all():
        raise FloatingPointError("candidate errors are non-finite")
    return result


def fit_conformal_radius_schedule(
    errors_by_step: np.ndarray,
    *,
    coverage: float,
    dt_seconds: float,
    target_speed_mps: float,
    calibration_count: int | None = None,
) -> dict[str, Any]:
    """Fit a horizon-dependent radius with a simultaneous coverage correction.

    First, each horizon receives a split-conformal marginal quantile.  The
    maximum normalized error over the complete horizon is then calibrated once
    more, which yields one schedule that is auditable for full-trajectory
    coverage rather than only per-step coverage.
    """

    errors = np.asarray(errors_by_step, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[0] <= 0 or errors.shape[1] <= 0:
        raise ValueError("errors_by_step must have shape [samples, horizon]")
    if not np.isfinite(errors).all() or np.any(errors < 0.0):
        raise ValueError("errors_by_step must be finite and non-negative")
    if not np.isfinite([coverage, dt_seconds, target_speed_mps]).all():
        raise ValueError("coverage, dt_seconds and target_speed_mps must be finite")
    if not 0.0 < float(coverage) < 1.0 or dt_seconds <= 0.0 or target_speed_mps <= 0.0:
        raise ValueError("coverage must be in (0, 1), and dynamics scales must be positive")
    count = int(errors.shape[0] if calibration_count is None else calibration_count)
    if count != errors.shape[0] or count <= 0:
        raise ValueError("calibration_count must match the number of error samples")

    marginal = np.asarray(
        [upper_conformal_quantile(errors[:, step], coverage) for step in range(errors.shape[1])],
        dtype=np.float64,
    )
    # An all-zero marginal score is valid.  The denominator only needs a
    # numerical floor for the normalized simultaneous score.
    denominator = np.maximum(marginal, 1.0e-9)
    simultaneous_scores = np.max(errors / denominator[None, :], axis=1)
    simultaneous_multiplier = upper_conformal_quantile(simultaneous_scores, coverage)
    radius = marginal * max(simultaneous_multiplier, 1.0)
    return {
        "method": "split_conformal_horizon_schedule_with_simultaneous_multiplier",
        "coverage": float(coverage),
        "target_speed_mps": float(target_speed_mps),
        "dt_seconds": float(dt_seconds),
        "calibration_count": count,
        "horizon_steps": int(errors.shape[1]),
        "marginal_radius_m_by_step": marginal.tolist(),
        "simultaneous_multiplier": float(simultaneous_multiplier),
        "radius_m_by_step": radius.tolist(),
        "calibration_min_error_m": float(np.min(errors)),
        "calibration_mean_error_m": float(np.mean(errors)),
        "calibration_max_error_m": float(np.max(errors)),
    }


def evaluate_tube_coverage(
    candidates: np.ndarray,
    targets: np.ndarray,
    radius_m_by_step: np.ndarray,
) -> dict[str, float]:
    """Evaluate per-step and full-trajectory candidate-set tube coverage."""

    errors = candidate_minimum_errors(candidates, targets)
    radius = np.asarray(radius_m_by_step, dtype=np.float64)
    if radius.ndim != 1 or radius.shape[0] != errors.shape[1] or not np.isfinite(radius).all() or np.any(radius < 0.0):
        raise ValueError("radius_m_by_step must be finite, non-negative and match the horizon")
    within = errors <= radius[None, :] + 1.0e-12
    return {
        "full_trajectory_coverage": float(np.mean(np.all(within, axis=1))),
        "horizon_mean_coverage": float(np.mean(within, axis=0).mean()),
        "horizon_min_coverage": float(np.min(np.mean(within, axis=0))),
        "horizon_max_coverage": float(np.max(np.mean(within, axis=0))),
        "mean_min_error_m": float(np.mean(errors)),
        "maximum_min_error_m": float(np.max(errors)),
    }


def evaluate_context_adaptive_tube_coverage(
    candidates: np.ndarray,
    targets: np.ndarray,
    radius_m_by_step: np.ndarray,
    context_scores: np.ndarray,
    *,
    uncertainty_gain: float,
) -> dict[str, float]:
    """Evaluate a tube inflated by a policy-safe per-window context score.

    The score is an observable uncertainty proxy, not a target-truth label.
    This routine reports empirical coverage only; it does not turn the tube
    into a distribution-free conditional guarantee.
    """

    errors = candidate_minimum_errors(candidates, targets)
    radius = np.asarray(radius_m_by_step, dtype=np.float64)
    scores = np.asarray(context_scores, dtype=np.float64)
    gain = float(uncertainty_gain)
    if (
        radius.ndim != 1
        or radius.shape[0] != errors.shape[1]
        or not np.isfinite(radius).all()
        or np.any(radius < 0.0)
        or scores.ndim != 1
        or scores.shape[0] != errors.shape[0]
        or not np.isfinite(scores).all()
        or not np.isfinite(gain)
        or gain < 0.0
    ):
        raise ValueError("adaptive tube inputs have incompatible or non-finite values")
    scores = np.clip(scores, 0.0, 1.0)
    per_sample_radius = radius[None, :] * (1.0 + gain * scores[:, None])
    within = errors <= per_sample_radius + 1.0e-12
    return {
        "full_trajectory_coverage": float(np.mean(np.all(within, axis=1))),
        "horizon_mean_coverage": float(np.mean(within, axis=0).mean()),
        "horizon_min_coverage": float(np.min(np.mean(within, axis=0))),
        "horizon_max_coverage": float(np.max(np.mean(within, axis=0))),
        "mean_min_error_m": float(np.mean(errors)),
        "maximum_min_error_m": float(np.max(errors)),
        "mean_effective_radius_m": float(np.mean(per_sample_radius)),
        "maximum_effective_radius_m": float(np.max(per_sample_radius)),
    }


def _extend_schedule(schedule: np.ndarray, required: int, *, dt_seconds: float, target_speed_mps: float) -> np.ndarray:
    if required <= schedule.size:
        return schedule[:required].copy()
    extra = np.arange(1, required - schedule.size + 1, dtype=np.float64)
    tail = schedule[-1] + extra * float(dt_seconds) * float(target_speed_mps)
    return np.concatenate([schedule, tail])


@dataclass(frozen=True)
class DelayAwareConformalReachableTube:
    """Frozen schedule used by runtime candidate budgeting and planning."""

    radius_m_by_step: tuple[float, ...]
    coverage: float
    dt_seconds: float
    target_speed_mps: float
    calibration_count: int
    simultaneous_multiplier: float
    uncertainty_gain: float = 0.25
    source_model_hash: str | None = None

    def __post_init__(self) -> None:
        radius = np.asarray(self.radius_m_by_step, dtype=np.float64)
        if radius.ndim != 1 or radius.size == 0 or not np.isfinite(radius).all() or np.any(radius < 0.0):
            raise ValueError("radius_m_by_step must be a non-empty finite non-negative vector")
        if not 0.0 < float(self.coverage) < 1.0:
            raise ValueError("coverage must lie in (0, 1)")
        if not np.isfinite([self.dt_seconds, self.target_speed_mps, self.simultaneous_multiplier]).all():
            raise ValueError("tube scales must be finite")
        if self.dt_seconds <= 0.0 or self.target_speed_mps <= 0.0 or self.simultaneous_multiplier <= 0.0:
            raise ValueError("tube scales must be positive")
        if int(self.calibration_count) <= 0:
            raise ValueError("calibration_count must be positive")
        if not np.isfinite(float(self.uncertainty_gain)) or float(self.uncertainty_gain) < 0.0:
            raise ValueError("uncertainty_gain must be finite and non-negative")
        object.__setattr__(self, "radius_m_by_step", tuple(float(value) for value in radius))

    @classmethod
    def from_summary(cls, summary: dict[str, Any]) -> "DelayAwareConformalReachableTube":
        values = dict(summary)
        if "radius_m_by_step" not in values:
            raise ValueError("calibration summary is missing radius_m_by_step")
        return cls(
            radius_m_by_step=tuple(float(value) for value in values["radius_m_by_step"]),
            coverage=float(values["coverage"]),
            dt_seconds=float(values["dt_seconds"]),
            target_speed_mps=float(values["target_speed_mps"]),
            calibration_count=int(values["calibration_count"]),
            simultaneous_multiplier=float(values["simultaneous_multiplier"]),
            uncertainty_gain=float(values.get("uncertainty_gain", 0.25)),
            source_model_hash=values.get("source_model_hash"),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "DelayAwareConformalReachableTube":
        source = Path(path).resolve()
        document = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(document, dict) and "tube" in document:
            document = document["tube"]
        if not isinstance(document, dict):
            raise ValueError("tube artifact must contain a JSON object")
        return cls.from_summary(document)

    def radius_by_step(
        self,
        horizon_steps: int,
        *,
        queue_length: int = 0,
        prediction_age_steps: int = 0,
        uncertainty_score: float = 0.0,
    ) -> np.ndarray:
        """Return the schedule aligned to delayed and cached planning state."""

        horizon = int(horizon_steps)
        queue = int(queue_length)
        age = int(prediction_age_steps)
        if horizon <= 0 or queue < 0 or age < 0:
            raise ValueError("horizon_steps must be positive and offsets non-negative")
        score = float(uncertainty_score)
        if not np.isfinite(score):
            raise ValueError("uncertainty_score must be finite")
        score = float(np.clip(score, 0.0, 1.0))
        offset = queue + age
        schedule = _extend_schedule(
            np.asarray(self.radius_m_by_step, dtype=np.float64),
            offset + horizon,
            dt_seconds=self.dt_seconds,
            target_speed_mps=self.target_speed_mps,
        )
        aligned = schedule[offset : offset + horizon]
        return aligned * (1.0 + float(self.uncertainty_gain) * score)

    def budget_score(self, *, scale_m: float, horizon_steps: int, queue_length: int = 0, prediction_age_steps: int = 0) -> float:
        """Normalize tube width to the [0, 1] uncertainty feature range."""

        scale = float(scale_m)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale_m must be finite and positive")
        radius = self.radius_by_step(
            horizon_steps,
            queue_length=queue_length,
            prediction_age_steps=prediction_age_steps,
        )
        return float(np.clip(np.max(radius) / scale, 0.0, 1.0))


__all__ = [
    "DelayAwareConformalReachableTube",
    "candidate_minimum_errors",
    "evaluate_context_adaptive_tube_coverage",
    "evaluate_tube_coverage",
    "fit_conformal_radius_schedule",
    "upper_conformal_quantile",
]
