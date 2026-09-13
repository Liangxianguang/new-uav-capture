"""Deterministic freshness--covariance fusion for public target beliefs.

The fusion rule is deliberately planner-side and policy-safe.  It consumes
only the per-defender public belief, confidence, age and covariance fields;
the environment's target state is never consulted.  Position age
compensation is intentionally not performed here because the simulator's
``time_aligned`` belief contract already advances stale means to the current
observation time.  Age is instead used to discount stale information and
inflate its reported covariance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PublicBeliefFusionResult:
    """Fused public belief and auditable reliability diagnostics."""

    position: np.ndarray
    velocity: np.ndarray
    covariance: np.ndarray
    normalized_weights: np.ndarray
    effective_sample_size: float
    weight_entropy: float
    mean_age_steps: float
    max_age_steps: float
    mean_covariance_trace_m2: float
    fallback_used: bool
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64)
        velocity = np.asarray(self.velocity, dtype=np.float64)
        covariance = np.asarray(self.covariance, dtype=np.float64)
        weights = np.asarray(self.normalized_weights, dtype=np.float64)
        if position.shape != (3,) or velocity.shape != (3,):
            raise ValueError("fused position and velocity must have shape [3]")
        if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
            raise ValueError("fused covariance must be a finite 3x3 matrix")
        if weights.ndim != 1 or not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise ValueError("normalized fusion weights must be a finite non-negative vector")
        if weights.size == 0 or not np.isclose(float(weights.sum()), 1.0, atol=1.0e-8):
            raise ValueError("normalized fusion weights must sum to one")
        if not np.isfinite(
            [
                self.effective_sample_size,
                self.weight_entropy,
                self.mean_age_steps,
                self.max_age_steps,
                self.mean_covariance_trace_m2,
            ]
        ).all():
            raise ValueError("fusion diagnostics must be finite")

    def as_dict(self, *, enabled: bool = True) -> dict[str, Any]:
        """Return scalar diagnostics suitable for step JSONL/TensorBoard."""

        return {
            "belief_fusion_enabled": 1.0 if enabled else 0.0,
            "belief_fusion_effective_sample_size": float(self.effective_sample_size),
            "belief_fusion_weight_entropy": float(self.weight_entropy),
            "belief_fusion_mean_age_steps": float(self.mean_age_steps),
            "belief_fusion_max_age_steps": float(self.max_age_steps),
            "belief_fusion_mean_covariance_trace_m2": float(self.mean_covariance_trace_m2),
            "belief_fusion_fallback_used": 1.0 if self.fallback_used else 0.0,
            "belief_fusion_fallback_reason": self.fallback_reason or "none",
        }


def _as_vector(value: Any, count: int, name: str, default: float) -> np.ndarray:
    if value is None:
        return np.full(count, float(default), dtype=np.float64)
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (count,):
        raise ValueError(f"{name} must have shape [{count}]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _regularized_covariances(
    value: Any,
    count: int,
    *,
    covariance_floor_m2: float,
) -> tuple[np.ndarray, bool]:
    missing = value is None
    if missing:
        raw = np.broadcast_to(
            np.eye(3, dtype=np.float64)[None, :, :] * float(covariance_floor_m2),
            (count, 3, 3),
        ).copy()
    else:
        raw = np.asarray(value, dtype=np.float64)
        if raw.shape != (count, 3, 3):
            raise ValueError(f"target_observation_covariance must have shape [{count}, 3, 3]")
        if not np.isfinite(raw).all():
            raise ValueError("target_observation_covariance must be finite")

    result = np.empty_like(raw)
    floor = float(covariance_floor_m2)
    for index in range(count):
        symmetric = 0.5 * (raw[index] + raw[index].T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        eigenvalues = np.maximum(eigenvalues, floor)
        result[index] = (eigenvectors * eigenvalues[None, :]) @ eigenvectors.T
    return result, missing


def fuse_public_beliefs(
    observation: dict[str, Any],
    *,
    age_decay: float = 0.15,
    age_inflation_m2: float = 0.05,
    dropout_inflation_m2: float = 0.25,
    covariance_floor_m2: float = 0.01,
    confidence_power: float = 1.0,
    min_effective_samples: float = 1.0,
    anchor_index: int | None = None,
) -> PublicBeliefFusionResult:
    """Fuse public beliefs with freshness and covariance reliability weights.

    ``anchor_index`` identifies the receiving defender for a distributed
    local planner.  If the effective sample size is below the declared
    threshold, the result deterministically falls back to that defender's
    public belief.  Centralized callers may omit it; in that case the most
    reliable source is used as the fallback anchor.
    """

    positive = {
        "age_decay": age_decay,
        "covariance_floor_m2": covariance_floor_m2,
        "confidence_power": confidence_power,
        "min_effective_samples": min_effective_samples,
    }
    for name, value in positive.items():
        if not np.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    non_negative = {
        "age_inflation_m2": age_inflation_m2,
        "dropout_inflation_m2": dropout_inflation_m2,
    }
    for name, value in non_negative.items():
        if not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    positions = np.asarray(observation["target_belief_positions"], dtype=np.float64)
    velocities = np.asarray(observation["target_belief_velocities"], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or velocities.shape != positions.shape:
        raise ValueError("target belief positions and velocities must have shape [defenders, 3]")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise ValueError("target beliefs must be finite")
    count = int(positions.shape[0])
    if count <= 0:
        raise ValueError("at least one public belief is required")
    if anchor_index is not None and not 0 <= int(anchor_index) < count:
        raise ValueError("anchor_index must identify a valid public belief")

    confidences = _as_vector(
        observation.get("target_observation_confidence"), count, "target_observation_confidence", 1.0
    )
    ages = _as_vector(observation.get("message_age_steps"), count, "message_age_steps", 0.0)
    confidences = np.maximum(confidences, 0.0)
    ages = np.maximum(ages, 0.0)
    covariances, covariance_missing = _regularized_covariances(
        observation.get("target_observation_covariance"),
        count,
        covariance_floor_m2=covariance_floor_m2,
    )
    dropout_value = observation.get("target_observation_dropout")
    if dropout_value is None:
        dropout = confidences <= 0.0
    else:
        dropout = np.asarray(dropout_value, dtype=bool)
        if dropout.shape != (count,):
            raise ValueError(f"target_observation_dropout must have shape [{count}]")

    inflated = covariances + (
        float(age_inflation_m2) * ages + float(dropout_inflation_m2) * dropout.astype(np.float64)
    )[:, None, None] * np.eye(3, dtype=np.float64)[None, :, :]
    traces = np.trace(inflated, axis1=1, axis2=2)
    reliability = (
        np.maximum(confidences, 0.0) ** float(confidence_power)
        * np.exp(-float(age_decay) * ages)
        / np.maximum(traces, 3.0 * float(covariance_floor_m2))
    )
    finite_reliability = np.isfinite(reliability)
    reliability = np.where(finite_reliability, np.maximum(reliability, 0.0), 0.0)
    fallback_reason: str | None = None
    if float(reliability.sum()) <= 1.0e-12:
        fallback_reason = "no_positive_reliability"
    else:
        normalized = reliability / float(reliability.sum())
        effective_sample_size = 1.0 / max(float(np.sum(normalized * normalized)), 1.0e-12)
        if effective_sample_size < float(min_effective_samples):
            fallback_reason = "insufficient_effective_samples"

    if fallback_reason is not None:
        if anchor_index is None:
            anchor = int(np.argmax(reliability)) if float(reliability.sum()) > 0.0 else 0
        else:
            anchor = int(anchor_index)
        normalized = np.zeros(count, dtype=np.float64)
        normalized[anchor] = 1.0
        fallback_used = True
    else:
        fallback_used = False

    fused_position = np.sum(positions * normalized[:, None], axis=0)
    fused_velocity = np.sum(velocities * normalized[:, None], axis=0)
    offsets = positions - fused_position[None, :]
    fused_covariance = np.sum(
        normalized[:, None, None]
        * (inflated + offsets[:, :, None] * offsets[:, None, :]),
        axis=0,
    )
    effective_sample_size = 1.0 / max(float(np.sum(normalized * normalized)), 1.0e-12)
    positive_weights = normalized[normalized > 0.0]
    weight_entropy = float(-np.sum(positive_weights * np.log(positive_weights)))
    mean_age = float(np.dot(normalized, ages))
    mean_trace = float(np.dot(normalized, np.trace(inflated, axis1=1, axis2=2)))
    if covariance_missing and fallback_reason is None:
        fallback_reason = "default_covariance"
    return PublicBeliefFusionResult(
        position=fused_position,
        velocity=fused_velocity,
        covariance=fused_covariance,
        normalized_weights=normalized,
        effective_sample_size=effective_sample_size,
        weight_entropy=weight_entropy,
        mean_age_steps=mean_age,
        max_age_steps=float(np.max(ages)),
        mean_covariance_trace_m2=mean_trace,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
    )


__all__ = ["PublicBeliefFusionResult", "fuse_public_beliefs"]
