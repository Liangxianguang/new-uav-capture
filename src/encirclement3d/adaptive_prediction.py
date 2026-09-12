"""Deterministic uncertainty-triggered prediction-budget scheduling.

UAKR is an inference policy, not a learned predictor.  Its inputs are all
available in the policy-safe observation, and every threshold/weight is
explicitly serializable for validation freezing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class AdaptivePredictionConfig:
    """Three-level candidate-budget and refresh policy."""

    low_threshold: float = 0.35
    high_threshold: float = 0.65
    low_k: int = 1
    medium_k: int = 4
    high_k: int = 8
    low_refresh_interval_steps: int = 4
    medium_refresh_interval_steps: int = 2
    high_refresh_interval_steps: int = 1
    max_cache_age_steps: int = 4
    hysteresis: float = 0.05
    covariance_scale: float = 0.50
    message_age_scale_steps: float = 8.0
    speed_ratio_scale: float = 0.80
    residual_scale_m: float = 0.50
    covariance_weight: float = 0.30
    age_weight: float = 0.20
    confidence_weight: float = 0.20
    speed_weight: float = 0.15
    cache_weight: float = 0.10
    residual_weight: float = 0.05
    tube_radius_scale_m: float = 2.0
    tube_weight: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.low_threshold) < float(self.high_threshold) <= 1.0:
            raise ValueError("adaptive thresholds must satisfy 0 <= low < high <= 1")
        for name in ("low_k", "medium_k", "high_k", "low_refresh_interval_steps", "medium_refresh_interval_steps", "high_refresh_interval_steps", "max_cache_age_steps"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= float(self.hysteresis) < 0.5:
            raise ValueError("hysteresis must lie in [0, 0.5)")
        scales = (
            self.covariance_scale,
            self.message_age_scale_steps,
            self.speed_ratio_scale,
            self.residual_scale_m,
            self.tube_radius_scale_m,
        )
        if any(not np.isfinite(float(value)) or float(value) <= 0.0 for value in scales):
            raise ValueError("adaptive feature scales must be finite and positive")
        weights = (
            self.covariance_weight,
            self.age_weight,
            self.confidence_weight,
            self.speed_weight,
            self.cache_weight,
            self.residual_weight,
            self.tube_weight,
        )
        if any(not np.isfinite(float(value)) or float(value) < 0.0 for value in weights):
            raise ValueError("adaptive feature weights must be finite and non-negative")
        if float(sum(weights)) <= 0.0:
            raise ValueError("adaptive feature weights must have a positive sum")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AdaptivePredictionConfig":
        values = dict(mapping)
        return cls(**values)


@dataclass(frozen=True)
class AdaptivePredictionDecision:
    """Auditable UAKR decision for one control step."""

    uncertainty_score: float
    bucket: str
    num_samples: int
    refresh_interval_steps: int
    refresh: bool
    forced_refresh: bool
    forced_refresh_reason: str | None
    components: dict[str, float]

    def __post_init__(self) -> None:
        if self.bucket not in {"low", "medium", "high"}:
            raise ValueError("bucket must be low, medium, or high")
        if not 0.0 <= float(self.uncertainty_score) <= 1.0:
            raise ValueError("uncertainty_score must lie in [0, 1]")
        if int(self.num_samples) <= 0 or int(self.refresh_interval_steps) <= 0:
            raise ValueError("num_samples and refresh_interval_steps must be positive")
        if not all(np.isfinite(float(value)) for value in self.components.values()):
            raise ValueError("adaptive components must be finite")

    @property
    def bucket_index(self) -> int:
        return {"low": 0, "medium": 1, "high": 2}[self.bucket]

    def as_dict(self) -> dict[str, Any]:
        return {
            "uncertainty_score": float(self.uncertainty_score),
            "bucket": self.bucket,
            "bucket_index": int(self.bucket_index),
            "num_samples": int(self.num_samples),
            "refresh_interval_steps": int(self.refresh_interval_steps),
            "refresh": bool(self.refresh),
            "forced_refresh": bool(self.forced_refresh),
            "forced_refresh_reason": self.forced_refresh_reason,
            "components": {key: float(value) for key, value in self.components.items()},
        }


class AdaptivePredictionPolicy:
    """Stateful hysteretic scheduler with no target-truth dependency."""

    def __init__(self, config: AdaptivePredictionConfig | None = None) -> None:
        self.config = config or AdaptivePredictionConfig()
        self._previous_bucket: str | None = None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "AdaptivePredictionPolicy":
        return cls(AdaptivePredictionConfig.from_mapping(mapping))

    def reset(self) -> None:
        self._previous_bucket = None

    @staticmethod
    def _mean_feature(value: Any, default: float) -> float:
        if value is None:
            return float(default)
        array = np.asarray(value, dtype=np.float64)
        finite = array[np.isfinite(array)]
        return float(np.mean(finite)) if finite.size else float(default)

    def _components(
        self,
        observation: Mapping[str, Any],
        *,
        cached_age_steps: int,
        previous_residual_m: float | None,
        reachable_tube_radius_m: float | None,
    ) -> dict[str, float]:
        covariance = np.asarray(
            observation.get("target_observation_covariance", np.zeros((1, 3, 3))),
            dtype=np.float64,
        )
        if covariance.ndim == 3 and covariance.shape[-2:] == (3, 3):
            covariance_trace = float(np.mean(np.trace(covariance, axis1=-2, axis2=-1)))
        else:
            covariance_trace = 0.0
        confidence = self._mean_feature(observation.get("target_observation_confidence"), 1.0)
        message_age = self._mean_feature(observation.get("message_age_steps"), 0.0)
        belief_velocity = np.asarray(
            observation.get("target_belief_velocities", np.zeros((1, 3))),
            dtype=np.float64,
        )
        if belief_velocity.ndim == 2 and belief_velocity.shape[-1] == 3 and belief_velocity.size:
            speed = float(np.mean(np.linalg.norm(belief_velocity, axis=-1)))
        else:
            speed = 0.0
        execution = observation.get("execution", {})
        max_speed = float(execution.get("max_speed_mps", 5.0)) if isinstance(execution, Mapping) else 5.0
        max_speed = max(max_speed, 1.0e-6)
        residual = 0.0 if previous_residual_m is None else max(float(previous_residual_m), 0.0)
        components = {
            "covariance": float(np.clip(covariance_trace / self.config.covariance_scale, 0.0, 1.0)),
            "message_age": float(np.clip(message_age / self.config.message_age_scale_steps, 0.0, 1.0)),
            "confidence_deficit": float(np.clip(1.0 - confidence, 0.0, 1.0)),
            "speed_ratio": float(np.clip((speed / max_speed) / self.config.speed_ratio_scale, 0.0, 1.0)),
            "cache_age": float(np.clip(float(cached_age_steps) / self.config.max_cache_age_steps, 0.0, 1.0)),
            "one_step_residual": float(np.clip(residual / self.config.residual_scale_m, 0.0, 1.0)),
        }
        if self.config.tube_weight > 0.0:
            tube_radius = 0.0 if reachable_tube_radius_m is None else max(float(reachable_tube_radius_m), 0.0)
            components["tube_width"] = float(
                np.clip(tube_radius / self.config.tube_radius_scale_m, 0.0, 1.0)
            )
        return components

    def _bucket_for_score(self, score: float) -> str:
        if score < self.config.low_threshold:
            bucket = "low"
        elif score < self.config.high_threshold:
            bucket = "medium"
        else:
            bucket = "high"
        previous = self._previous_bucket
        if previous is not None:
            rank = {"low": 0, "medium": 1, "high": 2}
            if rank[bucket] < rank[previous]:
                lower_threshold = self.config.low_threshold if previous == "medium" else self.config.high_threshold
                if score >= lower_threshold - self.config.hysteresis:
                    bucket = previous
        self._previous_bucket = bucket
        return bucket

    def decide(
        self,
        observation: Mapping[str, Any],
        *,
        cached_age_steps: int,
        has_cache: bool,
        previous_residual_m: float | None = None,
        reachable_tube_radius_m: float | None = None,
    ) -> AdaptivePredictionDecision:
        if int(cached_age_steps) < 0:
            raise ValueError("cached_age_steps must be non-negative")
        components = self._components(
            observation,
            cached_age_steps=cached_age_steps,
            previous_residual_m=previous_residual_m,
            reachable_tube_radius_m=reachable_tube_radius_m,
        )
        configured_weights = {
            "covariance": self.config.covariance_weight,
            "message_age": self.config.age_weight,
            "confidence_deficit": self.config.confidence_weight,
            "speed_ratio": self.config.speed_weight,
            "cache_age": self.config.cache_weight,
            "one_step_residual": self.config.residual_weight,
            "tube_width": self.config.tube_weight,
        }
        weights = np.asarray([configured_weights[name] for name in components], dtype=np.float64)
        values = np.asarray(list(components.values()), dtype=np.float64)
        score = float(np.clip(np.dot(weights, values) / weights.sum(), 0.0, 1.0))
        bucket = self._bucket_for_score(score)
        settings = {
            "low": (self.config.low_k, self.config.low_refresh_interval_steps),
            "medium": (self.config.medium_k, self.config.medium_refresh_interval_steps),
            "high": (self.config.high_k, self.config.high_refresh_interval_steps),
        }
        num_samples, refresh_interval = settings[bucket]
        forced_reason: str | None = None
        if not has_cache:
            forced_reason = "no_cache"
        elif int(cached_age_steps) >= self.config.max_cache_age_steps:
            forced_reason = "cache_age_limit"
        elif previous_residual_m is not None and float(previous_residual_m) > self.config.residual_scale_m:
            forced_reason = "residual_limit"
        refresh = forced_reason is not None or int(cached_age_steps) % int(refresh_interval) == 0
        return AdaptivePredictionDecision(
            uncertainty_score=score,
            bucket=bucket,
            num_samples=int(num_samples),
            refresh_interval_steps=int(refresh_interval),
            refresh=bool(refresh),
            forced_refresh=forced_reason is not None,
            forced_refresh_reason=forced_reason,
            components=components,
        )


__all__ = [
    "AdaptivePredictionConfig",
    "AdaptivePredictionDecision",
    "AdaptivePredictionPolicy",
]
