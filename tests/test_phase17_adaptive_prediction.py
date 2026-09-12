from __future__ import annotations

import numpy as np

from encirclement3d.adaptive_prediction import (
    AdaptivePredictionConfig,
    AdaptivePredictionPolicy,
)


def _observation(covariance_trace: float, confidence: float, age: float, speed: float) -> dict:
    return {
        "target_observation_covariance": np.eye(3, dtype=np.float64)[None] * (covariance_trace / 3.0),
        "target_observation_confidence": np.array([confidence], dtype=np.float64),
        "message_age_steps": np.array([age], dtype=np.float64),
        "target_belief_velocities": np.array([[speed, 0.0, 0.0]], dtype=np.float64),
        "execution": {"max_speed_mps": 5.0},
    }


def test_adaptive_policy_is_explainable_and_uses_three_budgets() -> None:
    policy = AdaptivePredictionPolicy()
    low = policy.decide(
        _observation(0.0, 1.0, 0.0, 0.0),
        cached_age_steps=0,
        has_cache=True,
    )
    high = policy.decide(
        _observation(2.0, 0.0, 16.0, 5.0),
        cached_age_steps=4,
        has_cache=True,
    )

    assert low.bucket == "low"
    assert low.num_samples == 1
    assert low.refresh_interval_steps == 4
    assert high.bucket == "high"
    assert high.num_samples == 8
    assert high.refresh_interval_steps == 1
    assert high.forced_refresh is True
    assert high.forced_refresh_reason == "cache_age_limit"
    assert set(low.components) == {
        "covariance",
        "message_age",
        "confidence_deficit",
        "speed_ratio",
        "cache_age",
        "one_step_residual",
    }


def test_adaptive_policy_refreshes_on_interval_and_residual_without_truth_fields() -> None:
    config = AdaptivePredictionConfig(max_cache_age_steps=6)
    policy = AdaptivePredictionPolicy(config)
    observation = _observation(0.0, 1.0, 0.0, 0.0)
    first = policy.decide(observation, cached_age_steps=0, has_cache=False)
    second = policy.decide(observation, cached_age_steps=1, has_cache=True)
    third = policy.decide(
        observation,
        cached_age_steps=1,
        has_cache=True,
        previous_residual_m=0.6,
    )

    assert first.refresh is True
    assert first.forced_refresh_reason == "no_cache"
    assert second.refresh is False
    assert third.refresh is True
    assert third.forced_refresh_reason == "residual_limit"
    assert "target_position" not in third.as_dict()
    assert "target_velocity" not in third.as_dict()
