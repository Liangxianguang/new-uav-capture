from __future__ import annotations

import numpy as np
import pytest

from encirclement3d.belief_fusion import fuse_public_beliefs
from encirclement3d.distributed_dn_mpc import DistributedDNMPCConfig, DistributedMinimaxDNMPC
from encirclement3d.minimax_mpc import (
    MinimaxMPCConfig,
    ScenarioMinimaxMPC,
    ScenarioTrajectorySet,
    belief_fusion_diagnostics,
    make_belief_candidate_set,
)


def _observation() -> dict[str, object]:
    return {
        "defender_positions": np.zeros((2, 3), dtype=np.float64),
        "defender_velocities": np.zeros((2, 3), dtype=np.float64),
        "target_belief_positions": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        "target_belief_velocities": np.array([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
        "target_observation_confidence": np.array([1.0, 1.0]),
        "message_age_steps": np.array([0.0, 4.0]),
        "target_observation_covariance": np.stack([0.1 * np.eye(3), 2.0 * np.eye(3)]),
        "world_lower_bounds": np.array([-10.0, -10.0, 0.5]),
        "world_upper_bounds": np.array([10.0, 10.0, 10.0]),
        "obstacles": [],
    }


def _scenarios() -> ScenarioTrajectorySet:
    trajectories = np.array(
        [
            [[1.0, 0.0, 1.0], [1.5, 0.0, 1.0], [2.0, 0.0, 1.0], [2.5, 0.0, 1.0]],
            [[1.0, 0.2, 1.0], [1.5, 0.2, 1.0], [2.0, 0.2, 1.0], [2.5, 0.2, 1.0]],
        ],
        dtype=np.float64,
    )
    return ScenarioTrajectorySet(trajectories, np.ones(2))


def test_fusion_downweights_old_high_covariance_source():
    result = fuse_public_beliefs(_observation())

    assert result.position[0] < 1.0
    assert result.velocity[0] < 1.5
    assert result.normalized_weights[0] > result.normalized_weights[1]
    assert result.mean_age_steps < 1.0
    assert result.mean_covariance_trace_m2 > 0.3
    assert not result.fallback_used


def test_fusion_anchor_fallback_is_deterministic():
    observation = _observation()
    observation["target_observation_confidence"] = np.array([1.0, 0.01])
    first = fuse_public_beliefs(
        observation,
        min_effective_samples=1.5,
        anchor_index=1,
    )
    second = fuse_public_beliefs(
        observation,
        min_effective_samples=1.5,
        anchor_index=1,
    )

    assert first.fallback_used
    assert first.fallback_reason == "insufficient_effective_samples"
    np.testing.assert_array_equal(first.normalized_weights, [0.0, 1.0])
    np.testing.assert_allclose(first.position, [10.0, 0.0, 0.0])
    np.testing.assert_allclose(first.position, second.position)


def test_missing_covariance_is_explicitly_reported_without_truth_access():
    observation = _observation()
    observation.pop("target_observation_covariance")
    result = fuse_public_beliefs(observation)

    assert result.fallback_reason == "default_covariance"
    assert np.isfinite(result.covariance).all()


def test_invalid_fusion_parameters_are_rejected():
    with pytest.raises(ValueError, match="age_decay"):
        fuse_public_beliefs(_observation(), age_decay=0.0)
    with pytest.raises(ValueError, match="dropout_inflation_m2"):
        fuse_public_beliefs(_observation(), dropout_inflation_m2=-1.0)


def test_legacy_diagnostics_are_disabled_and_new_mode_is_audited():
    observation = _observation()
    legacy = MinimaxMPCConfig(horizon_steps=4, control_horizon_steps=2)
    enabled = MinimaxMPCConfig(
        horizon_steps=4,
        control_horizon_steps=2,
        belief_fusion_mode="freshness_covariance",
        belief_fusion_min_effective_samples=1.0,
    )
    assert belief_fusion_diagnostics(observation, legacy)["belief_fusion_enabled"] == 0.0
    diagnostics = belief_fusion_diagnostics(observation, enabled)
    assert diagnostics["belief_fusion_enabled"] == 1.0
    assert diagnostics["belief_fusion_effective_sample_size"] > 1.0


def test_centralized_planner_and_belief_candidates_accept_fusion_mode():
    observation = _observation()
    config = MinimaxMPCConfig(
        horizon_steps=4,
        control_horizon_steps=2,
        max_role_variants=2,
        perimeter_scales=(1.0,),
        belief_fusion_mode="freshness_covariance",
    )
    candidates = make_belief_candidate_set(
        observation,
        horizon_steps=4,
        dt_seconds=0.1,
        max_speed_mps=5.0,
        candidate_count=2,
        belief_fusion_config=config,
    )
    plan = ScenarioMinimaxMPC(config).plan(observation, candidates)

    assert plan.diagnostics.belief_fusion_enabled == 1.0
    assert np.isfinite(plan.diagnostics.belief_fusion_effective_sample_size)


def test_distributed_planner_reports_fusion_diagnostics():
    observation = _observation()
    config = MinimaxMPCConfig(
        horizon_steps=4,
        control_horizon_steps=2,
        max_role_variants=2,
        perimeter_scales=(1.0,),
        belief_fusion_mode="freshness_covariance",
    )
    planner = DistributedMinimaxDNMPC(
        config,
        DistributedDNMPCConfig(communication_mode="none", max_iterations=1, local_timeout_ms=1000.0),
    )
    plan = planner.plan(observation, _scenarios(), step_index=0)

    assert plan.diagnostics.belief_fusion_enabled == 1.0
    assert np.isfinite(plan.diagnostics.belief_fusion_mean_age_steps)
