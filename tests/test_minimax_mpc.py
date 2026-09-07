from __future__ import annotations

import numpy as np
import pytest

from encirclement3d.minimax_mpc import (
    MinimaxMPCConfig,
    ScenarioMinimaxMPC,
    ScenarioTrajectorySet,
    aggregate_scenario_costs,
    make_belief_candidate_set,
)


def _observation() -> dict[str, object]:
    return {
        "defender_positions": np.array(
            [[-4.0, -2.0, 3.0], [-4.0, -0.7, 3.2], [-4.0, 0.7, 3.5], [-4.0, 2.0, 3.8]],
            dtype=np.float64,
        ),
        "defender_velocities": np.zeros((4, 3), dtype=np.float64),
        "target_belief_positions": np.tile(np.array([4.0, 0.0, 4.0]), (4, 1)),
        "target_belief_velocities": np.tile(np.array([0.5, 0.0, 0.0]), (4, 1)),
        "target_observation_confidence": np.ones(4),
        "message_age_steps": np.zeros(4),
        "world_lower_bounds": np.array([-10.0, -10.0, 0.5]),
        "world_upper_bounds": np.array([10.0, 10.0, 10.0]),
        "obstacles": [],
    }


def test_risk_aggregators_distinguish_expected_worst_and_cvar() -> None:
    costs = [1.0, 2.0, 10.0]
    weights = [0.8, 0.1, 0.1]
    assert aggregate_scenario_costs(costs, weights, "expected", 0.75) == pytest.approx(2.0)
    assert aggregate_scenario_costs(costs, weights, "worst_case", 0.75) == pytest.approx(10.0)
    assert aggregate_scenario_costs(costs, weights, "cvar", 0.75) == pytest.approx(5.0)


def test_candidate_contract_rejects_raw_candidates_in_planner() -> None:
    observation = _observation()
    candidates = ScenarioTrajectorySet(
        trajectories=np.repeat(np.array([[[4.0, 0.0, 4.0]]]), 2, axis=0),
        weights=np.ones(2),
        dynamics_status="raw",
    )
    planner = ScenarioMinimaxMPC(MinimaxMPCConfig(horizon_steps=1, control_horizon_steps=1))
    plan = planner.plan(observation, candidates, fallback_actions=np.zeros((4, 3)))
    assert plan.diagnostics.status == "fallback"
    assert "projected" in str(plan.diagnostics.fallback_reason)


def test_projected_scenario_planner_returns_common_bounded_action() -> None:
    observation = _observation()
    candidates = make_belief_candidate_set(
        observation,
        horizon_steps=4,
        dt_seconds=0.1,
        max_speed_mps=5.0,
        candidate_count=4,
    )
    planner = ScenarioMinimaxMPC(
        MinimaxMPCConfig(
            horizon_steps=4,
            control_horizon_steps=2,
            max_role_variants=2,
            perimeter_scales=(1.0,),
        )
    )
    plan = planner.plan(observation, candidates, fallback_actions=np.zeros((4, 3)))
    assert plan.diagnostics.status == "success"
    assert plan.action_sequence.shape == (4, 4, 3)
    assert np.isfinite(plan.action_sequence).all()
    assert float(np.linalg.norm(plan.actions, axis=1).max()) <= 5.0 + 1e-8
    assert len(plan.diagnostics.scenario_costs) == candidates.candidate_count


def test_belief_candidate_generation_is_deterministic_and_policy_safe() -> None:
    observation = _observation()
    first = make_belief_candidate_set(
        observation,
        horizon_steps=5,
        dt_seconds=0.1,
        max_speed_mps=5.0,
        candidate_count=8,
    )
    second = make_belief_candidate_set(
        observation,
        horizon_steps=5,
        dt_seconds=0.1,
        max_speed_mps=5.0,
        candidate_count=8,
    )
    np.testing.assert_allclose(first.trajectories, second.trajectories)
    assert first.dynamics_status == "projected"
