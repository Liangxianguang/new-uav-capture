from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv
from encirclement3d.minimax_mpc import (
    MinimaxMPCConfig,
    ScenarioMinimaxMPC,
    ScenarioTrajectorySet,
    aggregate_scenario_costs,
    evaluate_candidate_capture_distances,
    make_belief_candidate_set,
)
from scripts.evaluate_minimax_mpc import PredictionRuntime


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def test_candidate_distance_rollout_is_truth_free_and_reports_terminal_and_minimum() -> None:
    observation = _observation()
    scenarios = ScenarioTrajectorySet(
        trajectories=np.array(
            [
                [[-3.5, -2.0, 3.0], [-3.5, -2.0, 3.0]],
                [[-3.0, -2.0, 3.0], [-3.0, -2.0, 3.0]],
            ],
            dtype=np.float64,
        ),
        weights=np.ones(2),
        dynamics_status="projected",
    )
    actions = np.zeros((2, 4, 3), dtype=np.float64)
    distances = evaluate_candidate_capture_distances(
        observation,
        actions,
        scenarios,
        dt_seconds=0.1,
        max_speed_mps=5.0,
    )
    assert distances["terminal_distances_m"].shape == (2,)
    assert distances["minimum_distances_m"].shape == (2,)
    np.testing.assert_allclose(distances["terminal_distances_m"], [0.5, 1.0])
    np.testing.assert_allclose(distances["minimum_distances_m"], [0.5, 1.0])


def test_prediction_runtime_reuses_cached_candidates_and_records_age() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml").read_text(encoding="utf-8")
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=3, target_speed_scale=0.45)
    observation = env.reset(seed=648301)
    runtime = PredictionRuntime(
        env=env,
        source="belief",
        device=torch.device("cpu"),
        num_samples=4,
        refresh_interval_steps=2,
    )
    runtime.reset()

    first, first_latency, first_refreshed, first_age = runtime.predict(observation, planner_horizon=4)
    second, second_latency, second_refreshed, second_age = runtime.predict(observation, planner_horizon=4)
    third, _third_latency, third_refreshed, third_age = runtime.predict(observation, planner_horizon=4)

    assert first_refreshed is True
    assert first_age == 0
    assert second_refreshed is False
    assert second_age == 1
    assert second is first
    assert second_latency == pytest.approx(0.0)
    assert first_latency >= 0.0
    assert third_refreshed is True
    assert third_age == 0
    assert third is not first
