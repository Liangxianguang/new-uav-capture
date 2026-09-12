from __future__ import annotations

import numpy as np

from encirclement3d.reachability_interception import (
    minimum_arrival_time,
    planned_rnic_diagnostics,
    reachability_normalized_interception_cost,
    rnic_summary,
)


def test_minimum_arrival_time_respects_acceleration_and_speed_limits() -> None:
    accelerating = minimum_arrival_time(
        np.array([1.0]),
        np.array([0.0]),
        max_speed_mps=5.0,
        max_acceleration_mps2=2.0,
    )
    cruising = minimum_arrival_time(
        np.array([20.0]),
        np.array([5.0]),
        max_speed_mps=5.0,
        max_acceleration_mps2=2.0,
    )

    np.testing.assert_allclose(accelerating, [1.0])
    np.testing.assert_allclose(cruising, [4.0])


def test_rnic_returns_sequence_scenario_cost_and_auditable_slack() -> None:
    positions = np.zeros((1, 2, 1, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    targets = np.array([[[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]]], dtype=np.float64)

    cost, slack, arrival = reachability_normalized_interception_cost(
        positions,
        velocities,
        targets,
        dt_seconds=0.1,
        max_speed_mps=5.0,
        max_acceleration_mps2=2.0,
        time_margin_s=0.0,
        time_scale_s=0.5,
    )

    assert cost.shape == (1, 1)
    assert slack.shape == (1, 1, 2)
    assert arrival.shape == (1, 1, 2, 1)
    assert float(cost[0, 0]) > 0.0
    assert float(slack[0, 0, 0]) < 0.0
    summary = rnic_summary(slack)
    assert summary["minimum_best_slack_s"] < 0.0


def test_planned_rnic_diagnostics_reconstructs_selected_team_rollout() -> None:
    positions = np.zeros((2, 3), dtype=np.float64)
    actions = np.zeros((3, 2, 3), dtype=np.float64)
    actions[:, :, 0] = 1.0
    targets = np.zeros((2, 3, 3), dtype=np.float64)
    targets[:, :, 0] = 2.0

    diagnostics = planned_rnic_diagnostics(
        positions,
        actions,
        targets,
        dt_seconds=0.1,
        max_speed_mps=5.0,
        max_acceleration_mps2=6.0,
    )

    assert diagnostics["earliest_feasible_intercept_step"] >= 1.0
    assert 0.0 <= diagnostics["unreachable_slot_ratio"] <= 1.0
    assert 0.0 <= diagnostics["margin_violation_ratio"] <= 1.0
    assert diagnostics["maximum_arrival_time_s"] >= diagnostics["mean_arrival_time_s"]
