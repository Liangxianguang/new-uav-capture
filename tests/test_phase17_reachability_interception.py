from __future__ import annotations

import numpy as np

from encirclement3d.reachability_interception import (
    formation_slot_reachability_cost,
    minimum_arrival_time,
    planned_rnic_diagnostics,
    reachability_normalized_interception_cost,
    rnic_summary,
)


def test_formation_slot_cost_selects_the_assignment_with_smallest_shortfall() -> None:
    directions = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    target = np.zeros((1, 1, 3), dtype=np.float64)
    positions = directions[[1, 0, 2, 3]][None, None, :, :]
    velocities = np.zeros_like(positions)

    cost, slack, assignments, arrival = formation_slot_reachability_cost(
        positions,
        velocities,
        target,
        slot_radius_m=1.0,
        dt_seconds=0.1,
        max_speed_mps=5.0,
        max_acceleration_mps2=6.0,
        time_margin_s=0.15,
        time_scale_s=0.5,
        slot_directions=directions,
    )

    assert cost.shape == (1, 1)
    assert slack.shape == (1, 1, 1)
    assert assignments.shape == (1, 1, 1, 4)
    assert arrival.shape == (1, 1, 1, 4, 4)
    np.testing.assert_array_equal(assignments[0, 0, 0], [1, 0, 2, 3])
    assert float(cost[0, 0]) > 0.0


def test_formation_slot_cost_is_zero_for_reachable_slots_and_validates_limit() -> None:
    directions = np.array(
        [[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]],
        dtype=np.float64,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    positions = directions[None, None, :, :]
    target = np.zeros((2, 1, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    cost, slack, assignments, _arrival = formation_slot_reachability_cost(
        positions,
        velocities,
        target,
        slot_radius_m=1.0,
        dt_seconds=1.0,
        max_speed_mps=5.0,
        max_acceleration_mps2=6.0,
        time_margin_s=0.15,
        time_scale_s=0.5,
    )

    np.testing.assert_allclose(cost, 0.0)
    assert np.all(slack >= 0.15)
    assert np.all(assignments[0, :, 0] == np.arange(4))

    too_many = np.zeros((1, 1, 7, 3), dtype=np.float64)
    with np.testing.assert_raises(ValueError):
        formation_slot_reachability_cost(
            too_many,
            np.zeros_like(too_many),
            np.zeros((1, 1, 3), dtype=np.float64),
            slot_radius_m=1.0,
            dt_seconds=0.1,
            max_speed_mps=5.0,
            max_acceleration_mps2=6.0,
        )


def test_planned_formation_rnic_reports_assignment_switch_rate() -> None:
    positions = np.array(
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0]],
        dtype=np.float64,
    )
    actions = np.zeros((2, 4, 3), dtype=np.float64)
    targets = np.zeros((1, 2, 3), dtype=np.float64)
    diagnostics = planned_rnic_diagnostics(
        positions,
        actions,
        targets,
        dt_seconds=0.1,
        max_speed_mps=5.0,
        max_acceleration_mps2=6.0,
        cost_mode="formation_slot",
        slot_radius_m=1.0,
    )

    assert diagnostics["cost_mode"] == "formation_slot"
    assert diagnostics["assignment_switch_rate"] == 0.0
    assert 0.0 <= diagnostics["margin_violation_ratio"] <= 1.0


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


def test_target_tube_radius_makes_rnic_cost_more_conservative() -> None:
    positions = np.zeros((1, 2, 1, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    targets = np.array([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=np.float64)
    baseline, _slack, _arrival = reachability_normalized_interception_cost(
        positions,
        velocities,
        targets,
        dt_seconds=0.1,
        max_speed_mps=5.0,
        max_acceleration_mps2=2.0,
        time_margin_s=0.0,
        time_scale_s=0.5,
    )
    tube, _slack, _arrival = reachability_normalized_interception_cost(
        positions,
        velocities,
        targets,
        dt_seconds=0.1,
        max_speed_mps=5.0,
        max_acceleration_mps2=2.0,
        time_margin_s=0.0,
        time_scale_s=0.5,
        target_tube_radius_m=(0.2, 0.2),
    )
    assert float(tube[0, 0]) >= float(baseline[0, 0])


def test_gated_rnic_ignores_mild_slack_deficit_but_keeps_diagnostics() -> None:
    positions = np.zeros((1, 1, 1, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    targets = np.array([[[0.5, 0.0, 0.0]]], dtype=np.float64)

    ungated, slack, _arrival = reachability_normalized_interception_cost(
        positions,
        velocities,
        targets,
        dt_seconds=0.1,
        max_speed_mps=5.0,
        max_acceleration_mps2=2.0,
        time_margin_s=0.15,
        time_scale_s=0.5,
    )
    gated, gated_slack, _arrival = reachability_normalized_interception_cost(
        positions,
        velocities,
        targets,
        dt_seconds=0.1,
        max_speed_mps=5.0,
        max_acceleration_mps2=2.0,
        time_margin_s=0.15,
        time_scale_s=0.5,
        activation_slack_s=-0.75,
    )

    np.testing.assert_allclose(slack, gated_slack)
    assert float(gated[0, 0]) == 0.0
    assert float(ungated[0, 0]) > float(gated[0, 0])
