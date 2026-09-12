from __future__ import annotations

import numpy as np

from encirclement3d.execution_dynamics import ExecutionParameters
from encirclement3d.minimax_mpc import ScenarioTrajectorySet
from encirclement3d.queue_aware_rollout import (
    endpoint_error_diagnostics,
    prepare_queue_aware_observation,
    prefix_geometry_diagnostics,
    rollout_queue_prefix,
    shift_scenario_trajectory_set,
)


def _parameters() -> ExecutionParameters:
    return ExecutionParameters(
        enabled=True,
        dt_seconds=0.1,
        action_delay_steps=2,
        command_noise_std_mps=0.0,
        command_noise_bound_mps=0.0,
        clip_command_noise=True,
        velocity_time_constant_seconds=0.0,
        drag_coefficient=0.0,
        max_speed_mps=5.0,
        max_acceleration_mps2=100.0,
        mass_scale=1.0,
    )


def test_queue_prefix_uses_shared_execution_dynamics() -> None:
    positions = np.zeros((1, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    queue = [
        np.array([[1.0, 0.0, 0.0]], dtype=np.float64),
        np.array([[0.0, 1.0, 0.0]], dtype=np.float64),
    ]

    state = rollout_queue_prefix(positions, velocities, queue, _parameters())

    assert state.queue_length == 2
    assert state.first_controllable_step == 2
    np.testing.assert_allclose(state.prefix_positions[0], [[0.1, 0.0, 0.0]])
    np.testing.assert_allclose(state.prefix_positions[1], [[0.1, 0.1, 0.0]])
    np.testing.assert_allclose(state.delayed_velocities, [[0.0, 1.0, 0.0]])


def test_prepare_observation_removes_consumed_queue_and_records_audit_state() -> None:
    observation = {
        "defender_positions": np.zeros((1, 3), dtype=np.float64),
        "defender_velocities": np.zeros((1, 3), dtype=np.float64),
        "execution_action_queue": [[[1.0, 0.0, 0.0]]],
        "execution": {
            "enabled": True,
            "action_delay_steps": 1,
            "max_speed_mps": 5.0,
            "max_acceleration_mps2": 100.0,
        },
    }

    aligned, state = prepare_queue_aware_observation(observation, dt_seconds=0.1)

    assert aligned["execution_action_queue"] == []
    assert aligned["execution"]["action_queue"] == []
    assert aligned["qdr"]["queue_length"] == 1
    np.testing.assert_allclose(aligned["defender_positions"], [[0.1, 0.0, 0.0]])
    np.testing.assert_allclose(aligned["defender_velocities"], state.delayed_velocities)


def test_shift_scenario_paths_preserves_metadata_and_bounds_extension_speed() -> None:
    scenarios = ScenarioTrajectorySet(
        trajectories=np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0]],
            ],
            dtype=np.float64,
        ),
        weights=np.array([0.25, 0.75], dtype=np.float64),
        score_kind="calibrated_region",
        source_model_hash="phase17-test",
        timestamp_step=11,
    )

    shifted = shift_scenario_trajectory_set(
        scenarios,
        offset_steps=1,
        horizon_steps=3,
        dt_seconds=0.1,
        max_speed_mps=5.0,
    )

    assert shifted.trajectories.shape == (2, 3, 3)
    np.testing.assert_allclose(shifted.trajectories[0], [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    np.testing.assert_allclose(shifted.weights, scenarios.weights)
    assert shifted.score_kind == scenarios.score_kind
    assert shifted.source_model_hash == scenarios.source_model_hash
    assert shifted.timestamp_step == scenarios.timestamp_step


def test_prefix_geometry_diagnostics_uses_only_public_geometry() -> None:
    state = rollout_queue_prefix(
        np.zeros((2, 3), dtype=np.float64),
        np.zeros((2, 3), dtype=np.float64),
        [
            np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64),
        ],
        _parameters(),
    )
    observation = {
        "world_lower_bounds": np.array([-5.0, -5.0, -5.0]),
        "world_upper_bounds": np.array([5.0, 5.0, 5.0]),
        "obstacles": [],
    }

    diagnostics = prefix_geometry_diagnostics(
        state,
        observation,
        drone_radius_m=0.25,
        safety_margin_m=0.35,
    )

    assert diagnostics["minimum_boundary_margin_m"] > 0.0
    np.testing.assert_allclose(diagnostics["minimum_inter_agent_distance_m"], 0.2)


def test_endpoint_error_diagnostics_is_post_hoc_and_shape_checked() -> None:
    diagnostics = endpoint_error_diagnostics(
        np.zeros((2, 3), dtype=np.float64),
        np.zeros((2, 3), dtype=np.float64),
        np.array([[0.1, 0.0, 0.0], [0.0, -0.2, 0.0]], dtype=np.float64),
        np.array([[0.0, 0.3, 0.0], [0.0, 0.0, -0.4]], dtype=np.float64),
    )

    np.testing.assert_allclose(diagnostics["position_error_mean_m"], 0.15)
    np.testing.assert_allclose(diagnostics["position_error_max_m"], 0.2)
    np.testing.assert_allclose(diagnostics["velocity_error_mean_mps"], 0.35)
    np.testing.assert_allclose(diagnostics["velocity_error_max_mps"], 0.4)
