from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_minimax_mpc import _shift_warm_start_sequence  # noqa: E402

from encirclement3d.distributed_dn_mpc import (
    DistributedDNMPCConfig,
    DistributedMinimaxDNMPC,
)
from encirclement3d.minimax_mpc import MinimaxMPCConfig, ScenarioTrajectorySet


def _observation() -> dict[str, object]:
    return {
        "defender_positions": np.array(
            [[-4.0, -2.0, 3.0], [-4.0, -0.7, 3.2], [-4.0, 0.7, 3.5], [-4.0, 2.0, 3.8]],
            dtype=np.float64,
        ),
        "defender_velocities": np.zeros((4, 3), dtype=np.float64),
        "target_belief_positions": np.tile(np.array([4.0, 0.0, 4.0]), (4, 1)),
        "target_belief_velocities": np.tile(np.array([0.5, 0.0, 0.0]), (4, 1)),
        "world_lower_bounds": np.array([-10.0, -10.0, 0.5]),
        "world_upper_bounds": np.array([10.0, 10.0, 10.0]),
        "obstacles": [],
    }


def _scenarios(status: str = "projected") -> ScenarioTrajectorySet:
    path = np.array(
        [
            [[-3.5, -2.0, 3.0], [-3.0, -2.0, 3.0], [-2.5, -2.0, 3.0], [-2.0, -2.0, 3.0]],
            [[-3.5, 2.0, 3.0], [-3.0, 2.0, 3.0], [-2.5, 2.0, 3.0], [-2.0, 2.0, 3.0]],
        ],
        dtype=np.float64,
    )
    return ScenarioTrajectorySet(path, np.ones(2), dynamics_status=status)


def _planner(mode: str, **overrides: object) -> DistributedMinimaxDNMPC:
    config = MinimaxMPCConfig(
        horizon_steps=4,
        control_horizon_steps=2,
        max_role_variants=2,
        perimeter_scales=(1.0,),
    )
    distributed = DistributedDNMPCConfig(
        communication_mode=mode,
        max_iterations=1,
        local_timeout_ms=1000.0,
        **overrides,
    )
    return DistributedMinimaxDNMPC(config, distributed)


def test_ideal_communication_is_bounded_and_auditable() -> None:
    planner = _planner("ideal")
    plan = planner.plan(_observation(), _scenarios(), step_index=0, fallback_actions=np.zeros((4, 3)))
    diagnostics = plan.diagnostics
    assert plan.action_sequence.shape == (4, 4, 3)
    assert np.isfinite(plan.action_sequence).all()
    assert diagnostics.messages_attempted == 24
    assert diagnostics.messages_sent == diagnostics.messages_received == 24
    assert diagnostics.messages_dropped == 0
    assert diagnostics.max_message_age_steps == 0
    assert diagnostics.iterations == 1
    assert diagnostics.converged is False
    assert diagnostics.status == "not_converged"


def test_queue_aware_peer_rollout_keeps_single_agent_shape() -> None:
    observation = _observation()
    observation["execution"] = {
        "enabled": True,
        "action_delay_steps": 0,
        "max_speed_mps": 5.0,
        "max_acceleration_mps2": 6.0,
        "command_noise_std_mps": 0.0,
        "velocity_time_constant_seconds": 0.0,
        "drag_coefficient": 0.0,
    }
    observation["qdr"] = {"execution_aware_action_rollout": True}
    planner = _planner("ideal")

    plan = planner.plan(observation, _scenarios(), step_index=0)

    assert plan.diagnostics.status in {"not_converged", "success", "partial_fallback"}
    assert plan.diagnostics.status != "fallback"
    assert np.isfinite(plan.action_sequence).all()
    assert plan.diagnostics.qdr_suffix_gate_active is True
    assert plan.diagnostics.qdr_suffix_gate_exhausted is False


def test_qdr_execution_tube_is_explicit_and_auditable() -> None:
    observation = _observation()
    observation["execution"] = {
        "enabled": True,
        "action_delay_steps": 4,
        "max_speed_mps": 5.0,
        "max_acceleration_mps2": 6.0,
        "command_noise_std_mps": 0.08,
        "command_noise_bound_mps": 0.24,
        "velocity_time_constant_seconds": 0.2,
        "drag_coefficient": 0.0,
    }
    observation["qdr"] = {"execution_aware_action_rollout": True}

    nominal = _planner("ideal")
    nominal_plan = nominal.plan(observation, _scenarios(), step_index=0)
    assert nominal_plan.diagnostics.qdr_execution_tube_enabled is False
    assert nominal_plan.diagnostics.qdr_mean_execution_tube_radius_m == 0.0
    assert nominal_plan.diagnostics.qdr_max_execution_tube_radius_m == 0.0

    calibrated = _planner(
        "ideal",
        qdr_execution_tube_enabled=True,
        qdr_execution_tube_multiplier=2.0,
    )
    calibrated_plan = calibrated.plan(observation, _scenarios(), step_index=0)
    assert calibrated_plan.diagnostics.qdr_execution_tube_enabled is True
    assert calibrated_plan.diagnostics.qdr_execution_tube_multiplier == 2.0
    assert calibrated_plan.diagnostics.qdr_mean_execution_tube_radius_m > 0.0
    assert calibrated_plan.diagnostics.qdr_max_execution_tube_radius_m >= (
        calibrated_plan.diagnostics.qdr_mean_execution_tube_radius_m
    )

    frozen = _planner(
        "ideal",
        qdr_execution_tube_enabled=True,
        qdr_execution_tube_multiplier=2.0,
        qdr_execution_tube_radius_m_by_step=(0.1, 0.2, 0.3, 0.4),
    )
    frozen_plan = frozen.plan(observation, _scenarios(), step_index=0)
    assert frozen_plan.diagnostics.qdr_mean_execution_tube_radius_m == pytest.approx(0.25)
    assert frozen_plan.diagnostics.qdr_max_execution_tube_radius_m == pytest.approx(0.4)

    windowed = _planner(
        "ideal",
        qdr_execution_tube_enabled=True,
        qdr_execution_tube_multiplier=2.0,
        qdr_execution_tube_radius_m_by_step=(0.1, 0.2, 0.3, 0.4),
        qdr_execution_tube_active_steps=2,
    )
    windowed_plan = windowed.plan(observation, _scenarios(), step_index=0)
    assert windowed_plan.diagnostics.qdr_execution_tube_active_steps == 2
    assert windowed_plan.diagnostics.qdr_mean_execution_tube_radius_m == pytest.approx(0.075)
    assert windowed_plan.diagnostics.qdr_max_execution_tube_radius_m == pytest.approx(0.2)


def test_local_candidates_remove_duplicate_weighted_reference() -> None:
    observation = _observation()
    observation["execution"] = {
        "enabled": True,
        "action_delay_steps": 0,
        "max_speed_mps": 5.0,
        "max_acceleration_mps2": 6.0,
        "command_noise_std_mps": 0.0,
        "velocity_time_constant_seconds": 0.0,
        "drag_coefficient": 0.0,
    }
    observation["qdr"] = {"execution_aware_action_rollout": True}
    planner = _planner("ideal")
    path = np.tile(np.array([[1.0, 0.0, 3.0]], dtype=np.float64), (4, 1))
    scenarios = ScenarioTrajectorySet(path[None, ...], np.ones(1))

    candidates = planner._local_candidate_sequences(
        observation,
        scenarios,
        agent_id=0,
        own_position=observation["defender_positions"][0],
        known={},
    )

    # One unique target path plus the two QDR recovery candidates.
    assert len(candidates) == 3


def test_qdr_feasibility_first_preserves_selected_action_and_cost() -> None:
    observation = _observation()
    observation["execution"] = {
        "enabled": True,
        "action_delay_steps": 0,
        "max_speed_mps": 5.0,
        "max_acceleration_mps2": 6.0,
        "command_noise_std_mps": 0.0,
        "velocity_time_constant_seconds": 0.0,
        "drag_coefficient": 0.0,
    }
    observation["qdr"] = {"execution_aware_action_rollout": True}
    observation["obstacles"] = [
        {
            "center_xy": np.array([-2.8, -2.0]),
            "radius": 0.5,
            "height": 4.0,
            "shape": "cylinder",
        }
    ]

    full = _planner("ideal")
    fast = _planner("ideal", qdr_feasibility_first=True)
    full_plan = full.plan(observation, _scenarios(), step_index=0)
    fast_plan = fast.plan(observation, _scenarios(), step_index=0)

    np.testing.assert_array_equal(full_plan.action_sequence, fast_plan.action_sequence)
    np.testing.assert_allclose(
        full_plan.diagnostics.scenario_costs,
        fast_plan.diagnostics.scenario_costs,
        rtol=0.0,
        atol=0.0,
    )
    assert fast_plan.diagnostics.qdr_suffix_gate_rejected_candidates > 0


def test_queue_aware_local_obstacles_include_horizon_reachable_geometry() -> None:
    observation = _observation()
    observation["obstacles"] = [
        {
            "center_xy": np.array([5.0, -2.0]),
            "radius": 0.5,
            "height": 4.0,
            "shape": "cylinder",
        }
    ]
    observation["execution"] = {"enabled": True, "action_delay_steps": 0}
    observation["qdr"] = {"execution_aware_action_rollout": True}
    planner = _planner("ideal")

    visible = planner._local_obstacles(observation, observation["defender_positions"][0])

    assert len(visible) == 1


def test_qdr_exhaustion_soft_progress_policy_is_explicit_and_auditable() -> None:
    observation = _observation()
    observation["execution"] = {
        "enabled": True,
        "action_delay_steps": 0,
        "max_speed_mps": 5.0,
        "max_acceleration_mps2": 6.0,
        "command_noise_std_mps": 0.0,
        "velocity_time_constant_seconds": 0.0,
        "drag_coefficient": 0.0,
    }
    observation["qdr"] = {"execution_aware_action_rollout": True}
    # Enclose every finite candidate so the planner must enter the explicitly
    # diagnosed exhausted-candidate branch.
    observation["obstacles"] = [
        {
            "center_xy": np.array([-2.0, 0.0]),
            "radius": 20.0,
            "height": 20.0,
            "shape": "cylinder",
        }
    ]

    hard = _planner("ideal")
    hard_plan = hard.plan(observation, _scenarios(), step_index=0)
    soft = _planner("ideal", qdr_exhaustion_policy="normalized_soft_progress")
    soft_plan = soft.plan(observation, _scenarios(), step_index=0)

    assert hard_plan.diagnostics.qdr_suffix_gate_exhausted is True
    assert hard_plan.diagnostics.qdr_exhaustion_policy == "hard_min_violation"
    assert hard_plan.diagnostics.qdr_exhaustion_soft_fallback_count == 0
    assert soft_plan.diagnostics.qdr_suffix_gate_exhausted is True
    assert soft_plan.diagnostics.qdr_exhaustion_policy == "normalized_soft_progress"
    assert soft_plan.diagnostics.qdr_exhaustion_soft_fallback_count > 0
    assert np.isfinite(soft_plan.action_sequence).all()


def test_no_communication_does_not_create_peer_messages() -> None:
    planner = _planner("none")
    first = planner.plan(_observation(), _scenarios(), step_index=0)
    second = planner.plan(_observation(), _scenarios(), step_index=1)
    assert first.diagnostics.messages_attempted == 0
    assert first.diagnostics.messages_received == 0
    assert second.diagnostics.messages_sent == 0
    np.testing.assert_allclose(first.action_sequence, second.action_sequence)


def test_delayed_messages_report_age_after_delivery() -> None:
    planner = _planner("delayed", message_delay_steps=2, max_message_age_steps=4)
    first = planner.plan(_observation(), _scenarios(), step_index=0)
    second = planner.plan(_observation(), _scenarios(), step_index=1)
    third = planner.plan(_observation(), _scenarios(), step_index=2)
    assert first.diagnostics.messages_received == 0
    assert second.diagnostics.messages_received == 0
    assert third.diagnostics.messages_received > 0
    assert third.diagnostics.max_message_age_steps == 2


def test_asynchronous_communication_uses_deterministic_sender_phases() -> None:
    planner = _planner(
        "asynchronous",
        communication_interval_steps=2,
        message_delay_steps=1,
    )
    first = planner.plan(_observation(), _scenarios(), step_index=0)
    second = planner.plan(_observation(), _scenarios(), step_index=1)
    assert first.diagnostics.communication_mode == "asynchronous"
    assert first.diagnostics.messages_attempted < 24
    assert second.diagnostics.communication_mode == "asynchronous"
    assert second.diagnostics.messages_attempted == first.diagnostics.messages_attempted
    assert first.diagnostics.messages_received == 0
    assert second.diagnostics.messages_received > 0


def test_dropout_is_deterministic_and_raw_candidates_fallback() -> None:
    first = _planner("dropout", message_dropout_probability=0.5).plan(
        _observation(), _scenarios(), step_index=0
    )
    second = _planner("dropout", message_dropout_probability=0.5).plan(
        _observation(), _scenarios(), step_index=0
    )
    assert first.diagnostics.messages_dropped == second.diagnostics.messages_dropped
    assert first.diagnostics.messages_sent == second.diagnostics.messages_sent

    fallback = _planner("ideal").plan(
        _observation(),
        _scenarios(status="raw"),
        step_index=0,
        fallback_actions=np.ones((4, 3), dtype=np.float64),
    )
    assert fallback.diagnostics.status == "fallback"
    assert "projected" in str(fallback.diagnostics.fallback_reason)


def test_configuration_rejects_invalid_dropout_probability() -> None:
    with pytest.raises(ValueError, match="dropout"):
        DistributedDNMPCConfig(message_dropout_probability=1.1)


def test_configuration_rejects_unknown_qdr_exhaustion_policy() -> None:
    with pytest.raises(ValueError, match="qdr_exhaustion_policy"):
        DistributedDNMPCConfig(qdr_exhaustion_policy="unknown")


def test_receding_horizon_warm_start_shifts_executed_action() -> None:
    sequence = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3)
    shifted = _shift_warm_start_sequence(sequence)

    np.testing.assert_array_equal(shifted[:-1], sequence[1:])
    np.testing.assert_array_equal(shifted[-1], sequence[-1])


def test_distributed_formation_slot_rnic_scores_the_full_team_rollout() -> None:
    config = MinimaxMPCConfig(
        horizon_steps=4,
        control_horizon_steps=2,
        max_role_variants=2,
        perimeter_scales=(1.0,),
        reachability_normalized_cost_enabled=True,
        reachability_cost_mode="formation_slot",
        reachability_slot_radius_m=1.4,
        weight_reachability=1.0,
    )
    planner = DistributedMinimaxDNMPC(
        config,
        DistributedDNMPCConfig(
            communication_mode="delayed",
            message_delay_steps=1,
            max_iterations=1,
            local_timeout_ms=1000.0,
        ),
    )

    plan = planner.plan(_observation(), _scenarios(), step_index=0)

    assert plan.diagnostics.status in {"success", "not_converged", "partial_fallback"}
    assert np.isfinite(plan.diagnostics.scenario_costs).all()
    assert plan.diagnostics.scenario_costs[0] >= 0.0


def test_distributed_escape_gap_term_is_team_scored_and_audited() -> None:
    config = MinimaxMPCConfig(
        horizon_steps=4,
        control_horizon_steps=2,
        max_role_variants=2,
        perimeter_scales=(1.0,),
        escape_gap_cost_enabled=True,
        weight_escape_gap=0.3,
    )
    planner = DistributedMinimaxDNMPC(
        config,
        DistributedDNMPCConfig(
            communication_mode="ideal",
            max_iterations=1,
            local_timeout_ms=1000.0,
        ),
    )
    plan = planner.plan(_observation(), _scenarios(), step_index=0)
    assert plan.diagnostics.status in {"success", "not_converged", "partial_fallback"}
    assert np.isfinite(plan.diagnostics.scenario_costs).all()
    assert np.isfinite(plan.diagnostics.escape_gap_cost)
    assert np.isfinite(plan.diagnostics.escape_gap_coverage_ratio)
