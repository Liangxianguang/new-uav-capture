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


def test_receding_horizon_warm_start_shifts_executed_action() -> None:
    sequence = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3)
    shifted = _shift_warm_start_sequence(sequence)

    np.testing.assert_array_equal(shifted[:-1], sequence[1:])
    np.testing.assert_array_equal(shifted[-1], sequence[-1])
