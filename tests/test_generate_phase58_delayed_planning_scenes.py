from __future__ import annotations

from scripts.generate_phase58_delayed_planning_scenes import (
    BASE_EXECUTION,
    BASE_PURSUIT,
    BLOCKS,
    condition_for,
    split_for,
)


def test_phase58_delay_noise_covers_extended_levels_without_changing_sensing() -> None:
    condition_id, pursuit, execution, factors = condition_for(1, 29)

    assert condition_id == "delay10_noise0.16"
    assert pursuit == BASE_PURSUIT
    assert execution["action_delay_steps"] == 10
    assert execution["command_noise_std"] == 0.16
    assert factors["action_delay_steps"] == 10
    assert factors["command_noise_std"] == 0.16


def test_phase58_communication_block_records_all_execution_factors() -> None:
    condition_id, pursuit, execution, factors = condition_for(2, 59)

    assert "msgdelay" in condition_id and "dropout" in condition_id
    assert pursuit["message_delay_steps"] in {0, 2, 4, 6}
    assert pursuit["message_dropout_probability"] in {0.0, 0.1, 0.2, 0.3}
    assert execution["action_delay_steps"] == BASE_EXECUTION["action_delay_steps"]
    assert factors["drag_coefficient"] in {0.0, 0.05}


def test_phase58_joint_block_is_explicitly_stress_only() -> None:
    condition_id, pursuit, execution, factors = condition_for(3, 0)

    assert condition_id.startswith("delay8_noise0.12")
    assert execution["action_delay_steps"] == 8
    assert execution["command_noise_std"] == 0.12
    assert pursuit["message_delay_steps"] == 4
    assert pursuit["message_dropout_probability"] == 0.2
    assert factors["drag_coefficient"] == 0.05


def test_phase58_split_is_deterministic_and_balanced() -> None:
    values = [split_for(0, index) for index in range(6)]

    assert values == [
        "development_calibration",
        "development_confirmation",
        "locked_diagnostic",
        "development_calibration",
        "development_confirmation",
        "locked_diagnostic",
    ]
    assert len(BLOCKS) == 4


def test_phase60_rollout_policy_is_explicitly_overridable() -> None:
    from scripts.generate_phase58_delayed_planning_scenes import build_records
    from scripts.evaluate_s4_branching import load_protocol
    from evaluate_minimax_mpc import DEFAULT_ENVIRONMENT_CONFIG
    from pathlib import Path

    protocol = load_protocol(Path("configs/phase60_qdr_revalidation_protocol.yaml"))
    records = build_records(
        protocol,
        DEFAULT_ENVIRONMENT_CONFIG,
        groups_per_block=3,
        seed_start=991101,
        rollout_policy="phase60_qdr_revalidation_matrix",
    )

    assert {record["rollout_policy"] for record in records} == {
        "phase60_qdr_revalidation_matrix"
    }
