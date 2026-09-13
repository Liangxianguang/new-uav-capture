from __future__ import annotations

from scripts.generate_phase57_causal_factor_scenes import (
    BASE_EXECUTION,
    BASE_PURSUIT,
    condition_for,
    split_for,
)


def test_phase57_delay_noise_changes_only_execution_fields() -> None:
    condition_id, pursuit, execution = condition_for(1, 11)

    assert condition_id == "delay8_noise0.12"
    assert pursuit == BASE_PURSUIT
    assert execution["action_delay_steps"] == 8
    assert execution["command_noise_std"] == 0.12
    assert execution["pending_command_authority"] == BASE_EXECUTION["pending_command_authority"]


def test_phase57_communication_changes_only_declared_fields() -> None:
    condition_id, pursuit, execution = condition_for(2, 11)

    assert condition_id == "dropout0.10_msgdelay4_track0.10"
    assert pursuit["message_dropout_probability"] == 0.10
    assert pursuit["message_delay_steps"] == 4
    assert execution["velocity_time_constant_seconds"] == 0.10
    assert execution["action_delay_steps"] == BASE_EXECUTION["action_delay_steps"]
    assert execution["command_noise_std"] == BASE_EXECUTION["command_noise_std"]


def test_phase57_interaction_contains_both_execution_and_communication_factors() -> None:
    condition_id, pursuit, execution = condition_for(3, 15)

    assert condition_id == "delay8_noise0.12_dropout0.30_msgdelay4_track0.10"
    assert execution["action_delay_steps"] == 8
    assert execution["command_noise_std"] == 0.12
    assert pursuit["message_dropout_probability"] == 0.30
    assert pursuit["message_delay_steps"] == 4
    assert execution["velocity_time_constant_seconds"] == 0.10


def test_phase57_split_assignment_is_balanced_and_deterministic() -> None:
    values = [split_for(0, index) for index in range(6)]
    assert values == [
        "development_calibration",
        "development_confirmation",
        "locked_diagnostic",
        "development_calibration",
        "development_confirmation",
        "locked_diagnostic",
    ]
