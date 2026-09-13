from __future__ import annotations

from scripts.generate_phase56_strong_baseline_scenes import (
    BASE_EXECUTION,
    BASE_PURSUIT,
    _condition,
    _split_for,
)


def test_phase56_delay_noise_condition_changes_only_declared_execution_fields() -> None:
    condition_id, pursuit, execution = _condition(1, 19)

    assert condition_id == "delay8_noise0.12"
    assert pursuit == BASE_PURSUIT
    assert execution["action_delay_steps"] == 8
    assert execution["command_noise_std"] == 0.12
    assert execution["pending_command_authority"] == BASE_EXECUTION["pending_command_authority"]


def test_phase56_communication_condition_changes_only_declared_pursuit_execution_fields() -> None:
    condition_id, pursuit, execution = _condition(2, 11)

    assert condition_id == "dropout0.30_msgdelay4_track0.00"
    assert pursuit["message_dropout_probability"] == 0.30
    assert pursuit["message_delay_steps"] == 4
    assert execution["velocity_time_constant_seconds"] == 0.0
    assert execution["action_delay_steps"] == BASE_EXECUTION["action_delay_steps"]
    assert execution["command_noise_std"] == BASE_EXECUTION["command_noise_std"]


def test_phase56_split_assignment_is_deterministic() -> None:
    values = [_split_for(0, index) for index in range(6)]

    assert values == [
        "development_calibration",
        "development_confirmation",
        "locked_diagnostic",
        "development_calibration",
        "development_confirmation",
        "locked_diagnostic",
    ]

