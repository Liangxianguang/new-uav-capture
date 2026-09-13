from __future__ import annotations

import numpy as np

from src.encirclement3d.qdr_formalization import audit_qdr_time_index


def test_qdr_full_and_shifted_rollouts_are_equivalent() -> None:
    positions = np.zeros((2, 3), dtype=np.float64)
    velocities = np.ones((2, 3), dtype=np.float64)
    queue = np.arange(12, dtype=np.float64).reshape(2, 2, 3)
    suffix = np.arange(18, dtype=np.float64).reshape(3, 2, 3)

    result = audit_qdr_time_index(positions, velocities, queue, suffix)

    assert result.passed
    assert result.first_controllable_step == 2
    assert result.candidate_action_application_steps == (2, 3, 4)
    assert result.candidate_state_visibility_steps == (3, 4, 5)
    assert result.max_position_error_m <= 1.0e-9


def test_qdr_zero_delay_reduces_to_ordinary_rollout_contract() -> None:
    positions = np.zeros((1, 3), dtype=np.float64)
    velocities = np.zeros((1, 3), dtype=np.float64)
    suffix = np.ones((2, 1, 3), dtype=np.float64)

    result = audit_qdr_time_index(positions, velocities, [], suffix)

    assert result.passed
    assert result.first_controllable_step == 0
    assert result.candidate_action_application_steps == (0, 1)


def test_qdr_rejects_wrong_action_shape() -> None:
    positions = np.zeros((2, 3), dtype=np.float64)
    velocities = np.zeros((2, 3), dtype=np.float64)

    try:
        audit_qdr_time_index(positions, velocities, np.zeros((2, 3)), np.zeros((1, 2, 3)))
    except ValueError as error:
        assert "actions must have shape" in str(error)
    else:  # pragma: no cover
        raise AssertionError("invalid action shape must be rejected")

