from __future__ import annotations

import numpy as np
import pytest

from encirclement3d.feasible_consensus import (
    evaluate_fixed_consensus_slots,
    feasible_consensus_slot_gate,
)


def _slot_rollout() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directions = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=np.float64,
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    target = np.zeros((1, 3, 3), dtype=np.float64)
    positions = np.broadcast_to(1.4 * directions[None, None, :, :], (1, 3, 4, 3)).copy()
    velocities = np.zeros_like(positions)
    return positions, velocities, target


def test_fc_dbf_accepts_a_stable_slot_rollout() -> None:
    positions, velocities, targets = _slot_rollout()
    metrics = feasible_consensus_slot_gate(
        positions,
        velocities,
        targets,
        slot_radius_m=1.4,
        dt_seconds=0.1,
        max_speed_mps=5.0,
        max_acceleration_mps2=6.0,
        slot_tolerance_m=0.1,
        min_slot_slack_s=-0.1,
        gate_horizon_steps=3,
    )
    assert bool(np.asarray(metrics["feasible"])[0, 0]) is True
    assert float(np.asarray(metrics["max_slot_error_m"])[0, 0]) < 1.0e-10
    assert bool(metrics["gate_exhausted"]) is False


def test_fc_dbf_exposes_gate_exhaustion_without_fabricating_feasibility() -> None:
    positions, velocities, targets = _slot_rollout()
    positions = np.zeros_like(positions)
    metrics = feasible_consensus_slot_gate(
        positions,
        velocities,
        targets,
        slot_radius_m=1.4,
        dt_seconds=0.1,
        max_speed_mps=5.0,
        max_acceleration_mps2=6.0,
        slot_tolerance_m=0.01,
        min_slot_slack_s=-0.1,
        gate_horizon_steps=2,
    )
    assert bool(np.asarray(metrics["feasible"])[0, 0]) is False
    assert bool(metrics["gate_exhausted"]) is True


def test_fc_dbf_rejects_invalid_previous_assignment() -> None:
    positions, velocities, targets = _slot_rollout()
    with pytest.raises(ValueError, match="previous_assignment"):
        feasible_consensus_slot_gate(
            positions,
            velocities,
            targets,
            slot_radius_m=1.4,
            dt_seconds=0.1,
            max_speed_mps=5.0,
            max_acceleration_mps2=6.0,
            previous_assignment=np.asarray([0, 0, 1, 2]),
        )


def test_fc_dbf_fixed_consensus_token_matches_nominal_assignment() -> None:
    positions, velocities, targets = _slot_rollout()
    nominal = feasible_consensus_slot_gate(
        positions,
        velocities,
        targets,
        slot_radius_m=1.4,
        dt_seconds=0.1,
        max_speed_mps=5.0,
        max_acceleration_mps2=6.0,
        slot_tolerance_m=0.1,
        min_slot_slack_s=-0.1,
        gate_horizon_steps=3,
    )
    fixed = evaluate_fixed_consensus_slots(
        positions,
        velocities,
        targets,
        np.asarray(nominal["assignment_paths"], dtype=np.int64)[0],
        slot_radius_m=1.4,
        dt_seconds=0.1,
        max_speed_mps=5.0,
        max_acceleration_mps2=6.0,
        slot_tolerance_m=0.1,
        min_slot_slack_s=-0.1,
        gate_horizon_steps=3,
    )
    np.testing.assert_array_equal(fixed["feasible"], nominal["feasible"])
    np.testing.assert_allclose(fixed["cost"], nominal["cost"], atol=1.0e-10)
    assert np.asarray(fixed["assignment_paths"]).shape == (1, 1, 3, 4)
