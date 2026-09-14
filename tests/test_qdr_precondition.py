from __future__ import annotations

import numpy as np
import pytest

from encirclement3d.execution_dynamics import ExecutionParameters
from encirclement3d.qdr_precondition import (
    audit_qdr_precondition,
    audit_qdr_segment_liveness,
    classify_qdr_precondition,
    queue_prefix_risk_score,
    rollout_suffix_state,
)


def _diagnostic(barrier: float) -> dict[str, float | bool]:
    return {"minimum_prefix_barrier_m": barrier, "prefix_admissible": barrier >= 0.0}


def _parameters(*, enabled: bool = False) -> ExecutionParameters:
    return ExecutionParameters(
        enabled=enabled,
        dt_seconds=0.1,
        action_delay_steps=2,
        command_noise_std_mps=0.0,
        command_noise_bound_mps=0.0,
        clip_command_noise=True,
        velocity_time_constant_seconds=0.0,
        drag_coefficient=0.0,
        max_speed_mps=5.0,
        max_acceleration_mps2=6.0,
        mass_scale=1.0,
    )


def test_classification_distinguishes_safe_prefix_and_unsafe_suffix() -> None:
    result = classify_qdr_precondition(
        _diagnostic(0.25),
        _diagnostic(-0.10),
        authority_mode="immutable",
    )
    assert result.status == "prefix_safe_suffix_unsafe"
    assert not result.recovery_recommended


def test_immutable_unsafe_prefix_is_explicitly_unrecoverable() -> None:
    result = classify_qdr_precondition(
        _diagnostic(-0.25),
        _diagnostic(-0.10),
        authority_mode="immutable",
    )
    assert result.status == "prefix_unsafe_unrecoverable"
    assert not result.recovery_allowed
    assert not result.recovery_recommended


@pytest.mark.parametrize("authority", ["replace_nonexecuting", "flush_pending"])
def test_nonimmutable_authority_can_recommend_recovery(authority: str) -> None:
    result = classify_qdr_precondition(
        _diagnostic(-0.25),
        _diagnostic(0.20),
        authority_mode=authority,
    )
    assert result.status == "prefix_unsafe_recoverable"
    assert result.recovery_allowed
    assert result.recovery_recommended


def test_safe_prefix_and_suffix_is_feasible() -> None:
    result = classify_qdr_precondition(
        _diagnostic(0.25),
        _diagnostic(0.20),
        authority_mode="immutable",
    )
    assert result.status == "prefix_safe_suffix_safe"
    assert result.prefix_admissible and result.suffix_admissible


def test_rollout_suffix_uses_execution_dynamics_and_preserves_shape() -> None:
    positions = np.zeros((2, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    actions = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        ],
        dtype=np.float64,
    )
    state = rollout_suffix_state(positions, velocities, actions, _parameters())
    assert state.prefix_positions.shape == (2, 2, 3)
    np.testing.assert_allclose(state.delayed_positions, [[0.3, 0.0, 0.0], [0.0, 0.3, 0.0]])
    assert state.queue_length == 2


def test_audit_returns_suffix_geometry_and_assessment() -> None:
    positions = np.zeros((1, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    actions = np.asarray([[[0.0, 0.0, 0.0]]], dtype=np.float64)
    observation = {
        "world_lower_bounds": np.asarray([-10.0, -10.0, -10.0]),
        "world_upper_bounds": np.asarray([10.0, 10.0, 10.0]),
        "obstacles": [],
    }
    result = audit_qdr_precondition(
        prefix_diagnostics=_diagnostic(0.25),
        delayed_positions=positions,
        delayed_velocities=velocities,
        action_sequence=actions,
        observation=observation,
        parameters=_parameters(),
        drone_radius_m=0.25,
        safety_margin_m=0.35,
        authority_mode="immutable",
    )
    assert result["assessment"]["status"] == "prefix_safe_suffix_safe"
    assert result["suffix"]["prefix_admissible"]


def test_invalid_tolerance_and_action_shape_are_rejected() -> None:
    with pytest.raises(ValueError, match="tolerance_m"):
        classify_qdr_precondition(_diagnostic(0.0), _diagnostic(0.0), authority_mode="immutable", tolerance_m=-1.0)
    with pytest.raises(ValueError, match="action_sequence"):
        rollout_suffix_state(
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            _parameters(),
        )


def test_queue_prefix_risk_is_bounded_and_uses_public_clearance_only() -> None:
    safe = queue_prefix_risk_score(
        {"minimum_clearance_m": 2.0, "maximum_safety_margin_violation_m": 0.0},
        safety_margin_m=0.35,
    )
    near = queue_prefix_risk_score(
        {"minimum_clearance_m": 0.20, "maximum_safety_margin_violation_m": 0.15},
        safety_margin_m=0.35,
    )
    empty = queue_prefix_risk_score(
        {"minimum_clearance_m": float("inf"), "maximum_safety_margin_violation_m": 0.0},
        safety_margin_m=0.35,
    )
    assert safe == 0.0
    assert 0.0 < near <= 1.0
    assert empty == 0.0


def test_queue_prefix_risk_rejects_invalid_diagnostic_values() -> None:
    with pytest.raises(ValueError, match="minimum_clearance_m"):
        queue_prefix_risk_score({}, safety_margin_m=0.35)
    with pytest.raises(ValueError, match="scale_m"):
        queue_prefix_risk_score(
            {"minimum_clearance_m": 0.2},
            safety_margin_m=0.35,
            scale_m=0.0,
        )


def test_segment_liveness_audit_distinguishes_prefix_suffix_and_terminal() -> None:
    defenders = np.asarray(
        [
            [[0.0, 0.0, 0.0]],
            [[0.4, 0.0, 0.0]],
            [[0.8, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [
            [[1.2, 0.0, 0.0], [0.8, 0.0, 0.0], [0.8, 0.0, 0.0]],
            [[2.0, 0.0, 0.0], [1.6, 0.0, 0.0], [1.2, 0.0, 0.0]],
        ],
        dtype=np.float64,
    )
    result = audit_qdr_segment_liveness(
        prefix_diagnostics=_diagnostic(0.2),
        suffix_diagnostics=_diagnostic(0.1),
        defender_positions_path=defenders,
        candidate_target_paths=targets,
        capture_radius_m=0.05,
    )
    assert result["assessment"]["status"] == "terminal_candidate_feasible"
    assert result["assessment"]["earliest_any_candidate_capture_step"] == 3
    assert result["assessment"]["terminal_any_candidate_feasible"] is True
    assert result["assessment"]["terminal_all_candidate_feasible"] is False
    assert result["assessment"]["finite_progress_available"] is True


def test_segment_liveness_audit_prioritizes_prefix_failure_without_truth() -> None:
    result = audit_qdr_segment_liveness(
        prefix_diagnostics=_diagnostic(-0.1),
        suffix_diagnostics=_diagnostic(0.1),
        defender_positions_path=np.zeros((2, 1, 3), dtype=np.float64),
        candidate_target_paths=np.ones((1, 2, 3), dtype=np.float64),
        capture_radius_m=0.1,
    )
    assert result["assessment"]["status"] == "prefix_infeasible"
    assert result["assessment"]["prefix_feasible"] is False


def test_segment_liveness_audit_rejects_mismatched_horizon() -> None:
    with pytest.raises(ValueError, match="candidate_target_paths"):
        audit_qdr_segment_liveness(
            prefix_diagnostics=_diagnostic(0.1),
            suffix_diagnostics=_diagnostic(0.1),
            defender_positions_path=np.zeros((2, 1, 3), dtype=np.float64),
            candidate_target_paths=np.zeros((1, 3, 3), dtype=np.float64),
            capture_radius_m=0.1,
        )
