from __future__ import annotations

import numpy as np
import pytest

from encirclement3d.delay_aware_conformal_tube import (
    DelayAwareConformalReachableTube,
    candidate_minimum_errors,
    evaluate_tube_coverage,
    fit_conformal_radius_schedule,
    upper_conformal_quantile,
)


def test_upper_conformal_quantile_uses_finite_sample_index() -> None:
    assert upper_conformal_quantile(np.array([0.1, 0.2, 0.3, 0.4]), 0.75) == pytest.approx(0.4)


def test_fit_and_evaluate_schedule_reports_full_trajectory_coverage() -> None:
    candidates = np.zeros((4, 2, 3, 3), dtype=np.float64)
    targets = np.zeros((4, 3, 3), dtype=np.float64)
    targets[:, :, 0] = np.array([0.1, 0.2, 0.3])
    errors = candidate_minimum_errors(candidates, targets)
    summary = fit_conformal_radius_schedule(
        errors,
        coverage=0.75,
        dt_seconds=0.1,
        target_speed_mps=3.6,
    )
    coverage = evaluate_tube_coverage(candidates, targets, np.asarray(summary["radius_m_by_step"]))
    assert summary["horizon_steps"] == 3
    assert coverage["full_trajectory_coverage"] >= 0.75


def test_queue_and_uncertainty_align_and_inflate_radius() -> None:
    tube = DelayAwareConformalReachableTube(
        radius_m_by_step=(0.1, 0.2, 0.3, 0.4),
        coverage=0.9,
        dt_seconds=0.1,
        target_speed_mps=2.0,
        calibration_count=10,
        simultaneous_multiplier=1.0,
        uncertainty_gain=0.5,
    )
    nominal = tube.radius_by_step(2)
    delayed = tube.radius_by_step(2, queue_length=1, uncertainty_score=1.0)
    assert nominal.tolist() == pytest.approx([0.1, 0.2])
    assert delayed.tolist() == pytest.approx([0.2 * 1.5, 0.3 * 1.5])


def test_scenario_set_rejects_radius_schedule_with_wrong_horizon() -> None:
    from encirclement3d.minimax_mpc import ScenarioTrajectorySet

    with pytest.raises(ValueError, match="conformal_radius_by_step_m"):
        ScenarioTrajectorySet(
            trajectories=np.zeros((1, 2, 3)),
            weights=np.ones(1),
            conformal_radius_by_step_m=(0.1,),
        )


def test_scenario_set_truncate_slices_radius_schedule() -> None:
    from encirclement3d.minimax_mpc import ScenarioTrajectorySet

    scenarios = ScenarioTrajectorySet(
        trajectories=np.zeros((1, 3, 3)),
        weights=np.ones(1),
        conformal_radius_by_step_m=(0.1, 0.2, 0.3),
    )
    truncated = scenarios.truncate(2)
    assert truncated.conformal_radius_by_step_m == pytest.approx((0.1, 0.2))
