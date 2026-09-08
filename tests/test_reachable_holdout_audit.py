from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_execution_reachable_holdout import (
    evaluate_margin,
    fit_margin,
    variant_matches_calibration_contract,
)


def test_fit_margin_uses_simultaneous_order_statistic() -> None:
    errors = np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=np.float64)
    base = np.ones_like(errors)
    margin = fit_margin(errors, base, quantile=0.75)
    assert margin["simultaneous_multiplier"] == 3.0
    assert margin["frozen_radius_m_by_step"] == [3.0, 3.0]
    assert margin["calibration_coverage"] == 0.75


def test_evaluate_margin_rejects_out_of_calibration_samples() -> None:
    errors = np.asarray([[0.5], [2.0]], dtype=np.float64)
    frozen = np.asarray([1.0], dtype=np.float64)
    records = [
        {"in_calibration_domain": True},
        {"in_calibration_domain": False},
    ]
    summary = evaluate_margin(
        errors,
        frozen,
        records=records,
        target_coverage=0.9,
        policy_decision="reject_or_fallback",
    )
    assert summary["out_of_calibration_samples"] == 1
    assert summary["policy_decision"] == "reject_or_fallback"
    assert summary["actionable_coverage"] is None


def test_runtime_margin_uses_per_sample_base_radii() -> None:
    errors = np.asarray([[0.5], [1.0]], dtype=np.float64)
    base = np.asarray([[0.5], [0.5]], dtype=np.float64)
    records = [{"in_calibration_domain": True}, {"in_calibration_domain": True}]
    summary = evaluate_margin(
        errors,
        np.asarray([1.0], dtype=np.float64),
        records=records,
        target_coverage=0.9,
        policy_decision="allow",
        base_radii=base,
        runtime_multiplier=2.0,
    )
    assert summary["runtime_multiplier"] == 2.0
    assert summary["runtime_simultaneous_coverage"] == 1.0
    assert summary["runtime_coverage_pass"] is True


def test_declared_contract_rejects_changed_range_even_when_samples_overlap() -> None:
    calibration = {
        "action_delay_steps": 2,
        "command_noise_std": 0.08,
        "velocity_time_constant_seconds": 0.4,
        "max_speed_scale": 1.0,
        "max_acceleration_scale": 1.0,
        "mass_scale": 1.0,
        "drag_coefficient": 0.1,
        "max_speed_scale_range": [0.85, 0.98],
        "max_acceleration_scale_range": [0.70, 0.95],
        "mass_scale_range": [0.90, 1.10],
        "drag_coefficient_range": [0.05, 0.20],
    }
    changed = {**calibration, "drag_coefficient_range": [0.0375, 0.15]}
    assert not variant_matches_calibration_contract(changed, calibration)
