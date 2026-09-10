from __future__ import annotations

import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = PROJECT_ROOT / "scripts" / "analyze_execution_failures.py"
    spec = importlib.util.spec_from_file_location("analyze_execution_failures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failure_summary_separates_certificate_gap_from_actual_unsafe() -> None:
    module = _module()
    rows = [
        {
            "abort_required": False,
            "prefix_admissible": True,
            "fallback_used": True,
            "fallback_certificate_valid": True,
            "execution_rollout_certificate_valid": False,
            "continuous_segment_certificate_valid": False,
            "actual_post_robust_state_safe": True,
            "first_failed_barrier": "rollout:step[5]/sweep[4]/boundary_upper[2]/0",
            "fallback_candidate_type": "barrier_recovery",
            "action_execution_error_norm_mps": 0.2,
        },
        {
            "abort_required": True,
            "prefix_admissible": False,
            "fallback_used": True,
            "fallback_certificate_valid": False,
            "execution_rollout_certificate_valid": False,
            "continuous_segment_certificate_valid": False,
            "actual_post_robust_state_safe": False,
            "first_failed_barrier": "continuous:minimum_robust_barrier",
            "fallback_candidate_type": "zero_action",
            "action_execution_error_norm_mps": 0.4,
        },
    ]

    summary = module.summarize_steps(rows)

    assert summary["prefix_admissible"]["rate"] == 0.5
    assert summary["abort_required"]["count"] == 1
    assert summary["continuous_only_gap"]["count"] == 1
    assert summary["actual_post_robust_unsafe"]["count"] == 1
    assert summary["barrier_families"]["boundary"]["count"] == 1
    assert summary["barrier_families"]["continuous_contract"]["count"] == 1
    assert summary["barrier_horizons"]["5"]["count"] == 1


def test_barrier_family_handles_missing_labels() -> None:
    module = _module()
    assert module.barrier_family(None) == "none"
    assert module.barrier_family("rollout:obstacle[0]/0") == "obstacle"
    assert module.barrier_family("rollout:inter_agent[0,1]") == "inter_agent"
