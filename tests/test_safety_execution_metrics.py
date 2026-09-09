import importlib.util
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _metrics_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_safety_execution_metrics",
        PROJECT_ROOT / "scripts" / "evaluate_safety_execution.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wilson_interval_reports_episode_level_uncertainty() -> None:
    module = _metrics_module()
    interval = module._wilson_interval([True, False, True, False])

    assert interval["successes"] == 2
    assert interval["trials"] == 4
    assert interval["rate"] == 0.5
    assert 0.1 < interval["lower"] < 0.2
    assert 0.8 < interval["upper"] < 0.9


def test_execution_summary_reports_p99_latency_and_wilson_metrics() -> None:
    module = _metrics_module()
    rows = [
        {
            "safe_capture_success": True,
            "capture_event": True,
            "collision": False,
            "boundary_violation": False,
            "timeout": False,
            "capture_time_seconds": None,
            "min_clearance_m": 1.0,
            "command_certificate_valid_rate": 1.0,
            "execution_rollout_certificate_valid_rate": 1.0,
            "swept_volume_certificate_valid_rate": 1.0,
            "continuous_segment_certificate_valid_rate": 1.0,
            "executed_certificate_valid_rate": 1.0,
            "command_next_state_safe_rate": 1.0,
            "executed_next_state_safe_rate": 1.0,
            "executed_current_state_safe_rate": 1.0,
            "actual_post_state_safe_rate": 1.0,
            "actual_post_robust_state_safe_rate": 1.0,
            "mean_action_execution_error_norm_mps": 0.0,
            "mean_belief_goal_progress_m": 0.1,
            "mean_fallback_goal_progress_m": 0.0,
            "safety_latency_ms": 10.0,
            "minimum_command_next_barrier_m": 0.1,
            "minimum_execution_rollout_robust_barrier_m": 0.1,
            "minimum_swept_volume_robust_barrier_m": 0.1,
            "minimum_continuous_segment_robust_barrier_m": 0.1,
            "minimum_executed_next_barrier_m": 0.1,
            "minimum_actual_post_barrier_m": 0.1,
            "solver_fallback_count": 0,
            "fallback_certificate_valid": True,
        }
    ]
    steps = [{"safety_latency_ms": 10.0, "solver_backend": "test"}]

    summary = module.summarize(rows, steps)

    assert summary["safety_latency_ms"]["p99"] == 10.0
    assert summary["wilson_95"]["safe_capture_rate"]["trials"] == 1
    assert summary["wilson_95"]["safe_capture_rate"]["successes"] == 1


def test_resolve_episode_seeds_supports_reproducible_holdout_blocks() -> None:
    module = _metrics_module()
    evaluation = {"robust_safe_seed_protocol": {"episode_seeds": [10, 20, 30]}}

    assert module.resolve_episode_seeds(evaluation) == [10, 20, 30]
    assert module.resolve_episode_seeds(evaluation, seed_start=700001, seed_count=4) == [700001, 700002, 700003, 700004]
    assert module.resolve_episode_seeds(evaluation, seed_start=700001, seed_count=4, max_episodes=2) == [700001, 700002]
    with pytest.raises(ValueError, match="provided together"):
        module.resolve_episode_seeds(evaluation, seed_start=700001)
    with pytest.raises(ValueError, match="positive"):
        module.resolve_episode_seeds(evaluation, seed_start=700001, seed_count=0)
