from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from encirclement3d.residual_actor import RecurrentResidualActor  # noqa: E402
from train_phase79_dagger_residual import (  # noqa: E402
    _write_evaluation_progress,
    teacher_demo_is_accepted,
)


ROOT = Path(__file__).resolve().parents[1]


def test_residual_actor_is_bounded_and_recurrent() -> None:
    actor = RecurrentResidualActor(local_observation_dim=11, hidden_dim=16, residual_scale=2.5)
    hidden = actor.initial_hidden(4)
    local = torch.randn(4, 11)
    base = torch.full((4, 3), 1.0)
    action, next_hidden, residual = actor.step(local, base, hidden, action_scale=5.0)
    assert action.shape == (4, 3)
    assert residual.shape == (4, 3)
    assert next_hidden.shape == (4, 16)
    assert torch.all(action <= 5.0)
    assert torch.all(action >= -5.0)
    assert not torch.equal(hidden, next_hidden)


def test_phase79_requires_192_demos_and_keeps_phase78_contract() -> None:
    payload = yaml.safe_load(
        (ROOT / "configs" / "phase79_dnmcp_dagger_residual.yaml").read_text(encoding="utf-8")
    )
    assert payload["experiment"]["demo_episodes"] >= 192
    assert payload["experiment"]["dagger_rounds"] >= 1
    assert payload["experiment"]["episodes_per_round"] > 0
    assert payload["environment_overrides"]["task"]["pursuit"]["target_maneuver_crossing_gain"] == 0.0
    assert payload["environment_overrides"]["task"]["pursuit"]["target_maneuver_enable_reverse_lane_change"] is False
    assert payload["experiment"]["filtered_route_base"] is True


def test_phase79_conservative_candidate_is_validation_only_and_bounded() -> None:
    payload = yaml.safe_load(
        (ROOT / "configs" / "phase79_dnmcp_dagger_residual_conservative.yaml").read_text(encoding="utf-8")
    )
    pursuit = payload["environment_overrides"]["task"]["pursuit"]
    assert payload["not_a_locked_test"] is True
    assert payload["locked_test_tuning_forbidden"] is True
    assert pursuit["safety_margin"] == 1.0
    assert payload["experiment"]["residual_scale_mps"] == 0.5
    assert payload["experiment"]["demo_episodes"] >= 192


def test_teacher_quality_gate_rejects_unsafe_rollouts() -> None:
    assert teacher_demo_is_accepted(
        {
            "safe_capture_success": True,
            "collision": False,
            "boundary_violation": False,
            "timeout": False,
        }
    )
    assert not teacher_demo_is_accepted(
        {
            "safe_capture_success": False,
            "collision": True,
            "boundary_violation": False,
            "timeout": False,
        }
    )
    assert not teacher_demo_is_accepted(
        {
            "safe_capture_success": True,
            "collision": False,
            "boundary_violation": True,
            "timeout": False,
        }
    )


def test_evaluation_progress_snapshot_is_atomic_and_calibration_only(tmp_path) -> None:
    path = tmp_path / "evaluation_progress.json"
    rows = [
        {
            "safe_capture_success": True,
            "collision": False,
            "boundary_violation": False,
            "timeout": False,
        }
    ]
    _write_evaluation_progress(
        path,
        target_episodes=2,
        rows=rows,
        route_latencies=[1.0],
        actor_latencies=[2.0],
        safety_latencies=[3.0],
        total_latencies=[6.0],
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["locked_test_used"] is False
    assert payload["episodes_completed"] == 1
    assert payload["complete"] is False
    assert payload["latency_ms"]["total"] == {"p50": 6.0, "p95": 6.0, "p99": 6.0}
    assert not list(tmp_path.glob(".evaluation_progress.json.tmp"))


def test_phase81_transition_enables_quality_gate_and_cpu_thread_cap() -> None:
    payload = yaml.safe_load(
        (ROOT / "configs" / "phase81_dnmcp_dagger_residual_nominal_plus.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert payload["experiment"]["demo_episodes"] >= 192
    assert payload["experiment"]["quality_gate_initial_demos"] is True
    assert payload["experiment"]["torch_threads"] == 1
