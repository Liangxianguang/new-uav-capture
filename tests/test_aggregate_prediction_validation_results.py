from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_prediction_validation_results import (
    load_groups,
    summarize_runs,
)
from scripts.aggregate_prediction_seed_results import load_group_records


def _write_run(root: Path, condition: str, seed: int, min_fde: float, epochs: int = 3) -> Path:
    path = root / f"phase15_condition_{condition}_seed{seed}"
    path.mkdir()
    metadata = {
        "model": "gru",
        "action_conditioning": condition,
        "train_dataset": "results/phase15_s4_branching_predictor_v3/train/dataset.npz",
        "validation_dataset": "results/phase15_s4_branching_predictor_v3/validation/dataset.npz",
        "arguments": {
            "epochs": epochs,
            "batch_size": 128,
            "hidden_dim": 32,
            "num_layers": 1,
            "learning_rate": 0.001,
            "seed": seed,
        },
    }
    final = {
        "epoch": epochs,
        "loss": 0.1,
        "model_min_ade": min_fde / 2.0,
        "model_min_fde": min_fde,
        "model_ade_top1": min_fde / 2.0,
        "model_fde_top1": min_fde,
        "model_energy_score": 1.0,
        "model_candidate_spread": 0.0,
        "candidate_speed_feasible_fraction": 0.8,
        "candidate_acceleration_feasible_fraction": 0.0,
        "candidate_obstacle_clear_fraction": 0.9,
        "candidate_feasible_fraction": 0.0,
        "calibrated_coverage_full_trajectory": 0.9,
        "single_sample_latency_p95_ms": 1.0,
    }
    (path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (path / "training.json").write_text(json.dumps([final]), encoding="utf-8")
    return path


def test_validation_aggregator_pairs_matching_seeds(tmp_path: Path) -> None:
    root = tmp_path / "validation_aggregation_test_artifacts"
    root.mkdir()
    for seed, value in ((1, 1.0), (2, 1.2)):
        _write_run(root, "both", seed, value)
        _write_run(root, "history", seed, value + 0.5)
    groups = load_groups([
        f"both={root.as_posix()}/phase15_condition_both_seed*",
        f"history={root.as_posix()}/phase15_condition_history_seed*",
    ])
    import numpy as np

    summary = summarize_runs(groups["both"], np.random.default_rng(1), 100)
    assert summary["metrics"]["model_min_fde"]["mean"] == pytest.approx(1.1)
    assert [run.seed for run in groups["history"]] == [1, 2]


def test_validation_aggregator_rejects_locked_test_reference(tmp_path: Path) -> None:
    path = _write_run(tmp_path, "both", 1, 1.0)
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["validation_dataset"] = "locked_test/dataset.npz"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="locked-test"):
        load_groups([f"both={path.parent.as_posix()}/*"])


def test_locked_test_aggregator_accepts_explicit_condition_groups(tmp_path: Path) -> None:
    expected = {
        "raw": {"min_fde": 1.0},
        "projected": {
            "min_fde": 0.9,
            "min_ade": 0.4,
            "coverage_full_trajectory": 0.9,
            "candidate_feasible_fraction": 0.8,
        },
        "latency": {"single_sample_latency_p95_ms": 1.0},
    }
    for condition in ("both", "future"):
        directory = tmp_path / f"action_condition_{condition}_seed1"
        directory.mkdir()
        (directory / "locked_test_metrics.json").write_text(json.dumps(expected), encoding="utf-8")
    grouped = load_group_records(
        [
            f"both={tmp_path.as_posix()}/action_condition_both_seed*",
            f"future={tmp_path.as_posix()}/action_condition_future_seed*",
        ],
        (
            "raw.min_fde",
            "projected.min_fde",
            "projected.min_ade",
            "projected.coverage_full_trajectory",
            "projected.candidate_feasible_fraction",
            "latency.single_sample_latency_p95_ms",
        ),
    )
    assert sorted(grouped) == ["both", "future"]
    assert grouped["both"][1]["projected.min_fde"] == pytest.approx(0.9)
