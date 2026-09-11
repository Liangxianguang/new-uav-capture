from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_prediction_selection_results import load_groups, summarize_group


def _write_selection_run(root: Path, group: str, seed: int, min_fde: float) -> Path:
    training = root / f"training_{group}_seed{seed}"
    training.mkdir()
    training_metadata = {
        "model": "gru",
        "model_backend": "gru_gaussian",
        "action_conditioning": group,
        "train_dataset": "results/phase16/train/dataset.npz",
        "validation_dataset": "results/phase16/validation/dataset.npz",
        "arguments": {"seed": seed},
    }
    (training / "metadata.json").write_text(json.dumps(training_metadata), encoding="utf-8")
    result = root / f"selection_{group}_seed{seed}"
    result.mkdir()
    metrics = {
        "min_ade": min_fde / 2.0,
        "min_fde": min_fde,
        "energy_score": 1.0,
        "coverage_full_trajectory": 0.9,
        "conformal_full_trajectory_coverage_error": 0.0,
        "candidate_feasible_fraction": 0.8,
        "candidate_obstacle_clear_fraction": 0.9,
        "branch_top1_accuracy": 0.8,
        "branch_uniform_vote_accuracy": 0.8,
        "branch_any_candidate_coverage": 0.9,
        "branch_bimodal_candidate_fraction": 0.2,
    }
    document = {
        "evaluation_split": "validation_second_half_episode_seeds",
        "locked_test_dataset": None,
        "training_output": str(training),
        "calibration": {"source": "validation_first_half_episode_seeds_only"},
        "num_samples": 1,
        "sampling_steps": 4,
        "projection_iterations": 4,
        "raw": {**metrics, "min_fde": min_fde + 0.1},
        "projected": metrics,
        "latency": {
            "single_sample_latency_p50_ms": 1.0,
            "single_sample_latency_p95_ms": 2.0,
            "single_sample_latency_p99_ms": 3.0,
        },
    }
    (result / "validation_selection_metrics.json").write_text(json.dumps(document), encoding="utf-8")
    return result


def test_selection_aggregator_uses_validation_only_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "selection"
    root.mkdir()
    for seed, value in ((1, 1.0), (2, 1.2)):
        _write_selection_run(root, "both", seed, value)
        _write_selection_run(root, "future", seed, value + 0.5)
    groups = load_groups(
        [
            f"both={root.as_posix()}/selection_both_seed*",
            f"future={root.as_posix()}/selection_future_seed*",
        ]
    )
    import numpy as np

    summary = summarize_group(groups["both"], np.random.default_rng(1), 100)
    assert summary["metrics"]["projected.min_fde"]["mean"] == pytest.approx(1.1)
    assert summary["action_conditioning"] == "both"


def test_selection_aggregator_rejects_locked_test_reference(tmp_path: Path) -> None:
    root = tmp_path / "selection"
    root.mkdir()
    result = _write_selection_run(root, "both", 1, 1.0)
    artifact = result / "validation_selection_metrics.json"
    document = json.loads(artifact.read_text(encoding="utf-8"))
    document["locked_test_dataset"] = "results/phase16/locked_test/dataset.npz"
    artifact.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="locked-test"):
        load_groups([f"both={result.parent.as_posix()}/selection_both_seed*"])
