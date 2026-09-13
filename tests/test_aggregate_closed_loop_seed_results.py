from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.aggregate_closed_loop_seed_results import (
    RunArtifact,
    grouped_metric_matrix,
    paired_comparison,
    paired_method_comparison,
    summarize_group,
)


def _artifact(label: str, seed: int, safe_capture: list[bool]) -> RunArtifact:
    rows = []
    steps = []
    for index, value in enumerate(safe_capture):
        rows.append(
            {
                "episode_index": index,
                "safe_capture_success": value,
                "capture_event": value,
                "collision": False,
                "boundary_violation": False,
                "timeout": not value,
                "capture_time_seconds": 1.0 if value else float("nan"),
                "min_clearance_m": 0.5,
                "future_action_condition_available_rate": 0.9,
                "mean_prediction_age_steps": 0.0,
            }
        )
        steps.append(
            {
                "predictor_latency_ms": 1.0,
                "planner_latency_ms": 2.0,
                "safety_latency_ms": 0.5,
                "total_control_latency_ms": 3.5,
            }
        )
    return RunArtifact(
        label=label,
        seed=seed,
        path=Path(f"{label}_seed{seed}"),
        scene_hash="same-frozen-scenes",
        methods={"distributed_delayed": rows},
        steps={"distributed_delayed": steps},
    )


def test_closed_loop_summary_and_paired_bootstrap_keep_episode_pairing() -> None:
    gru = [_artifact("gru", 1, [True, False]), _artifact("gru", 2, [True, True])]
    official = [_artifact("official", 1, [True, True]), _artifact("official", 2, [True, False])]
    rng = np.random.default_rng(19)

    summary = summarize_group(gru, rng, bootstrap_samples=200)
    safe_capture = summary["distributed_delayed"]["episode_metrics"]["safe_capture_success"]
    assert safe_capture["mean"] == 0.75
    assert safe_capture["bootstrap_95_ci"][0] <= safe_capture["mean"] <= safe_capture["bootstrap_95_ci"][1]
    assert summary["distributed_delayed"]["pooled_step_latency_ms"]["total_control_latency_ms"]["p95"] == 3.5

    comparison = paired_comparison(
        gru,
        official,
        "distributed_delayed",
        np.random.default_rng(23),
        bootstrap_samples=200,
    )
    delta = comparison["episode_metrics"]["safe_capture_success"]
    assert comparison["shared_training_seeds"] == [1, 2]
    assert delta["mean_delta_candidate_minus_reference"] == 0.0
    assert delta["paired_bootstrap_95_ci"][0] <= 0.0 <= delta["paired_bootstrap_95_ci"][1]


def test_mirror_group_bootstrap_averages_upper_lower_members(tmp_path: Path) -> None:
    scene_path = tmp_path / "gru_seed1" / "scenes.jsonl"
    scene_path.parent.mkdir()
    scene_path.write_text(
        "\n".join(
            json.dumps(
                {"episode_index": index, "mirror_group_id": index // 2},
                separators=(",", ":"),
            )
            for index in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    artifact = RunArtifact(
        label="gru",
        seed=1,
        path=scene_path.parent,
        scene_hash="same-frozen-scenes",
        methods={
            "B0": [
                {"episode_index": 0, "safe_capture_success": True},
                {"episode_index": 1, "safe_capture_success": False},
                {"episode_index": 2, "safe_capture_success": True},
                {"episode_index": 3, "safe_capture_success": True},
            ]
        },
        steps={"B0": [{"total_control_latency_ms": 1.0, "qdr_latency_ms": 0.0}]},
    )
    groups, matrix = grouped_metric_matrix([artifact], "B0", "safe_capture_success")
    assert groups == [0, 1]
    assert matrix.shape == (1, 2)
    assert matrix.tolist() == [[0.5, 1.0]]


def test_paired_method_comparison_uses_same_mirror_units(tmp_path: Path) -> None:
    scene_path = tmp_path / "gru_seed1" / "scenes.jsonl"
    scene_path.parent.mkdir()
    scene_path.write_text(
        "\n".join(
            json.dumps({"episode_index": index, "mirror_group_id": index // 2})
            for index in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    rows = {
        "B0": [
            {"episode_index": 0, "safe_capture_success": False},
            {"episode_index": 1, "safe_capture_success": False},
        ],
        "B1": [
            {"episode_index": 0, "safe_capture_success": True},
            {"episode_index": 1, "safe_capture_success": True},
        ],
    }
    artifact = RunArtifact(
        label="gru",
        seed=1,
        path=scene_path.parent,
        scene_hash="same-frozen-scenes",
        methods=rows,
        steps={method: [{"total_control_latency_ms": 1.0} for _ in range(2)] for method in rows},
    )
    result = paired_method_comparison(
        [artifact], [artifact], "B0", "B1", np.random.default_rng(3), bootstrap_samples=20, bootstrap_unit="mirror_group"
    )
    assert result["episode_metrics"]["safe_capture_success"]["mean_delta_candidate_minus_reference"] == 1.0
