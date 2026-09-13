from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_phase59_repair import (  # noqa: E402
    budget_summary,
    grouped_subset_metric_matrix,
)
from aggregate_closed_loop_seed_results import RunArtifact  # noqa: E402


def test_phase59_budget_audit_detects_unrealized_gru_style_budget(tmp_path: Path) -> None:
    method_dir = tmp_path / "R2_phase59_queue_cbf_k4"
    method_dir.mkdir(parents=True)
    (method_dir / "steps.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "episode_index": 0,
                    "candidate_budget_requested": 4.0,
                    "candidate_count": 1.0,
                }
            )
            for _ in range(3)
        ),
        encoding="utf-8",
    )
    artifact = RunArtifact(
        label="repair",
        seed=727201,
        path=tmp_path,
        scene_hash="scene",
        methods={},
        steps={},
    )

    audit = budget_summary([artifact], "R2_phase59_queue_cbf_k4", [0])

    assert audit["requested_min"] == 4
    assert audit["realized_min"] == 1
    assert audit["realized_rate"] == 0.0
    assert audit["mismatch_steps"] == 3


def test_phase59_block_slice_does_not_reuse_the_full_episode_matrix(tmp_path: Path) -> None:
    method_dir = tmp_path / "R0_phase59_qdr_baseline"
    method_dir.mkdir(parents=True)
    episodes = [
        {"episode_index": 0, "safe_capture_success": 1.0},
        {"episode_index": 1, "safe_capture_success": 1.0},
        {"episode_index": 2, "safe_capture_success": 0.0},
        {"episode_index": 3, "safe_capture_success": 0.0},
    ]
    (method_dir / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in episodes), encoding="utf-8"
    )
    (tmp_path / "scenes.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "episode_index": index,
                    "mirror_group_id": index // 2,
                }
            )
            for index in range(4)
        ),
        encoding="utf-8",
    )
    artifact = RunArtifact(
        label="repair",
        seed=727201,
        path=tmp_path,
        scene_hash="scene",
        methods={"R0_phase59_qdr_baseline": episodes},
        steps={},
    )

    groups, matrix = grouped_subset_metric_matrix(
        [artifact],
        "R0_phase59_qdr_baseline",
        "safe_capture_success",
        [2, 3],
    )

    assert groups == [1]
    assert matrix.tolist() == [[0.0]]
