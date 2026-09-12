from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_uakr_local_reliability import analyze


def test_local_reliability_audit_splits_by_canonical_groups(tmp_path: Path) -> None:
    run = tmp_path / "run"
    method = run / "distributed_delayed"
    method.mkdir(parents=True)
    episodes = []
    steps = []
    for index in range(8):
        episodes.append({"episode_index": index})
        steps.append(
            {
                "episode_index": index,
                "adaptive_uncertainty_score": index / 10.0,
                "adaptive_prediction_residual_m": index / 20.0,
                "safety_independent_next_state_safe": index < 4,
                "safety_independent_current_state_safe": index < 4,
            }
        )
    (method / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in episodes), encoding="utf-8")
    (method / "steps.jsonl").write_text("\n".join(json.dumps(row) for row in steps), encoding="utf-8")
    scenes = tmp_path / "scenes.jsonl"
    scenes.write_text(
        "\n".join(
            json.dumps({"episode_index": index, "mirror_group_id": f"g{index // 2}"})
            for index in range(8)
        ),
        encoding="utf-8",
    )

    result = analyze([run], scenes, split_seed=3)

    assert result["group_count"] == 4
    assert result["calibration_group_count"] == 2
    assert result["confirmation_group_count"] == 2
    assert result["splits"]["calibration"]["uncertainty"]["next_state_violation"]["step_count"] == 4
    assert result["splits"]["confirmation"]["uncertainty"]["next_state_violation"]["step_count"] == 4
