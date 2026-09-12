from __future__ import annotations

import json
from pathlib import Path

from scripts.split_scene_manifest_by_groups import split_scene_manifest


def test_split_scene_manifest_is_disjoint_and_complete(tmp_path: Path) -> None:
    scenes = tmp_path / "scenes.jsonl"
    rows = [
        {"episode_index": index, "mirror_group_id": f"group-{index // 2}"}
        for index in range(8)
    ]
    scenes.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    artifact = tmp_path / "artifact.json"
    artifact.write_text(
        json.dumps(
            {
                "split_metadata": {
                    "calibration_groups": ["group-0", "group-1"],
                    "confirmation_groups": ["group-2", "group-3"],
                }
            }
        ),
        encoding="utf-8",
    )
    calibration = tmp_path / "calibration.jsonl"
    confirmation = tmp_path / "confirmation.jsonl"

    result = split_scene_manifest(scenes, artifact, calibration, confirmation)

    assert result["calibration_episode_count"] == 4
    assert result["confirmation_episode_count"] == 4
    assert len(calibration.read_text(encoding="utf-8").splitlines()) == 4
    assert len(confirmation.read_text(encoding="utf-8").splitlines()) == 4
