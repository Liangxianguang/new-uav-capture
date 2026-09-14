from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.extract_scene_split import extract_split


def _write_scene_file(path: Path) -> None:
    rows = []
    for group, split in ((1, "development_calibration"), (2, "locked_diagnostic")):
        for member, bias in enumerate(("upper", "lower")):
            rows.append(
                {
                    "mirror_group_id": group,
                    "mirror_pair_member": member,
                    "defender_bias": bias,
                    "evaluation_split": split,
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_extracts_complete_non_locked_split(tmp_path: Path) -> None:
    scenes = tmp_path / "scenes.jsonl"
    output = tmp_path / "calibration" / "scenes.jsonl"
    _write_scene_file(scenes)
    manifest = extract_split(scenes, output, evaluation_split="development_calibration")
    assert manifest["episodes"] == 2
    assert manifest["mirror_groups"] == 1
    assert manifest["locked_test_included"] is False
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_locked_split_requires_explicit_opt_in(tmp_path: Path) -> None:
    scenes = tmp_path / "scenes.jsonl"
    _write_scene_file(scenes)
    with pytest.raises(ValueError, match="locked_diagnostic"):
        extract_split(scenes, tmp_path / "locked" / "scenes.jsonl", evaluation_split="locked_diagnostic")
