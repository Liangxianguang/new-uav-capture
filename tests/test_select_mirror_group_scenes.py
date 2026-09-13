from __future__ import annotations

import pytest

from scripts.select_mirror_group_scenes import select_mirror_group_block


def _records() -> list[dict[str, object]]:
    return [
        {"episode_index": 0, "mirror_group_id": 10, "scene_block": "validation"},
        {"episode_index": 1, "mirror_group_id": 10, "scene_block": "validation"},
        {"episode_index": 2, "mirror_group_id": 11, "scene_block": "validation"},
        {"episode_index": 3, "mirror_group_id": 11, "scene_block": "validation"},
        {"episode_index": 4, "mirror_group_id": 12, "scene_block": "validation"},
        {"episode_index": 5, "mirror_group_id": 12, "scene_block": "validation"},
    ]


def test_selection_keeps_complete_ordered_mirror_groups() -> None:
    selected, metadata = select_mirror_group_block(
        _records(), drop_first_groups=1, expected_split="validation"
    )
    assert [record["episode_index"] for record in selected] == [2, 3, 4, 5]
    assert metadata["selected_mirror_group_count"] == 2
    assert metadata["locked_test_included"] is False


def test_selection_rejects_incomplete_groups() -> None:
    records = _records()[:-1]
    with pytest.raises(ValueError, match="two members"):
        select_mirror_group_block(records, drop_first_groups=0)


def test_selection_rejects_empty_holdout() -> None:
    with pytest.raises(ValueError, match="leaves no mirror groups"):
        select_mirror_group_block(_records(), drop_first_groups=3)
