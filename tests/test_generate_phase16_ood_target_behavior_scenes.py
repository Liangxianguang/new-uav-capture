from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_phase16_ood_target_behavior_scenes import OOD_BEHAVIORS, validate_records  # noqa: E402


def test_behavior_ood_conditions_are_distinct() -> None:
    assert len(OOD_BEHAVIORS) == 2
    assert OOD_BEHAVIORS[0]["target_branch_defender_lookahead_seconds"] != OOD_BEHAVIORS[1]["target_branch_defender_lookahead_seconds"]


def test_validate_records_accepts_behavior_mirrors() -> None:
    obstacle = {"shape": "wall", "center_xy": [0.0, 0.0], "radius": 0.55, "height": 9.45, "half_extents_xy": [0.55, 4.25]}
    records = []
    for group, behavior in enumerate(OOD_BEHAVIORS):
        for member, bias in enumerate(("upper", "lower")):
            records.append({"mirror_group_id": group, "defender_bias": bias, "layout_seed": group + 1, "target_speed_scale": 0.65, "ood_behavior": behavior, "scenario": {"obstacles": [obstacle]}})
    validate_records(records)
