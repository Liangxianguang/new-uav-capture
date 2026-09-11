from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_phase16_ood_target_speed_scenes import (  # noqa: E402
    OOD_SPEED_SCALES,
    TRAINING_SPEED_SCALES,
    validate_records,
)


def test_target_speed_ranges_are_disjoint() -> None:
    assert set(OOD_SPEED_SCALES).isdisjoint(TRAINING_SPEED_SCALES)
    assert min(OOD_SPEED_SCALES) > max(TRAINING_SPEED_SCALES)


def test_validate_records_accepts_both_speed_levels_and_mirrors() -> None:
    obstacle = {
        "shape": "wall",
        "center_xy": [0.0, 0.0],
        "radius": 0.55,
        "height": 9.45,
        "half_extents_xy": [0.55, 4.25],
    }
    records = []
    for group, speed in enumerate(OOD_SPEED_SCALES):
        for member, bias in enumerate(("upper", "lower")):
            records.append(
                {
                    "mirror_group_id": group,
                    "defender_bias": bias,
                    "layout_seed": group + 1,
                    "target_speed_scale": speed,
                    "scenario": {"obstacles": [obstacle]},
                }
            )
    validate_records(records)
