from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_phase16_ood_geometry_scenes import OOD_GEOMETRY, TRAINING_GEOMETRY, validate_records


def test_geometry_ranges_are_disjoint() -> None:
    for name, (_train_low, train_high) in TRAINING_GEOMETRY.items():
        ood_low, _ood_high = OOD_GEOMETRY[name]
        assert ood_low > train_high


def test_validate_records_accepts_mirrored_ood_geometry() -> None:
    obstacle = {
        "shape": "wall",
        "center_xy": [0.0, 0.0],
        "radius": 0.75,
        "height": 9.7,
        "half_extents_xy": [0.75, 4.40],
    }
    records = [
        {"mirror_group_id": 0, "defender_bias": "upper", "layout_seed": 1, "scenario": {"obstacles": [obstacle]}},
        {"mirror_group_id": 0, "defender_bias": "lower", "layout_seed": 1, "scenario": {"obstacles": [obstacle]}},
    ]
    validate_records(records)
