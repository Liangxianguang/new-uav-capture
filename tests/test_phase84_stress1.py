from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_phase82_nominal_plus1_scenes as generator  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def test_phase84_stress1_smoke_preserves_calibration_contract(tmp_path: Path) -> None:
    manifest = generator.generate(
        ROOT / "configs" / "phase84_stress1_single_factor.yaml",
        ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml",
        tmp_path / "phase84_stress1_smoke",
        episodes_per_variant=2,
        allow_small=True,
    )
    assert manifest["dataset_version"] == "phase84.stress1.v1"
    assert manifest["total_scenes"] == 6
    assert manifest["variant_counts"] == {
        "target_speed_stress": 2,
        "obstacle_near_stress": 2,
        "formation_tight_stress": 2,
    }
    records = [
        json.loads(line)
        for line in (tmp_path / "phase84_stress1_smoke" / "scenes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert all(item["difficulty"] == "stress1" for item in records)
    assert all(item["scene_block"] == "phase84_stress1_calibration" for item in records)
    assert all(item["mirror_group_id"].startswith("phase84-stress1-") for item in records)
    assert all(not item["target_crossing_required"] for item in records)
    assert all(
        not item["pursuit_overrides"]["target_maneuver_enable_reverse_lane_change"]
        for item in records
    )


def test_phase84_stress1_is_at_least_300_scene_calibration() -> None:
    payload = yaml.safe_load(
        (ROOT / "configs" / "phase84_stress1_single_factor.yaml").read_text(encoding="utf-8")
    )
    assert payload["episodes_per_variant"] * len(payload["variants"]) >= 300
    assert payload["base_profile"]["difficulty_label"] == "stress1"
    assert payload["base_profile"]["pursuit_overrides"]["target_maneuver_crossing_gain"] == 0.0
    assert payload["base_profile"]["pursuit_overrides"]["target_maneuver_enable_reverse_lane_change"] is False
    assert payload["variants"]["target_speed_stress"]["target_speed_scales"] == [0.65, 0.70]
    assert payload["variants"]["obstacle_near_stress"]["target_obstacle_clearance_max_m"] == 2.0
    assert payload["variants"]["formation_tight_stress"]["defender_y_scale"] == 0.58
