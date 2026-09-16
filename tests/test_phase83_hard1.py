from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_phase82_nominal_plus1_scenes as generator  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def test_phase83_hard1_smoke_preserves_outward_target_contract(tmp_path: Path) -> None:
    manifest = generator.generate(
        ROOT / "configs" / "phase83_hard1_single_factor.yaml",
        ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml",
        tmp_path / "phase83_hard1_smoke",
        episodes_per_variant=2,
        allow_small=True,
    )
    assert manifest["dataset_version"] == "phase83.hard1.v1"
    assert manifest["total_scenes"] == 6
    assert manifest["variant_counts"] == {
        "target_speed": 2,
        "obstacle_near": 2,
        "formation_tight": 2,
    }
    assert manifest["mirror_groups"] == 3
    assert manifest["all_mirror_groups_have_two_members"] is True
    assert manifest["initial_escape_clearance_certified_rate"] == 1.0

    records = [
        json.loads(line)
        for line in (tmp_path / "phase83_hard1_smoke" / "scenes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert all(not item["target_crossing_required"] for item in records)
    assert all(
        not item["pursuit_overrides"]["target_maneuver_enable_reverse_lane_change"]
        for item in records
    )
    obstacle_near = [item for item in records if item["variant"] == "obstacle_near"]
    assert max(item["target_initial_obstacle_clearance_m"] for item in obstacle_near) <= 2.40 + 1.0e-9


def test_phase83_hard1_uses_only_gate_passing_factors() -> None:
    payload = yaml.safe_load(
        (ROOT / "configs" / "phase83_hard1_single_factor.yaml").read_text(encoding="utf-8")
    )
    assert payload["episodes_per_variant"] * len(payload["variants"]) >= 300
    assert payload["variants"]["target_speed"]["target_speed_scales"] == [0.55, 0.65]
    assert payload["variants"]["obstacle_near"]["target_obstacle_clearance_max_m"] == 2.40
    assert payload["variants"]["formation_tight"]["defender_y_scale"] == 0.65
    assert payload["base_profile"]["pursuit_overrides"]["target_maneuver_enable_reverse_lane_change"] is False
    assert payload["base_profile"]["execution"]["action_delay_steps"] == 1
