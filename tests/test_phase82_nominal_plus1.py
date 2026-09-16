from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_phase82_nominal_plus1_scenes as phase82  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def test_phase82_smoke_has_single_factor_blocks_and_contract(tmp_path: Path) -> None:
    manifest = phase82.generate(
        ROOT / "configs" / "phase82_nominal_plus1.yaml",
        ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml",
        tmp_path / "phase82_smoke",
        episodes_per_variant=2,
        allow_small=True,
    )
    assert manifest["total_scenes"] == 10
    assert manifest["variant_counts"] == {
        "command_noise": 2,
        "action_delay": 2,
        "maneuver_frequency": 2,
        "obstacle_near": 2,
        "formation_tight": 2,
    }
    assert manifest["mirror_groups"] == 5
    assert manifest["all_mirror_groups_have_two_members"] is True
    assert manifest["initial_escape_clearance_certified_rate"] == 1.0
    assert manifest["contract"]["one_factor_per_variant"] is True

    records = [
        json.loads(line)
        for line in (tmp_path / "phase82_smoke" / "scenes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(records) == 10
    assert all(not item["target_crossing_required"] for item in records)
    assert all(
        not item["pursuit_overrides"]["target_maneuver_enable_reverse_lane_change"]
        for item in records
    )
    assert {item["variant"] for item in records} == {
        "command_noise",
        "action_delay",
        "maneuver_frequency",
        "obstacle_near",
        "formation_tight",
    }
    obstacle_near = [item for item in records if item["variant"] == "obstacle_near"]
    assert max(item["target_initial_obstacle_clearance_m"] for item in obstacle_near) <= 2.40 + 1.0e-9
    tight = [item for item in records if item["variant"] == "formation_tight"]
    assert min(item["formation_spacing_min_m"] for item in tight) < 1.30


def test_phase82_protocol_changes_one_factor_per_block() -> None:
    payload = yaml.safe_load(
        (ROOT / "configs" / "phase82_nominal_plus1.yaml").read_text(encoding="utf-8")
    )
    base = payload["base_profile"]
    variants = payload["variants"]
    assert payload["episodes_per_variant"] >= 100
    assert len(variants) * payload["episodes_per_variant"] >= 300
    assert variants["command_noise"]["execution"]["command_noise_std_mps"] == 0.030
    assert variants["action_delay"]["execution"]["action_delay_steps"] == 2
    assert variants["maneuver_frequency"]["pursuit_overrides"]["target_maneuver_replan_interval_steps"] == 6
    assert variants["obstacle_near"]["initial_side_distances"] != base["initial_side_distances"]
    assert variants["obstacle_near"]["target_obstacle_clearance_max_m"] == 2.40
    assert variants["formation_tight"]["defender_y_scale"] == 0.65
    assert all(
        payload["base_profile"]["pursuit_overrides"]["target_maneuver_enable_reverse_lane_change"] is False
        for _ in [0]
    )
