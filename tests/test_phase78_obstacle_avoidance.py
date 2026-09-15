from __future__ import annotations

import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import generate_phase78_obstacle_avoidance_scenes as phase78  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def test_phase78_smoke_is_outward_and_non_crossing(tmp_path: Path) -> None:
    manifest = phase78.generate(
        ROOT / "configs" / "phase78_obstacle_avoidance_first.yaml",
        ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml",
        tmp_path / "phase78_smoke",
        episodes_per_difficulty=2,
        allow_small=True,
    )
    assert manifest["total_scenes"] == 6
    assert manifest["difficulty_counts"] == {"easy": 2, "nominal": 2, "hard": 2}
    assert manifest["target_crossing_required_rate"] == 0.0
    assert manifest["initial_escape_clearance_certified_rate"] == 1.0
    records = [
        json.loads(line)
        for line in (tmp_path / "phase78_smoke" / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert all(not bool(item["target_crossing_required"]) for item in records)
    assert all(bool(item["avoidance_certificate"]["initial_escape_clearance_certified"]) for item in records)
    assert all(
        not bool(item["pursuit_overrides"]["target_maneuver_enable_reverse_lane_change"])
        for item in records
    )


def test_phase78_protocol_declares_obstacle_avoidance_first() -> None:
    payload = yaml.safe_load(
        (ROOT / "configs" / "phase78_obstacle_avoidance_first.yaml").read_text(encoding="utf-8")
    )
    assert payload["common"]["target_crossing_required"] is False
    assert payload["common"]["initial_escape_lookahead_steps"] == 4
    assert all(
        profile["pursuit_overrides"]["target_maneuver_obstacle_avoidance_gain"] > 0.0
        and profile["pursuit_overrides"]["target_maneuver_enable_reverse_lane_change"] is False
        for profile in payload["difficulty_profiles"].values()
    )


def test_phase78_seed_offset_creates_independent_reproducible_pool(tmp_path: Path) -> None:
    first = phase78.generate(
        ROOT / "configs" / "phase80_progressive_hard_v2.yaml",
        ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml",
        tmp_path / "pool_a",
        episodes_per_difficulty=2,
        allow_small=True,
        seed_offset=0,
    )
    second = phase78.generate(
        ROOT / "configs" / "phase80_progressive_hard_v2.yaml",
        ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml",
        tmp_path / "pool_b",
        episodes_per_difficulty=2,
        allow_small=True,
        seed_offset=1_000_000,
    )
    repeat = phase78.generate(
        ROOT / "configs" / "phase80_progressive_hard_v2.yaml",
        ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml",
        tmp_path / "pool_c",
        episodes_per_difficulty=2,
        allow_small=True,
        seed_offset=1_000_000,
    )
    assert first["seed_offset"] == 0
    assert second["seed_offset"] == 1_000_000
    assert second["scene_file_sha256"] != first["scene_file_sha256"]
    assert repeat["scene_file_sha256"] == second["scene_file_sha256"]
