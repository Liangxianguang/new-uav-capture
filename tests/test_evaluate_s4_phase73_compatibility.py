from __future__ import annotations

from pathlib import Path

from scripts.evaluate_s4_closed_loop import config_for_phase73_spec, read_scenes


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml"
def _write_nested_phase73_scene(path: Path) -> None:
    path.write_text(
        '{"episode_index": 0, "scenario": {}, "spec": {'
        '"episode_seed": 731202, "layout_seed": 1731202, '
        '"target_speed_scale": 0.65, "defender_side": "left", '
        '"initial_side_distance": 6.5, "target_motion_mode": "adaptive_maneuvering", '
        '"target_crossing_required": true, "observation_condition": "delayed_noisy", '
        '"pursuit_overrides": {"message_delay_steps": 4}, '
        '"obstacle_count": 3, "mirror_pair_member": "left", '
        '"mirror_group_id": "phase73-test-0000", "scene_block": "development_validation"}}\n',
        encoding="utf-8",
    )


def test_phase73_nested_scene_spec_is_normalized_without_rewriting(tmp_path: Path) -> None:
    scene_path = tmp_path / "scenes.jsonl"
    _write_nested_phase73_scene(scene_path)
    before = scene_path.read_text(encoding="utf-8")
    records = read_scenes(scene_path, 1)
    assert len(records) == 1
    assert records[0]["target_crossing_required"] is True
    assert records[0]["target_motion_mode"] == "adaptive_maneuvering"
    assert records[0]["pursuit_overrides"]["message_delay_steps"] in {2, 4}
    assert records[0]["defender_bias"] in {"upper", "lower"}
    assert records[0]["mirror_group_id"] == "phase73-test-0000"
    assert scene_path.read_text(encoding="utf-8") == before


def test_phase73_config_keeps_maneuvering_target_instead_of_s4_branching(tmp_path: Path) -> None:
    scene_path = tmp_path / "scenes.jsonl"
    _write_nested_phase73_scene(scene_path)
    record = read_scenes(scene_path, 1)[0]
    config = config_for_phase73_spec(ENVIRONMENT, record, None)
    pursuit = config["task"]["pursuit"]
    assert pursuit["target_motion_mode"] == "adaptive_maneuvering"
    assert pursuit["target_flee_gain"] == 0.05
    assert pursuit["target_heading_persistence"] == 4.0
    assert config["world"]["max_steps"] == 250
