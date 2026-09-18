"""Adapt the retained Phase 73 development block to the Phase 86 evaluator schema."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    adapted = []
    for index, row in enumerate(rows):
        spec = row["spec"]
        item = copy.deepcopy(row)
        item["dataset_index"] = index
        item["episode_index"] = index
        item["episode_seed"] = int(spec["episode_seed"])
        item["layout_seed"] = int(spec["layout_seed"])
        item["defender_side"] = spec["defender_side"]
        item["initial_side_distance"] = float(spec["initial_side_distance"])
        item["target_speed_scale"] = float(spec["target_speed_scale"])
        item["target_motion_mode"] = spec["target_motion_mode"]
        item["target_crossing_required"] = True
        item["observation_condition"] = spec["observation_condition"]
        item["pursuit_overrides"] = copy.deepcopy(spec["pursuit_overrides"])
        item["obstacle_count"] = int(spec["obstacle_count"])
        item["difficulty"] = "route_exercise"
        item["variant"] = "route_exercise"
        item["single_factor"] = "target_crossing_required"
        item["single_factor_value"] = True
        item["scene_block"] = "phase86_route_exercise"
        item["execution"] = {"action_delay_steps": 0, "command_noise_std_mps": 0.0}
        item["mirror_group_id"] = str(spec["mirror_group_id"])
        item["mirror_pair_member"] = spec["mirror_pair_member"]
        item["spec"] = copy.deepcopy(item)
        adapted.append(item)
    scene_file = output / "scenes.jsonl"
    scene_file.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in adapted), encoding="utf-8")
    manifest = {
        "dataset_name": "phase86.route_exercise.development.v1",
        "dataset_version": "phase86.route_exercise.v1",
        "phase": "development_calibration_only",
        "not_a_locked_test": True,
        "locked_test": False,
        "locked_test_modified": False,
        "total_scenes": len(adapted),
        "mirror_groups": len({item["mirror_group_id"] for item in adapted}),
        "source_phase73_scene_sha256": sha256(args.input),
        "scene_file_sha256": sha256(scene_file),
        "target_crossing_required": True,
        "direct_path_blocked": True,
        "minimum_bypass_routes": 2,
        "source_scene_file": str(args.input.resolve()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
