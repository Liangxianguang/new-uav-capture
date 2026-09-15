from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
import generate_phase77_calibrated_difficulty_scenes as phase77  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def test_target_maneuver_candidate_filter_rejects_boundary_rollouts() -> None:
    config = yaml.safe_load((ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml").read_text(encoding="utf-8"))
    config["task"]["pursuit"]["target_maneuver_feasibility_margin_m"] = 0.60
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.65)
    env.reset(seed=771201)
    env.target_position = np.array([9.7, 0.0, 5.0], dtype=np.float64)
    env.target_velocity.fill(0.0)
    env.target_acceleration.fill(0.0)
    env.target_escape_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    candidates = env._target_maneuver_candidates(env.defender_positions, env.defender_velocities)
    evaluated = [
        env._evaluate_target_maneuver_candidate(item, env.defender_positions, env.defender_velocities)
        for item in candidates
    ]
    assert any(not bool(item["feasible"]) for item in evaluated)
    assert any(str(item["feasibility_failure"]) == "boundary_clearance" for item in evaluated)


def test_phase77_smoke_generator_keeps_three_mirror_pools(tmp_path: Path) -> None:
    manifest = phase77.generate(
        ROOT / "configs" / "phase77_calibrated_difficulty.yaml",
        ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml",
        ROOT / "configs" / "phase73_crossing_scene_dataset.yaml",
        tmp_path / "phase77_smoke",
        episodes_per_difficulty=2,
        allow_small=True,
    )
    assert manifest["total_scenes"] == 6
    assert manifest["difficulty_counts"] == {"easy": 2, "nominal": 2, "hard": 2}
    assert manifest["all_mirror_groups_have_two_members"] is True
    scene_lines = (tmp_path / "phase77_smoke" / "scenes.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(scene_lines) == 6
    assert {str(json.loads(line)["difficulty"]) for line in scene_lines} == {"easy", "nominal", "hard"}
