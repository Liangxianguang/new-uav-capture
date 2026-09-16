from pathlib import Path

import numpy as np
import yaml

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_world_boundary_accounting_separates_target_and_defender() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml").read_text(
            encoding="utf-8"
        )
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    env.reset(seed=820101)
    target = np.array([env.upper[0] + 1.0, env.target_position[1], env.target_position[2]], dtype=np.float64)
    target_velocity = np.zeros((1, 3), dtype=np.float64)
    env._enforce_world_bounds(target[None, :], target_velocity, entity="target")
    assert env.world_violation_steps == 1
    assert env.target_world_violation_steps == 1
    assert env.defender_world_violation_steps == 0
    assert env.first_target_boundary_violation_step == 1
    assert env.first_defender_boundary_violation_step is None

    defender = env.defender_positions.copy()
    defender[0, 1] = env.lower[1] - 1.0
    defender_velocity = np.zeros_like(defender)
    env._enforce_world_bounds(defender, defender_velocity, entity="defender")
    assert env.world_violation_steps == 2
    assert env.target_world_violation_steps == 1
    assert env.defender_world_violation_steps == 1
    assert env.first_defender_boundary_violation_step == 1
