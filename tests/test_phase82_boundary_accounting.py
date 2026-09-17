from pathlib import Path

import numpy as np
import pytest
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


def test_predictive_target_boundary_direction_uses_braking_horizon() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["task"]["pursuit"].update(
        {
            "target_motion_mode": "adaptive_maneuvering",
            "target_maneuver_predictive_boundary_recovery": True,
            "target_maneuver_predictive_boundary_lookahead_steps": 24,
        }
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.65)
    env.reset(seed=820102)
    env.target_position = np.array([7.0, 0.0, 3.7], dtype=np.float64)
    env.target_velocity = np.array([2.4, 0.0, -1.2], dtype=np.float64)
    direction = env._target_maneuver_boundary_direction()
    assert direction[0] < 0.0
    assert direction[2] > 0.0


@pytest.mark.parametrize("axis,sign", [(axis, sign) for axis in range(3) for sign in (-1, 1)])
def test_target_world_clipping_accounts_for_all_six_boundary_directions(axis: int, sign: int) -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml").read_text(
            encoding="utf-8"
        )
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    env.reset(seed=820200 + axis * 2 + (sign > 0))
    target = env.target_position.copy()
    target[axis] = env.upper[axis] + 1.0 if sign > 0 else env.lower[axis] - 1.0
    velocity = np.zeros((1, 3), dtype=np.float64)

    env._enforce_world_bounds(target[None, :], velocity, entity="target")

    expected = env.upper[axis] if sign > 0 else env.lower[axis]
    assert target[axis] == expected
    assert env.target_world_violation_steps == 1
    assert env.first_target_boundary_violation_step == 1
    assert env._target_boundary_clearance(target[0]) < 0.0


def test_target_boundary_guard_clips_an_outward_command_before_crossing() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml").read_text(
            encoding="utf-8"
        )
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.75)
    env.reset(seed=820210)
    safe_lower, safe_upper = env._target_safe_bounds()
    env.target_position = np.array([safe_upper[0] - 0.05, 0.0, 5.0], dtype=np.float64)
    env.target_velocity = np.array([3.6, 0.0, 0.0], dtype=np.float64)
    env.target_acceleration.fill(0.0)

    command = env._constrain_target_command(np.array([3.6, 0.0, 0.0]), enforce_maneuver_limits=False)

    assert env.target_command_clipped
    assert command[0] < 0.0


def test_target_candidate_rejection_uses_effective_boundary_clearance() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.65)
    env.reset(seed=820220)
    _safe_lower, safe_upper = env._target_safe_bounds()
    env.target_position = np.array([safe_upper[0] - 0.05, 0.0, 5.0], dtype=np.float64)
    env.target_velocity = np.array([3.6, 0.0, 0.0], dtype=np.float64)
    env.target_acceleration.fill(0.0)
    candidate = {"mode": "straight_flee", "direction": np.array([1.0, 0.0, 0.0]), "speed_scale": 1.0, "route": "direct"}

    evaluated = env._evaluate_target_maneuver_candidate(
        candidate,
        env.defender_positions,
        env.defender_velocities,
    )

    assert not evaluated["feasible"]
    assert evaluated["required_boundary_clearance"] == 0.0
    assert evaluated["min_boundary"] <= 0.0
    assert evaluated["feasibility_failure"] == "boundary_clearance"


def test_unsafe_target_fallback_is_marked_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.65)
    env.reset(seed=820230)

    candidate = {
        "mode": "straight_flee",
        "direction": np.array([1.0, 0.0, 0.0]),
        "speed_scale": 1.0,
        "route": "direct",
    }
    monkeypatch.setattr(env, "_target_maneuver_candidates", lambda *_args: [candidate])

    def rejected(item: dict, *_args: np.ndarray) -> dict:
        return {
            **item,
            "score": -1000.0,
            "min_distance": 0.0,
            "terminal_distance": 0.0,
            "min_clearance": -1.0,
            "min_boundary": -1.0,
            "required_clearance": 0.6,
            "required_boundary_clearance": 0.0,
            "first_clearance": -1.0,
            "first_boundary": -1.0,
            "feasible_prefix_steps": 0,
            "feasible": False,
            "feasibility_failure": "boundary_clearance",
            "estimated_capture_time_seconds": 0.0,
            "escape_gap_rad": 0.0,
            "crossing_alignment": 0.0,
            "obstacle_avoidance_alignment": 0.0,
        }

    monkeypatch.setattr(env, "_evaluate_target_maneuver_candidate", rejected)
    env._adaptive_maneuvering_target_action()

    assert env.target_maneuver_fallback_count == 1
    assert env.target_maneuver_candidate_invalid
    assert env.target_maneuver_last_feasibility_failure == "boundary_clearance"


def test_target_invalid_episode_is_separate_from_defender_collision() -> None:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml").read_text(
            encoding="utf-8"
        )
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    env.reset(seed=820240)
    target = np.array([env.upper[0] + 1.0, 0.0, 5.0], dtype=np.float64)
    env._enforce_world_bounds(target[None, :], np.zeros((1, 3), dtype=np.float64), entity="target")
    env.target_position = target
    env.defender_positions = np.array(
        [[-8.0, -8.0, 2.0], [-8.0, 8.0, 2.0], [0.0, -8.0, 8.0], [0.0, 8.0, 8.0]],
        dtype=np.float64,
    )
    env.defender_velocities.fill(0.0)

    _observation, _reward, terminated, _truncated, info = env.step(np.zeros((4, 3)))

    assert terminated
    assert info["target_invalid_episode"]
    assert info["target_boundary_violation"]
    assert not info["defender_physical_collision"]
    assert not info["defender_safety_failure"]
    assert not info["collision"]
    assert not info["task_valid_for_policy_evaluation"]
    assert info["termination_reason"] == "target_boundary_violation"
