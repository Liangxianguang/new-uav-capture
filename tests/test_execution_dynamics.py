from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def execution_config(**overrides: object) -> dict:
    config = load_config()
    config["dynamics"]["execution"].update({"enabled": True, **overrides})
    return config


def test_disabled_execution_preserves_ideal_velocity_contract() -> None:
    config = load_config()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    env.reset(seed=730101)
    actions = np.full((4, 3), [1.2, -0.4, 0.2], dtype=np.float64)

    _observation, _reward, _terminated, _truncated, info = env.step(actions)

    np.testing.assert_allclose(env.defender_velocities, actions)
    assert info["execution_enabled"] is False
    assert info["action_execution_error_norm"] == pytest.approx(0.0)
    assert info["mean_action_execution_error_norm"] == pytest.approx(0.0)


def test_action_delay_is_applied_before_velocity_execution() -> None:
    config = execution_config(
        action_delay_steps=2,
        velocity_time_constant_seconds=0.0,
        max_acceleration_scale=100.0,
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    env.reset(seed=730102)
    actions = np.full((4, 3), [1.0, 0.0, 0.0], dtype=np.float64)

    env.step(actions)
    np.testing.assert_allclose(env.defender_velocities, 0.0)
    env.step(actions)
    np.testing.assert_allclose(env.defender_velocities, 0.0)
    env.step(actions)
    np.testing.assert_allclose(env.defender_velocities, actions)


def test_execution_respects_acceleration_and_velocity_tracking() -> None:
    config = execution_config(
        velocity_time_constant_seconds=1.0,
        max_acceleration_scale=0.5,
        mass_scale=1.0,
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    env.reset(seed=730103)
    actions = np.full((4, 3), [5.0, 0.0, 0.0], dtype=np.float64)

    env.step(actions)
    first_speed = np.linalg.norm(env.defender_velocities, axis=1)
    assert np.all(first_speed <= 0.5 * float(config["world"]["dt"]) * float(config["agents"]["defender_max_acceleration"]) + 1e-9)
    env.step(actions)
    second_speed = np.linalg.norm(env.defender_velocities, axis=1)
    assert np.all(second_speed >= first_speed)
    assert np.all(second_speed <= float(config["agents"]["defender_max_speed"]) + 1e-9)


def test_execution_noise_and_randomization_are_seed_reproducible() -> None:
    config = execution_config(
        command_noise_std=0.15,
        randomize_per_episode=True,
        max_speed_scale_range=[0.8, 0.9],
        max_acceleration_scale_range=[0.7, 0.9],
        mass_scale_range=[0.9, 1.1],
        drag_coefficient_range=[0.0, 0.2],
    )
    first = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    second = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    first.reset(seed=730104)
    second.reset(seed=730104)
    actions = np.full((4, 3), [2.0, -0.5, 0.4], dtype=np.float64)

    first_observation, *_ = first.step(actions)
    second_observation, *_ = second.step(actions)

    np.testing.assert_allclose(first.defender_velocities, second.defender_velocities)
    assert first_observation["execution"] == second_observation["execution"]
    assert np.isfinite(first.action_execution_error_norm)
