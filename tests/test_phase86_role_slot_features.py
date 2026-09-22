"""Tests for the development-only fixed role/slot observation extension."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from encirclement3d.observation_encoding import policy_observations
from encirclement3d.pursuit_controllers import PublicBeliefRouteIntentController
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = PROJECT_ROOT / "configs" / "phase85_target_contract_repaired_environment.yaml"


def _environment(*, role_slots: bool) -> dict:
    config = yaml.safe_load(ENVIRONMENT.read_text(encoding="utf-8"))
    config["task"]["policy_obstacle_geometry"] = "shape_extents_and_type"
    config["task"]["pursuit"]["include_prediction_features"] = True
    if role_slots:
        config["task"]["policy_role_slot_features"] = True
        config["task"]["policy_role_slot_radius_m"] = 1.25
    return config


def test_role_slot_features_are_opt_in_and_append_seven_values() -> None:
    plain = CaptureRadiusPursuit3DEnv(_environment(role_slots=False), obstacle_count=3, target_speed_scale=0.0)
    role = CaptureRadiusPursuit3DEnv(_environment(role_slots=True), obstacle_count=3, target_speed_scale=0.0)
    plain_observation = plain.reset(seed=861201)
    role_observation = role.reset(seed=861201)

    plain_values = policy_observations(plain, plain_observation)
    role_values = policy_observations(role, role_observation)

    assert role_values.shape == (4, plain_values.shape[1] + 7)
    np.testing.assert_allclose(role_values[:, : plain_values.shape[1]], plain_values)
    np.testing.assert_allclose(role_values[:, -4:], np.eye(4, dtype=np.float32))
    assert np.isfinite(role_values).all()


def test_role_slot_features_do_not_read_hidden_target_truth() -> None:
    config = _environment(role_slots=True)
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.0)
    observation = env.reset(seed=861202)
    baseline = policy_observations(env, observation)
    env.target_position += np.array([4.0, -3.0, 2.0])
    repeated = policy_observations(env, env.observe())
    np.testing.assert_allclose(repeated, baseline)


def test_route_teacher_accepts_a_fixed_interceptor_for_warmup() -> None:
    env = CaptureRadiusPursuit3DEnv(_environment(role_slots=True), obstacle_count=3, target_speed_scale=0.05)
    observation = env.reset(seed=861203)
    teacher = PublicBeliefRouteIntentController(env, interceptor_id=0)
    action = teacher.act(observation)
    assert action.shape == (4, 3)
    assert np.isfinite(action).all()
