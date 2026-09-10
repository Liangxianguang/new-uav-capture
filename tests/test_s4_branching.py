from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv
from encirclement3d.showcase import (
    prepare_showcase_episode,
    s4_adaptive_branching_scenario,
    s4_branch_route_metrics,
    validate_s4_branching_scenario,
)
from scripts.collect_s4_branching_dataset import balanced_sampling_weights


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    config = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["task"]["pursuit"]["target_motion_mode"] = "adaptive_branching"
    return config


def test_s4_scenario_keeps_both_exits_conservatively_reachable() -> None:
    config = load_config()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=1, target_speed_scale=0.65)
    scenario = s4_adaptive_branching_scenario(env, layout_seed=715001, defender_bias="upper")
    validate_s4_branching_scenario(env, scenario)
    routes = s4_branch_route_metrics(env, scenario)
    assert scenario.scenario_type == "s4_branching"
    assert routes["branch_route_feasible"] is True
    assert routes["target_branch_route_feasible"] == {"lower": True, "upper": True}
    assert all(all(values) for values in routes["defender_branch_route_feasible"].values())


def test_s4_branch_selection_flips_for_mirrored_defender_pressure_without_observation_leak() -> None:
    config = load_config()
    signs: list[int] = []
    for bias in ("upper", "lower"):
        env = CaptureRadiusPursuit3DEnv(config, obstacle_count=1, target_speed_scale=0.65)
        scenario = s4_adaptive_branching_scenario(env, layout_seed=715002, defender_bias=bias)
        observation = prepare_showcase_episode(env, scenario, seed=715102)
        assert not any("branch" in key for key in observation)
        env.target_position[0] = float(env.pursuit["target_branch_decision_x"])
        action = env._target_action()
        assert np.isfinite(action).all()
        assert np.linalg.norm(action) <= float(env.agents["target_max_speed"]) * env.target_speed_scale + 1e-9
        assert env.target_branch_sign in {-1, 1}
        signs.append(int(env.target_branch_sign))
    assert signs == [-1, 1]


def test_s4_target_rollout_remains_speed_bounded_and_deterministic() -> None:
    config = load_config()
    first = CaptureRadiusPursuit3DEnv(config, obstacle_count=1, target_speed_scale=0.75)
    second = CaptureRadiusPursuit3DEnv(config, obstacle_count=1, target_speed_scale=0.75)
    scenario = s4_adaptive_branching_scenario(first, layout_seed=715003, defender_bias="upper")
    prepare_showcase_episode(first, scenario, seed=715103)
    prepare_showcase_episode(second, scenario, seed=715103)
    for _ in range(120):
        first.step(np.zeros((4, 3)))
        second.step(np.zeros((4, 3)))
        np.testing.assert_allclose(first.target_position, second.target_position)
        assert np.linalg.norm(first.target_velocity) <= float(first.agents["target_max_speed"]) + 1e-9
        assert np.all(first.target_position >= first.lower - 1e-9)
        assert np.all(first.target_position <= first.upper + 1e-9)
        if first.target_branch_sign is not None:
            break
    assert first.target_branch_sign == -1


def test_s4_balanced_sampler_equalizes_observed_policy_branch_strata() -> None:
    weights, counts = balanced_sampling_weights(
        np.array([0, 0, 0, 1, 1], dtype=np.int8),
        np.array([-1, -1, 1, -1, -1], dtype=np.int8),
    )
    assert counts == {"0": 2, "1": 1, "10": 2}
    assert np.mean(weights) == 1.0
    strata = np.array([0, 0, 1, 10, 10])
    masses = [float(weights[strata == key].sum()) for key in (0, 1, 10)]
    np.testing.assert_allclose(masses, np.full(3, masses[0]))
