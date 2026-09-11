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
from scripts.collect_s4_branching_dataset import episode_spec
from scripts.split_s4_branching_dataset import assign_splits


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


def test_s4_randomized_geometry_stays_valid_and_changes_with_layout_seed() -> None:
    config = load_config()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=1, target_speed_scale=0.65)
    variation = {
        "wall_half_extent_x_m": [0.45, 0.65],
        "wall_half_extent_y_m": [3.85, 4.35],
        "wall_height_m": [9.25, 9.55],
        "target_initial_x_m": [-4.25, -3.45],
        "target_initial_y_m": [-0.40, 0.40],
        "target_altitude_m": [4.30, 5.70],
        "defender_x_offset_m": [0.0, 0.35],
        "defender_y_scale": [0.92, 1.08],
        "defender_z_offset_m": [-0.30, 0.30],
    }
    first = s4_adaptive_branching_scenario(env, layout_seed=715010, defender_bias="upper", variation=variation)
    second = s4_adaptive_branching_scenario(env, layout_seed=715011, defender_bias="upper", variation=variation)
    validate_s4_branching_scenario(env, first)
    validate_s4_branching_scenario(env, second)
    assert not np.allclose(first.defender_positions, second.defender_positions)
    assert first.obstacles[0].half_extents_xy is not None
    assert second.obstacles[0].half_extents_xy is not None
    assert not np.allclose(first.obstacles[0].half_extents_xy, second.obstacles[0].half_extents_xy)


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


def test_s4_v3_episode_specs_bind_mirror_members_to_one_layout() -> None:
    collection = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "phase15_s4_branching_train_v3.yaml").read_text(encoding="utf-8")
    )
    upper = episode_spec(collection, 0)
    lower = episode_spec(collection, 1)
    assert upper["layout_seed"] == lower["layout_seed"]
    assert upper["mirror_group_id"] == lower["mirror_group_id"]
    assert upper["defender_bias"] == "upper"
    assert lower["defender_bias"] == "lower"
    assert upper["target_speed_scale"] == lower["target_speed_scale"]
    assert upper["observation_condition"] == lower["observation_condition"]
    assert upper["rollout_policy"] == lower["rollout_policy"]


def test_s4_scene_split_never_separates_mirror_groups() -> None:
    records = [
        {
            "episode_index": group * 2 + member,
            "layout_seed": 100 + group,
            "mirror_group_id": group,
            "defender_bias": "upper" if member == 0 else "lower",
        }
        for group in range(3)
        for member in range(2)
    ]
    splits, assignment = assign_splits(
        records,
        {"train": 2, "validation": 2, "locked_test": 2},
        seed=17,
    )
    assert all(len(value) == 2 for value in splits.values())
    assert len(set(assignment.values())) == 3
    for group in range(3):
        assert len({assignment[str(group)]}) == 1
