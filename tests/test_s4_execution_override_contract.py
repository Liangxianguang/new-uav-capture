from __future__ import annotations

from pathlib import Path

import yaml

from scripts.evaluate_s4_branching import config_for_spec
from scripts.evaluate_s4_closed_loop import apply_phase17_execution_mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_spec_execution_overrides_reach_environment_config() -> None:
    protocol = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "phase15_s4_branching_pilot.yaml").read_text(encoding="utf-8")
    )
    spec = {
        "target_speed_scale": 0.65,
        "pursuit_overrides": {},
        "execution_overrides": {
            "enabled": True,
            "action_delay_steps": 2,
            "pending_command_authority": "immutable",
            "command_noise_std": 0.08,
            "command_noise_bound_sigma": 3.0,
            "clip_command_noise": True,
            "velocity_time_constant_seconds": 0.40,
            "drag_coefficient": 0.10,
            "max_speed_scale": 1.0,
            "max_acceleration_scale": 1.0,
            "mass_scale": 1.0,
            "random_seed_offset": 104729,
            "randomize_per_episode": False,
            "max_speed_scale_range": [1.0, 1.0],
            "max_acceleration_scale_range": [1.0, 1.0],
            "mass_scale_range": [1.0, 1.0],
            "drag_coefficient_range": [0.0, 0.0],
        },
    }
    config = config_for_spec(
        PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml",
        protocol,
        spec,
        max_steps=None,
    )
    assert config["dynamics"]["execution"]["enabled"] is True
    assert config["dynamics"]["execution"]["action_delay_steps"] == 2
    assert config["dynamics"]["execution"]["command_noise_std"] == 0.08


def test_phase17_defaults_do_not_overwrite_frozen_execution_override() -> None:
    protocol = yaml.safe_load(
        (PROJECT_ROOT / "configs" / "phase15_s4_branching_pilot.yaml").read_text(encoding="utf-8")
    )
    spec = {
        "target_speed_scale": 0.65,
        "pursuit_overrides": {},
        "execution_overrides": {"enabled": True, "action_delay_steps": 2, "command_noise_std": 0.08},
    }
    config = config_for_spec(
        PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml",
        protocol,
        spec,
        max_steps=None,
    )
    apply_phase17_execution_mapping(
        config,
        {"enabled": True, "action_delay_steps": 2, "command_noise_std": 0.0},
        frozen_scene_record=spec,
    )
    assert config["dynamics"]["execution"]["command_noise_std"] == 0.08
