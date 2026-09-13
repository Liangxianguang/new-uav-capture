from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from scripts.evaluate_s4_branching import config_for_spec
from scripts.evaluate_s4_closed_loop import (
    apply_execution_cli_overrides,
    apply_phase17_execution_mapping,
    configure_torch_threads,
)


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


def test_execution_cli_overrides_are_recordable_and_enable_execution() -> None:
    effective = apply_execution_cli_overrides(
        {"enabled": False, "action_delay_steps": 2, "command_noise_std": 0.0},
        delay_steps=4,
        noise_std_mps=0.12,
        noise_bound_sigma=2.5,
        tracking_time_constant_s=0.4,
        drag_coefficient=0.1,
    )
    assert effective == {
        "enabled": True,
        "action_delay_steps": 4,
        "command_noise_std": 0.12,
        "command_noise_bound_sigma": 2.5,
        "velocity_time_constant_seconds": 0.4,
        "drag_coefficient": 0.1,
    }


def test_execution_cli_overrides_reject_negative_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        apply_execution_cli_overrides({}, noise_std_mps=-0.01)


def test_torch_thread_configuration_validates_and_records_observed_values(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: calls.append(("interop", value)))
    monkeypatch.setattr(torch, "set_num_threads", lambda value: calls.append(("intra", value)))
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)

    observed = configure_torch_threads(1, 1)

    assert calls == [("interop", 1), ("intra", 1)]
    assert observed == {"torch_num_threads": 1, "torch_num_interop_threads": 1}


def test_torch_thread_configuration_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        configure_torch_threads(0, None)
