from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import yaml

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml"


def _config() -> dict:
    config = yaml.safe_load(ENV_CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["dynamics"]["execution"].update(
        {
            "enabled": True,
            "action_delay_steps": 3,
            "pending_command_authority": "replace_nonexecuting",
            "queue_token_contract_enabled": True,
            "queue_token_max_override_slots": 1,
        }
    )
    config["world"]["max_steps"] = 10
    return config


def test_environment_exposes_and_consumes_queue_token_ack() -> None:
    env = CaptureRadiusPursuit3DEnv(_config(), obstacle_count=1, target_speed_scale=0.5)
    observation = env.reset(seed=20260914)
    token = observation["execution"]["queue_token"]
    assert token["pending_length"] == 3
    assert observation["execution"]["queue_token_contract_enabled"] is True

    action = np.zeros((4, 3), dtype=np.float64)
    next_observation, _reward, _terminated, _truncated, info = env.step(
        action,
        command_authority={
            "mode": "replace_nonexecuting",
            "emergency_brake": True,
            "expected_queue_token": token,
        },
    )
    ack = info["queue_authority_ack"]
    assert ack["accepted"] is True
    assert ack["applied"] is True
    assert ack["overridden_slots"] == 1
    assert ack["reason"] == "bounded_recovery_applied"
    assert next_observation["execution"]["queue_token"]["generation"] == 2

    _next, _reward, _terminated, _truncated, stale_info = env.step(
        action,
        command_authority={
            "mode": "replace_nonexecuting",
            "emergency_brake": True,
            "expected_queue_token": token,
        },
    )
    stale_ack = stale_info["queue_authority_ack"]
    assert stale_ack["accepted"] is False
    assert stale_ack["applied"] is False
    assert stale_ack["reason"] == "stale_queue_token"
