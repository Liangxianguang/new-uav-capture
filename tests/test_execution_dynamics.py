from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from encirclement3d.execution_dynamics import (
    CommandAuthorityDirective,
    ExecutionParameters,
    apply_command_authority,
    advance_execution,
    position_uncertainty_radii,
    rollout_execution,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv
from encirclement3d.safety_certificate import (
    check_execution_rollout_safety,
    check_execution_continuous_segment_safety,
    check_execution_swept_volume_safety,
    execution_barrier_values,
    execution_barrier_values_with_action_jacobian,
)
from encirclement3d.safety_qp import RobustCBFQPConfig, RobustCBFQPFilter


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


def test_command_authority_contract_only_replaces_authorized_queue_entries() -> None:
    queued = [
        np.full((4, 3), [1.0, 0.0, 0.0], dtype=np.float64),
        np.full((4, 3), [2.0, 0.0, 0.0], dtype=np.float64),
    ]

    immutable, directive, slots = apply_command_authority(
        queued,
        CommandAuthorityDirective(mode="immutable", emergency_brake=True),
        allowed_mode="immutable",
    )
    np.testing.assert_allclose(immutable, queued)
    assert directive.emergency_brake is True
    assert slots == 0

    tail, _directive, slots = apply_command_authority(
        queued,
        {"mode": "replace_nonexecuting", "emergency_brake": True},
        allowed_mode="replace_nonexecuting",
    )
    np.testing.assert_allclose(tail[0], queued[0])
    np.testing.assert_allclose(tail[1], 0.0)
    assert slots == 1

    flushed, _directive, slots = apply_command_authority(
        queued,
        {"mode": "flush_pending", "emergency_brake": True},
        allowed_mode="flush_pending",
    )
    np.testing.assert_allclose(flushed, 0.0)
    assert slots == 2
    with pytest.raises(ValueError, match="escalation"):
        apply_command_authority(
            queued,
            {"mode": "flush_pending", "emergency_brake": True},
            allowed_mode="immutable",
        )


def test_environment_applies_only_configured_queue_flush_authority() -> None:
    config = execution_config(
        action_delay_steps=2,
        pending_command_authority="flush_pending",
        velocity_time_constant_seconds=0.0,
        max_acceleration_scale=100.0,
    )
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    env.reset(seed=730105)
    forward = np.full((4, 3), [1.0, 0.0, 0.0], dtype=np.float64)
    env.step(forward)
    env.step(forward)
    _observation, _reward, _terminated, _truncated, info = env.step(
        np.zeros_like(forward),
        command_authority={"mode": "flush_pending", "emergency_brake": True},
    )

    np.testing.assert_allclose(env.last_delayed_actions, 0.0)
    assert info["emergency_brake_requested"] is True
    assert info["queue_override_slots"] == 2


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


def test_shared_execution_rollout_matches_delay_and_tracking_contract() -> None:
    parameters = ExecutionParameters(
        enabled=True,
        dt_seconds=0.1,
        action_delay_steps=1,
        command_noise_std_mps=0.0,
        command_noise_bound_mps=0.0,
        clip_command_noise=True,
        velocity_time_constant_seconds=0.2,
        drag_coefficient=0.0,
        max_speed_mps=5.0,
        max_acceleration_mps2=6.0,
        mass_scale=1.0,
    )
    positions = np.zeros((4, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    queued = [np.zeros_like(positions)]
    command = np.full_like(positions, [1.0, 0.0, 0.0])

    rollout_positions, rollout_velocities, steps = rollout_execution(
        positions,
        velocities,
        queued,
        command,
        parameters,
        horizon_steps=2,
    )

    np.testing.assert_allclose(steps[0].executed, 0.0)
    np.testing.assert_allclose(steps[1].executed[:, 0], 0.5)
    np.testing.assert_allclose(rollout_velocities[1, :, 0], 0.5)
    np.testing.assert_allclose(rollout_positions[1, :, 0], 0.05)
    assert position_uncertainty_radii(parameters, 4, 2).shape == (2, 4)


def test_execution_aware_qp_and_certificate_share_queue_contract() -> None:
    config = load_config()
    config["world"]["max_steps"] = 20
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    positions = np.array(
        [[-4.0, -4.0, 4.0], [-4.0, 4.0, 4.0], [4.0, -4.0, 4.0], [4.0, 4.0, 4.0]],
        dtype=np.float64,
    )
    observation = {
        "defender_positions": positions,
        "defender_velocities": np.zeros((4, 3), dtype=np.float64),
        "world_lower_bounds": np.array([-10.0, -10.0, 0.5], dtype=np.float64),
        "world_upper_bounds": np.array([10.0, 10.0, 10.0], dtype=np.float64),
        "obstacles": [],
    }
    observation.update(
        {
            "world_lower_bounds": env.lower.copy(),
            "world_upper_bounds": env.upper.copy(),
            "execution": {
                "enabled": True,
                "action_delay_steps": 1,
                "action_queue": [np.zeros((4, 3), dtype=np.float64)],
                "max_speed_mps": 5.0,
                "max_acceleration_mps2": 6.0,
                "mass_scale": 1.0,
                "drag_coefficient": 0.0,
                "velocity_time_constant_seconds": 0.2,
                "command_noise_std_mps": 0.0,
                "command_noise_bound_sigma": 3.0,
                "clip_command_noise": True,
            },
        }
    )
    qp_config = RobustCBFQPConfig(
        safety_margin_m=0.10,
        disturbance_margin_m=0.0,
        observation_error_margin_m=0.0,
        delay_margin_m=0.0,
        execution_margin_m=0.0,
        slack_enabled=False,
        fallback_policy="barrier_recovery",
    )
    desired = np.full((4, 3), [1.0, 0.0, 0.0], dtype=np.float64)
    actions, diagnostics = RobustCBFQPFilter(env, qp_config).filter(desired, observation)
    certificate = check_execution_rollout_safety(
        observation,
        actions,
        dt=env.dt,
        drone_radius=float(env.agents["drone_radius"]),
        max_speed_mps=qp_config.max_speed_mps,
        max_acceleration_mps2=qp_config.max_acceleration_mps2,
        safety_margin_m=qp_config.safety_margin_m,
        robust_margin_m=qp_config.robust_margin_m,
        action_change_limit_mps=qp_config.action_change_limit_mps,
    )

    assert diagnostics.solver_success
    assert diagnostics.certificate_valid
    assert certificate.valid
    assert certificate.rollout_state_safe


def test_execution_fallback_uses_barrier_direction_for_recovery() -> None:
    config = load_config()
    config["world"]["max_steps"] = 20
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    obstacle = {
        "shape": "cylinder",
        "center_xy": np.array([0.0, 0.0]),
        "radius": 1.0,
        "height": 10.0,
    }
    observation = {
        "defender_positions": np.array(
            [[1.30, 0.0, 4.0], [-4.0, 4.0, 4.0], [4.0, -4.0, 4.0], [4.0, 4.0, 4.0]],
            dtype=np.float64,
        ),
        "defender_velocities": np.zeros((4, 3), dtype=np.float64),
        "world_lower_bounds": env.lower.copy(),
        "world_upper_bounds": env.upper.copy(),
        "obstacles": [obstacle],
        "execution": {
            "enabled": True,
            "action_delay_steps": 0,
            "action_queue": [],
            "max_speed_mps": 5.0,
            "max_acceleration_mps2": 6.0,
            "mass_scale": 1.0,
            "drag_coefficient": 0.0,
            "velocity_time_constant_seconds": 0.0,
            "command_noise_std_mps": 0.0,
            "command_noise_bound_sigma": 3.0,
            "clip_command_noise": True,
        },
    }
    qp_config = RobustCBFQPConfig(
        safety_margin_m=0.10,
        disturbance_margin_m=0.0,
        observation_error_margin_m=0.0,
        delay_margin_m=0.0,
        execution_margin_m=0.0,
        slack_enabled=False,
        fallback_policy="barrier_recovery",
    )

    actions, diagnostics = RobustCBFQPFilter(env, qp_config).filter(
        np.zeros((4, 3), dtype=np.float64), observation
    )

    assert diagnostics.fallback_used
    assert diagnostics.recovery_action_used
    assert actions[0, 0] > 0.0
    assert diagnostics.action_correction_norm > 0.0


def test_execution_fallback_marks_independently_certified_candidate() -> None:
    config = load_config()
    config["world"]["max_steps"] = 20
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)
    observation = {
        "defender_positions": np.array(
            [[-4.0, -4.0, 4.0], [-4.0, 4.0, 4.0], [4.0, -4.0, 4.0], [4.0, 4.0, 4.0]],
            dtype=np.float64,
        ),
        "defender_velocities": np.zeros((4, 3), dtype=np.float64),
        "world_lower_bounds": env.lower.copy(),
        "world_upper_bounds": env.upper.copy(),
        "obstacles": [],
        "execution": {
            "enabled": True,
            "action_delay_steps": 0,
            "action_queue": [],
            "max_speed_mps": 5.0,
            "max_acceleration_mps2": 6.0,
            "mass_scale": 1.0,
            "drag_coefficient": 0.0,
            "velocity_time_constant_seconds": 0.0,
            "command_noise_std_mps": 0.0,
            "command_noise_bound_sigma": 3.0,
            "clip_command_noise": True,
        },
    }
    filter_instance = RobustCBFQPFilter(
        env,
        RobustCBFQPConfig(
            safety_margin_m=0.10,
            disturbance_margin_m=0.0,
            observation_error_margin_m=0.0,
            delay_margin_m=0.0,
            execution_margin_m=0.0,
            slack_enabled=False,
            fallback_policy="barrier_recovery",
        ),
    )
    actions, diagnostics = filter_instance._execution_fallback(
        np.full((4, 3), [1.0, 0.0, 0.0], dtype=np.float64),
        observation,
        0.0,
        reason="test_solver_failure",
        category="execution_rollout_infeasible",
        precondition_valid=True,
        directive=CommandAuthorityDirective(mode="immutable", emergency_brake=False),
    )

    np.testing.assert_allclose(actions, 0.0)
    assert diagnostics.fallback_used
    assert diagnostics.certificate_valid
    assert diagnostics.status == "fallback_certified"


def test_execution_swept_volume_certificate_reports_sampling_contract() -> None:
    observation = {
        "defender_positions": np.array(
            [[-4.0, -4.0, 4.0], [-4.0, 4.0, 4.0], [4.0, -4.0, 4.0], [4.0, 4.0, 4.0]],
            dtype=np.float64,
        ),
        "defender_velocities": np.zeros((4, 3), dtype=np.float64),
        "world_lower_bounds": np.array([-10.0, -10.0, 0.5], dtype=np.float64),
        "world_upper_bounds": np.array([10.0, 10.0, 10.0], dtype=np.float64),
        "obstacles": [],
        "execution": {
            "enabled": True,
            "action_delay_steps": 0,
            "action_queue": [],
            "max_speed_mps": 5.0,
            "max_acceleration_mps2": 6.0,
            "mass_scale": 1.0,
            "drag_coefficient": 0.0,
            "velocity_time_constant_seconds": 0.0,
            "command_noise_std_mps": 0.0,
            "command_noise_bound_sigma": 3.0,
            "clip_command_noise": True,
        },
    }
    certificate = check_execution_swept_volume_safety(
        observation,
        np.zeros((4, 3), dtype=np.float64),
        dt=0.1,
        drone_radius=0.25,
        safety_margin_m=0.10,
        robust_margin_m=0.0,
        horizon_steps=3,
        subdivisions_per_step=4,
    )

    assert certificate.valid
    assert certificate.swept_volume_safe
    assert certificate.sample_count == 13
    assert certificate.subdivisions_per_step == 4


def test_continuous_segment_certificate_detects_interagent_midsegment_crossing() -> None:
    observation = {
        "defender_positions": np.array([[-0.3, 0.0, 4.0], [0.3, 0.0, 4.0]], dtype=np.float64),
        "defender_velocities": np.zeros((2, 3), dtype=np.float64),
        "world_lower_bounds": np.array([-10.0, -10.0, 0.5], dtype=np.float64),
        "world_upper_bounds": np.array([10.0, 10.0, 10.0], dtype=np.float64),
        "obstacles": [],
        "execution": {
            "enabled": True,
            "action_delay_steps": 0,
            "action_queue": [],
            "max_speed_mps": 5.0,
            "max_acceleration_mps2": 100.0,
            "mass_scale": 1.0,
            "drag_coefficient": 0.0,
            "velocity_time_constant_seconds": 0.0,
            "command_noise_std_mps": 0.0,
            "command_noise_bound_sigma": 3.0,
            "clip_command_noise": True,
        },
    }
    certificate = check_execution_continuous_segment_safety(
        observation,
        np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]], dtype=np.float64),
        dt=0.1,
        drone_radius=0.1,
        safety_margin_m=0.0,
        robust_margin_m=0.0,
        horizon_steps=1,
    )

    assert not certificate.valid
    assert not certificate.continuous_segment_safe
    assert certificate.minimum_robust_barrier_m < 0.0


def test_analytic_execution_barrier_jacobian_matches_finite_difference() -> None:
    observation = {
        "defender_positions": np.array([[-4.0, -4.0, 4.0], [4.0, 4.0, 4.0]], dtype=np.float64),
        "defender_velocities": np.zeros((2, 3), dtype=np.float64),
        "world_lower_bounds": np.array([-10.0, -10.0, 0.5], dtype=np.float64),
        "world_upper_bounds": np.array([10.0, 10.0, 10.0], dtype=np.float64),
        "obstacles": [],
        "execution": {
            "enabled": True,
            "action_delay_steps": 1,
            "action_queue": [np.array([[0.2, -0.1, 0.1], [-0.2, 0.1, -0.1]], dtype=np.float64)],
            "max_speed_mps": 5.0,
            "max_acceleration_mps2": 6.0,
            "mass_scale": 1.0,
            "drag_coefficient": 0.02,
            "velocity_time_constant_seconds": 0.2,
            "command_noise_std_mps": 0.0,
            "command_noise_bound_sigma": 3.0,
            "clip_command_noise": True,
        },
    }
    action = np.array([[0.7, -0.4, 0.2], [-0.5, 0.3, -0.1]], dtype=np.float64)
    kwargs = {
        "dt": 0.1,
        "drone_radius": 0.25,
        "safety_margin_m": 0.1,
        "robust_margin_m": 0.0,
        "horizon_steps": 4,
    }
    _nominal, analytic_values, jacobian, _assumptions = execution_barrier_values_with_action_jacobian(
        observation, action, **kwargs
    )
    _nominal, reference_values, _assumptions = execution_barrier_values(observation, action, **kwargs)
    assert list(analytic_values) == list(reference_values)
    np.testing.assert_allclose(
        [analytic_values[name] for name in analytic_values],
        [reference_values[name] for name in reference_values],
        atol=1.0e-12,
    )

    finite_difference = np.zeros_like(jacobian)
    flat_action = action.reshape(-1)
    step = 1.0e-5
    names = list(analytic_values)
    for index in range(flat_action.size):
        plus = flat_action.copy()
        minus = flat_action.copy()
        plus[index] += step
        minus[index] -= step
        _nominal, plus_values, _assumptions = execution_barrier_values(
            observation, plus.reshape(action.shape), **kwargs
        )
        _nominal, minus_values, _assumptions = execution_barrier_values(
            observation, minus.reshape(action.shape), **kwargs
        )
        finite_difference[:, index] = np.asarray(
            [(plus_values[name] - minus_values[name]) / (2.0 * step) for name in names],
            dtype=np.float64,
        )
    np.testing.assert_allclose(jacobian, finite_difference, atol=2.0e-5, rtol=2.0e-5)
