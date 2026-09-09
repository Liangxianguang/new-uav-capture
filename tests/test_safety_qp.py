from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv
from encirclement3d.safety_certificate import check_one_step_safety
from encirclement3d.safety_qp import RobustCBFQPConfig, RobustCBFQPFilter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_CONFIG = PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml"


def _env() -> CaptureRadiusPursuit3DEnv:
    config = yaml.safe_load(ENV_CONFIG.read_text(encoding="utf-8"))
    config["world"]["max_steps"] = 20
    return CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.1)


def _observation(
    positions: np.ndarray,
    *,
    velocities: np.ndarray | None = None,
    obstacles: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    position_array = np.asarray(positions, dtype=np.float64)
    return {
        "defender_positions": position_array,
        "defender_velocities": (
            np.zeros_like(position_array)
            if velocities is None
            else np.asarray(velocities, dtype=np.float64)
        ),
        "world_lower_bounds": np.array([-10.0, -10.0, 0.5], dtype=np.float64),
        "world_upper_bounds": np.array([10.0, 10.0, 10.0], dtype=np.float64),
        "obstacles": [] if obstacles is None else obstacles,
    }


def _filter_config(**overrides: object) -> RobustCBFQPConfig:
    values: dict[str, object] = {
        "gamma": 0.25,
        "safety_margin_m": 0.10,
        "disturbance_margin_m": 0.0,
        "observation_error_margin_m": 0.0,
        "delay_margin_m": 0.0,
        "execution_margin_m": 0.0,
        "max_speed_mps": 5.0,
        "max_acceleration_mps2": 6.0,
        "slack_weight": 1.0e8,
        "solver_tolerance": 1.0e-7,
    }
    values.update(overrides)
    return RobustCBFQPConfig(**values)


def test_execution_projection_recovers_feasible_linearized_point_after_dykstra_limit() -> None:
    env = _env()
    filter_instance = RobustCBFQPFilter(
        env,
        _filter_config(execution_projection_iterations=1),
    )
    target = np.zeros(2, dtype=np.float64)
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]], dtype=np.float64)
    lower_rhs = np.array([1.0, 1.0, 3.0, 2.0], dtype=np.float64)
    action, success, iterations, message = filter_instance._project_linearized_halfspaces(
        target,
        matrix,
        lower_rhs,
        np.full(2, -10.0),
        np.full(2, 10.0),
    )

    assert success
    assert message == "highs_linearized_feasibility_recovery"
    assert iterations >= 1
    assert np.min(matrix @ action - lower_rhs) >= -1.0e-7


def test_execution_projection_uses_euclidean_speed_ball_not_inscribed_box() -> None:
    env = _env()
    filter_instance = RobustCBFQPFilter(
        env,
        _filter_config(enforce_action_change=False, max_speed_mps=5.0),
    )
    velocities = np.zeros((4, 3), dtype=np.float64)
    lower, upper = filter_instance._execution_action_bounds(velocities)

    np.testing.assert_allclose(lower, -5.0)
    np.testing.assert_allclose(upper, 5.0)
    target = np.zeros(12, dtype=np.float64)
    target[:3] = [5.0, 5.0, 0.0]
    action, success, _iterations, _message = filter_instance._project_linearized_halfspaces(
        target,
        np.empty((0, 12), dtype=np.float64),
        np.empty(0, dtype=np.float64),
        lower,
        upper,
        speed_max_mps=5.0,
    )

    assert success
    assert np.linalg.norm(action[:3]) == pytest.approx(5.0)
    assert np.max(np.linalg.norm(action.reshape(-1, 3), axis=1)) <= 5.0 + 1.0e-9


def test_qp_preserves_nominal_action_when_all_barriers_are_inactive() -> None:
    env = _env()
    observation = _observation(
        np.array(
            [[-4.0, -4.0, 4.0], [-4.0, 4.0, 4.0], [4.0, -4.0, 4.0], [4.0, 4.0, 4.0]],
        )
    )
    desired = np.array([[0.2, 0.1, 0.0], [-0.1, 0.2, 0.0], [0.1, -0.2, 0.0], [-0.2, -0.1, 0.0]])
    actions, diagnostics = RobustCBFQPFilter(env, _filter_config()).filter(desired, observation)

    assert diagnostics.solver_success
    assert not diagnostics.fallback_used
    np.testing.assert_allclose(actions, desired, atol=1.0e-5)
    assert diagnostics.certificate_valid


def test_qp_projects_action_away_from_obstacle() -> None:
    env = _env()
    obstacle = {
        "shape": "cylinder",
        "center_xy": np.array([0.0, 0.0]),
        "radius": 1.0,
        "height": 10.0,
    }
    observation = _observation(
        np.array([[1.40, 3.0, 4.0], [3.0, 3.0, 4.0], [3.0, -3.0, 4.0], [-3.0, -3.0, 4.0]]),
        obstacles=[obstacle],
    )
    desired = np.array([[-5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    actions, diagnostics = RobustCBFQPFilter(env, _filter_config()).filter(desired, observation)
    certificate = check_one_step_safety(
        observation,
        actions,
        dt=env.dt,
        drone_radius=0.25,
        max_speed_mps=5.0,
        max_acceleration_mps2=6.0,
        safety_margin_m=0.10,
        robust_margin_m=0.0,
    )

    assert diagnostics.solver_success
    assert actions[0, 0] > desired[0, 0]
    assert certificate.valid
    assert certificate.next_min_barrier_m >= -1.0e-6


def test_qp_projects_action_inside_world_boundary() -> None:
    env = _env()
    observation = _observation(
        np.array([[9.64, -4.0, 4.0], [-4.0, 4.0, 4.0], [4.0, -4.0, 4.0], [4.0, 4.0, 4.0]])
    )
    desired = np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    actions, diagnostics = RobustCBFQPFilter(env, _filter_config()).filter(desired, observation)
    certificate = check_one_step_safety(
        observation,
        actions,
        dt=env.dt,
        drone_radius=0.25,
        max_speed_mps=5.0,
        max_acceleration_mps2=6.0,
        safety_margin_m=0.10,
        robust_margin_m=0.0,
    )

    assert diagnostics.solver_success
    assert actions[0, 0] < desired[0, 0]
    assert certificate.valid
    assert certificate.next_min_barrier_m >= -1.0e-6


def test_qp_accepts_legal_axis_aligned_speed_under_euclidean_bound() -> None:
    env = _env()
    observation = _observation(
        np.array(
            [[-4.0, -4.0, 4.0], [-4.0, 4.0, 4.0], [4.0, -4.0, 4.0], [4.0, 4.0, 4.0]],
        ),
        velocities=np.array(
            [[4.4, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ),
    )
    desired = np.array(
        [[5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    actions, diagnostics = RobustCBFQPFilter(env, _filter_config()).filter(desired, observation)

    assert diagnostics.solver_success
    assert not diagnostics.fallback_used
    assert float(np.linalg.norm(actions[0])) <= 5.0 + 1.0e-6
    np.testing.assert_allclose(actions[0], desired[0], atol=1.0e-5)


def test_velocity_level_filter_can_disable_unmodelled_action_change_limit() -> None:
    env = _env()
    observation = _observation(
        np.array(
            [[-4.0, -4.0, 4.0], [-4.0, 4.0, 4.0], [4.0, -4.0, 4.0], [4.0, 4.0, 4.0]],
        ),
        velocities=np.array(
            [[4.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ),
    )
    desired = np.array(
        [[-4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    actions, diagnostics = RobustCBFQPFilter(
        env,
        _filter_config(enforce_action_change=False),
    ).filter(desired, observation)

    assert diagnostics.solver_success
    assert not diagnostics.fallback_used
    np.testing.assert_allclose(actions, desired, atol=1.0e-5)


def test_qp_separates_close_defenders() -> None:
    env = _env()
    observation = _observation(
        np.array([[0.0, 0.0, 4.0], [0.80, 0.0, 4.0], [4.0, 4.0, 4.0], [-4.0, -4.0, 4.0]])
    )
    desired = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    actions, diagnostics = RobustCBFQPFilter(env, _filter_config()).filter(desired, observation)
    certificate = check_one_step_safety(
        observation,
        actions,
        dt=env.dt,
        drone_radius=0.25,
        max_speed_mps=5.0,
        max_acceleration_mps2=6.0,
        safety_margin_m=0.10,
        robust_margin_m=0.0,
    )

    assert diagnostics.solver_success
    assert actions[0, 0] < desired[0, 0]
    assert actions[1, 0] > desired[1, 0]
    assert certificate.valid


def test_qp_fallback_is_explicit_for_inconsistent_action_bounds() -> None:
    env = _env()
    observation = _observation(
        np.array([[-4.0, -4.0, 4.0], [-4.0, 4.0, 4.0], [4.0, -4.0, 4.0], [4.0, 4.0, 4.0]]),
        velocities=np.full((4, 3), 100.0),
    )
    desired = np.ones((4, 3), dtype=np.float64)
    actions, diagnostics = RobustCBFQPFilter(env, _filter_config()).filter(desired, observation)

    assert diagnostics.fallback_used
    assert not diagnostics.certificate_valid
    assert diagnostics.status == "fallback"
    np.testing.assert_allclose(actions, 0.0)


def test_certificate_rejects_action_that_leaves_safe_set() -> None:
    observation = _observation(
        np.array([[9.64, -4.0, 4.0], [-4.0, 4.0, 4.0], [4.0, -4.0, 4.0], [4.0, 4.0, 4.0]])
    )
    result = check_one_step_safety(
        observation,
        np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        dt=0.1,
        drone_radius=0.25,
        max_speed_mps=5.0,
        max_acceleration_mps2=6.0,
        safety_margin_m=0.10,
        robust_margin_m=0.0,
    )

    assert not result.valid
    assert "next_state_outside_safe_set" in result.violations


def test_config_rejects_invalid_fallback_policy() -> None:
    with pytest.raises(ValueError, match="fallback_policy"):
        RobustCBFQPConfig(fallback_policy="ignore")


def test_config_records_full_horizon_brake_policy_explicitly() -> None:
    enabled = RobustCBFQPConfig(execution_emergency_brake_on_full_horizon_failure=True)
    disabled = RobustCBFQPConfig(execution_emergency_brake_on_full_horizon_failure=False)

    assert enabled.execution_emergency_brake_on_full_horizon_failure is True
    assert disabled.execution_emergency_brake_on_full_horizon_failure is False


def test_config_rejects_unknown_execution_linearization_backend() -> None:
    with pytest.raises(ValueError, match="execution_linearization_backend"):
        RobustCBFQPConfig(execution_linearization_backend="unknown")


def test_config_rejects_negative_execution_active_margin() -> None:
    with pytest.raises(ValueError, match="execution_linearization_active_margin_m"):
        RobustCBFQPConfig(execution_linearization_active_margin_m=-0.1)


def test_qp_diagnostics_persist_barriers_and_residuals() -> None:
    env = _env()
    observation = _observation(
        np.array(
            [[-4.0, -4.0, 4.0], [-4.0, 4.0, 4.0], [4.0, -4.0, 4.0], [4.0, 4.0, 4.0]],
        )
    )
    desired = np.zeros((4, 3), dtype=np.float64)
    _actions, diagnostics = RobustCBFQPFilter(env, _filter_config()).filter(desired, observation)

    assert diagnostics.failure_category == "none"
    assert diagnostics.precondition_valid
    assert diagnostics.constraint_count == len(diagnostics.barrier_values_m or {})
    assert diagnostics.constraint_count == len(diagnostics.constraint_residuals_m or {})
    assert diagnostics.minimum_constraint_residual >= -1.0e-7


def test_precondition_failure_is_distinguished_from_solver_failure() -> None:
    env = _env()
    observation = _observation(
        np.array(
            [[-4.0, -4.0, 1.1], [-4.0, 4.0, 4.0], [4.0, -4.0, 4.0], [4.0, 4.0, 4.0]],
        )
    )
    config = _filter_config(
        disturbance_margin_m=0.10,
        observation_error_margin_m=0.06,
        delay_margin_m=0.20,
        execution_margin_m=0.10,
        fallback_policy="barrier_recovery",
    )
    _actions, diagnostics = RobustCBFQPFilter(env, config).filter(np.zeros((4, 3)), observation)

    assert diagnostics.fallback_used
    assert diagnostics.failure_category == "precondition_invalid"
    assert not diagnostics.precondition_valid
    assert diagnostics.recovery_action_used
    assert diagnostics.fallback_reason is not None
    assert diagnostics.solver_success is False
    assert diagnostics.solver_message.startswith("current_state_outside_robust_safe_set")
