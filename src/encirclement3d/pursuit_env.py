"""Partially observable 3D capture-radius pursuit-evasion environment.

This module is intentionally separate from the tetrahedral containment
benchmark. A capture is a geometric event: one defender reaches the configured
capture radius without a safety failure. The policy observation never exposes
the target's true state; the target belief is formed only from local detections
and delayed teammate messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from encirclement3d.execution_dynamics import (
    apply_command_authority,
    apply_command_authority_with_token,
    advance_execution,
    clip_rows,
    command_authority_directive,
    commit_delayed_command_with_token,
    move_toward_velocity,
    parameters_from_observation,
    QueueToken,
    validate_command_authority_mode,
)

TETRAHEDRON_DIRECTIONS = np.array(
    [
        [1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ],
    dtype=np.float64,
)
TETRAHEDRON_DIRECTIONS /= np.linalg.norm(TETRAHEDRON_DIRECTIONS, axis=1, keepdims=True)


@dataclass(frozen=True)
class CylinderObstacle:
    """Static obstacle used by the pursuit benchmark.

    The historical name is retained for compatibility.  ``shape='cylinder'``
    uses ``radius``; ``shape='box'`` additionally uses ``half_extents_xy``.
    Keeping one observation-compatible record lets the benchmark introduce
    boxes and wall segments without changing the actor input dimension.
    """

    center_xy: np.ndarray
    radius: float
    height: float
    shape: str = "cylinder"
    half_extents_xy: np.ndarray | None = None


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9:
        return vector / norm
    if fallback is None:
        return np.zeros_like(vector)
    return fallback.copy()


_PURSUIT_DEFAULTS: dict[str, Any] = {
    "capture_radius": 0.80,
    "spawn_distance": 4.80,
    "detection_range": 7.50,
    "visibility_cosine_threshold": -1.0,
    "detection_dropout_probability": 0.15,
    "observation_noise_std": 0.03,
    "message_delay_steps": 2,
    "message_dropout_probability": 0.05,
    "communication_link_dropout_probability": 0.0,
    "maximum_message_age_steps": 60,
    "observation_delay_steps": 0,
    "belief_update_mode": "legacy",
    "belief_stale_velocity_decay": 1.0,
    "belief_velocity_decay_start_age_steps": 0,
    "detection_loss_burst_probability": 0.0,
    "detection_loss_burst_duration_steps": 1,
    "observation_confidence_decay": 0.97,
    "observation_covariance_growth": 0.25,
    "include_prediction_features": False,
    "include_uncertainty_features": False,
    "prediction_horizon_seconds": 0.55,
    "prediction_uncertainty_base": 0.08,
    "target_defender_avoidance_distance": 2.20,
    "target_defender_avoidance_gain": 4.00,
    "target_obstacle_avoidance_distance": 2.20,
    "target_obstacle_avoidance_gain": 3.50,
    "target_boundary_margin": 1.20,
    "target_boundary_gain": 7.00,
    "target_heading_persistence": 0.55,
    "target_motion_mode": "flee_persistence",
    "target_turn_interval_steps": 12,
    "target_s_curve_amplitude": 0.85,
    "target_s_curve_frequency": 0.18,
    "target_burst_period_steps": 30,
    "target_burst_duration_steps": 8,
    "target_burst_speed_scale": 1.25,
    "target_adaptive_heading_weight": 0.20,
    "target_adaptive_obstacle_weight": 4.00,
    "target_adaptive_boundary_weight": 2.00,
    "target_adaptive_defender_weight": 1.00,
    "target_adaptive_lookahead_steps": 8,
    # Maneuvering Adversary v2.  These settings control a hidden, physically
    # constrained target controller.  It observes delayed/noisy defender
    # tracks rather than simulator truth; ``adaptive_adversarial`` remains the
    # explicit oracle stress-test mode.
    "target_maneuver_observation_delay_steps": 2,
    "target_maneuver_observation_noise_std": 0.12,
    "target_maneuver_observation_dropout_probability": 0.10,
    "target_maneuver_replan_interval_steps": 8,
    "target_maneuver_min_hold_steps": 6,
    "target_maneuver_horizon_steps": 12,
    "target_maneuver_max_turn_rate_rad_s": 1.05,
    "target_maneuver_max_jerk_mps3": 12.0,
    "target_maneuver_route_margin_m": 1.00,
    "target_maneuver_route_boundary_buffer_m": 0.50,
    "target_maneuver_feasibility_margin_m": 0.60,
    "target_maneuver_distance_weight": 1.00,
    "target_maneuver_terminal_weight": 1.50,
    "target_maneuver_clearance_weight": 2.50,
    "target_maneuver_boundary_weight": 1.50,
    "target_maneuver_gap_weight": 0.75,
    "target_maneuver_smoothness_weight": 0.25,
    "target_maneuver_switch_bonus": 0.08,
    "target_maneuver_burst_speed_scale": 1.15,
    "target_maneuver_boundary_recovery_speed_scale": 0.75,
    "target_maneuver_predictive_boundary_recovery": False,
    "target_maneuver_safety_first_fallback": False,
    "target_maneuver_predictive_boundary_lookahead_steps": 12,
    "target_maneuver_crossing_gain": 0.0,
    "target_maneuver_obstacle_avoidance_gain": 0.0,
    "target_maneuver_enable_reverse_lane_change": True,
    # S4 uses a committed, geometry-aware exit selection. The selected exit
    # is simulator-private: it is never included in ``observe``.
    "target_branch_decision_x": -3.20,
    "target_branch_exit_offset_y": 5.30,
    "target_branch_waypoint_x": 0.85,
    "target_branch_goal_x": 7.50,
    "target_branch_defender_lookahead_seconds": 0.60,
    "target_branch_commit_margin_seconds": 0.0,
    "target_flee_gain": 1.00,
    "target_vertical_gain": 0.20,
    "controller_obstacle_avoidance_distance": 2.00,
    "controller_obstacle_avoidance_gain": 2.80,
    "controller_inter_agent_distance": 1.00,
    "controller_inter_agent_gain": 1.20,
    "safety_margin": 0.20,
    "capture_bonus": 25.00,
    "collision_penalty": 15.00,
    "progress_reward_weight": 3.00,
    "distance_reward_weight": 0.12,
    "coverage_reward_weight": 0.15,
    "defender_boundary_margin": 1.25,
    "defender_boundary_proximity_weight": 1.50,
    "defender_boundary_progress_weight": 0.80,
    "max_observation_obstacles": 3,
    "obstacle_profile": "cylinders",
    "map_seed_offset": 0,
}

# This is intentionally a lightweight execution model, not a replacement for
# a flight controller or a full airframe model. It lets the pursuit benchmark
# test whether a policy that was trained with ideal velocity commands remains
# safe when commands are delayed, noisy, and tracked imperfectly.
_EXECUTION_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "random_seed_offset": 104729,
    "action_delay_steps": 0,
    "pending_command_authority": "immutable",
    "queue_token_contract_enabled": False,
    "queue_token_max_override_slots": None,
    "command_noise_std": 0.0,
    "command_noise_bound_sigma": 3.0,
    "clip_command_noise": True,
    "velocity_time_constant_seconds": 0.0,
    "drag_coefficient": 0.0,
    "max_speed_scale": 1.0,
    "max_acceleration_scale": 1.0,
    "mass_scale": 1.0,
    "randomize_per_episode": False,
    "max_speed_scale_range": (1.0, 1.0),
    "max_acceleration_scale_range": (1.0, 1.0),
    "mass_scale_range": (1.0, 1.0),
    "drag_coefficient_range": (0.0, 0.0),
}

_TARGET_MOTION_MODES = {
    "flee_persistence",
    "random_turn",
    "s_curve",
    "burst",
    "boundary_escape",
    "adaptive_adversarial",
    "adaptive_branching",
    "adaptive_maneuvering",
}
_MANEUVER_MODES = {
    "straight_flee",
    "lateral_jink",
    "obstacle_bypass",
    "reverse_lane_change",
    "vertical_escape",
    "speed_burst",
    "boundary_recovery",
    "obstacle_recovery",
}
_OBSTACLE_PROFILES = {"cylinders", "boxes", "walls", "narrow_channels", "mixed"}
_BELIEF_UPDATE_MODES = {"legacy", "zero_velocity", "constant_velocity", "time_aligned"}


def pursuit_settings(task: dict[str, Any]) -> dict[str, Any]:
    configured = task.get("pursuit", {})
    if not isinstance(configured, dict):
        raise ValueError("task.pursuit must be a mapping.")
    unknown = sorted(set(configured).difference(_PURSUIT_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown task.pursuit settings: {', '.join(unknown)}")
    settings = {**_PURSUIT_DEFAULTS, **configured}
    positive = {
        "capture_radius",
        "spawn_distance",
        "detection_range",
        "target_defender_avoidance_distance",
        "target_obstacle_avoidance_distance",
        "target_boundary_margin",
        "controller_obstacle_avoidance_distance",
        "controller_inter_agent_distance",
        "max_observation_obstacles",
        "prediction_horizon_seconds",
    }
    for name in positive:
        if float(settings[name]) <= 0.0:
            raise ValueError(f"task.pursuit.{name} must be positive.")
    if not -1.0 <= float(settings["visibility_cosine_threshold"]) <= 1.0:
        raise ValueError("task.pursuit.visibility_cosine_threshold must be in [-1, 1].")
    for name in (
        "message_dropout_probability",
        "communication_link_dropout_probability",
        "detection_dropout_probability",
        "detection_loss_burst_probability",
    ):
        if not 0.0 <= float(settings[name]) < 1.0:
            raise ValueError(f"task.pursuit.{name} must be in [0, 1).")
    if (
        int(settings["message_delay_steps"]) < 0
        or int(settings["observation_delay_steps"]) < 0
        or int(settings["maximum_message_age_steps"]) <= 0
    ):
        raise ValueError("Message/observation delays must be non-negative and maximum message age must be positive.")
    if int(settings["detection_loss_burst_duration_steps"]) <= 0:
        raise ValueError("task.pursuit.detection_loss_burst_duration_steps must be positive.")
    if not 0.0 < float(settings["observation_confidence_decay"]) <= 1.0:
        raise ValueError("task.pursuit.observation_confidence_decay must be in (0, 1].")
    if float(settings["observation_covariance_growth"]) < 0.0:
        raise ValueError("task.pursuit.observation_covariance_growth must be non-negative.")
    if float(settings["prediction_uncertainty_base"]) < 0.0:
        raise ValueError("task.pursuit.prediction_uncertainty_base must be non-negative.")
    if not 0.0 <= float(settings["belief_stale_velocity_decay"]) <= 1.0:
        raise ValueError("task.pursuit.belief_stale_velocity_decay must be in [0, 1].")
    if int(settings["belief_velocity_decay_start_age_steps"]) < 0:
        raise ValueError("task.pursuit.belief_velocity_decay_start_age_steps must be non-negative.")
    if str(settings["target_motion_mode"]) not in _TARGET_MOTION_MODES:
        raise ValueError(
            "task.pursuit.target_motion_mode must be one of "
            + ", ".join(sorted(_TARGET_MOTION_MODES))
            + "."
        )
    if str(settings["obstacle_profile"]) not in _OBSTACLE_PROFILES:
        raise ValueError(
            "task.pursuit.obstacle_profile must be one of "
            + ", ".join(sorted(_OBSTACLE_PROFILES))
            + "."
        )
    if str(settings["belief_update_mode"]) not in _BELIEF_UPDATE_MODES:
        raise ValueError(
            "task.pursuit.belief_update_mode must be one of "
            + ", ".join(sorted(_BELIEF_UPDATE_MODES))
            + "."
        )
    for name in (
        "target_turn_interval_steps",
        "target_burst_period_steps",
        "target_burst_duration_steps",
    ):
        if int(settings[name]) <= 0:
            raise ValueError(f"task.pursuit.{name} must be positive.")
    for name in (
        "target_s_curve_amplitude",
        "target_s_curve_frequency",
        "target_burst_speed_scale",
        "target_adaptive_heading_weight",
        "target_adaptive_obstacle_weight",
        "target_adaptive_boundary_weight",
        "target_adaptive_defender_weight",
        "target_branch_defender_lookahead_seconds",
        "target_branch_commit_margin_seconds",
    ):
        if float(settings[name]) < 0.0:
            raise ValueError(f"task.pursuit.{name} must be non-negative.")
    if int(settings["target_adaptive_lookahead_steps"]) <= 0:
        raise ValueError("task.pursuit.target_adaptive_lookahead_steps must be positive.")
    if int(settings["target_maneuver_observation_delay_steps"]) < 0:
        raise ValueError("task.pursuit.target_maneuver_observation_delay_steps must be non-negative.")
    if not 0.0 <= float(settings["target_maneuver_observation_dropout_probability"]) < 1.0:
        raise ValueError("task.pursuit.target_maneuver_observation_dropout_probability must be in [0, 1).")
    if int(settings["target_maneuver_replan_interval_steps"]) not in range(6, 11):
        raise ValueError("task.pursuit.target_maneuver_replan_interval_steps must be in [6, 10].")
    if int(settings["target_maneuver_min_hold_steps"]) < 1:
        raise ValueError("task.pursuit.target_maneuver_min_hold_steps must be positive.")
    if int(settings["target_maneuver_predictive_boundary_lookahead_steps"]) <= 0:
        raise ValueError("task.pursuit.target_maneuver_predictive_boundary_lookahead_steps must be positive.")
    if not 8 <= int(settings["target_maneuver_horizon_steps"]) <= 16:
        raise ValueError("task.pursuit.target_maneuver_horizon_steps must be in [8, 16].")
    if float(settings["target_maneuver_max_turn_rate_rad_s"]) <= 0.0:
        raise ValueError("task.pursuit.target_maneuver_max_turn_rate_rad_s must be positive.")
    if float(settings["target_maneuver_max_jerk_mps3"]) <= 0.0:
        raise ValueError("task.pursuit.target_maneuver_max_jerk_mps3 must be positive.")
    for name in (
        "target_maneuver_observation_noise_std",
        "target_maneuver_route_margin_m",
        "target_maneuver_route_boundary_buffer_m",
        "target_maneuver_feasibility_margin_m",
        "target_maneuver_distance_weight",
        "target_maneuver_terminal_weight",
        "target_maneuver_clearance_weight",
        "target_maneuver_boundary_weight",
        "target_maneuver_gap_weight",
        "target_maneuver_smoothness_weight",
        "target_maneuver_switch_bonus",
        "target_maneuver_burst_speed_scale",
        "target_maneuver_boundary_recovery_speed_scale",
        "target_maneuver_crossing_gain",
        "target_maneuver_obstacle_avoidance_gain",
    ):
        if float(settings[name]) < 0.0:
            raise ValueError(f"task.pursuit.{name} must be non-negative.")
    if float(settings["target_maneuver_burst_speed_scale"]) <= 0.0:
        raise ValueError("task.pursuit.target_maneuver_burst_speed_scale must be positive.")
    if float(settings["target_maneuver_boundary_recovery_speed_scale"]) <= 0.0:
        raise ValueError("task.pursuit.target_maneuver_boundary_recovery_speed_scale must be positive.")
    if not isinstance(settings["target_maneuver_enable_reverse_lane_change"], bool):
        raise ValueError("task.pursuit.target_maneuver_enable_reverse_lane_change must be boolean.")
    for name in (
        "target_maneuver_predictive_boundary_recovery",
        "target_maneuver_safety_first_fallback",
    ):
        if not isinstance(settings[name], bool):
            raise ValueError(f"task.pursuit.{name} must be boolean.")
    if float(settings["target_branch_exit_offset_y"]) <= 0.0:
        raise ValueError("task.pursuit.target_branch_exit_offset_y must be positive.")
    if float(settings["target_branch_waypoint_x"]) <= 0.0:
        raise ValueError("task.pursuit.target_branch_waypoint_x must be positive.")
    if not np.isfinite(
        [
            float(settings["target_branch_decision_x"]),
            float(settings["target_branch_goal_x"]),
        ]
    ).all():
        raise ValueError("task.pursuit target branch x coordinates must be finite.")
    if int(settings["target_burst_duration_steps"]) > int(settings["target_burst_period_steps"]):
        raise ValueError("target_burst_duration_steps cannot exceed target_burst_period_steps.")
    if int(settings["map_seed_offset"]) < 0:
        raise ValueError("task.pursuit.map_seed_offset must be non-negative.")
    return settings


def _finite_range(value: Any, name: str, minimum: float) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"dynamics.execution.{name} must contain exactly two values.")
    low, high = (float(value[0]), float(value[1]))
    if not np.isfinite([low, high]).all() or low < minimum or high < low:
        raise ValueError(f"dynamics.execution.{name} must satisfy {minimum} <= low <= high.")
    return low, high


def execution_settings(dynamics: dict[str, Any]) -> dict[str, Any]:
    """Validate lightweight command-execution settings without changing observations."""
    configured = dynamics.get("execution", {})
    if not isinstance(configured, dict):
        raise ValueError("dynamics.execution must be a mapping.")
    unknown = sorted(set(configured).difference(_EXECUTION_DEFAULTS))
    if unknown:
        raise ValueError(f"Unknown dynamics.execution settings: {', '.join(unknown)}")
    settings = {**_EXECUTION_DEFAULTS, **configured}
    if int(settings["random_seed_offset"]) < 0:
        raise ValueError("dynamics.execution.random_seed_offset must be non-negative.")
    if int(settings["action_delay_steps"]) < 0:
        raise ValueError("dynamics.execution.action_delay_steps must be non-negative.")
    settings["pending_command_authority"] = validate_command_authority_mode(
        settings["pending_command_authority"]
    )
    if not isinstance(settings["queue_token_contract_enabled"], (bool, np.bool_)):
        raise ValueError("dynamics.execution.queue_token_contract_enabled must be boolean.")
    override_limit = settings["queue_token_max_override_slots"]
    if override_limit is not None:
        if isinstance(override_limit, bool) or int(override_limit) != float(override_limit) or int(override_limit) < 0:
            raise ValueError(
                "dynamics.execution.queue_token_max_override_slots must be a non-negative integer or null."
            )
        settings["queue_token_max_override_slots"] = int(override_limit)
    for name in (
        "command_noise_std",
        "command_noise_bound_sigma",
        "velocity_time_constant_seconds",
        "drag_coefficient",
    ):
        if not np.isfinite(float(settings[name])) or float(settings[name]) < 0.0:
            raise ValueError(f"dynamics.execution.{name} must be finite and non-negative.")
    for name in ("max_speed_scale", "max_acceleration_scale", "mass_scale"):
        if not np.isfinite(float(settings[name])) or float(settings[name]) <= 0.0:
            raise ValueError(f"dynamics.execution.{name} must be finite and positive.")
    settings["max_speed_scale_range"] = _finite_range(settings["max_speed_scale_range"], "max_speed_scale_range", 1e-9)
    settings["max_acceleration_scale_range"] = _finite_range(
        settings["max_acceleration_scale_range"], "max_acceleration_scale_range", 1e-9
    )
    settings["mass_scale_range"] = _finite_range(settings["mass_scale_range"], "mass_scale_range", 1e-9)
    settings["drag_coefficient_range"] = _finite_range(
        settings["drag_coefficient_range"], "drag_coefficient_range", 0.0
    )
    return settings


@dataclass(frozen=True)
class PursuitEpisodeMetrics:
    minimum_target_distance: float
    nearest_defender: int
    collision: bool
    physical_target_contact: bool
    min_clearance: float


@dataclass(frozen=True)
class _BeliefPacket:
    """Timestamped local detection or delayed teammate message."""

    delivery_step: int
    receiver: int
    source: int
    timestamp_step: int
    position: np.ndarray
    velocity: np.ndarray
    confidence: float
    covariance: np.ndarray
    via_message: bool


@dataclass(frozen=True)
class _BeliefSnapshot:
    """Local belief state retained for a bounded fixed-lag update."""

    step: int
    positions: np.ndarray
    velocities: np.ndarray
    confidences: np.ndarray
    covariances: np.ndarray
    timestamps: np.ndarray


class CaptureRadiusPursuit3DEnv:
    """Cooperative 3D pursuit task with partial target observations.

    The environment stores target ground truth internally for simulation and a
    centralized critic, but observe exposes only defender states, obstacle
    geometry, local target beliefs, visibility flags, and message age.
    """

    def __init__(self, config: dict[str, Any], obstacle_count: int, target_speed_scale: float = 1.0):
        self.config = config
        self.world = config["world"]
        self.agents = config["agents"]
        self.task = config["task"]
        self.pursuit = pursuit_settings(self.task)
        self.execution = execution_settings(config.get("dynamics", {}))
        self.obstacle_count = int(obstacle_count)
        self.target_speed_scale = float(target_speed_scale)
        self.n_defenders = int(self.agents["defenders"])
        if self.n_defenders != 4:
            raise ValueError("CaptureRadiusPursuit3DEnv currently requires four homogeneous defenders.")

        self.dt = float(self.world["dt"])
        self.max_steps = int(self.world["max_steps"])
        half_extent = float(self.world["half_extent_xy"])
        self.lower = np.array([-half_extent, -half_extent, float(self.world["minimum_altitude"])], dtype=np.float64)
        self.upper = np.array([half_extent, half_extent, float(self.world["height"])], dtype=np.float64)
        self.rng = np.random.default_rng()
        self.execution_rng = np.random.default_rng()

        self.execution_action_queue: list[np.ndarray] = []
        self.execution_queue_token = QueueToken(generation=0, issued_step=0, pending_length=0)
        self.execution_max_speed = float(self.agents["defender_max_speed"])
        self.execution_max_acceleration = float(self.agents["defender_max_acceleration"])
        self.execution_mass_scale = 1.0
        self.execution_drag_coefficient = 0.0
        self.last_desired_actions = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.last_delayed_actions = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.last_executed_actions = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.last_command_authority_mode = str(self.execution["pending_command_authority"])
        self.last_emergency_brake_requested = False
        self.last_queue_override_slots = 0
        self.last_queue_authority_ack: dict[str, Any] | None = None
        self.action_execution_error_norm = 0.0
        self.action_execution_error_sum = 0.0
        self.action_execution_steps = 0

        self.obstacles: list[CylinderObstacle] = []
        self.defender_positions = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.defender_velocities = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.target_position = np.zeros(3, dtype=np.float64)
        self.target_velocity = np.zeros(3, dtype=np.float64)
        self.target_acceleration = np.zeros(3, dtype=np.float64)
        self.target_escape_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.target_branch_sign: int | None = None
        self.target_branch_decision_step: int | None = None
        self.target_branch_scores = np.full(2, np.nan, dtype=np.float64)
        self.target_maneuver_mode = "straight_flee"
        self.target_maneuver_mode_start_step = 0
        self.target_maneuver_last_replan_step = -1
        self.target_maneuver_direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.target_maneuver_speed_scale = 1.0
        self.target_maneuver_route = "direct"
        self.target_maneuver_switch_count = 0
        self.target_maneuver_estimated_capture_time_seconds = 0.0
        self.target_maneuver_escape_gap_rad = float(2.0 * np.pi)
        self.target_maneuver_decision_scores: dict[str, float] = {}
        self.target_maneuver_mode_counts: dict[str, int] = {mode: 0 for mode in sorted(_MANEUVER_MODES)}
        self.target_maneuver_feasible_candidate_count = 0
        self.target_maneuver_rejected_candidate_count = 0
        self.target_maneuver_fallback_count = 0
        self.target_maneuver_last_feasibility_failure = "none"
        self.target_maneuver_observation_queue: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.target_maneuver_observed_positions = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.target_maneuver_observed_velocities = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.target_maneuver_observation_ages = np.zeros(self.n_defenders, dtype=np.int64)
        self.target_belief_positions = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.target_belief_velocities = np.zeros((self.n_defenders, 3), dtype=np.float64)
        self.target_visible = np.zeros(self.n_defenders, dtype=bool)
        self.target_observation_confidence = np.zeros(self.n_defenders, dtype=np.float64)
        self.target_observation_timestamps = np.full(self.n_defenders, -1, dtype=np.int64)
        self.target_observation_covariance = np.zeros((self.n_defenders, 3, 3), dtype=np.float64)
        self.detection_loss_burst_remaining = np.zeros(self.n_defenders, dtype=np.int64)
        self.message_age_steps = np.full(self.n_defenders, int(self.pursuit["maximum_message_age_steps"]), dtype=np.int64)
        self._message_queue: list[_BeliefPacket] = []

        self.step_count = 0
        self.collision_steps = 0
        self.world_violation_steps = 0
        self.target_world_violation_steps = 0
        self.target_boundary_violation_steps = 0
        self.target_obstacle_violation_steps = 0
        self.defender_world_violation_steps = 0
        self.first_target_boundary_violation_step: int | None = None
        self.first_defender_boundary_violation_step: int | None = None
        self.target_maneuver_candidate_invalid = False
        self.target_command_clipped = False
        self.target_command_clipped_steps = 0
        self.min_clearance = float("inf")
        self.min_target_boundary_clearance = float("inf")
        self.min_defender_boundary_clearance = float("inf")
        self.capture_time_seconds: float | None = None
        self.capturing_defender_id: int | None = None
        self.history: list[dict[str, np.ndarray | float | int | str]] = []

    def reset(self, seed: int, record_history: bool = False) -> dict[str, Any]:
        self.rng = np.random.default_rng(seed)
        self.execution_rng = np.random.default_rng(
            int(seed) + int(self.execution["random_seed_offset"])
        )
        self.step_count = 0
        self.collision_steps = 0
        self.world_violation_steps = 0
        self.target_world_violation_steps = 0
        self.target_boundary_violation_steps = 0
        self.target_obstacle_violation_steps = 0
        self.defender_world_violation_steps = 0
        self.first_target_boundary_violation_step = None
        self.first_defender_boundary_violation_step = None
        self.target_maneuver_candidate_invalid = False
        self.target_command_clipped = False
        self.target_command_clipped_steps = 0
        self.min_clearance = float("inf")
        self.min_target_boundary_clearance = float("inf")
        self.min_defender_boundary_clearance = float("inf")
        self.capture_time_seconds = None
        self.capturing_defender_id = None
        self.history = []
        self._message_queue = []
        self.last_desired_actions.fill(0.0)
        self.last_delayed_actions.fill(0.0)
        self.last_executed_actions.fill(0.0)
        self.last_command_authority_mode = str(self.execution["pending_command_authority"])
        self.last_emergency_brake_requested = False
        self.last_queue_override_slots = 0
        self.last_queue_authority_ack = None
        self.action_execution_error_norm = 0.0
        self.action_execution_error_sum = 0.0
        self.action_execution_steps = 0
        self.execution_max_speed = float(self.agents["defender_max_speed"])
        self.execution_max_acceleration = float(self.agents["defender_max_acceleration"])
        self.execution_mass_scale = 1.0
        self.execution_drag_coefficient = 0.0
        if bool(self.execution["randomize_per_episode"]):
            self.execution_max_speed *= self.execution_rng.uniform(*self.execution["max_speed_scale_range"])
            self.execution_max_acceleration *= self.execution_rng.uniform(*self.execution["max_acceleration_scale_range"])
            self.execution_mass_scale = self.execution_rng.uniform(*self.execution["mass_scale_range"])
            self.execution_drag_coefficient = self.execution_rng.uniform(*self.execution["drag_coefficient_range"])
        else:
            self.execution_max_speed *= float(self.execution["max_speed_scale"])
            self.execution_max_acceleration *= float(self.execution["max_acceleration_scale"])
            self.execution_mass_scale = float(self.execution["mass_scale"])
            self.execution_drag_coefficient = float(self.execution["drag_coefficient"])
        self.execution_action_queue = [
            np.zeros((self.n_defenders, 3), dtype=np.float64)
            for _ in range(int(self.execution["action_delay_steps"]))
        ]
        self.execution_queue_token = QueueToken(
            generation=0,
            issued_step=0,
            pending_length=len(self.execution_action_queue),
        )

        self.target_position = np.array(
            [
                self.rng.uniform(-2.0, 2.0),
                self.rng.uniform(-2.0, 2.0),
                self.rng.uniform(3.0, min(7.0, self.upper[2] - 1.5)),
            ],
            dtype=np.float64,
        )
        self.target_velocity.fill(0.0)
        self.target_acceleration.fill(0.0)
        self.target_escape_direction = _unit(
            self.rng.normal(0.0, 1.0, size=3),
            fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        self.target_branch_sign = None
        self.target_branch_decision_step = None
        self.target_branch_scores.fill(np.nan)
        self.target_maneuver_mode = "straight_flee"
        self.target_maneuver_mode_start_step = 0
        self.target_maneuver_last_replan_step = -1
        self.target_maneuver_direction = self.target_escape_direction.copy()
        self.target_maneuver_speed_scale = 1.0
        self.target_maneuver_route = "direct"
        self.target_maneuver_switch_count = 0
        self.target_maneuver_estimated_capture_time_seconds = 0.0
        self.target_maneuver_escape_gap_rad = float(2.0 * np.pi)
        self.target_maneuver_decision_scores = {}
        self.target_maneuver_mode_counts = {mode: 0 for mode in sorted(_MANEUVER_MODES)}
        self.target_maneuver_feasible_candidate_count = 0
        self.target_maneuver_rejected_candidate_count = 0
        self.target_maneuver_fallback_count = 0
        self.target_maneuver_last_feasibility_failure = "none"
        self.defender_positions = self.target_position + TETRAHEDRON_DIRECTIONS * float(self.pursuit["spawn_distance"])
        self.defender_positions += self.rng.normal(0.0, 0.20, size=self.defender_positions.shape)
        self.defender_positions = np.clip(self.defender_positions, self.lower + 0.6, self.upper - 0.6)
        self.defender_velocities.fill(0.0)
        map_seed_offset = int(self.pursuit["map_seed_offset"])
        if map_seed_offset:
            map_rng = self.rng
            self.rng = np.random.default_rng(seed + map_seed_offset)
            self.obstacles = self._sample_obstacles()
            self.rng = map_rng
        else:
            self.obstacles = self._sample_obstacles()

        self._update_boundary_clearance_metrics()

        if str(self.pursuit["target_motion_mode"]) == "adaptive_maneuvering":
            self._reset_target_maneuver_observation()
        else:
            # Do not consume RNG draws or alter historical target behavior for
            # the existing motion modes, including locked-test modes.
            self.target_maneuver_observation_queue = []
            self.target_maneuver_observed_positions = self.defender_positions.copy()
            self.target_maneuver_observed_velocities = self.defender_velocities.copy()
            self.target_maneuver_observation_ages.fill(0)

        self.target_belief_positions[:] = 0.0
        self.target_belief_velocities[:] = 0.0
        self.message_age_steps[:] = int(self.pursuit["maximum_message_age_steps"])
        self.target_observation_confidence.fill(0.0)
        self.target_observation_timestamps.fill(-1)
        self.target_observation_covariance[:] = np.eye(3, dtype=np.float64)[None, :, :] * float(
            self.pursuit["observation_covariance_growth"]
        )
        self.detection_loss_burst_remaining.fill(0)
        self._update_target_beliefs()
        if record_history:
            self._record_history()
        return self.observe()

    def observe(self) -> dict[str, Any]:
        """Return policy-safe partial observations without target ground truth."""
        predicted_positions, predicted_uncertainties = self._predict_target_beliefs()
        return {
            "defender_positions": self.defender_positions.copy(),
            "defender_velocities": self.defender_velocities.copy(),
            "obstacles": [
                {
                    "center_xy": obstacle.center_xy.copy(),
                    "radius": float(obstacle.radius),
                    "height": float(obstacle.height),
                    "shape": obstacle.shape,
                    "half_extents_xy": (
                        None
                        if obstacle.half_extents_xy is None
                        else obstacle.half_extents_xy.copy()
                    ),
                }
                for obstacle in self.obstacles
            ],
            "target_belief_positions": self.target_belief_positions.copy(),
            "target_belief_velocities": self.target_belief_velocities.copy(),
            "target_visible": self.target_visible.copy(),
            "target_observation_confidence": self.target_observation_confidence.copy(),
            "target_observation_timestamps": self.target_observation_timestamps.copy(),
            "target_observation_covariance": self.target_observation_covariance.copy(),
            "target_observation_age_steps": np.maximum(
                self.step_count - self.target_observation_timestamps,
                0,
            ).astype(np.int64),
            "message_age_steps": self.message_age_steps.copy(),
            "target_prediction_positions": predicted_positions,
            "target_prediction_uncertainties": predicted_uncertainties,
            "step": int(self.step_count),
            "execution": {
                "enabled": bool(self.execution["enabled"]),
                "action_delay_steps": int(self.execution["action_delay_steps"]),
                "action_queue": [item.copy() for item in self.execution_action_queue],
                "queue_token": self.execution_queue_token.as_dict(),
                "queue_token_contract_enabled": bool(self.execution["queue_token_contract_enabled"]),
                "queue_token_max_override_slots": self.execution["queue_token_max_override_slots"],
                "pending_command_authority": str(self.execution["pending_command_authority"]),
                "last_command_authority_mode": str(self.last_command_authority_mode),
                "last_emergency_brake_requested": bool(self.last_emergency_brake_requested),
                "last_queue_override_slots": int(self.last_queue_override_slots),
                "max_speed_mps": float(self.execution_max_speed),
                "max_acceleration_mps2": float(self.execution_max_acceleration),
                "mass_scale": float(self.execution_mass_scale),
                "drag_coefficient": float(self.execution_drag_coefficient),
                "velocity_time_constant_seconds": float(self.execution["velocity_time_constant_seconds"]),
                "command_noise_std_mps": float(self.execution["command_noise_std"]),
                "command_noise_bound_sigma": float(self.execution["command_noise_bound_sigma"]),
                "command_noise_bound_mps": float(
                    self.execution["command_noise_std"] * self.execution["command_noise_bound_sigma"]
                ),
                "clip_command_noise": bool(self.execution["clip_command_noise"]),
                "action_execution_error_norm": float(self.action_execution_error_norm),
                "mean_action_execution_error_norm": float(
                    self.action_execution_error_sum / max(self.action_execution_steps, 1)
                ),
            },
        }

    def centralized_state(self) -> np.ndarray:
        """Training-only global state for a centralized critic."""
        extent = float(self.world["half_extent_xy"])
        max_obstacles = int(self.pursuit["max_observation_obstacles"])
        obstacle_features = np.zeros((max_obstacles, 5), dtype=np.float32)
        for index, obstacle in enumerate(sorted(self.obstacles, key=lambda item: float(item.radius))[:max_obstacles]):
            obstacle_features[index] = np.array(
                [
                    obstacle.center_xy[0] / extent,
                    obstacle.center_xy[1] / extent,
                    obstacle.radius / extent,
                    obstacle.height / extent,
                    1.0,
                ],
                dtype=np.float32,
            )
        values = np.concatenate(
            [
                (self.defender_positions / extent).reshape(-1),
                (self.defender_velocities / float(self.agents["defender_max_speed"])).reshape(-1),
                self.target_position / extent,
                self.target_velocity / float(self.agents["target_max_speed"]),
                obstacle_features.reshape(-1),
                np.array([self.step_count / max(self.max_steps, 1)], dtype=np.float32),
            ]
        )
        return values.astype(np.float32)

    def policy_observations(self, observation: dict[str, Any] | None = None) -> np.ndarray:
        """Encode fixed-size decentralized observations for the shared actor."""
        current = self.observe() if observation is None else observation
        positions = np.asarray(current["defender_positions"], dtype=np.float32)
        velocities = np.asarray(current["defender_velocities"], dtype=np.float32)
        beliefs = np.asarray(current["target_belief_positions"], dtype=np.float32)
        belief_velocities = np.asarray(current["target_belief_velocities"], dtype=np.float32)
        visible = np.asarray(current["target_visible"], dtype=np.float32)
        confidence = np.asarray(current["target_observation_confidence"], dtype=np.float32)
        covariance = np.asarray(current["target_observation_covariance"], dtype=np.float32)
        message_age = np.asarray(current["message_age_steps"], dtype=np.float32)
        prediction_positions = np.asarray(current["target_prediction_positions"], dtype=np.float32)
        prediction_uncertainties = np.asarray(current["target_prediction_uncertainties"], dtype=np.float32)
        obstacles = list(current["obstacles"])
        extent = float(self.world["half_extent_xy"])
        max_obstacles = int(self.pursuit["max_observation_obstacles"])
        rows: list[np.ndarray] = []
        for index in range(self.n_defenders):
            teammate_indices = [other for other in range(self.n_defenders) if other != index]
            relative_teammates = (positions[teammate_indices] - positions[index]).reshape(-1) / extent
            relative_teammate_velocities = (
                velocities[teammate_indices] - velocities[index]
            ).reshape(-1) / float(self.agents["defender_max_speed"])
            nearest = sorted(
                obstacles,
                key=lambda obstacle: float(
                    np.linalg.norm(np.asarray(obstacle["center_xy"], dtype=np.float32) - positions[index, :2])
                    - float(obstacle["radius"])
                ),
            )[:max_obstacles]
            obstacle_features = np.zeros((max_obstacles, 5), dtype=np.float32)
            for obstacle_index, obstacle in enumerate(nearest):
                center = np.asarray(obstacle["center_xy"], dtype=np.float32)
                obstacle_features[obstacle_index] = np.array(
                    [
                        (center[0] - positions[index, 0]) / extent,
                        (center[1] - positions[index, 1]) / extent,
                        (0.5 * float(obstacle["height"]) - positions[index, 2]) / extent,
                        float(obstacle["radius"]) / extent,
                        float(obstacle["height"]) / extent,
                    ],
                    dtype=np.float32,
                )
            rows.append(
                np.concatenate(
                    [
                        velocities[index] / float(self.agents["defender_max_speed"]),
                        (beliefs[index] - positions[index]) / extent,
                        belief_velocities[index] / float(self.agents["target_max_speed"]),
                        np.array(
                            [
                                visible[index],
                                min(
                                    message_age[index] / float(self.pursuit["maximum_message_age_steps"]),
                                    1.0,
                                ),
                            ],
                            dtype=np.float32,
                        ),
                        (
                            np.concatenate(
                                [
                                    np.array([confidence[index]], dtype=np.float32),
                                    np.clip(np.diag(covariance[index]) / max(extent**2, 1e-9), 0.0, 1.0),
                                ]
                            )
                            if bool(self.pursuit["include_uncertainty_features"])
                            else np.empty(0, dtype=np.float32)
                        ),
                        (
                            np.concatenate(
                                [
                                    (prediction_positions[index] - positions[index]) / extent,
                                    np.array([prediction_uncertainties[index] / extent], dtype=np.float32),
                                ]
                            )
                            if bool(self.pursuit["include_prediction_features"])
                            else np.empty(0, dtype=np.float32)
                        ),
                        relative_teammates,
                        relative_teammate_velocities,
                        obstacle_features.reshape(-1),
                    ]
                )
            )
        values = np.stack(rows).astype(np.float32)
        if not np.isfinite(values).all():
            raise RuntimeError("Partial observation encoder emitted a non-finite value.")
        return values

    def prediction_feature_slice(self) -> slice:
        """Return the four-column prediction block in the actor observation.

        The block is deliberately fixed-width: three relative predicted
        position values followed by one scalar uncertainty. A learned
        predictor can replace this block without changing the frozen actor
        input dimension.
        """
        if not bool(self.pursuit["include_prediction_features"]):
            raise ValueError("Prediction features are disabled in this environment configuration.")
        start = 3 + 3 + 3 + 2
        if bool(self.pursuit["include_uncertainty_features"]):
            start += 4
        return slice(start, start + 4)

    def step(
        self,
        defender_actions: np.ndarray,
        record_history: bool = False,
        command_authority: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        actions = np.asarray(defender_actions, dtype=np.float64)
        if actions.shape != (self.n_defenders, 3):
            raise ValueError(f"Expected actions with shape {(self.n_defenders, 3)}, got {actions.shape}.")
        previous_distance = self._target_distances().min()
        previous_defender_boundary_clearance = self._defender_boundary_clearance()
        actions = self._clip_rows(actions, float(self.agents["defender_max_speed"]))
        self._apply_defender_actions(actions, command_authority=command_authority)

        previous_target_velocity = self.target_velocity.copy()
        target_action = self._target_action()
        target_action = self._constrain_target_command(
            target_action,
            enforce_maneuver_limits=str(self.pursuit["target_motion_mode"]) == "adaptive_maneuvering",
        )
        self.target_velocity = self._move_toward_velocity(
            self.target_velocity[None, :],
            target_action[None, :],
            max_delta=float(self.agents["target_max_acceleration"]) * self.dt,
        )[0]
        self.target_velocity = self._clip_rows(
            self.target_velocity[None, :],
            float(self.agents["target_max_speed"]),
        )[0]
        self.target_acceleration = (self.target_velocity - previous_target_velocity) / max(self.dt, 1e-9)
        self.target_position += self.target_velocity * self.dt
        self._enforce_world_bounds(
            self.target_position[None, :],
            self.target_velocity[None, :],
            entity="target",
        )

        self.step_count += 1
        self._update_boundary_clearance_metrics()
        defender_boundary_clearance = self._defender_boundary_clearance()
        if self._target_boundary_clearance(self.target_position) <= 0.0:
            self.target_boundary_violation_steps += 1
            if self.first_target_boundary_violation_step is None:
                self.first_target_boundary_violation_step = int(self.step_count)
        if self._target_obstacle_clearance() <= 0.0:
            self.target_obstacle_violation_steps += 1
        self._update_target_beliefs()
        metrics = self._metrics()
        self.min_clearance = min(self.min_clearance, metrics.min_clearance)
        if metrics.collision:
            self.collision_steps += 1

        capture_event = bool(metrics.minimum_target_distance <= float(self.pursuit["capture_radius"]))
        target_boundary_violation = bool(self.target_boundary_violation_steps > 0)
        target_obstacle_violation = bool(self.target_obstacle_violation_steps > 0)
        target_invalid_episode = bool(
            target_boundary_violation
            or target_obstacle_violation
            or self.target_maneuver_candidate_invalid
        )
        defender_physical_collision = bool(metrics.collision)
        defender_boundary_violation = bool(self.defender_world_violation_steps > 0)
        defender_safety_failure = bool(defender_physical_collision or defender_boundary_violation)
        # Keep the aggregate termination behavior, but expose target contract
        # failures separately so they cannot be reported as defender failures.
        safety_failure = bool(target_invalid_episode or defender_safety_failure)
        safe_capture = bool(capture_event and not safety_failure)
        if safe_capture:
            self.capture_time_seconds = float(self.step_count * self.dt)
            self.capturing_defender_id = int(metrics.nearest_defender)

        if target_boundary_violation:
            termination_reason = "target_boundary_violation"
        elif target_obstacle_violation:
            termination_reason = "target_obstacle_violation"
        elif self.target_maneuver_candidate_invalid:
            termination_reason = "target_candidate_invalid"
        elif defender_safety_failure:
            termination_reason = "safety_failure"
        elif safe_capture:
            termination_reason = "safe_capture"
        else:
            termination_reason = "running"
        terminated = bool(safety_failure or safe_capture)
        truncated = bool(not terminated and self.step_count >= self.max_steps)
        if truncated:
            termination_reason = "timeout"

        progress = float(previous_distance - metrics.minimum_target_distance)
        coverage = self._coverage_score()
        reward_components = {
            "progress": float(self.pursuit["progress_reward_weight"]) * progress,
            "distance": -float(self.pursuit["distance_reward_weight"]) * metrics.minimum_target_distance,
            "coverage": float(self.pursuit["coverage_reward_weight"]) * coverage,
            "capture": float(self.pursuit["capture_bonus"]) if safe_capture else 0.0,
            # A target-contract violation terminates the sample, but is not a
            # defender action failure and must not train the defender to avoid it.
            "safety": -float(self.pursuit["collision_penalty"]) if defender_safety_failure else 0.0,
            "boundary_proximity": -float(self.pursuit["defender_boundary_proximity_weight"])
            * max(0.0, float(self.pursuit["defender_boundary_margin"]) - defender_boundary_clearance),
            "boundary_progress": float(self.pursuit["defender_boundary_progress_weight"])
            * (defender_boundary_clearance - previous_defender_boundary_clearance),
        }
        reward = float(sum(reward_components.values()))

        if record_history:
            self._record_history()
        info = {
            "success": safe_capture,
            "safe_capture_success": safe_capture,
            "capture_event": capture_event,
            "capture_time_seconds": self.capture_time_seconds,
            "capturing_defender_id": self.capturing_defender_id,
            "nearest_target_distance": float(metrics.minimum_target_distance),
            "nearest_defender": int(metrics.nearest_defender),
            "relative_speed_at_capture": (
                float(np.linalg.norm(self.defender_velocities[metrics.nearest_defender] - self.target_velocity))
                if capture_event
                else None
            ),
            "collision": defender_safety_failure,
            "safety_failure": safety_failure,
            "defender_physical_collision": defender_physical_collision,
            "defender_safety_failure": defender_safety_failure,
            "collision_steps": int(self.collision_steps),
            "physical_target_contact": bool(metrics.physical_target_contact),
            "world_violation_steps": int(self.world_violation_steps),
            "target_world_violation_steps": int(self.target_world_violation_steps),
            "target_boundary_violation_steps": int(self.target_boundary_violation_steps),
            "target_obstacle_violation_steps": int(self.target_obstacle_violation_steps),
            "defender_world_violation_steps": int(self.defender_world_violation_steps),
            "target_boundary_violation": target_boundary_violation,
            "target_obstacle_violation": target_obstacle_violation,
            "target_invalid_episode": target_invalid_episode,
            "target_candidate_invalid": bool(self.target_maneuver_candidate_invalid),
            "defender_boundary_violation": defender_boundary_violation,
            "boundary_violation": bool(target_boundary_violation or defender_boundary_violation),
            "task_valid_for_policy_evaluation": not target_invalid_episode,
            "first_target_boundary_violation_step": self.first_target_boundary_violation_step,
            "first_defender_boundary_violation_step": self.first_defender_boundary_violation_step,
            "minimum_target_boundary_clearance_m": float(self.min_target_boundary_clearance),
            "minimum_defender_boundary_clearance_m": float(self.min_defender_boundary_clearance),
            "target_command_clipped": bool(self.target_command_clipped),
            "target_command_clipped_steps": int(self.target_command_clipped_steps),
            "target_candidate_fallback_count": int(self.target_maneuver_fallback_count),
            "target_candidate_rejection_reason": str(self.target_maneuver_last_feasibility_failure),
            "min_clearance": float(metrics.min_clearance),
            "min_clearance_so_far": float(self.min_clearance),
            "termination_reason": termination_reason,
            "reward_components": reward_components,
            "target_visible_fraction": float(np.mean(self.target_visible)),
            "mean_message_age_steps": float(np.mean(self.message_age_steps)),
            "mean_observation_confidence": float(np.mean(self.target_observation_confidence)),
            "mean_observation_age_steps": float(
                np.mean(np.maximum(self.step_count - self.target_observation_timestamps, 0))
            ),
            "mean_observation_covariance_trace": float(
                np.mean(np.trace(self.target_observation_covariance, axis1=1, axis2=2))
            ),
            "target_branch_sign": self.target_branch_sign,
            "target_branch_decision_step": self.target_branch_decision_step,
            "target_branch_scores_seconds": self.target_branch_scores.tolist(),
            "target_maneuver_mode": self.target_maneuver_mode,
            "target_maneuver_mode_start_step": int(self.target_maneuver_mode_start_step),
            "target_maneuver_last_replan_step": int(self.target_maneuver_last_replan_step),
            "target_maneuver_speed_scale": float(self.target_maneuver_speed_scale),
            "target_maneuver_route": self.target_maneuver_route,
            "target_maneuver_switch_count": int(self.target_maneuver_switch_count),
            "target_maneuver_estimated_capture_time_seconds": float(
                self.target_maneuver_estimated_capture_time_seconds
            ),
            "target_maneuver_escape_gap_rad": float(self.target_maneuver_escape_gap_rad),
            "target_maneuver_decision_scores": {
                str(key): float(value) for key, value in self.target_maneuver_decision_scores.items()
            },
            "target_maneuver_feasible_candidate_count": int(self.target_maneuver_feasible_candidate_count),
            "target_maneuver_rejected_candidate_count": int(self.target_maneuver_rejected_candidate_count),
            "target_maneuver_fallback_count": int(self.target_maneuver_fallback_count),
            "target_maneuver_last_feasibility_failure": str(self.target_maneuver_last_feasibility_failure),
            "target_maneuver_mode_counts": {
                str(key): int(value) for key, value in self.target_maneuver_mode_counts.items()
            },
            "target_maneuver_observation_age_steps": self.target_maneuver_observation_ages.tolist(),
            "target_maneuver_observation_delay_steps": int(
                self.pursuit["target_maneuver_observation_delay_steps"]
            ),
            "target_maneuver_max_turn_rate_rad_s": float(
                self.pursuit["target_maneuver_max_turn_rate_rad_s"]
            ),
            "target_maneuver_max_jerk_mps3": float(self.pursuit["target_maneuver_max_jerk_mps3"]),
            "capture_radius": float(self.pursuit["capture_radius"]),
            "execution_enabled": bool(self.execution["enabled"]),
            "action_execution_error_norm": float(self.action_execution_error_norm),
            "mean_action_execution_error_norm": float(
                self.action_execution_error_sum / max(self.action_execution_steps, 1)
            ),
            "last_desired_action_norm": float(np.mean(np.linalg.norm(self.last_desired_actions, axis=1))),
            "last_delayed_action_norm": float(np.mean(np.linalg.norm(self.last_delayed_actions, axis=1))),
            "last_executed_action_norm": float(np.mean(np.linalg.norm(self.last_executed_actions, axis=1))),
            "command_authority_mode": str(self.last_command_authority_mode),
            "emergency_brake_requested": bool(self.last_emergency_brake_requested),
            "queue_override_slots": int(self.last_queue_override_slots),
            "queue_authority_ack": self.last_queue_authority_ack,
        }
        return self.observe(), reward, terminated, truncated, info

    def _apply_defender_actions(
        self,
        actions: np.ndarray,
        *,
        command_authority: dict[str, Any] | None = None,
    ) -> None:
        desired = self._clip_rows(
            np.asarray(actions, dtype=np.float64),
            float(self.agents["defender_max_speed"]),
        )
        self.last_desired_actions = desired.copy()
        directive = command_authority_directive(
            command_authority,
            allowed_mode=str(self.execution["pending_command_authority"]),
        )
        self.last_command_authority_mode = directive.mode
        self.last_emergency_brake_requested = bool(directive.emergency_brake)
        self.last_queue_override_slots = 0
        self.last_queue_authority_ack = None
        if not bool(self.execution["enabled"]):
            # Preserve the historical ideal velocity-level benchmark exactly
            # unless the experimental execution model is explicitly enabled.
            delayed = desired
            executed = desired.copy()
        else:
            if bool(self.execution["queue_token_contract_enabled"]):
                token_directive: Any = directive
                configured_override_limit = self.execution["queue_token_max_override_slots"]
                if configured_override_limit is not None and directive.max_override_slots is None:
                    token_directive = {
                        **directive.as_dict(),
                        "max_override_slots": int(configured_override_limit),
                    }
                (
                    self.execution_action_queue,
                    self.execution_queue_token,
                    authority_ack,
                ) = apply_command_authority_with_token(
                    self.execution_action_queue,
                    token_directive,
                    allowed_mode=str(self.execution["pending_command_authority"]),
                    current_token=self.execution_queue_token,
                )
                self.last_queue_authority_ack = authority_ack.as_dict()
                self.last_queue_override_slots = int(authority_ack.overridden_slots)
                (
                    self.execution_action_queue,
                    delayed,
                    self.execution_queue_token,
                ) = commit_delayed_command_with_token(
                    self.execution_action_queue,
                    desired,
                    self.execution_queue_token,
                    next_step=self.step_count + 1,
                )
            else:
                self.execution_action_queue, _resolved, self.last_queue_override_slots = apply_command_authority(
                    self.execution_action_queue,
                    directive,
                    allowed_mode=str(self.execution["pending_command_authority"]),
                )
                if self.execution_action_queue:
                    self.execution_action_queue.append(desired.copy())
                    delayed = self.execution_action_queue.pop(0)
                else:
                    delayed = desired
            self.last_delayed_actions = delayed.copy()
            parameters = parameters_from_observation(self.observe(), self.dt)
            noise = self.execution_rng.normal(
                0.0,
                float(self.execution["command_noise_std"]),
                size=delayed.shape,
            )
            executed = advance_execution(
                self.defender_velocities,
                delayed,
                parameters,
                noise=noise,
            ).executed

        self.last_delayed_actions = delayed.copy()
        self.last_executed_actions = executed.copy()
        self.action_execution_error_norm = float(np.mean(np.linalg.norm(executed - desired, axis=1)))
        self.action_execution_error_sum += self.action_execution_error_norm
        self.action_execution_steps += 1
        self.defender_velocities = executed
        self.defender_positions += self.defender_velocities * self.dt
        self._enforce_world_bounds(
            self.defender_positions,
            self.defender_velocities,
            entity="defender",
        )

    def _target_action(self) -> np.ndarray:
        if str(self.pursuit["target_motion_mode"]) == "adaptive_maneuvering":
            return self._adaptive_maneuvering_target_action()
        if str(self.pursuit["target_motion_mode"]) == "adaptive_adversarial":
            return self._adaptive_adversarial_target_action()
        if str(self.pursuit["target_motion_mode"]) == "adaptive_branching":
            return self._adaptive_branching_target_action()

        desired = float(self.pursuit["target_heading_persistence"]) * self.target_escape_direction
        for defender_position in self.defender_positions:
            delta = self.target_position - defender_position
            distance = float(np.linalg.norm(delta))
            if distance < float(self.pursuit["target_defender_avoidance_distance"]):
                desired += (
                    _unit(delta)
                    * (float(self.pursuit["target_defender_avoidance_distance"]) - distance)
                    * float(self.pursuit["target_defender_avoidance_gain"])
                )
        for obstacle in self.obstacles:
            clearance, normal = self._cylinder_clearance_and_normal(self.target_position, obstacle)
            if clearance < float(self.pursuit["target_obstacle_avoidance_distance"]):
                desired += (
                    normal
                    * (float(self.pursuit["target_obstacle_avoidance_distance"]) - clearance)
                    * float(self.pursuit["target_obstacle_avoidance_gain"])
                )
        defender_centroid = self.defender_positions.mean(axis=0)
        desired += float(self.pursuit["target_flee_gain"]) * _unit(
            self.target_position - defender_centroid,
            fallback=self.target_escape_direction,
        )
        desired[2] += float(self.pursuit["target_vertical_gain"]) * np.sin(0.11 * self.step_count)
        margin = float(self.pursuit["target_boundary_margin"])
        for axis in range(3):
            if self.target_position[axis] < self.lower[axis] + margin:
                desired[axis] += float(self.pursuit["target_boundary_gain"])
            if self.target_position[axis] > self.upper[axis] - margin:
                desired[axis] -= float(self.pursuit["target_boundary_gain"])
        mode = str(self.pursuit["target_motion_mode"])
        if mode == "random_turn" and self.step_count > 0 and self.step_count % int(self.pursuit["target_turn_interval_steps"]) == 0:
            self.target_escape_direction = _unit(
                self.rng.normal(0.0, 1.0, size=3),
                fallback=self.target_escape_direction,
            )
            desired = 0.35 * desired + self.target_escape_direction
        elif mode == "s_curve":
            lateral = np.array(
                [-self.target_escape_direction[1], self.target_escape_direction[0], 0.0],
                dtype=np.float64,
            )
            lateral = _unit(lateral, fallback=np.array([0.0, 1.0, 0.0]))
            desired += lateral * float(self.pursuit["target_s_curve_amplitude"]) * np.sin(
                float(self.pursuit["target_s_curve_frequency"]) * self.step_count
            )
        elif mode == "burst":
            burst_period = int(self.pursuit["target_burst_period_steps"])
            burst_duration = int(self.pursuit["target_burst_duration_steps"])
            if self.step_count % burst_period < burst_duration:
                desired *= float(self.pursuit["target_burst_speed_scale"])
        elif mode == "boundary_escape":
            boundary_direction = np.zeros(3, dtype=np.float64)
            for axis in range(3):
                distance_to_lower = self.target_position[axis] - self.lower[axis]
                distance_to_upper = self.upper[axis] - self.target_position[axis]
                if distance_to_lower < distance_to_upper:
                    boundary_direction[axis] -= 1.0 / max(distance_to_lower, 0.5)
                else:
                    boundary_direction[axis] += 1.0 / max(distance_to_upper, 0.5)
            desired += 0.35 * _unit(boundary_direction, fallback=self.target_escape_direction)

        direction = _unit(desired, fallback=self.target_escape_direction)
        self.target_escape_direction = direction
        speed_scale = self.target_speed_scale
        if mode == "burst":
            burst_period = int(self.pursuit["target_burst_period_steps"])
            burst_duration = int(self.pursuit["target_burst_duration_steps"])
            if self.step_count % burst_period < burst_duration:
                speed_scale *= float(self.pursuit["target_burst_speed_scale"])
        return direction * float(self.agents["target_max_speed"]) * speed_scale

    def _reset_target_maneuver_observation(self) -> None:
        """Initialize the target's private delayed/noisy defender sensor."""

        delay = int(self.pursuit["target_maneuver_observation_delay_steps"])
        noise_std = float(self.pursuit["target_maneuver_observation_noise_std"])
        positions = self.defender_positions + self.rng.normal(
            0.0,
            noise_std,
            size=self.defender_positions.shape,
        )
        velocities = self.defender_velocities + self.rng.normal(
            0.0,
            2.0 * noise_std,
            size=self.defender_velocities.shape,
        )
        valid = np.ones(self.n_defenders, dtype=bool)
        snapshot = (positions.copy(), velocities.copy(), valid.copy())
        # The target has an initial track, but it is delayed by the configured
        # amount.  Filling the queue with the same initial packet avoids an
        # artificial zero-information transient at reset.
        self.target_maneuver_observation_queue = [
            (snapshot[0].copy(), snapshot[1].copy(), snapshot[2].copy())
            for _ in range(delay + 1)
        ]
        self.target_maneuver_observed_positions = positions.copy()
        self.target_maneuver_observed_velocities = velocities.copy()
        self.target_maneuver_observation_ages = np.full(self.n_defenders, delay, dtype=np.int64)

    def _observe_defenders_for_maneuver(self) -> tuple[np.ndarray, np.ndarray]:
        """Return delayed/noisy defender tracks for the non-oracle adversary.

        This routine is deliberately separate from ``observe``.  It models the
        target's own imperfect sensor and never returns simulator truth to the
        maneuver selector.  Missing packets are propagated with the previous
        estimated velocity, so a dropout cannot silently reveal a fresh true
        position.
        """

        noise_std = float(self.pursuit["target_maneuver_observation_noise_std"])
        positions = self.defender_positions + self.rng.normal(
            0.0,
            noise_std,
            size=self.defender_positions.shape,
        )
        velocities = self.defender_velocities + self.rng.normal(
            0.0,
            2.0 * noise_std,
            size=self.defender_velocities.shape,
        )
        valid = self.rng.random(self.n_defenders) >= float(
            self.pursuit["target_maneuver_observation_dropout_probability"]
        )
        self.target_maneuver_observation_queue.append(
            (positions.copy(), velocities.copy(), valid.copy())
        )
        delay = int(self.pursuit["target_maneuver_observation_delay_steps"])
        delayed_positions, delayed_velocities, delayed_valid = self.target_maneuver_observation_queue.pop(0)

        for index in range(self.n_defenders):
            if bool(delayed_valid[index]):
                self.target_maneuver_observed_positions[index] = delayed_positions[index]
                self.target_maneuver_observed_velocities[index] = delayed_velocities[index]
                self.target_maneuver_observation_ages[index] = delay
            else:
                self.target_maneuver_observed_positions[index] += (
                    self.target_maneuver_observed_velocities[index] * self.dt
                )
                self.target_maneuver_observed_velocities[index] *= 0.90
                self.target_maneuver_observation_ages[index] += 1
        return (
            self.target_maneuver_observed_positions.copy(),
            self.target_maneuver_observed_velocities.copy(),
        )

    @staticmethod
    def _rotate_unit_vector_toward(
        current: np.ndarray,
        desired: np.ndarray,
        max_angle_rad: float,
    ) -> np.ndarray:
        """Rotate ``current`` toward ``desired`` by at most ``max_angle_rad``."""

        current_unit = _unit(np.asarray(current, dtype=np.float64))
        desired_unit = _unit(np.asarray(desired, dtype=np.float64), fallback=current_unit)
        if float(np.linalg.norm(current_unit)) <= 1e-9:
            return desired_unit
        cosine = float(np.clip(np.dot(current_unit, desired_unit), -1.0, 1.0))
        angle = float(np.arccos(cosine))
        if angle <= max_angle_rad:
            return desired_unit
        axis = np.cross(current_unit, desired_unit)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-9:
            basis_index = int(np.argmin(np.abs(current_unit)))
            basis = np.zeros(3, dtype=np.float64)
            basis[basis_index] = 1.0
            axis = np.cross(current_unit, basis)
            axis_norm = float(np.linalg.norm(axis))
        axis /= max(axis_norm, 1e-9)
        theta = min(float(max_angle_rad), angle)
        rotated = (
            current_unit * np.cos(theta)
            + np.cross(axis, current_unit) * np.sin(theta)
            + axis * float(np.dot(axis, current_unit)) * (1.0 - np.cos(theta))
        )
        return _unit(rotated, fallback=current_unit)

    def _limit_target_command(
        self,
        current_velocity: np.ndarray,
        current_acceleration: np.ndarray,
        desired_velocity: np.ndarray,
    ) -> np.ndarray:
        """Apply target speed, turn-rate, acceleration, and jerk limits."""

        max_speed = float(self.agents["target_max_speed"])
        max_acceleration = float(self.agents["target_max_acceleration"])
        desired = self._clip_rows(np.asarray(desired_velocity, dtype=np.float64)[None, :], max_speed)[0]
        current = np.asarray(current_velocity, dtype=np.float64)
        acceleration = np.asarray(current_acceleration, dtype=np.float64)
        current_speed = float(np.linalg.norm(current))
        desired_speed = float(np.linalg.norm(desired))
        if current_speed > 1e-9 and desired_speed > 1e-9:
            desired_direction = self._rotate_unit_vector_toward(
                current,
                desired,
                float(self.pursuit["target_maneuver_max_turn_rate_rad_s"]) * self.dt,
            )
            desired = desired_direction * desired_speed

        desired_acceleration = (desired - current) / max(self.dt, 1e-9)
        desired_acceleration = self._clip_rows(
            desired_acceleration[None, :],
            max_acceleration,
        )[0]
        acceleration_delta = desired_acceleration - acceleration
        max_jerk_delta = float(self.pursuit["target_maneuver_max_jerk_mps3"]) * self.dt
        acceleration_delta = self._clip_rows(
            acceleration_delta[None, :],
            max_jerk_delta,
        )[0]
        limited_acceleration = self._clip_rows(
            (acceleration + acceleration_delta)[None, :],
            max_acceleration,
        )[0]
        proposed = self._clip_rows(
            (current + limited_acceleration * self.dt)[None, :],
            max_speed,
        )[0]
        # Jerk limiting can tilt the acceleration away from the desired
        # velocity, while turn-rate projection can in turn change the
        # implied acceleration.  Alternate the two convex-like projections a
        # few times so the returned command satisfies both constraints, not
        # merely the direction constraint in isolation.
        max_turn_angle = float(self.pursuit["target_maneuver_max_turn_rate_rad_s"]) * self.dt
        max_jerk_delta = float(self.pursuit["target_maneuver_max_jerk_mps3"]) * self.dt
        for _ in range(6):
            proposed_speed = float(np.linalg.norm(proposed))
            if current_speed > 1e-9 and proposed_speed > 1e-9:
                proposed_direction = self._rotate_unit_vector_toward(
                    current,
                    proposed,
                    max_turn_angle,
                )
                proposed = proposed_direction * proposed_speed
            proposed_acceleration = (proposed - current) / max(self.dt, 1e-9)
            proposed_acceleration = self._clip_rows(
                proposed_acceleration[None, :],
                max_acceleration,
            )[0]
            acceleration_delta = self._clip_rows(
                (proposed_acceleration - acceleration)[None, :],
                max_jerk_delta,
            )[0]
            proposed = current + (acceleration + acceleration_delta) * self.dt
            proposed = self._clip_rows(proposed[None, :], max_speed)[0]
        return proposed

    def _target_maneuver_escape_gap(
        self,
        direction: np.ndarray,
        defender_positions: np.ndarray,
    ) -> float:
        """Return the angular escape gap containing a candidate direction."""

        relative = np.asarray(defender_positions, dtype=np.float64) - self.target_position[None, :]
        horizontal_norm = np.linalg.norm(relative[:, :2], axis=1)
        angles = np.mod(np.arctan2(relative[:, 1], relative[:, 0]), 2.0 * np.pi)
        angles = np.sort(angles[horizontal_norm > 1e-9])
        if angles.size == 0:
            return float(2.0 * np.pi)
        gaps = np.diff(np.concatenate([angles, angles[:1] + 2.0 * np.pi]))
        candidate_angle = float(np.mod(np.arctan2(direction[1], direction[0]), 2.0 * np.pi))
        for index, gap in enumerate(gaps):
            start = float(angles[index])
            if candidate_angle >= start and candidate_angle <= start + float(gap):
                return float(gap)
        return float(np.max(gaps))

    def _target_maneuver_route_candidates(
        self,
        forward: np.ndarray,
        lateral: np.ndarray,
    ) -> list[tuple[str, np.ndarray]]:
        """Build left/right/top obstacle-bypass directions from known geometry."""

        margin = float(self.pursuit["target_maneuver_route_margin_m"])
        route_boundary_buffer = max(
            0.0,
            float(self.pursuit["target_maneuver_route_boundary_buffer_m"]),
        )
        route_boundary_buffer = min(
            route_boundary_buffer,
            0.49 * float(self.upper[2] - self.lower[2]),
        )
        routes: list[tuple[str, np.ndarray]] = []
        for obstacle_index, obstacle in enumerate(self.obstacles):
            half = (
                np.array([obstacle.radius, obstacle.radius], dtype=np.float64)
                if obstacle.half_extents_xy is None
                else np.asarray(obstacle.half_extents_xy, dtype=np.float64)
            )
            forward_extent = float(np.abs(forward[:2]) @ half)
            lateral_extent = float(np.abs(lateral[:2]) @ half)
            center = np.array(
                [obstacle.center_xy[0], obstacle.center_xy[1], self.target_position[2]],
                dtype=np.float64,
            )
            for sign, label in ((-1.0, "left"), (1.0, "right")):
                waypoint = center + forward * (forward_extent + margin)
                waypoint += lateral * sign * (lateral_extent + margin)
                waypoint[2] = float(np.clip(waypoint[2], self.lower[2] + 0.5, self.upper[2] - 0.5))
                routes.append((f"obstacle_{obstacle_index}_{label}", _unit(waypoint - self.target_position)))
            top = center.copy()
            top[2] = min(float(obstacle.height) + margin, self.upper[2] - 0.5)
            top[2] = float(
                np.clip(
                    top[2],
                    self.lower[2] + route_boundary_buffer,
                    self.upper[2] - route_boundary_buffer,
                )
            )
            routes.append((f"obstacle_{obstacle_index}_top", _unit(top - self.target_position)))
        return routes

    def _target_obstacle_avoidance_direction(self) -> np.ndarray:
        """Return the local direction pointing away from the nearest obstacle.

        This prior is deliberately geometric and target-private.  It prevents
        the adversary from selecting an evasive action that heads into the
        obstacle field merely because that action creates a larger defender
        escape gap.  If no obstacle is present, the nominal escape direction
        remains the fallback.
        """

        if not self.obstacles:
            return _unit(self.target_escape_direction, fallback=np.array([1.0, 0.0, 0.0]))
        target_xy = np.asarray(self.target_position[:2], dtype=np.float64)
        nearest = min(
            self.obstacles,
            key=lambda obstacle: float(np.linalg.norm(np.asarray(obstacle.center_xy, dtype=np.float64) - target_xy)),
        )
        away_xy = target_xy - np.asarray(nearest.center_xy, dtype=np.float64)
        away = np.array([away_xy[0], away_xy[1], 0.0], dtype=np.float64)
        if float(np.linalg.norm(away)) <= 1.0e-9:
            return _unit(self.target_escape_direction, fallback=np.array([1.0, 0.0, 0.0]))
        return _unit(away, fallback=self.target_escape_direction)

    def _target_radius(self) -> float:
        """Return the configured target radius, with point-target compatibility."""

        radius = float(self.agents.get("target_radius", 0.0))
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("agents.target_radius must be finite and non-negative.")
        return radius

    def _target_safe_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the target center bounds after radius and margin erosion."""

        erosion = self._target_radius() + float(self.pursuit["target_boundary_margin"])
        lower = self.lower + erosion
        upper = self.upper - erosion
        if np.any(lower >= upper):
            raise ValueError("target radius and boundary margin leave no valid target state space.")
        return lower, upper

    def _target_boundary_clearance(self, position: np.ndarray) -> float:
        """Return signed clearance from the target's effective boundary.

        A positive value is inside the target safety set.  The configured
        target margin is included here so candidate rollout and runtime
        accounting use exactly the same boundary semantics.
        """

        position = np.asarray(position, dtype=np.float64)
        lower, upper = self._target_safe_bounds()
        return float(min(np.min(position - lower), np.min(upper - position)))

    def _defender_boundary_clearance(self) -> float:
        return float(
            min(
                np.min(self.defender_positions - self.lower),
                np.min(self.upper - self.defender_positions),
            )
        )

    def _target_obstacle_clearance(self) -> float:
        return float(
            min(
                (self._obstacle_clearance(self.target_position, obstacle) for obstacle in self.obstacles),
                default=float("inf"),
            )
        )

    def _update_boundary_clearance_metrics(self) -> None:
        self.min_target_boundary_clearance = min(
            self.min_target_boundary_clearance,
            self._target_boundary_clearance(self.target_position),
        )
        self.min_defender_boundary_clearance = min(
            self.min_defender_boundary_clearance,
            self._defender_boundary_clearance(),
        )

    def _target_inward_direction(self, position: np.ndarray) -> np.ndarray:
        """Return an inward direction for a predicted target state."""

        position = np.asarray(position, dtype=np.float64)
        lower, upper = self._target_safe_bounds()
        direction = np.zeros(3, dtype=np.float64)
        for axis in range(3):
            if position[axis] <= lower[axis]:
                direction[axis] += 1.0
            elif position[axis] >= upper[axis]:
                direction[axis] -= 1.0
        return _unit(direction, fallback=self.target_escape_direction)

    def _target_maneuver_boundary_direction(self) -> np.ndarray:
        """Return an inward direction when the current target state is risky.

        The predictive branch is intentionally target-private.  It models the
        target noticing that its bounded dynamics cannot safely maintain the
        current heading until the next nominal maneuver decision; it does not
        reveal target state to the defender policy.
        """

        boundary_direction = np.zeros(3, dtype=np.float64)
        boundary_trigger = max(
            0.50,
            float(np.linalg.norm(self.target_velocity)) * self.dt * 2.0,
        )
        predictive_recovery = bool(self.pursuit["target_maneuver_predictive_boundary_recovery"])
        lookahead_distance = (
            float(self.pursuit["target_maneuver_predictive_boundary_lookahead_steps"]) * self.dt
            if predictive_recovery
            else 0.0
        )
        safe_lower, safe_upper = self._target_safe_bounds()
        for axis in range(3):
            lower_distance = float(self.target_position[axis] - safe_lower[axis])
            upper_distance = float(safe_upper[axis] - self.target_position[axis])
            lower_projected = lower_distance + min(float(self.target_velocity[axis]), 0.0) * lookahead_distance
            upper_projected = upper_distance - max(float(self.target_velocity[axis]), 0.0) * lookahead_distance
            if lower_distance < boundary_trigger or lower_projected < boundary_trigger:
                boundary_direction[axis] += 1.0
            if upper_distance < boundary_trigger or upper_projected < boundary_trigger:
                boundary_direction[axis] -= 1.0
        return boundary_direction

    def _target_maneuver_candidates(
        self,
        defender_positions: np.ndarray,
        defender_velocities: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Return physically plausible maneuver candidates from private tracks."""

        del defender_velocities  # The rollout evaluator uses them directly.
        centroid = np.mean(defender_positions, axis=0)
        away = _unit(self.target_position - centroid, fallback=self.target_escape_direction)
        forward = _unit(self.target_velocity, fallback=self.target_escape_direction)
        horizontal_forward = _unit(
            np.array([forward[0], forward[1], 0.0], dtype=np.float64),
            fallback=np.array([1.0, 0.0, 0.0]),
        )
        lateral = _unit(
            np.array([-horizontal_forward[1], horizontal_forward[0], 0.0], dtype=np.float64),
            fallback=np.array([0.0, 1.0, 0.0]),
        )
        candidates: list[dict[str, Any]] = [
            {"mode": "straight_flee", "direction": away, "speed_scale": 1.0, "route": "direct"},
            {
                "mode": "lateral_jink",
                "direction": _unit(0.55 * away + 0.84 * lateral),
                "speed_scale": 1.0,
                "route": "lateral_positive",
            },
            {
                "mode": "lateral_jink",
                "direction": _unit(0.55 * away - 0.84 * lateral),
                "speed_scale": 1.0,
                "route": "lateral_negative",
            },
            {
                "mode": "reverse_lane_change",
                "direction": _unit(-0.20 * forward + 0.98 * lateral),
                "speed_scale": 1.0,
                "route": "lane_change_positive",
            },
            {
                "mode": "reverse_lane_change",
                "direction": _unit(-0.20 * forward - 0.98 * lateral),
                "speed_scale": 1.0,
                "route": "lane_change_negative",
            },
            {
                "mode": "vertical_escape",
                "direction": _unit(0.78 * away + 0.63 * np.array([0.0, 0.0, 1.0])),
                "speed_scale": 1.0,
                "route": "climb",
            },
            {
                "mode": "vertical_escape",
                "direction": _unit(0.78 * away - 0.63 * np.array([0.0, 0.0, 1.0])),
                "speed_scale": 1.0,
                "route": "descend",
            },
            {
                "mode": "speed_burst",
                "direction": away,
                "speed_scale": float(self.pursuit["target_maneuver_burst_speed_scale"]),
                "route": "burst",
            },
        ]
        if not bool(self.pursuit.get("target_maneuver_enable_reverse_lane_change", True)):
            candidates = [item for item in candidates if item["mode"] != "reverse_lane_change"]
        for route, direction in self._target_maneuver_route_candidates(horizontal_forward, lateral):
            candidates.append(
                {
                    "mode": "obstacle_bypass",
                    "direction": direction,
                    "speed_scale": 1.0,
                    "route": route,
                }
            )
        boundary_direction = self._target_maneuver_boundary_direction()
        if float(np.linalg.norm(boundary_direction)) > 1.0e-9:
            candidates.append(
                {
                    "mode": "boundary_recovery",
                    "direction": _unit(boundary_direction, fallback=away),
                    "speed_scale": float(
                        self.pursuit["target_maneuver_boundary_recovery_speed_scale"]
                    ),
                    "route": "boundary_recovery",
                }
            )
        return candidates

    def _evaluate_target_maneuver_candidate(
        self,
        candidate: dict[str, Any],
        defender_positions: np.ndarray,
        defender_velocities: np.ndarray,
    ) -> dict[str, Any]:
        """Roll out one candidate and return survival-oriented diagnostics."""

        horizon = int(self.pursuit["target_maneuver_horizon_steps"])
        max_speed = float(self.agents["target_max_speed"])
        target_speed = min(
            max_speed,
            max_speed * float(self.target_speed_scale) * float(candidate["speed_scale"]),
        )
        target_position = self.target_position.copy()
        target_velocity = self.target_velocity.copy()
        target_acceleration = self.target_acceleration.copy()
        min_distance = float("inf")
        terminal_distance = float("inf")
        min_clearance = float("inf")
        min_boundary = float("inf")
        first_clearance = float("inf")
        first_boundary = float("inf")
        feasible_prefix_steps = 0
        failure_reason = "none"
        required_clearance = max(
            float(self.pursuit["target_maneuver_feasibility_margin_m"]),
            float(self.agents["drone_radius"]) + float(self.pursuit["safety_margin"]),
        )
        for timestep in range(horizon):
            previous_position = target_position.copy()
            desired_velocity = np.asarray(candidate["direction"], dtype=np.float64) * target_speed
            command = self._limit_target_command(
                target_velocity,
                target_acceleration,
                desired_velocity,
            )
            next_velocity = self._move_toward_velocity(
                target_velocity[None, :],
                command[None, :],
                max_delta=float(self.agents["target_max_acceleration"]) * self.dt,
            )[0]
            next_acceleration = (next_velocity - target_velocity) / max(self.dt, 1e-9)
            target_position = target_position + next_velocity * self.dt
            target_velocity = next_velocity
            target_acceleration = next_acceleration
            predicted_defenders = defender_positions + defender_velocities * ((timestep + 1) * self.dt)
            distances = np.linalg.norm(predicted_defenders - target_position[None, :], axis=1)
            min_distance = min(min_distance, float(np.min(distances)))
            terminal_distance = float(np.min(distances))
            segment_clearance = float("inf")
            for fraction in (0.0, 0.5, 1.0):
                segment_position = previous_position + float(fraction) * (target_position - previous_position)
                segment_clearance = min(
                    segment_clearance,
                    min(
                        (
                            float(self._obstacle_clearance(segment_position, obstacle))
                            for obstacle in self.obstacles
                        ),
                        default=float("inf"),
                    ),
                )
            min_clearance = min(min_clearance, segment_clearance)
            for obstacle in self.obstacles:
                min_clearance = min(min_clearance, self._obstacle_clearance(target_position, obstacle))
            step_boundary = self._target_boundary_clearance(target_position)
            min_boundary = min(min_boundary, step_boundary)
            if timestep == 0:
                first_clearance = float(segment_clearance)
                first_boundary = step_boundary
            step_clearance = float(segment_clearance)
            if step_clearance >= required_clearance and step_boundary > 0.0:
                feasible_prefix_steps += 1
            elif failure_reason == "none":
                failure_reason = (
                    "obstacle_clearance"
                    if step_clearance < required_clearance
                    else "boundary_clearance"
                )
        estimated_capture_time = max(
            (min_distance - float(self.pursuit["capture_radius"]))
            / max(float(self.agents["defender_max_speed"]) - target_speed, 0.25),
            0.0,
        )
        escape_gap = self._target_maneuver_escape_gap(candidate["direction"], defender_positions)
        direction_change = 1.0 - float(
            np.clip(
                np.dot(
                    _unit(self.target_velocity, fallback=self.target_escape_direction),
                    _unit(candidate["direction"], fallback=self.target_escape_direction),
                ),
                -1.0,
                1.0,
            )
        )
        crossing_alignment = float(
            np.dot(
                _unit(np.asarray(candidate["direction"], dtype=np.float64), fallback=self.target_escape_direction),
                _unit(self.target_escape_direction, fallback=np.array([1.0, 0.0, 0.0])),
            )
        )
        obstacle_avoidance_alignment = float(
            np.dot(
                _unit(np.asarray(candidate["direction"], dtype=np.float64), fallback=self.target_escape_direction),
                self._target_obstacle_avoidance_direction(),
            )
        )
        score = (
            float(self.pursuit["target_maneuver_distance_weight"]) * min_distance
            + float(self.pursuit["target_maneuver_terminal_weight"]) * terminal_distance
            + float(self.pursuit["target_maneuver_clearance_weight"]) * min_clearance
            + float(self.pursuit["target_maneuver_boundary_weight"]) * min_boundary
            + float(self.pursuit["target_maneuver_gap_weight"]) * escape_gap
            + float(self.pursuit["target_maneuver_smoothness_weight"]) * estimated_capture_time
            - float(self.pursuit["target_maneuver_smoothness_weight"]) * direction_change
            + float(self.pursuit["target_maneuver_crossing_gain"]) * crossing_alignment
            + float(self.pursuit["target_maneuver_obstacle_avoidance_gain"]) * obstacle_avoidance_alignment
        )
        feasible = bool(min_clearance >= required_clearance and min_boundary > 0.0)
        if not feasible:
            score -= 1.0e3 + 1.0e2 * max(
                required_clearance - min_clearance if np.isfinite(min_clearance) else 0.0,
                -min_boundary,
                0.0,
            )
        return {
            **candidate,
            "score": float(score),
            "min_distance": float(min_distance),
            "terminal_distance": float(terminal_distance),
            "min_clearance": float(min_clearance),
            "min_boundary": float(min_boundary),
            "required_clearance": float(required_clearance),
            "required_boundary_clearance": 0.0,
            "first_clearance": float(first_clearance),
            "first_boundary": float(first_boundary),
            "feasible_prefix_steps": int(feasible_prefix_steps),
            "feasible": feasible,
            "feasibility_failure": "none" if feasible else str(failure_reason),
            "estimated_capture_time_seconds": float(estimated_capture_time),
            "escape_gap_rad": float(escape_gap),
            "crossing_alignment": float(crossing_alignment),
            "obstacle_avoidance_alignment": float(obstacle_avoidance_alignment),
        }

    def _adaptive_maneuvering_target_action(self) -> np.ndarray:
        """Choose a hidden, delayed-state, finite-horizon evasion maneuver."""

        defender_positions, defender_velocities = self._observe_defenders_for_maneuver()
        interval = int(self.pursuit["target_maneuver_replan_interval_steps"])
        predictive_boundary_risk = bool(
            self.pursuit["target_maneuver_predictive_boundary_recovery"]
            and np.linalg.norm(self._target_maneuver_boundary_direction()) > 1.0e-9
        )
        should_replan = (
            self.target_maneuver_last_replan_step < 0
            or self.step_count - self.target_maneuver_last_replan_step >= interval
            or predictive_boundary_risk
            or self.target_maneuver_last_feasibility_failure != "none"
            or self.target_maneuver_mode == "obstacle_recovery"
        )
        if should_replan:
            candidates = self._target_maneuver_candidates(defender_positions, defender_velocities)
            evaluated = [
                self._evaluate_target_maneuver_candidate(item, defender_positions, defender_velocities)
                for item in candidates
            ]
            if self.obstacles and not any(bool(item["feasible"]) for item in evaluated):
                # When every evasive maneuver has an unsafe horizon, first
                # remove kinetic energy instead of accelerating deeper into
                # the obstacle field.  This emergency candidate is injected
                # only after normal non-trivial maneuvers have all failed.
                obstacle_recovery = {
                    "mode": "obstacle_recovery",
                    "direction": self._target_obstacle_avoidance_direction(),
                    "speed_scale": 0.0,
                    "route": "obstacle_recovery_brake",
                }
                evaluated.append(
                    self._evaluate_target_maneuver_candidate(
                        obstacle_recovery,
                        defender_positions,
                        defender_velocities,
                    )
                )
            feasible_evaluated = [item for item in evaluated if bool(item["feasible"])]
            self.target_maneuver_feasible_candidate_count = len(feasible_evaluated)
            self.target_maneuver_rejected_candidate_count = len(evaluated) - len(feasible_evaluated)
            if not feasible_evaluated:
                self.target_maneuver_fallback_count += 1
                required_clearance = max(
                    float(self.pursuit["target_maneuver_feasibility_margin_m"]),
                    float(self.agents["drone_radius"]) + float(self.pursuit["safety_margin"]),
                )
                immediate_safe = [
                    item
                    for item in evaluated
                    # The long-horizon margin is a candidate preference and
                    # rejection diagnostic.  A fallback is still valid when
                    # its realized next step stays physically clear of both
                    # obstacles and the effective target boundary.
                    if float(item["first_clearance"]) > 0.0
                    and float(item["first_boundary"]) > 0.0
                ]
                fallback_pool = immediate_safe or evaluated
                if immediate_safe:
                    fallback = max(
                        fallback_pool,
                        key=lambda item: (
                            int(item["feasible_prefix_steps"]),
                            float(item["first_clearance"]),
                            float(item["first_boundary"]),
                            float(item["min_clearance"]),
                            float(item["score"]),
                        ),
                    )
                else:
                    # An unsafe fallback must never be silently promoted to a
                    # valid target trajectory.  Keep the best diagnostic
                    # action so the caller can inspect the failure, but mark
                    # the episode invalid for policy evaluation.
                    if bool(self.pursuit["target_maneuver_safety_first_fallback"]):
                        fallback = max(
                            fallback_pool,
                            key=lambda item: (
                                int(item["feasible_prefix_steps"]),
                                float(item["first_clearance"]),
                                float(item["first_boundary"]),
                                float(item["min_clearance"]),
                                float(item["min_boundary"]),
                                float(item["score"]),
                            ),
                        )
                    else:
                        fallback = max(
                            fallback_pool,
                            key=lambda item: (
                                int(item["feasible_prefix_steps"]),
                                float(item["first_clearance"]),
                                float(item["first_boundary"]),
                                float(item["score"]),
                            ),
                        )
                    self.target_maneuver_candidate_invalid = True
                self.target_maneuver_last_feasibility_failure = str(fallback["feasibility_failure"])
            else:
                fallback = None
                self.target_maneuver_last_feasibility_failure = "none"
            self.target_maneuver_decision_scores = {
                mode: max(
                    float(item["score"])
                    for item in evaluated
                    if item["mode"] == mode
                )
                for mode in sorted(_MANEUVER_MODES)
                if any(item["mode"] == mode for item in evaluated)
            }
            boundary_recovery_items = [
                item for item in evaluated if item["mode"] == "boundary_recovery"
            ]
            feasible_boundary_recovery = [
                item for item in boundary_recovery_items if bool(item["feasible"])
            ]
            force_boundary_recovery = bool(
                self.pursuit["target_maneuver_predictive_boundary_recovery"]
                and predictive_boundary_risk
                and feasible_boundary_recovery
            )
            hold_steps = self.step_count - int(self.target_maneuver_mode_start_step)
            eligible = feasible_evaluated if feasible_evaluated else [fallback]
            if hold_steps < int(self.pursuit["target_maneuver_min_hold_steps"]):
                if feasible_evaluated:
                    held_feasible = [item for item in feasible_evaluated if item["mode"] == self.target_maneuver_mode]
                    eligible = held_feasible or feasible_evaluated
                elif fallback is not None and fallback["mode"] == self.target_maneuver_mode:
                    eligible = [fallback]
            if force_boundary_recovery:
                eligible = feasible_boundary_recovery
            current_mode = self.target_maneuver_mode
            for item in eligible:
                if item["mode"] != current_mode:
                    item["score"] += float(self.pursuit["target_maneuver_switch_bonus"])
            selected = max(eligible, key=lambda item: (float(item["score"]), str(item["route"])))
            if selected["mode"] != self.target_maneuver_mode:
                self.target_maneuver_switch_count += 1
                self.target_maneuver_mode_start_step = int(self.step_count)
            self.target_maneuver_mode = str(selected["mode"])
            self.target_maneuver_mode_counts[self.target_maneuver_mode] += 1
            self.target_maneuver_direction = _unit(
                np.asarray(selected["direction"], dtype=np.float64),
                fallback=self.target_escape_direction,
            )
            self.target_maneuver_speed_scale = float(selected["speed_scale"])
            self.target_maneuver_route = str(selected["route"])
            self.target_maneuver_last_replan_step = int(self.step_count)
            self.target_maneuver_estimated_capture_time_seconds = float(
                selected["estimated_capture_time_seconds"]
            )
            self.target_maneuver_escape_gap_rad = float(selected["escape_gap_rad"])
        self.target_escape_direction = self.target_maneuver_direction.copy()
        return (
            self.target_maneuver_direction
            * float(self.agents["target_max_speed"])
            * float(self.target_speed_scale)
            * float(self.target_maneuver_speed_scale)
        )

    def _constrain_target_command(
        self,
        target_action: np.ndarray,
        *,
        enforce_maneuver_limits: bool = True,
    ) -> np.ndarray:
        """Constrain a target command and recover before it exits the safe set."""

        if enforce_maneuver_limits:
            limited = self._limit_target_command(
                self.target_velocity,
                self.target_acceleration,
                target_action,
            )
        else:
            limited = np.asarray(target_action, dtype=np.float64).copy()
        max_delta = float(self.agents["target_max_acceleration"]) * self.dt
        predicted_velocity = self._move_toward_velocity(
            self.target_velocity[None, :],
            limited[None, :],
            max_delta=max_delta,
        )[0]
        predicted_position = self.target_position + predicted_velocity * self.dt
        if self._target_boundary_clearance(predicted_position) > 0.0:
            return limited

        # A target command may be dynamically limited too late to preserve
        # the effective boundary margin.  Replace it with an inward command;
        # if even that cannot recover in one step, the episode is explicitly
        # marked invalid by the post-step contract check.
        recovery_direction = self._target_inward_direction(predicted_position)
        recovery_speed = float(self.agents["target_max_speed"]) * max(
            float(self.target_speed_scale),
            0.1,
        )
        recovery_action = recovery_direction * recovery_speed
        recovery = (
            self._limit_target_command(
                self.target_velocity,
                self.target_acceleration,
                recovery_action,
            )
            if enforce_maneuver_limits
            else recovery_action
        )
        self.target_command_clipped = True
        self.target_command_clipped_steps += 1
        return recovery

    def _adaptive_branching_target_action(self) -> np.ndarray:
        """Commit to the less interceptable S4 wall exit using simulator state.

        This is an unseen adversary, analogous to the existing adaptive target:
        defender state is used only to generate the target's behavior. The
        branch label and scores deliberately remain absent from ``observe`` so
        controllers must infer the maneuver from their observation histories.
        """

        target_speed = float(self.agents["target_max_speed"]) * float(self.target_speed_scale)
        decision_x = float(self.pursuit["target_branch_decision_x"])
        if self.target_branch_sign is None and self.target_position[0] >= decision_x:
            lookahead = float(self.pursuit["target_branch_defender_lookahead_seconds"])
            projected_defenders = self.defender_positions + self.defender_velocities * lookahead
            waypoint_x = float(self.pursuit["target_branch_waypoint_x"])
            exit_offset_y = float(self.pursuit["target_branch_exit_offset_y"])
            scores: list[float] = []
            for sign in (-1, 1):
                exit_point = np.array(
                    [-waypoint_x, sign * exit_offset_y, self.target_position[2]], dtype=np.float64
                )
                target_arrival = float(np.linalg.norm(exit_point - self.target_position)) / max(target_speed, 1e-9)
                defender_arrival = np.linalg.norm(projected_defenders - exit_point[None, :], axis=1) / max(
                    float(self.agents["defender_max_speed"]), 1e-9
                )
                # Larger margin means the closest defender reaches that exit
                # later relative to the target, and is therefore safer.
                scores.append(float(np.min(defender_arrival) - target_arrival))
            self.target_branch_scores[:] = scores
            lower_score, upper_score = scores
            margin = float(self.pursuit["target_branch_commit_margin_seconds"])
            if upper_score > lower_score + margin:
                self.target_branch_sign = 1
            elif lower_score > upper_score + margin:
                self.target_branch_sign = -1
            else:
                # Stable tie break keeps reset/replay deterministic while
                # mirrored non-tie layouts still select mirrored exits.
                self.target_branch_sign = 1
            self.target_branch_decision_step = int(self.step_count)

        if self.target_branch_sign is None:
            desired = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            sign = int(self.target_branch_sign)
            waypoint_x = float(self.pursuit["target_branch_waypoint_x"])
            exit_y = sign * float(self.pursuit["target_branch_exit_offset_y"])
            if self.target_position[0] < -waypoint_x:
                goal = np.array([-waypoint_x, exit_y, self.target_position[2]], dtype=np.float64)
            elif self.target_position[0] < waypoint_x:
                goal = np.array([waypoint_x, exit_y, self.target_position[2]], dtype=np.float64)
            else:
                goal = np.array(
                    [float(self.pursuit["target_branch_goal_x"]), exit_y, self.target_position[2]],
                    dtype=np.float64,
                )
            desired = _unit(goal - self.target_position, fallback=self.target_escape_direction)

        for defender_position in self.defender_positions:
            delta = self.target_position - defender_position
            distance = float(np.linalg.norm(delta))
            if distance < float(self.pursuit["target_defender_avoidance_distance"]):
                desired += (
                    _unit(delta)
                    * (float(self.pursuit["target_defender_avoidance_distance"]) - distance)
                    * float(self.pursuit["target_defender_avoidance_gain"])
                )
        for obstacle in self.obstacles:
            clearance, normal = self._cylinder_clearance_and_normal(self.target_position, obstacle)
            if clearance < float(self.pursuit["target_obstacle_avoidance_distance"]):
                desired += (
                    normal
                    * (float(self.pursuit["target_obstacle_avoidance_distance"]) - clearance)
                    * float(self.pursuit["target_obstacle_avoidance_gain"])
                )
        margin = float(self.pursuit["target_boundary_margin"])
        for axis in range(3):
            if self.target_position[axis] < self.lower[axis] + margin:
                desired[axis] += float(self.pursuit["target_boundary_gain"])
            if self.target_position[axis] > self.upper[axis] - margin:
                desired[axis] -= float(self.pursuit["target_boundary_gain"])
        self.target_escape_direction = _unit(desired, fallback=self.target_escape_direction)
        return self.target_escape_direction * target_speed

    def _adaptive_adversarial_target_action(self) -> np.ndarray:
        """Choose a feasible one-step escape direction from public geometry.

        This policy is intentionally environment-internal. It uses the true
        simulator state only to generate an unseen adversary; no selected
        direction, target state, or policy id is exposed through ``observe``.
        """

        centroid = self.defender_positions.mean(axis=0)
        candidates: list[np.ndarray] = [
            _unit(self.target_position - centroid, fallback=self.target_escape_direction),
            self.target_escape_direction.copy(),
        ]
        for defender_position in self.defender_positions:
            candidates.append(
                _unit(self.target_position - defender_position, fallback=self.target_escape_direction)
            )
        for axis in range(3):
            basis = np.zeros(3, dtype=np.float64)
            basis[axis] = 1.0
            candidates.extend((basis, -basis))
        candidates.extend(TETRAHEDRON_DIRECTIONS.copy())

        unique: list[np.ndarray] = []
        for candidate in candidates:
            direction = _unit(np.asarray(candidate, dtype=np.float64), fallback=self.target_escape_direction)
            if not any(float(np.dot(direction, previous)) > 1.0 - 1.0e-9 for previous in unique):
                unique.append(direction)

        target_speed = float(self.agents["target_max_speed"]) * float(self.target_speed_scale)
        max_delta = float(self.agents["target_max_acceleration"]) * float(self.dt)
        best_score = float("-inf")
        best_direction = self.target_escape_direction.copy()
        feasible_direction_found = False
        candidate_clearances: list[tuple[np.ndarray, float, float]] = []
        for direction in unique:
            desired_velocity = direction * target_speed
            predicted_velocity = self._move_toward_velocity(
                self.target_velocity[None, :],
                desired_velocity[None, :],
                max_delta=max_delta,
            )[0]
            predicted_position = self.target_position + predicted_velocity * float(self.dt)
            defender_distance = float(np.min(np.linalg.norm(self.defender_positions - predicted_position, axis=1)))
            simulated_position = self.target_position.copy()
            simulated_velocity = self.target_velocity.copy()
            obstacle_clearance = float("inf")
            boundary_clearance = float("inf")
            for _ in range(int(self.pursuit["target_adaptive_lookahead_steps"])):
                simulated_velocity = self._move_toward_velocity(
                    simulated_velocity[None, :],
                    desired_velocity[None, :],
                    max_delta=max_delta,
                )[0]
                simulated_position = simulated_position + simulated_velocity * float(self.dt)
                obstacle_clearance = min(
                    obstacle_clearance,
                    min(
                        (self._obstacle_clearance(simulated_position, obstacle) for obstacle in self.obstacles),
                        default=float("inf"),
                    ),
                )
                boundary_clearance = min(
                    boundary_clearance,
                    float(
                        min(
                            np.min(simulated_position - self.lower),
                            np.min(self.upper - simulated_position),
                        )
                    ),
                )
            candidate_clearances.append((direction, obstacle_clearance, boundary_clearance))
            if obstacle_clearance < 0.0 or boundary_clearance < 0.0:
                continue
            score = (
                float(self.pursuit["target_adaptive_defender_weight"]) * defender_distance
                + float(self.pursuit["target_adaptive_obstacle_weight"]) * obstacle_clearance
                + float(self.pursuit["target_adaptive_boundary_weight"]) * boundary_clearance
                + float(self.pursuit["target_adaptive_heading_weight"]) * float(
                    np.dot(direction, self.target_escape_direction)
                )
            )
            if score > best_score:
                best_score = score
                best_direction = direction
                feasible_direction_found = True

        if not feasible_direction_found:
            # The current state may already be close to a corner. Preserve
            # physical validity by choosing the direction with the largest
            # boundary/obstacle clearance instead of forcing an escape move.
            best_direction = max(
                candidate_clearances,
                key=lambda item: min(item[1], item[2]),
            )[0]

        self.target_escape_direction = _unit(best_direction, fallback=self.target_escape_direction)
        return self.target_escape_direction * target_speed

    def _update_target_beliefs(self) -> None:
        self.target_visible[:] = False
        delivered_this_step = np.zeros(self.n_defenders, dtype=bool)
        pending: list[_BeliefPacket] = []
        for packet in self._message_queue:
            if packet.delivery_step <= self.step_count:
                receiver = int(packet.receiver)
                delivered_this_step[receiver] |= self._deliver_belief_packet(packet)
            else:
                pending.append(packet)
        self._message_queue = pending

        for index in range(self.n_defenders):
            if self._target_is_visible(index):
                noise = self.rng.normal(0.0, float(self.pursuit["observation_noise_std"]), size=3)
                position = self.target_position + noise
                velocity = self.target_velocity + self.rng.normal(
                    0.0,
                    float(self.pursuit["observation_noise_std"]) / max(self.dt, 1e-9),
                    size=3,
                )
                self.target_visible[index] = True
                distance = float(np.linalg.norm(self.target_position - self.defender_positions[index]))
                confidence = float(
                    np.clip(
                        1.0 - distance / max(float(self.pursuit["detection_range"]), 1e-9),
                        0.05,
                        1.0,
                    )
                )
                covariance = np.eye(3, dtype=np.float64) * max(
                    float(self.pursuit["observation_noise_std"]) ** 2,
                    1e-8,
                )
                packet = _BeliefPacket(
                    delivery_step=self.step_count + int(self.pursuit["observation_delay_steps"]),
                    receiver=index,
                    source=index,
                    timestamp_step=self.step_count,
                    position=position.copy(),
                    velocity=velocity.copy(),
                    confidence=confidence,
                    covariance=covariance,
                    via_message=False,
                )
                if packet.delivery_step <= self.step_count:
                    delivered_this_step[index] |= self._deliver_belief_packet(packet)
                else:
                    self._message_queue.append(packet)
                for receiver in range(self.n_defenders):
                    if receiver == index:
                        continue
                    if (
                        self.rng.random() < float(self.pursuit["message_dropout_probability"])
                        or self.rng.random() < float(self.pursuit["communication_link_dropout_probability"])
                    ):
                        continue
                    self._message_queue.append(
                        _BeliefPacket(
                            delivery_step=self.step_count + int(self.pursuit["message_delay_steps"]),
                            receiver=receiver,
                            source=index,
                            timestamp_step=self.step_count,
                            position=position.copy(),
                            velocity=velocity.copy(),
                            confidence=confidence,
                            covariance=covariance.copy(),
                            via_message=True,
                        )
                    )
            elif not (
                str(self.pursuit["belief_update_mode"]) in {"zero_velocity", "constant_velocity", "time_aligned"}
                and delivered_this_step[index]
            ):
                belief_age = max(self.step_count - int(self.target_observation_timestamps[index]), 0)
                if (
                    str(self.pursuit["belief_update_mode"]) == "time_aligned"
                    and int(self.target_observation_timestamps[index]) >= 0
                    and belief_age >= int(self.pursuit["belief_velocity_decay_start_age_steps"])
                ):
                    self.target_belief_velocities[index] *= float(self.pursuit["belief_stale_velocity_decay"])
                self.target_belief_positions[index] += self.target_belief_velocities[index] * self.dt
                self.message_age_steps[index] = min(
                    self.message_age_steps[index] + 1,
                    int(self.pursuit["maximum_message_age_steps"]),
                )
                self.target_observation_confidence[index] *= float(self.pursuit["observation_confidence_decay"])
                self.target_observation_covariance[index] += np.eye(3, dtype=np.float64) * float(
                    self.pursuit["observation_covariance_growth"]
                )
        self._append_belief_snapshot()

    def _append_belief_snapshot(self) -> None:
        """Retain only locally available belief states for fixed-lag updates."""
        history: list[_BeliefSnapshot] = [] if self.step_count == 0 else list(getattr(self, "_belief_history", []))
        history.append(
            _BeliefSnapshot(
                step=int(self.step_count),
                positions=self.target_belief_positions.copy(),
                velocities=self.target_belief_velocities.copy(),
                confidences=self.target_observation_confidence.copy(),
                covariances=self.target_observation_covariance.copy(),
                timestamps=self.target_observation_timestamps.copy(),
            )
        )
        oldest_step = int(self.step_count) - int(self.pursuit["maximum_message_age_steps"])
        self._belief_history = [snapshot for snapshot in history if snapshot.step >= oldest_step]

    def _belief_snapshot_at(self, step: int) -> _BeliefSnapshot | None:
        for snapshot in reversed(getattr(self, "_belief_history", [])):
            if snapshot.step == step:
                return snapshot
        return None

    def _time_aligned_packet_state(
        self,
        packet: _BeliefPacket,
        receiver: int,
        age_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Fuse a delayed packet at its timestamp and propagate it to now.

        The optional prior is a receiver-local state saved at the packet's
        timestamp. It contains no simulator truth. If no valid prior exists,
        the fixed-lag update reduces to constant-velocity propagation of the
        received measurement.
        """
        position = packet.position.copy()
        velocity = packet.velocity.copy()
        confidence = float(packet.confidence)
        covariance = packet.covariance.copy()
        snapshot = self._belief_snapshot_at(int(packet.timestamp_step))
        if snapshot is not None and int(snapshot.timestamps[receiver]) >= 0:
            prior_covariance = snapshot.covariances[receiver]
            # The simulator's measurement and process covariance are diagonal
            # by construction. A per-axis gain is therefore the exact update
            # for this belief model and avoids dispatching to a platform BLAS
            # inverse merely to invert a 3 x 3 diagonal matrix.
            prior_variance = np.maximum(np.diag(prior_covariance), 1e-12)
            measurement_variance = np.maximum(np.diag(covariance), 1e-12)
            gain = prior_variance / (prior_variance + measurement_variance)
            position = snapshot.positions[receiver] + gain * (position - snapshot.positions[receiver])
            velocity = snapshot.velocities[receiver] + gain * (velocity - snapshot.velocities[receiver])
            covariance = np.diag((1.0 - gain) * prior_variance)
            confidence = max(confidence, float(snapshot.confidences[receiver]))
        position += velocity * (age_steps * self.dt)
        confidence *= float(self.pursuit["observation_confidence_decay"]) ** age_steps
        covariance += np.eye(3, dtype=np.float64) * (
            float(self.pursuit["observation_covariance_growth"]) * age_steps
        )
        return position, velocity, confidence, covariance

    def _deliver_belief_packet(self, packet: _BeliefPacket) -> bool:
        """Apply one packet only if it is newer than the receiver's belief.

        ``legacy`` retains the original benchmark semantics. ``zero_velocity``
        and ``constant_velocity`` are transparent diagnostic baselines.
        ``time_aligned`` updates a saved local state at the packet timestamp,
        then propagates the fused belief and uncertainty to the current
        decision step. All paths retain the source timestamp, so an actor can
        still distinguish a current estimate from an old measurement.
        """
        receiver = int(packet.receiver)
        if packet.timestamp_step < int(self.target_observation_timestamps[receiver]):
            return False
        age_steps = max(self.step_count - int(packet.timestamp_step), 0)
        mode = str(self.pursuit["belief_update_mode"])
        if mode == "zero_velocity":
            self.target_belief_positions[receiver] = packet.position
            self.target_belief_velocities[receiver] = np.zeros(3, dtype=np.float64)
            self.target_observation_confidence[receiver] = float(packet.confidence)
            self.target_observation_covariance[receiver] = packet.covariance.copy()
        elif mode == "constant_velocity":
            self.target_belief_positions[receiver] = packet.position + packet.velocity * (age_steps * self.dt)
            self.target_belief_velocities[receiver] = packet.velocity
            self.target_observation_confidence[receiver] = float(packet.confidence) * float(
                self.pursuit["observation_confidence_decay"]
            ) ** age_steps
            self.target_observation_covariance[receiver] = packet.covariance + np.eye(3, dtype=np.float64) * (
                float(self.pursuit["observation_covariance_growth"]) * age_steps
            )
        elif mode == "time_aligned":
            position, velocity, confidence, covariance = self._time_aligned_packet_state(packet, receiver, age_steps)
            self.target_belief_positions[receiver] = position
            self.target_belief_velocities[receiver] = velocity
            self.target_observation_confidence[receiver] = confidence
            self.target_observation_covariance[receiver] = covariance
        else:
            self.target_belief_positions[receiver] = packet.position
            self.target_belief_velocities[receiver] = packet.velocity
            self.target_observation_confidence[receiver] = float(packet.confidence)
            self.target_observation_covariance[receiver] = packet.covariance.copy()
        self.target_observation_timestamps[receiver] = int(packet.timestamp_step)
        self.message_age_steps[receiver] = min(
            age_steps,
            int(self.pursuit["maximum_message_age_steps"]),
        )
        return True

    def _predict_target_beliefs(self) -> tuple[np.ndarray, np.ndarray]:
        """Predict each local target belief without reading simulator target truth.

        This conservative constant-velocity predictor is the V1 interface for
        trajectory prediction. A learned predictor can replace its output as
        long as it consumes only the same per-defender belief history.
        """
        horizon = float(self.pursuit["prediction_horizon_seconds"])
        positions = self.target_belief_positions + horizon * self.target_belief_velocities
        ages = self.message_age_steps.astype(np.float64) * self.dt
        uncertainty = float(self.pursuit["prediction_uncertainty_base"]) + ages * float(
            self.agents["target_max_speed"]
        )
        return positions.copy(), uncertainty.astype(np.float64, copy=False)

    def _target_is_visible(self, defender_index: int) -> bool:
        if self.detection_loss_burst_remaining[defender_index] > 0:
            self.detection_loss_burst_remaining[defender_index] -= 1
            return False
        if self.rng.random() < float(self.pursuit["detection_loss_burst_probability"]):
            self.detection_loss_burst_remaining[defender_index] = max(
                int(self.pursuit["detection_loss_burst_duration_steps"]) - 1,
                0,
            )
            return False
        if self.rng.random() < float(self.pursuit["detection_dropout_probability"]):
            return False
        origin = self.defender_positions[defender_index]
        delta = self.target_position - origin
        distance = float(np.linalg.norm(delta))
        if distance > float(self.pursuit["detection_range"]):
            return False
        velocity = self.defender_velocities[defender_index]
        if np.linalg.norm(velocity) > 1e-6:
            heading_cosine = float(np.sum(_unit(velocity) * _unit(delta)))
            if heading_cosine < float(self.pursuit["visibility_cosine_threshold"]):
                return False
        return not any(self._segment_blocked_by_cylinder(origin, self.target_position, obstacle) for obstacle in self.obstacles)

    @staticmethod
    def _segment_blocked_by_cylinder(start: np.ndarray, end: np.ndarray, obstacle: CylinderObstacle) -> bool:
        if obstacle.shape != "cylinder":
            for fraction in np.linspace(0.0, 1.0, 41):
                point = start + fraction * (end - start)
                clearance, _normal = CaptureRadiusPursuit3DEnv._box_clearance_and_normal(point, obstacle)
                if clearance <= 0.0:
                    return True
            return False
        direction_xy = end[:2] - start[:2]
        squared_length = float(np.sum(direction_xy * direction_xy))
        if squared_length < 1e-12:
            return False
        interpolation = float(
            np.clip(
                np.sum((obstacle.center_xy - start[:2]) * direction_xy) / squared_length,
                0.0,
                1.0,
            )
        )
        closest_xy = start[:2] + interpolation * direction_xy
        if float(np.linalg.norm(closest_xy - obstacle.center_xy)) > obstacle.radius:
            return False
        height_at_closest = float(start[2] + interpolation * (end[2] - start[2]))
        return 0.0 <= height_at_closest <= obstacle.height

    def _metrics(self) -> PursuitEpisodeMetrics:
        target_distances = self._target_distances()
        nearest_defender = int(np.argmin(target_distances))
        radius = float(self.agents["drone_radius"])
        clearances: list[float] = []
        for position in self.defender_positions:
            for obstacle in self.obstacles:
                clearance, _normal = self._cylinder_clearance_and_normal(position, obstacle)
                clearances.append(clearance - radius)
        for first in range(self.n_defenders):
            for second in range(first + 1, self.n_defenders):
                clearances.append(
                    float(np.linalg.norm(self.defender_positions[first] - self.defender_positions[second]) - 2.0 * radius)
                )
        min_clearance = float(min(clearances)) if clearances else float("inf")
        return PursuitEpisodeMetrics(
            minimum_target_distance=float(target_distances[nearest_defender]),
            nearest_defender=nearest_defender,
            collision=bool(min_clearance < 0.0),
            physical_target_contact=bool(float(target_distances[nearest_defender]) <= 2.0 * radius),
            min_clearance=min_clearance,
        )

    def _coverage_score(self) -> float:
        vectors = self.defender_positions - self.target_position[None, :]
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
        # Four defenders make this tiny pairwise calculation. Elementwise
        # summation avoids dispatching to a BLAS/OpenMP runtime during a
        # PyTorch training process.
        pairwise = np.sum(vectors[:, None, :] * vectors[None, :, :], axis=2)
        upper = pairwise[np.triu_indices(self.n_defenders, k=1)]
        return float(np.clip(-np.mean(upper), -1.0, 1.0))

    def _target_distances(self) -> np.ndarray:
        return np.linalg.norm(self.defender_positions - self.target_position[None, :], axis=1)

    def _sample_obstacles(self) -> list[CylinderObstacle]:
        obstacles: list[CylinderObstacle] = []
        protected_points = np.vstack([self.defender_positions, self.target_position[None, :]])
        profile = str(self.pursuit["obstacle_profile"])
        for _ in range(self.obstacle_count):
            for _attempt in range(200):
                shape = profile
                if profile == "mixed":
                    shape = str(self.rng.choice(["cylinder", "box", "walls"]))
                elif profile == "narrow_channels":
                    shape = "wall"
                if shape == "cylinders":
                    shape = "cylinder"
                if shape == "boxes":
                    shape = "box"
                if shape == "walls":
                    shape = "wall"
                radius = float(self.rng.uniform(0.65, 1.15))
                height = float(self.rng.uniform(3.0, 7.0))
                center_xy = self.rng.uniform(-7.5, 7.5, size=2)
                half_extents_xy: np.ndarray | None = None
                if shape == "box":
                    half_extents_xy = self.rng.uniform(0.65, 1.5, size=2).astype(np.float64)
                    radius = float(np.max(half_extents_xy))
                elif shape == "wall":
                    long_extent = float(self.rng.uniform(3.0, 5.0))
                    short_extent = float(self.rng.uniform(0.25, 0.45))
                    if self.rng.random() < 0.5:
                        half_extents_xy = np.array([long_extent, short_extent], dtype=np.float64)
                    else:
                        half_extents_xy = np.array([short_extent, long_extent], dtype=np.float64)
                    radius = short_extent
                    if profile == "narrow_channels":
                        long_extent = float(self.rng.uniform(3.5, 5.5))
                        short_extent = float(self.rng.uniform(0.30, 0.40))
                        half_extents_xy = np.array([long_extent, short_extent], dtype=np.float64)
                        radius = short_extent
                candidate = CylinderObstacle(
                    center_xy=center_xy,
                    radius=radius,
                    height=height,
                    shape="cylinder" if shape == "cylinder" else shape,
                    half_extents_xy=half_extents_xy,
                )
                if self._obstacle_clear_of_points(candidate, protected_points) and all(
                    self._obstacle_horizontal_separation(candidate, existing) >= 1.0
                    for existing in obstacles
                ):
                    obstacles.append(candidate)
                    break
            else:
                raise RuntimeError("Unable to sample a non-overlapping pursuit obstacle layout.")
        return obstacles

    def _obstacle_clear_of_points(self, obstacle: CylinderObstacle, points: np.ndarray) -> bool:
        clearances = [self._obstacle_clearance(point, obstacle) for point in points]
        return bool(all(clearance > 1.8 for clearance in clearances))

    @staticmethod
    def _box_clearance_and_normal(position: np.ndarray, obstacle: CylinderObstacle) -> tuple[float, np.ndarray]:
        if obstacle.half_extents_xy is None:
            raise ValueError("Box/wall obstacle requires half_extents_xy.")
        center = np.array([obstacle.center_xy[0], obstacle.center_xy[1], obstacle.height * 0.5], dtype=np.float64)
        half = np.array([obstacle.half_extents_xy[0], obstacle.half_extents_xy[1], obstacle.height * 0.5], dtype=np.float64)
        delta = np.asarray(position, dtype=np.float64) - center
        signed = np.abs(delta) - half
        outside = np.maximum(signed, 0.0)
        outside_norm = float(np.linalg.norm(outside))
        if outside_norm > 1e-9:
            normal = outside * np.sign(delta)
            return outside_norm, _unit(normal, fallback=np.array([1.0, 0.0, 0.0]))
        penetration = float(np.min(-signed))
        axis = int(np.argmin(-signed))
        normal = np.zeros(3, dtype=np.float64)
        normal[axis] = 1.0 if delta[axis] >= 0.0 else -1.0
        return -penetration, normal

    def _obstacle_clearance(self, position: np.ndarray, obstacle: CylinderObstacle) -> float:
        if obstacle.shape == "cylinder":
            clearance, _normal = self._cylinder_clearance_and_normal(position, obstacle)
            return float(clearance)
        clearance, _normal = self._box_clearance_and_normal(position, obstacle)
        return float(clearance)

    @staticmethod
    def _obstacle_horizontal_separation(first: CylinderObstacle, second: CylinderObstacle) -> float:
        first_half = (
            np.array([first.radius, first.radius], dtype=np.float64)
            if first.half_extents_xy is None
            else np.asarray(first.half_extents_xy, dtype=np.float64)
        )
        second_half = (
            np.array([second.radius, second.radius], dtype=np.float64)
            if second.half_extents_xy is None
            else np.asarray(second.half_extents_xy, dtype=np.float64)
        )
        delta = np.abs(first.center_xy - second.center_xy) - first_half - second_half
        gap = np.maximum(delta, 0.0)
        if np.any(delta <= 0.0):
            # If the axis-aligned footprints overlap in one direction, use
            # the separating gap in the other direction.  This is less
            # conservative than bounding every wall by a large circle and
            # keeps narrow-channel maps sampleable.
            return float(np.max(gap))
        return float(np.linalg.norm(gap))

    def _cylinder_clearance_and_normal(self, position: np.ndarray, obstacle: CylinderObstacle) -> tuple[float, np.ndarray]:
        if obstacle.shape != "cylinder":
            return self._box_clearance_and_normal(position, obstacle)
        xy_delta = position[:2] - obstacle.center_xy
        xy_norm = float(np.linalg.norm(xy_delta))
        radial_normal = _unit(
            np.array([xy_delta[0], xy_delta[1], 0.0]),
            fallback=np.array([1.0, 0.0, 0.0]),
        )
        radial_gap = xy_norm - obstacle.radius
        if 0.0 <= position[2] <= obstacle.height:
            return radial_gap, radial_normal
        nearest_z = 0.0 if position[2] < 0.0 else obstacle.height
        vertical_gap = abs(position[2] - nearest_z)
        if radial_gap <= 0.0:
            normal = np.array([0.0, 0.0, -1.0 if position[2] < 0.0 else 1.0])
            return vertical_gap, normal
        clearance = float(np.hypot(radial_gap, vertical_gap))
        normal = _unit(radial_normal * radial_gap + np.array([0.0, 0.0, position[2] - nearest_z]))
        return clearance, normal

    def _enforce_world_bounds(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        *,
        entity: str | None = None,
    ) -> None:
        if entity not in {None, "target", "defender"}:
            raise ValueError(f"unknown world-bound entity: {entity}")
        inferred_entity = entity
        if inferred_entity is None and positions.shape[0] == self.n_defenders:
            inferred_entity = "defender"
        for axis in range(3):
            below = positions[:, axis] < self.lower[axis]
            above = positions[:, axis] > self.upper[axis]
            if bool(np.any(below | above)):
                count = int(np.count_nonzero(below | above))
                self.world_violation_steps += count
                if inferred_entity == "target":
                    self.target_world_violation_steps += count
                    if self.first_target_boundary_violation_step is None:
                        self.first_target_boundary_violation_step = int(self.step_count + 1)
                elif inferred_entity == "defender":
                    self.defender_world_violation_steps += count
                    if self.first_defender_boundary_violation_step is None:
                        self.first_defender_boundary_violation_step = int(self.step_count + 1)
            positions[below, axis] = self.lower[axis]
            positions[above, axis] = self.upper[axis]
            velocities[below | above, axis] *= -0.4

    @staticmethod
    def _clip_rows(values: np.ndarray, max_norm: float) -> np.ndarray:
        return clip_rows(values, max_norm)

    @staticmethod
    def _move_toward_velocity(current: np.ndarray, desired: np.ndarray, max_delta: float) -> np.ndarray:
        return move_toward_velocity(current, desired, max_delta)

    def _record_history(self) -> None:
        self.history.append(
            {
                "defender_positions": self.defender_positions.copy(),
                "target_position": self.target_position.copy(),
                "target_velocity": self.target_velocity.copy(),
                "target_acceleration": self.target_acceleration.copy(),
                "belief_positions": self.target_belief_positions.copy(),
                "capture_radius": float(self.pursuit["capture_radius"]),
                "target_branch_sign": self.target_branch_sign,
                "target_branch_decision_step": self.target_branch_decision_step,
                "target_maneuver_mode": self.target_maneuver_mode,
                "target_maneuver_route": self.target_maneuver_route,
                "step": int(self.step_count),
            }
        )
