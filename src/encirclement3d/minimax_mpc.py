"""Finite-shooting scenario risk-sensitive MPC for target encirclement.

This module intentionally implements the first, centralized diagnostic planner.
It consumes only policy-safe observations and dynamics-projected target
candidate trajectories.  Obstacle and inter-agent terms are rollout penalties;
the existing local CBF remains the post-planner safety layer until the robust
QP contract is implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Literal

import numpy as np

from .pursuit_env import TETRAHEDRON_DIRECTIONS, _unit
from .reachability_interception import reachability_normalized_interception_cost


RiskMode = Literal["expected", "worst_case", "cvar"]
ProjectionStatus = Literal["raw", "projected"]


@dataclass(frozen=True)
class ScenarioTrajectorySet:
    """Planner-facing absolute target trajectory candidates.

    ``trajectories[k, t]`` is the predicted target position at the future
    timestep ``t + 1``.  The planner refuses ``raw`` candidates because the
    diffusion predictor can violate the shared velocity/acceleration contract
    before projection.
    """

    trajectories: np.ndarray
    weights: np.ndarray
    score_kind: str = "uniform_uncalibrated"
    dynamics_status: ProjectionStatus = "projected"
    conformal_radius_m: float | None = None
    conformal_radius_by_step_m: tuple[float, ...] | None = None
    source_model_hash: str | None = None
    timestamp_step: int = -1

    def __post_init__(self) -> None:
        trajectories = np.asarray(self.trajectories, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        if trajectories.ndim != 3 or trajectories.shape[-1] != 3 or trajectories.shape[0] <= 0:
            raise ValueError("trajectories must have shape [candidates, horizon, 3].")
        if trajectories.shape[1] <= 0 or not np.isfinite(trajectories).all():
            raise ValueError("trajectories must be finite and have a positive horizon.")
        if weights.shape != (trajectories.shape[0],) or not np.isfinite(weights).all():
            raise ValueError("weights must have shape [candidates] and be finite.")
        if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
            raise ValueError("weights must be non-negative with a positive sum.")
        if self.score_kind not in {"uniform_uncalibrated", "calibrated_region", "calibrated_probability"}:
            raise ValueError(f"Unsupported score_kind: {self.score_kind}")
        if self.dynamics_status not in {"raw", "projected"}:
            raise ValueError(f"Unsupported dynamics_status: {self.dynamics_status}")
        if self.conformal_radius_m is not None and (
            not np.isfinite(float(self.conformal_radius_m)) or float(self.conformal_radius_m) < 0.0
        ):
            raise ValueError("conformal_radius_m must be finite and non-negative when supplied.")
        if self.conformal_radius_by_step_m is not None:
            radius = np.asarray(self.conformal_radius_by_step_m, dtype=np.float64)
            if (
                radius.ndim != 1
                or radius.shape[0] != trajectories.shape[1]
                or not np.isfinite(radius).all()
                or np.any(radius < 0.0)
            ):
                raise ValueError("conformal_radius_by_step_m must match the candidate horizon and be non-negative.")
            object.__setattr__(self, "conformal_radius_by_step_m", tuple(float(value) for value in radius))
            if self.conformal_radius_m is None:
                object.__setattr__(self, "conformal_radius_m", float(np.mean(radius)))

    @property
    def candidate_count(self) -> int:
        return int(np.asarray(self.trajectories).shape[0])

    @property
    def horizon_steps(self) -> int:
        return int(np.asarray(self.trajectories).shape[1])

    @property
    def normalized_weights(self) -> np.ndarray:
        weights = np.asarray(self.weights, dtype=np.float64)
        return weights / float(weights.sum())

    def truncate(self, horizon_steps: int) -> "ScenarioTrajectorySet":
        if horizon_steps <= 0:
            raise ValueError("horizon_steps must be positive.")
        if self.horizon_steps < horizon_steps:
            raise ValueError("Candidate horizon is shorter than the requested planner horizon.")
        return ScenarioTrajectorySet(
            trajectories=np.asarray(self.trajectories)[:, :horizon_steps].copy(),
            weights=np.asarray(self.weights).copy(),
            score_kind=self.score_kind,
            dynamics_status=self.dynamics_status,
            conformal_radius_m=(
                None
                if self.conformal_radius_by_step_m is not None
                else self.conformal_radius_m
            ),
            conformal_radius_by_step_m=(
                None
                if self.conformal_radius_by_step_m is None
                else self.conformal_radius_by_step_m[:horizon_steps]
            ),
            source_model_hash=self.source_model_hash,
            timestamp_step=self.timestamp_step,
        )


@dataclass(frozen=True)
class MinimaxMPCConfig:
    """Configuration for the centralized finite-shooting planner."""

    dt_seconds: float = 0.1
    horizon_steps: int = 8
    control_horizon_steps: int = 3
    max_speed_mps: float = 5.0
    action_change_limit_mps: float | None = None
    capture_radius_m: float = 0.8
    drone_radius_m: float = 0.25
    safety_margin_m: float = 0.35
    minimum_inter_agent_distance_m: float = 1.0
    risk_mode: RiskMode = "worst_case"
    cvar_alpha: float = 0.75
    role_perimeter_m: float = 1.4
    slot_gain: float = 2.5
    target_velocity_gain: float = 1.0
    weight_distance: float = 1.0
    weight_terminal_distance: float = 3.0
    weight_capture_hinge: float = 1.5
    weight_formation: float = 0.25
    weight_control: float = 0.015
    weight_control_change: float = 0.02
    weight_obstacle: float = 25.0
    weight_boundary: float = 25.0
    weight_inter_agent: float = 25.0
    weight_relative_speed: float = 0.02
    reachability_normalized_cost_enabled: bool = False
    weight_reachability: float = 0.0
    reachability_time_margin_s: float = 0.15
    reachability_time_scale_s: float = 0.50
    reachability_max_acceleration_mps2: float = 6.0
    max_role_variants: int = 4
    perimeter_scales: tuple[float, ...] = (0.75, 1.0)

    def __post_init__(self) -> None:
        positive = {
            "dt_seconds": self.dt_seconds,
            "horizon_steps": self.horizon_steps,
            "control_horizon_steps": self.control_horizon_steps,
            "max_speed_mps": self.max_speed_mps,
            "capture_radius_m": self.capture_radius_m,
            "drone_radius_m": self.drone_radius_m,
            "minimum_inter_agent_distance_m": self.minimum_inter_agent_distance_m,
            "role_perimeter_m": self.role_perimeter_m,
            "slot_gain": self.slot_gain,
            "max_role_variants": self.max_role_variants,
        }
        for name, value in positive.items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.control_horizon_steps > self.horizon_steps:
            raise ValueError("control_horizon_steps cannot exceed horizon_steps.")
        if self.action_change_limit_mps is not None and (
            not np.isfinite(float(self.action_change_limit_mps))
            or float(self.action_change_limit_mps) <= 0.0
        ):
            raise ValueError("action_change_limit_mps must be finite and positive when provided.")
        if self.risk_mode not in {"expected", "worst_case", "cvar"}:
            raise ValueError("risk_mode must be expected, worst_case, or cvar.")
        if not 0.0 <= float(self.cvar_alpha) < 1.0:
            raise ValueError("cvar_alpha must lie in [0, 1).")
        if not self.perimeter_scales or any(float(value) <= 0.0 for value in self.perimeter_scales):
            raise ValueError("perimeter_scales must contain positive values.")
        weights = (
            self.weight_distance,
            self.weight_terminal_distance,
            self.weight_capture_hinge,
            self.weight_formation,
            self.weight_control,
            self.weight_control_change,
            self.weight_obstacle,
            self.weight_boundary,
            self.weight_inter_agent,
            self.weight_relative_speed,
            self.weight_reachability,
        )
        if any(not np.isfinite(float(value)) or float(value) < 0.0 for value in weights):
            raise ValueError("Cost weights must be finite and non-negative.")
        if not np.isfinite(float(self.reachability_time_margin_s)) or float(self.reachability_time_margin_s) < 0.0:
            raise ValueError("reachability_time_margin_s must be finite and non-negative")
        if not np.isfinite(float(self.reachability_time_scale_s)) or float(self.reachability_time_scale_s) <= 0.0:
            raise ValueError("reachability_time_scale_s must be finite and positive")
        if not np.isfinite(float(self.reachability_max_acceleration_mps2)) or float(self.reachability_max_acceleration_mps2) <= 0.0:
            raise ValueError("reachability_max_acceleration_mps2 must be finite and positive")

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "MinimaxMPCConfig":
        """Build a validated config from a YAML-style planner mapping."""

        values = dict(mapping)
        # YAML keeps a human-readable implementation label alongside numeric
        # planner parameters; it is metadata rather than a dataclass field.
        values.pop("implementation", None)
        if "perimeter_scales" in values:
            values["perimeter_scales"] = tuple(float(value) for value in values["perimeter_scales"])
        return cls(**values)


@dataclass(frozen=True)
class MinimaxMPCDiagnostics:
    status: str
    risk_mode: str
    selected_sequence_index: int
    candidate_sequence_count: int
    scenario_costs: tuple[float, ...]
    objective_value: float
    expected_cost: float
    worst_case_cost: float
    cvar_cost: float
    latency_ms: float
    max_rollout_constraint_violation: float
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "risk_mode": self.risk_mode,
            "selected_sequence_index": self.selected_sequence_index,
            "candidate_sequence_count": self.candidate_sequence_count,
            "scenario_costs": list(self.scenario_costs),
            "objective_value": self.objective_value,
            "expected_cost": self.expected_cost,
            "worst_case_cost": self.worst_case_cost,
            "cvar_cost": self.cvar_cost,
            "latency_ms": self.latency_ms,
            "max_rollout_constraint_violation": self.max_rollout_constraint_violation,
            "fallback_reason": self.fallback_reason,
        }


@dataclass(frozen=True)
class MinimaxMPCPlan:
    actions: np.ndarray
    action_sequence: np.ndarray
    diagnostics: MinimaxMPCDiagnostics

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float64)
        sequence = np.asarray(self.action_sequence, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[-1] != 3:
            raise ValueError("actions must have shape [defenders, 3].")
        if sequence.ndim != 3 or sequence.shape[1:] != actions.shape:
            raise ValueError("action_sequence must have shape [horizon, defenders, 3].")
        if not np.isfinite(actions).all() or not np.isfinite(sequence).all():
            raise ValueError("Plan actions must be finite.")


def _clip_rows(values: np.ndarray, max_norm: float) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    return rows * np.minimum(1.0, float(max_norm) / np.maximum(norms, 1e-12))


def _project_actions(
    desired: np.ndarray,
    previous: np.ndarray,
    *,
    max_speed_mps: float,
    action_change_limit_mps: float | None,
) -> np.ndarray:
    """Project commands onto the shared speed and command-increment contract."""

    action = np.asarray(desired, dtype=np.float64)
    previous_action = np.asarray(previous, dtype=np.float64)
    if action.shape != previous_action.shape or action.shape[-1] != 3:
        raise ValueError("desired and previous actions must share a final xyz dimension.")
    if action_change_limit_mps is None:
        return _clip_rows(action, max_speed_mps)

    # Alternating projection onto the per-axis change box and the Euclidean
    # speed ball. Their intersection contains the observed velocity under the
    # environment contract, and the small fixed budget is deterministic.
    limit = float(action_change_limit_mps)
    lower = previous_action - limit
    upper = previous_action + limit
    projected = action.copy()
    for _ in range(12):
        projected = np.clip(projected, lower, upper)
        projected = _clip_rows(projected, max_speed_mps)
    return np.clip(projected, lower, upper)


def _belief_reference(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    beliefs = np.asarray(observation["target_belief_positions"], dtype=np.float64)
    velocities = np.asarray(observation["target_belief_velocities"], dtype=np.float64)
    confidences = np.asarray(observation.get("target_observation_confidence", np.ones(len(beliefs))), dtype=np.float64)
    ages = np.asarray(observation.get("message_age_steps", np.zeros(len(beliefs))), dtype=np.float64)
    if beliefs.ndim != 2 or beliefs.shape[1] != 3 or velocities.shape != beliefs.shape:
        raise ValueError("Belief positions and velocities must have shape [defenders, 3].")
    weights = np.maximum(confidences, 1e-3) / (1.0 + np.maximum(ages, 0.0))
    weights /= max(float(weights.sum()), 1e-12)
    return np.sum(beliefs * weights[:, None], axis=0), np.sum(velocities * weights[:, None], axis=0)


def _obstacle_clearance(position: np.ndarray, obstacle: dict[str, Any]) -> float:
    point = np.asarray(position, dtype=np.float64)
    center_xy = np.asarray(obstacle["center_xy"], dtype=np.float64)
    height = float(obstacle["height"])
    shape = str(obstacle.get("shape", "cylinder"))
    if shape == "cylinder":
        radial = float(np.linalg.norm(point[:2] - center_xy) - float(obstacle["radius"]))
        vertical = max(-point[2], point[2] - height, 0.0)
        if vertical == 0.0:
            return radial
        return vertical if radial <= 0.0 else float(np.hypot(radial, vertical))
    half = obstacle.get("half_extents_xy")
    if half is None:
        half = [float(obstacle["radius"]), float(obstacle["radius"])]
    center = np.array([center_xy[0], center_xy[1], height * 0.5], dtype=np.float64)
    half_extent = np.array([float(half[0]), float(half[1]), height * 0.5], dtype=np.float64)
    signed = np.abs(point - center) - half_extent
    outside = np.maximum(signed, 0.0)
    outside_norm = float(np.linalg.norm(outside))
    return outside_norm if outside_norm > 0.0 else -float(np.max(-signed))


def _weighted_cvar(values: np.ndarray, weights: np.ndarray, alpha: float) -> float:
    """Return upper-tail weighted CVaR for a finite cost vector."""

    costs = np.asarray(values, dtype=np.float64)
    probabilities = np.asarray(weights, dtype=np.float64)
    probabilities = probabilities / float(probabilities.sum())
    tail_mass = max(1.0 - float(alpha), 1e-12)
    order = np.argsort(-costs)
    remaining = tail_mass
    total = 0.0
    for index in order:
        mass = min(float(probabilities[index]), remaining)
        total += mass * float(costs[index])
        remaining -= mass
        if remaining <= 1e-12:
            break
    return float(total / tail_mass)


def aggregate_scenario_costs(values: Iterable[float], weights: Iterable[float], risk_mode: RiskMode, cvar_alpha: float) -> float:
    """Aggregate scenario costs using the planner's declared risk objective."""

    costs = np.asarray(list(values), dtype=np.float64)
    probabilities = np.asarray(list(weights), dtype=np.float64)
    if costs.ndim != 1 or costs.size == 0 or not np.isfinite(costs).all():
        raise ValueError("Scenario costs must be a non-empty finite vector.")
    if probabilities.shape != costs.shape or np.any(probabilities < 0.0) or float(probabilities.sum()) <= 0.0:
        raise ValueError("Scenario weights must match costs and have a positive sum.")
    probabilities = probabilities / float(probabilities.sum())
    if risk_mode == "expected":
        return float(np.dot(probabilities, costs))
    if risk_mode == "worst_case":
        return float(np.max(costs))
    if risk_mode == "cvar":
        return _weighted_cvar(costs, probabilities, cvar_alpha)
    raise ValueError(f"Unsupported risk mode: {risk_mode}")


def evaluate_candidate_capture_distances(
    observation: dict[str, Any],
    action_sequence: np.ndarray,
    scenarios: ScenarioTrajectorySet,
    *,
    dt_seconds: float,
    max_speed_mps: float,
) -> dict[str, np.ndarray]:
    """Evaluate planned candidate-level terminal and closest distances.

    The rollout uses only the planner observation, the selected nominal action
    sequence, and projected candidate paths.  It is therefore suitable for
    diagnostics without exposing the environment's true target trajectory.
    ``terminal_distances_m`` is the nearest-defender distance at the end of
    the planned horizon; ``minimum_distances_m`` is its minimum over that
    horizon.  Higher values are worse for both metrics.
    """

    if dt_seconds <= 0.0 or max_speed_mps <= 0.0:
        raise ValueError("dt_seconds and max_speed_mps must be positive.")
    actions = np.asarray(action_sequence, dtype=np.float64)
    paths = np.asarray(scenarios.trajectories, dtype=np.float64)
    positions = np.asarray(observation["defender_positions"], dtype=np.float64).copy()
    if actions.ndim != 3 or actions.shape[-1] != 3:
        raise ValueError("action_sequence must have shape [horizon, defenders, 3].")
    if paths.ndim != 3 or paths.shape[-1] != 3 or paths.shape[1] != actions.shape[0]:
        raise ValueError("Candidate paths and action sequence have incompatible horizons.")
    if positions.ndim != 2 or positions.shape[-1] != 3 or positions.shape[0] != actions.shape[1]:
        raise ValueError("Observation and action sequence have incompatible defender shapes.")

    minimum_distances = np.full(paths.shape[0], np.inf, dtype=np.float64)
    terminal_distances = np.full(paths.shape[0], np.inf, dtype=np.float64)
    for timestep, target_positions in enumerate(paths.transpose(1, 0, 2)):
        action = _clip_rows(actions[timestep], max_speed_mps)
        positions += action * float(dt_seconds)
        distances = np.linalg.norm(positions[None, :, :] - target_positions[:, None, :], axis=-1)
        nearest_distances = np.min(distances, axis=1)
        minimum_distances = np.minimum(minimum_distances, nearest_distances)
        if timestep == actions.shape[0] - 1:
            terminal_distances = nearest_distances
    if not np.isfinite(minimum_distances).all() or not np.isfinite(terminal_distances).all():
        raise FloatingPointError("Candidate distance rollout produced non-finite values.")
    return {
        "terminal_distances_m": terminal_distances,
        "minimum_distances_m": minimum_distances,
    }


def make_belief_candidate_set(
    observation: dict[str, Any],
    *,
    horizon_steps: int,
    dt_seconds: float,
    max_speed_mps: float,
    candidate_count: int = 8,
    lateral_spread_m: float = 0.6,
) -> ScenarioTrajectorySet:
    """Create a projected constant-velocity candidate set for smoke baselines.

    This is deliberately a baseline, not a replacement for the diffusion
    predictor. It is useful for checking planner mechanics without reading
    target truth or loading a checkpoint.
    """

    if horizon_steps <= 0 or candidate_count <= 0 or dt_seconds <= 0.0:
        raise ValueError("horizon_steps, candidate_count and dt_seconds must be positive.")
    position, velocity = _belief_reference(observation)
    velocity = _clip_rows(velocity[None, :], max_speed_mps)[0]
    direction = _unit(velocity, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    lateral = _unit(np.cross(direction, np.array([0.0, 0.0, 1.0])), fallback=np.array([0.0, 1.0, 0.0]))
    vertical = _unit(np.cross(direction, lateral), fallback=np.array([0.0, 0.0, 1.0]))
    offsets = np.zeros((candidate_count, 3), dtype=np.float64)
    for index in range(candidate_count):
        angle = 2.0 * np.pi * index / max(candidate_count, 1)
        offsets[index] = float(lateral_spread_m) * (
            np.cos(angle) * lateral + 0.5 * np.sin(angle) * vertical
        )
    times = (np.arange(horizon_steps, dtype=np.float64) + 1.0)[:, None]
    trajectories = position[None, None, :] + times[None, :, :] * float(dt_seconds) * velocity[None, None, :]
    trajectories = np.repeat(trajectories, candidate_count, axis=0) + offsets[:, None, :]
    return ScenarioTrajectorySet(
        trajectories=trajectories,
        weights=np.ones(candidate_count, dtype=np.float64),
        score_kind="uniform_uncalibrated",
        dynamics_status="projected",
    )


class ScenarioMinimaxMPC:
    """Centralized finite-shooting scenario MPC diagnostic planner."""

    def __init__(self, config: MinimaxMPCConfig) -> None:
        self.config = config

    def plan(
        self,
        observation: dict[str, Any],
        scenarios: ScenarioTrajectorySet,
        *,
        fallback_actions: np.ndarray | None = None,
    ) -> MinimaxMPCPlan:
        started = perf_counter()
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        velocities = np.asarray(observation["defender_velocities"], dtype=np.float64)
        if positions.ndim != 2 or positions.shape[-1] != 3 or velocities.shape != positions.shape:
            raise ValueError("Defender positions and velocities must have shape [defenders, 3].")
        if scenarios.dynamics_status != "projected":
            return self._fallback(
                positions,
                velocities,
                fallback_actions,
                started,
                scenarios,
                "planner requires dynamics-projected candidates",
            )
        if scenarios.horizon_steps < self.config.horizon_steps:
            return self._fallback(
                positions,
                velocities,
                fallback_actions,
                started,
                scenarios,
                "candidate horizon is shorter than planner horizon",
            )
        try:
            candidates = scenarios.truncate(self.config.horizon_steps)
            sequences = self._candidate_action_sequences(observation, candidates, velocities)
            if not sequences:
                raise RuntimeError("finite-shooting candidate set is empty")
            scenario_cost_matrix, constraint_violations = self._scenario_cost_matrix(
                observation,
                np.stack(sequences, axis=0),
                np.asarray(candidates.trajectories, dtype=np.float64),
            )
            if not np.isfinite(scenario_cost_matrix).all():
                raise FloatingPointError("scenario cost matrix contains non-finite values")
            objectives = np.asarray(
                [
                    aggregate_scenario_costs(
                        row,
                        candidates.normalized_weights,
                        self.config.risk_mode,
                        self.config.cvar_alpha,
                    )
                    for row in scenario_cost_matrix
                ],
                dtype=np.float64,
            )
            selected = int(np.argmin(objectives))
            selected_costs = scenario_cost_matrix[selected]
            expected = aggregate_scenario_costs(
                selected_costs,
                candidates.normalized_weights,
                "expected",
                self.config.cvar_alpha,
            )
            worst = aggregate_scenario_costs(
                selected_costs,
                candidates.normalized_weights,
                "worst_case",
                self.config.cvar_alpha,
            )
            cvar = aggregate_scenario_costs(
                selected_costs,
                candidates.normalized_weights,
                "cvar",
                self.config.cvar_alpha,
            )
            sequence = sequences[selected]
            diagnostics = MinimaxMPCDiagnostics(
                status="success",
                risk_mode=self.config.risk_mode,
                selected_sequence_index=selected,
                candidate_sequence_count=len(sequences),
                scenario_costs=tuple(float(value) for value in selected_costs),
                objective_value=float(objectives[selected]),
                expected_cost=float(expected),
                worst_case_cost=float(worst),
                cvar_cost=float(cvar),
                latency_ms=(perf_counter() - started) * 1000.0,
                max_rollout_constraint_violation=float(np.max(constraint_violations[selected])),
            )
            return MinimaxMPCPlan(
                actions=sequence[0].copy(),
                action_sequence=sequence.copy(),
                diagnostics=diagnostics,
            )
        except (FloatingPointError, RuntimeError, ValueError, KeyError, IndexError) as error:
            return self._fallback(positions, velocities, fallback_actions, started, scenarios, str(error))

    def _fallback(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        fallback_actions: np.ndarray | None,
        started: float,
        scenarios: ScenarioTrajectorySet,
        reason: str,
    ) -> MinimaxMPCPlan:
        if fallback_actions is None:
            _target, target_velocity = _belief_reference(
                {
                    "target_belief_positions": np.repeat(positions.mean(axis=0, keepdims=True), positions.shape[0], axis=0),
                    "target_belief_velocities": np.zeros_like(positions),
                }
            )
            del target_velocity
            actions = np.zeros_like(positions)
        else:
            actions = np.asarray(fallback_actions, dtype=np.float64)
            if actions.shape != positions.shape or not np.isfinite(actions).all():
                actions = np.zeros_like(positions)
        actions = _project_actions(
            actions,
            np.asarray(velocities, dtype=np.float64),
            max_speed_mps=self.config.max_speed_mps,
            action_change_limit_mps=self.config.action_change_limit_mps,
        )
        sequence = np.repeat(actions[None, :, :], self.config.horizon_steps, axis=0)
        diagnostics = MinimaxMPCDiagnostics(
            status="fallback",
            risk_mode=self.config.risk_mode,
            selected_sequence_index=-1,
            candidate_sequence_count=0,
            scenario_costs=tuple(),
            objective_value=float("nan"),
            expected_cost=float("nan"),
            worst_case_cost=float("nan"),
            cvar_cost=float("nan"),
            latency_ms=(perf_counter() - started) * 1000.0,
            max_rollout_constraint_violation=0.0,
            fallback_reason=reason,
        )
        return MinimaxMPCPlan(actions=actions, action_sequence=sequence, diagnostics=diagnostics)

    def _candidate_action_sequences(
        self,
        observation: dict[str, Any],
        scenarios: ScenarioTrajectorySet,
        initial_velocity: np.ndarray,
    ) -> list[np.ndarray]:
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        _belief, belief_velocity = _belief_reference(observation)
        first_target = scenarios.trajectories[:, 0]
        distances = np.linalg.norm(positions[:, None, :] - first_target[None, :, :], axis=-1)
        ordered_interceptors = list(np.argsort(np.min(distances, axis=1)).astype(int))
        role_ids = ordered_interceptors[: min(self.config.max_role_variants, positions.shape[0])]
        if not role_ids:
            role_ids = [int(np.argmin(np.linalg.norm(positions - first_target.mean(axis=0), axis=1)))]

        sequences: list[np.ndarray] = []
        reference_path = np.average(scenarios.trajectories, axis=0, weights=scenarios.normalized_weights)
        for target_path in [reference_path, *list(scenarios.trajectories)]:
            for interceptor_id in role_ids:
                for perimeter_scale in self.config.perimeter_scales:
                    sequences.append(
                        self._track_path(
                            positions,
                            initial_velocity,
                            belief_velocity,
                            target_path,
                            interceptor_id,
                            float(perimeter_scale),
                        )
                    )
        return sequences

    def _track_path(
        self,
        initial_positions: np.ndarray,
        initial_velocity: np.ndarray,
        target_velocity: np.ndarray,
        target_path: np.ndarray,
        interceptor_id: int,
        perimeter_scale: float,
    ) -> np.ndarray:
        positions = np.asarray(initial_positions, dtype=np.float64).copy()
        previous_action = np.asarray(initial_velocity, dtype=np.float64).copy()
        target_path = np.asarray(target_path, dtype=np.float64)
        actions: list[np.ndarray] = []
        perimeter = self.config.role_perimeter_m * float(perimeter_scale)
        for timestep in range(self.config.horizon_steps):
            if timestep >= self.config.control_horizon_steps and actions:
                action = actions[-1].copy()
            else:
                target = target_path[timestep]
                if timestep > 0:
                    target_velocity_step = (target_path[timestep] - target_path[timestep - 1]) / self.config.dt_seconds
                else:
                    target_velocity_step = np.asarray(target_velocity, dtype=np.float64)
                action = np.zeros_like(positions)
                for defender_id, position in enumerate(positions):
                    if defender_id == interceptor_id:
                        target_point = target
                    else:
                        direction = TETRAHEDRON_DIRECTIONS[defender_id]
                        target_point = target + direction * perimeter
                    desired = self.config.slot_gain * (target_point - position)
                    desired += self.config.target_velocity_gain * target_velocity_step
                    action[defender_id] = desired
                action = _project_actions(
                    action,
                    previous_action,
                    max_speed_mps=self.config.max_speed_mps,
                    action_change_limit_mps=self.config.action_change_limit_mps,
                )
            actions.append(action)
            positions = positions + action * self.config.dt_seconds
            previous_action = action
        return np.stack(actions, axis=0)

    def _scenario_cost_matrix(
        self,
        observation: dict[str, Any],
        action_sequences: np.ndarray,
        target_paths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate all shooting sequences and scenarios in one vectorized pass."""

        sequences = np.asarray(action_sequences, dtype=np.float64)
        paths = np.asarray(target_paths, dtype=np.float64)
        if sequences.ndim != 4 or paths.ndim != 3 or sequences.shape[-1] != 3 or paths.shape[-1] != 3:
            raise ValueError("action_sequences and target_paths have incompatible shapes.")
        sequence_count, horizon, defender_count, _ = sequences.shape
        candidate_count, candidate_horizon, _ = paths.shape
        if horizon != candidate_horizon or horizon != self.config.horizon_steps:
            raise ValueError("Action and target horizons must match planner horizon.")
        initial_positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        positions = np.broadcast_to(initial_positions[None, :, :], (sequence_count, defender_count, 3)).copy()
        previous_action = np.broadcast_to(
            np.asarray(observation["defender_velocities"], dtype=np.float64)[None, :, :],
            (sequence_count, defender_count, 3),
        ).copy()
        scenario_costs = np.zeros((sequence_count, candidate_count), dtype=np.float64)
        constraint_violations = np.zeros((sequence_count, candidate_count), dtype=np.float64)
        position_paths: list[np.ndarray] = []
        velocity_paths: list[np.ndarray] = []
        lower = np.asarray(observation.get("world_lower_bounds", [-np.inf, -np.inf, -np.inf]), dtype=np.float64)
        upper = np.asarray(observation.get("world_upper_bounds", [np.inf, np.inf, np.inf]), dtype=np.float64)
        if "world_lower_bounds" not in observation:
            # The public observation intentionally does not expose bounds;
            # callers may attach them from the environment for diagnostics.
            lower = np.full(3, -np.inf, dtype=np.float64)
            upper = np.full(3, np.inf, dtype=np.float64)
        for timestep in range(horizon):
            action = _project_actions(
                sequences[:, timestep],
                previous_action,
                max_speed_mps=self.config.max_speed_mps,
                action_change_limit_mps=self.config.action_change_limit_mps,
            )
            positions += action * self.config.dt_seconds
            position_paths.append(positions.copy())
            velocity_paths.append(action.copy())
            target = paths[:, timestep]
            delta = positions[:, None, :, :] - target[None, :, None, :]
            distances = np.linalg.norm(delta, axis=-1)
            minimum_distance = np.min(distances, axis=2)
            mean_distance = np.mean(distances, axis=2)
            scenario_costs += self.config.weight_distance * mean_distance
            scenario_costs += self.config.weight_capture_hinge * np.maximum(
                minimum_distance - self.config.capture_radius_m, 0.0
            ) ** 2
            if timestep == horizon - 1:
                scenario_costs += self.config.weight_terminal_distance * minimum_distance
            nearest = np.argmin(distances, axis=2)
            blocker_mask = np.ones((sequence_count, candidate_count, defender_count), dtype=bool)
            blocker_mask[
                np.arange(sequence_count)[:, None],
                np.arange(candidate_count)[None, :],
                nearest,
            ] = False
            formation_error = np.where(
                blocker_mask,
                np.abs(distances - self.config.role_perimeter_m),
                0.0,
            )
            scenario_costs += self.config.weight_formation * formation_error.sum(axis=2) / max(defender_count - 1, 1)
            scenario_costs += self.config.weight_control * np.sum(action * action, axis=(1, 2))[:, None] / max(defender_count, 1)
            change = action - previous_action
            scenario_costs += self.config.weight_control_change * np.sum(change * change, axis=(1, 2))[:, None] / max(defender_count, 1)
            scenario_costs += self.config.weight_relative_speed * np.mean(np.linalg.norm(change, axis=2), axis=1)[:, None]
            previous_action = action

            for obstacle in observation.get("obstacles", []):
                shape = str(obstacle.get("shape", "cylinder"))
                center_xy = np.asarray(obstacle["center_xy"], dtype=np.float64)
                height = float(obstacle["height"])
                if shape == "cylinder":
                    radial = np.linalg.norm(positions[..., :2] - center_xy, axis=-1) - float(obstacle["radius"])
                    vertical = np.maximum.reduce((-positions[..., 2], positions[..., 2] - height, np.zeros_like(radial)))
                    clearance = np.where(
                        vertical == 0.0,
                        radial,
                        np.where(radial <= 0.0, vertical, np.hypot(radial, vertical)),
                    )
                else:
                    half = obstacle.get("half_extents_xy")
                    if half is None:
                        half = [float(obstacle["radius"]), float(obstacle["radius"])]
                    center = np.array([center_xy[0], center_xy[1], height * 0.5])
                    half_extent = np.array([float(half[0]), float(half[1]), height * 0.5])
                    signed = np.abs(positions - center) - half_extent
                    outside = np.maximum(signed, 0.0)
                    outside_norm = np.linalg.norm(outside, axis=-1)
                    clearance = np.where(outside_norm > 0.0, outside_norm, -np.max(-signed, axis=-1))
                violation = np.maximum(self.config.safety_margin_m - (clearance - self.config.drone_radius_m), 0.0)
                scenario_costs += self.config.weight_obstacle * np.sum(violation * violation, axis=1)[:, None]
                constraint_violations = np.maximum(constraint_violations, np.max(violation, axis=1)[:, None])

            lower_violation = np.maximum(lower[None, None, :] - positions, 0.0)
            upper_violation = np.maximum(positions - upper[None, None, :], 0.0)
            boundary_violation = np.max(np.maximum(lower_violation, upper_violation), axis=(1, 2))
            scenario_costs += self.config.weight_boundary * boundary_violation[:, None] ** 2
            constraint_violations = np.maximum(constraint_violations, boundary_violation[:, None])
            for first in range(defender_count):
                for second in range(first + 1, defender_count):
                    inter_agent_violation = np.maximum(
                        self.config.minimum_inter_agent_distance_m
                        - np.linalg.norm(positions[:, first] - positions[:, second], axis=1),
                        0.0,
                    )
                    scenario_costs += self.config.weight_inter_agent * inter_agent_violation[:, None] ** 2
                    constraint_violations = np.maximum(
                        constraint_violations,
                        inter_agent_violation[:, None],
                    )
        if self.config.reachability_normalized_cost_enabled:
            reachability_cost, _best_slack, _arrival_times = reachability_normalized_interception_cost(
                np.stack(position_paths, axis=1),
                np.stack(velocity_paths, axis=1),
                paths,
                dt_seconds=self.config.dt_seconds,
                max_speed_mps=self.config.max_speed_mps,
                max_acceleration_mps2=self.config.reachability_max_acceleration_mps2,
                time_margin_s=self.config.reachability_time_margin_s,
                time_scale_s=self.config.reachability_time_scale_s,
                target_tube_radius_m=candidates.conformal_radius_by_step_m,
            )
            scenario_costs += self.config.weight_reachability * reachability_cost
        return scenario_costs, constraint_violations


__all__ = [
    "MinimaxMPCConfig",
    "MinimaxMPCDiagnostics",
    "MinimaxMPCPlan",
    "ScenarioMinimaxMPC",
    "ScenarioTrajectorySet",
    "aggregate_scenario_costs",
    "evaluate_candidate_capture_distances",
    "make_belief_candidate_set",
]
