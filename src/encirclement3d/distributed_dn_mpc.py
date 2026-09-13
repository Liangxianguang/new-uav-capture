"""Finite-candidate distributed best-response DN-MPC.

The implementation is deliberately explicit about its information boundary.
Each defender chooses a local action sequence from its own state, local
obstacles, shared projected target candidates, and the latest received peer
messages.  The message model is stateful so delayed and dropped updates can
be evaluated in the same closed-loop rollout as the ideal-communication
oracle.  Target truth is never read here.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np

from .minimax_mpc import (
    MinimaxMPCConfig,
    ScenarioTrajectorySet,
    aggregate_scenario_costs,
    belief_fusion_diagnostics,
    belief_reference,
)
from .escape_gap import escape_gap_metrics
from .feasible_consensus import (
    evaluate_fixed_consensus_slots,
    fc_dbf_summary,
    feasible_consensus_slot_gate,
)
from .reachability_interception import (
    formation_slot_reachability_cost,
    reachability_normalized_interception_cost,
)
from .pursuit_env import TETRAHEDRON_DIRECTIONS, _unit
from .execution_dynamics import (
    parameters_from_observation,
    rollout_action_candidates,
    rollout_action_sequence,
)


CommunicationMode = Literal["none", "ideal", "delayed", "dropout"]
_COMMUNICATION_MODES = {"none", "ideal", "delayed", "dropout"}


def _qdr_execution_aware(observation: dict[str, Any]) -> bool:
    metadata = observation.get("qdr", {})
    return isinstance(metadata, dict) and bool(
        metadata.get("execution_aware_action_rollout", False)
    )


def _rollout_local_action_candidates(
    observation: dict[str, Any],
    actions: np.ndarray,
    agent_id: int,
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return position/velocity paths for own commands under the QDR contract."""

    positions = np.asarray(observation["defender_positions"], dtype=np.float64)
    velocities = np.asarray(observation["defender_velocities"], dtype=np.float64)
    parameters = parameters_from_observation(observation, float(dt_seconds))
    candidate_actions = np.asarray(actions, dtype=np.float64)
    initial_position = np.broadcast_to(
        positions[agent_id], (candidate_actions.shape[0], 3)
    ).copy()
    initial_velocity = np.broadcast_to(
        velocities[agent_id], (candidate_actions.shape[0], 3)
    ).copy()
    position_paths, velocity_paths, _steps = rollout_action_candidates(
        initial_position,
        initial_velocity,
        candidate_actions,
        parameters,
    )
    return position_paths, velocity_paths


def _rollout_peer_action_path(
    observation: dict[str, Any],
    message: _PlannerMessage,
    action_sequence: np.ndarray | None,
    dt_seconds: float,
    horizon_steps: int,
    max_speed_mps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out a peer message with the same execution convention as own actions."""

    if action_sequence is None:
        actions = np.broadcast_to(
            np.asarray(message.velocity, dtype=np.float64),
            (int(horizon_steps), 3),
        ).copy()
    else:
        actions = _clip_rows(np.asarray(action_sequence, dtype=np.float64), max_speed_mps)
    if _qdr_execution_aware(observation):
        parameters = parameters_from_observation(observation, float(dt_seconds))
        position_path, velocity_path, _steps = rollout_action_sequence(
            np.asarray(message.position, dtype=np.float64)[None, :],
            np.asarray(message.velocity, dtype=np.float64)[None, :],
            actions[:, None, :],
            parameters,
        )
        # The helper rolls a one-agent batch and therefore returns [H, 1, 3].
        # Local distributed costs use the peer convention [H, 3], matching the
        # ideal-execution branch below and avoiding an accidental H-by-H
        # broadcast in the candidate/peer subtraction.
        return position_path[:, 0, :], velocity_path[:, 0, :]
    return (
        np.asarray(message.position, dtype=np.float64)[None, :]
        + np.cumsum(actions * float(dt_seconds), axis=0),
        actions,
    )


def _rollout_full_action_sequence(
    observation: dict[str, Any],
    sequences: np.ndarray,
    dt_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out a complete team command sequence for cooperative costs."""

    actions = np.asarray(sequences, dtype=np.float64)
    positions = np.asarray(observation["defender_positions"], dtype=np.float64)
    velocities = np.asarray(observation["defender_velocities"], dtype=np.float64)
    if _qdr_execution_aware(observation):
        return rollout_action_sequence(
            positions,
            velocities,
            actions,
            parameters_from_observation(observation, float(dt_seconds)),
        )[:2]
    return (
        positions[None, :, :] + np.cumsum(actions * float(dt_seconds), axis=0),
        actions,
    )


@dataclass(frozen=True)
class DistributedDNMPCConfig:
    """Communication and local-search parameters for distributed DN-MPC."""

    communication_mode: CommunicationMode = "ideal"
    max_iterations: int = 3
    convergence_tolerance_mps: float = 0.15
    communication_interval_steps: int = 1
    message_delay_steps: int = 0
    message_dropout_probability: float = 0.0
    max_message_age_steps: int = 8
    local_timeout_ms: float = 50.0
    max_local_candidate_paths: int = 4
    local_obstacle_range_m: float = 8.0
    bytes_per_float: int = 8
    fallback_policy: str = "fallback_actions_then_previous_sequence"

    def __post_init__(self) -> None:
        if self.communication_mode not in _COMMUNICATION_MODES:
            raise ValueError(f"Unsupported communication_mode: {self.communication_mode}")
        positive_integer = {
            "max_iterations": self.max_iterations,
            "communication_interval_steps": self.communication_interval_steps,
            "max_message_age_steps": self.max_message_age_steps,
            "max_local_candidate_paths": self.max_local_candidate_paths,
            "bytes_per_float": self.bytes_per_float,
        }
        for name, value in positive_integer.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive.")
        positive_float = {
            "convergence_tolerance_mps": self.convergence_tolerance_mps,
            "local_timeout_ms": self.local_timeout_ms,
            "local_obstacle_range_m": self.local_obstacle_range_m,
        }
        for name, value in positive_float.items():
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if int(self.message_delay_steps) < 0:
            raise ValueError("message_delay_steps must be non-negative.")
        if not 0.0 <= float(self.message_dropout_probability) <= 1.0:
            raise ValueError("message_dropout_probability must lie in [0, 1].")
        if not str(self.fallback_policy).strip():
            raise ValueError("fallback_policy must be non-empty.")

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "DistributedDNMPCConfig":
        values = dict(mapping)
        return cls(**values)


@dataclass(frozen=True)
class DistributedDNMPCDiagnostics:
    status: str
    communication_mode: str
    iterations: int
    converged: bool
    local_solver_failures: int
    messages_attempted: int
    messages_sent: int
    messages_received: int
    messages_dropped: int
    message_bytes_sent: int
    message_bytes_received: int
    max_message_age_steps: int
    mean_message_age_steps: float
    scenario_costs: tuple[float, ...]
    objective_value: float
    expected_cost: float
    worst_case_cost: float
    cvar_cost: float
    latency_ms: float
    max_action_delta_mps: float
    fallback_reason: str | None = None
    qdr_suffix_gate_active: bool = False
    qdr_suffix_gate_exhausted: bool = False
    qdr_suffix_gate_rejected_candidates: int = 0
    escape_gap_cost: float = float("nan")
    escape_gap_max_rad: float = float("nan")
    escape_gap_escape_rad: float = float("nan")
    escape_gap_coverage_ratio: float = float("nan")
    escape_gap_violation_rate: float = float("nan")
    fc_dbf_enabled: bool = False
    fc_dbf_feasible: bool = False
    fc_dbf_min_slot_slack_s: float = float("nan")
    fc_dbf_max_slot_error_m: float = float("nan")
    fc_dbf_mean_slot_error_m: float = float("nan")
    fc_dbf_mean_slot_progress_m: float = float("nan")
    fc_dbf_assignment_switch_rate: float = float("nan")
    fc_dbf_feasible_rate: float = float("nan")
    fc_dbf_gate_exhausted: bool = False
    fc_dbf_cost: float = float("nan")
    belief_fusion_enabled: float = 0.0
    belief_fusion_effective_sample_size: float = float("nan")
    belief_fusion_weight_entropy: float = float("nan")
    belief_fusion_mean_age_steps: float = float("nan")
    belief_fusion_max_age_steps: float = float("nan")
    belief_fusion_mean_covariance_trace_m2: float = float("nan")
    belief_fusion_fallback_used: float = 0.0
    belief_fusion_fallback_reason: str = "disabled"

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "communication_mode": self.communication_mode,
            "iterations": self.iterations,
            "converged": self.converged,
            "local_solver_failures": self.local_solver_failures,
            "messages_attempted": self.messages_attempted,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "messages_dropped": self.messages_dropped,
            "message_bytes_sent": self.message_bytes_sent,
            "message_bytes_received": self.message_bytes_received,
            "max_message_age_steps": self.max_message_age_steps,
            "mean_message_age_steps": self.mean_message_age_steps,
            "scenario_costs": list(self.scenario_costs),
            "objective_value": self.objective_value,
            "expected_cost": self.expected_cost,
            "worst_case_cost": self.worst_case_cost,
            "cvar_cost": self.cvar_cost,
            "latency_ms": self.latency_ms,
            "max_action_delta_mps": self.max_action_delta_mps,
            "fallback_reason": self.fallback_reason,
            "qdr_suffix_gate_active": self.qdr_suffix_gate_active,
            "qdr_suffix_gate_exhausted": self.qdr_suffix_gate_exhausted,
            "qdr_suffix_gate_rejected_candidates": self.qdr_suffix_gate_rejected_candidates,
            "escape_gap_cost": self.escape_gap_cost,
            "escape_gap_max_rad": self.escape_gap_max_rad,
            "escape_gap_escape_rad": self.escape_gap_escape_rad,
            "escape_gap_coverage_ratio": self.escape_gap_coverage_ratio,
            "escape_gap_violation_rate": self.escape_gap_violation_rate,
            "fc_dbf_enabled": self.fc_dbf_enabled,
            "fc_dbf_feasible": self.fc_dbf_feasible,
            "fc_dbf_min_slot_slack_s": self.fc_dbf_min_slot_slack_s,
            "fc_dbf_max_slot_error_m": self.fc_dbf_max_slot_error_m,
            "fc_dbf_mean_slot_error_m": self.fc_dbf_mean_slot_error_m,
            "fc_dbf_mean_slot_progress_m": self.fc_dbf_mean_slot_progress_m,
            "fc_dbf_assignment_switch_rate": self.fc_dbf_assignment_switch_rate,
            "fc_dbf_feasible_rate": self.fc_dbf_feasible_rate,
            "fc_dbf_gate_exhausted": self.fc_dbf_gate_exhausted,
            "fc_dbf_cost": self.fc_dbf_cost,
            "belief_fusion_enabled": self.belief_fusion_enabled,
            "belief_fusion_effective_sample_size": self.belief_fusion_effective_sample_size,
            "belief_fusion_weight_entropy": self.belief_fusion_weight_entropy,
            "belief_fusion_mean_age_steps": self.belief_fusion_mean_age_steps,
            "belief_fusion_max_age_steps": self.belief_fusion_max_age_steps,
            "belief_fusion_mean_covariance_trace_m2": self.belief_fusion_mean_covariance_trace_m2,
            "belief_fusion_fallback_used": self.belief_fusion_fallback_used,
            "belief_fusion_fallback_reason": self.belief_fusion_fallback_reason,
        }


@dataclass(frozen=True)
class DistributedDNMPCPlan:
    actions: np.ndarray
    action_sequence: np.ndarray
    diagnostics: DistributedDNMPCDiagnostics

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float64)
        sequence = np.asarray(self.action_sequence, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[-1] != 3:
            raise ValueError("actions must have shape [defenders, 3].")
        if sequence.ndim != 3 or sequence.shape[1:] != actions.shape:
            raise ValueError("action_sequence must have shape [horizon, defenders, 3].")
        if not np.isfinite(actions).all() or not np.isfinite(sequence).all():
            raise ValueError("Plan actions must be finite.")


@dataclass
class _PlannerMessage:
    sender: int
    receiver: int
    sent_step: int
    delivery_step: int
    position: np.ndarray
    velocity: np.ndarray
    action_sequence: np.ndarray


def _clip_rows(values: np.ndarray, max_norm: float) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=-1, keepdims=True)
    return rows * np.minimum(1.0, float(max_norm) / np.maximum(norms, 1e-12))


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


class DistributedMinimaxDNMPC:
    """Sequential best-response planner with explicit bounded communication."""

    def __init__(self, config: MinimaxMPCConfig, distributed: DistributedDNMPCConfig) -> None:
        self.config = config
        self.distributed = distributed
        self.reset()

    def reset(self) -> None:
        self._defender_count: int | None = None
        self._inbox: list[dict[int, _PlannerMessage]] = []
        self._pending: list[_PlannerMessage] = []
        self._last_step = -1
        self._stats: dict[str, Any] = {}
        self._fc_dbf_previous_assignment: np.ndarray | None = None
        self._fc_dbf_last_metrics: dict[str, np.ndarray | float | bool] | None = None
        self._fc_dbf_last_gate_exhausted = False

    def plan(
        self,
        observation: dict[str, Any],
        scenarios: ScenarioTrajectorySet,
        *,
        step_index: int = 0,
        previous_action_sequence: np.ndarray | None = None,
        fallback_actions: np.ndarray | None = None,
    ) -> DistributedDNMPCPlan:
        started = perf_counter()
        positions = np.asarray(observation.get("defender_positions"), dtype=np.float64)
        velocities = np.asarray(observation.get("defender_velocities"), dtype=np.float64)
        if positions.ndim != 2 or positions.shape[-1] != 3 or velocities.shape != positions.shape:
            raise ValueError("Defender positions and velocities must have shape [defenders, 3].")
        if int(step_index) < 0:
            raise ValueError("step_index must be non-negative.")
        self._prepare_step(int(step_index), positions.shape[0])
        self._stats = {
            "messages_attempted": 0,
            "messages_sent": 0,
            "messages_received": 0,
            "messages_dropped": 0,
            "message_bytes_sent": 0,
            "message_bytes_received": 0,
            "age_samples": [],
            "fc_dbf_local_checks": 0,
            "fc_dbf_incomplete_consensus": 0,
            "fc_dbf_gate_exhausted": 0,
            "qdr_suffix_gate_checks": 0,
            "qdr_suffix_gate_exhausted": 0,
            "qdr_suffix_gate_rejected_candidates": 0,
        }
        self._fc_dbf_last_metrics = None
        self._fc_dbf_last_gate_exhausted = False
        if scenarios.dynamics_status != "projected":
            return self._fallback(
                positions,
                fallback_actions,
                started,
                "planner requires dynamics-projected candidates",
            )
        if scenarios.horizon_steps < self.config.horizon_steps:
            return self._fallback(
                positions,
                fallback_actions,
                started,
                "candidate horizon is shorter than planner horizon",
            )
        try:
            candidates = scenarios.truncate(self.config.horizon_steps)
            previous = self._validated_previous_sequence(previous_action_sequence, positions.shape[0])
            sequences = self._initial_sequences(observation, candidates, positions, previous)
            self._broadcast_sequences(int(step_index), positions, velocities, sequences)
            self._deliver_messages(int(step_index))

            local_failures = 0
            first_failure: str | None = None
            converged = False
            iterations = 0
            max_delta = float("inf")
            for iteration in range(self.distributed.max_iterations):
                iterations = iteration + 1
                before = sequences.copy()
                for agent_id in range(positions.shape[0]):
                    known = self._known_peer_messages(agent_id, int(step_index))
                    try:
                        local_candidates = self._local_candidate_sequences(
                            observation,
                            candidates,
                            agent_id,
                            positions[agent_id],
                            known,
                        )
                        if not local_candidates:
                            raise RuntimeError("local finite-shooting candidate set is empty")
                        peer_sequences = {
                            peer_id: message.action_sequence
                            for peer_id, message in known.items()
                            if message.action_sequence.shape == sequences[:, peer_id].shape
                        }
                        selected, _costs = self._select_local_sequence(
                            observation,
                            candidates,
                            agent_id,
                            local_candidates,
                            known,
                            peer_sequences,
                        )
                        if (perf_counter() - started) * 1000.0 > self.distributed.local_timeout_ms * (
                            iteration * positions.shape[0] + agent_id + 1
                        ):
                            raise TimeoutError("distributed planner exceeded local time budget")
                        sequences[:, agent_id] = selected
                    except (FloatingPointError, KeyError, RuntimeError, TimeoutError, ValueError) as error:
                        local_failures += 1
                        if previous is not None:
                            sequences[:, agent_id] = previous[:, agent_id]
                        if first_failure is None:
                            first_failure = str(error)
                    self._broadcast_one(
                        int(step_index),
                        positions,
                        velocities,
                        agent_id,
                        sequences[:, agent_id],
                    )
                    self._deliver_messages(int(step_index))
                max_delta = float(np.max(np.linalg.norm(sequences - before, axis=2)))
                if max_delta <= self.distributed.convergence_tolerance_mps:
                    converged = True
                    break

            if not np.isfinite(sequences).all():
                raise FloatingPointError("distributed action sequence contains non-finite values")
            scenario_costs = self._team_scenario_costs(observation, candidates, sequences, int(step_index))
            expected = aggregate_scenario_costs(
                scenario_costs,
                candidates.normalized_weights,
                "expected",
                self.config.cvar_alpha,
            )
            worst = aggregate_scenario_costs(
                scenario_costs,
                candidates.normalized_weights,
                "worst_case",
                self.config.cvar_alpha,
            )
            cvar = aggregate_scenario_costs(
                scenario_costs,
                candidates.normalized_weights,
                "cvar",
                self.config.cvar_alpha,
            )
            objective = aggregate_scenario_costs(
                scenario_costs,
                candidates.normalized_weights,
                self.config.risk_mode,
                self.config.cvar_alpha,
            )
            if local_failures:
                status = "partial_fallback"
            elif converged:
                status = "success"
            else:
                status = "not_converged"
            escape_gap_summary = self._escape_gap_summary(
                observation,
                sequences,
                np.asarray(candidates.trajectories, dtype=np.float64),
                candidates.normalized_weights,
            )
            fc_summary: dict[str, Any] = {}
            if self._fc_dbf_last_metrics is not None:
                critical_scenario = int(np.argmax(scenario_costs)) if scenario_costs.size else 0
                fc_summary = fc_dbf_summary(
                    self._fc_dbf_last_metrics,
                    0,
                    critical_scenario,
                )
                fc_summary["fc_dbf_gate_exhausted"] = bool(self._fc_dbf_last_gate_exhausted)
                if not self._fc_dbf_last_gate_exhausted:
                    assignments = np.asarray(self._fc_dbf_last_metrics["assignments"], dtype=np.int64)
                    self._fc_dbf_previous_assignment = assignments[0, critical_scenario].copy()
            fusion_summary = belief_fusion_diagnostics(observation, self.config)
            diagnostics = self._diagnostics(
                status=status,
                iterations=iterations,
                converged=converged,
                local_solver_failures=local_failures,
                scenario_costs=scenario_costs,
                objective=objective,
                expected=expected,
                worst=worst,
                cvar=cvar,
                latency_ms=(perf_counter() - started) * 1000.0,
                max_delta=max_delta,
                fallback_reason=(first_failure if local_failures else None),
                qdr_suffix_gate_active=_qdr_execution_aware(observation),
                qdr_suffix_gate_exhausted=bool(self._stats["qdr_suffix_gate_exhausted"]),
                qdr_suffix_gate_rejected_candidates=int(
                    self._stats["qdr_suffix_gate_rejected_candidates"]
                ),
                **escape_gap_summary,
                **fc_summary,
                **fusion_summary,
            )
            return DistributedDNMPCPlan(
                actions=sequences[0].copy(),
                action_sequence=sequences.copy(),
                diagnostics=diagnostics,
            )
        except (FloatingPointError, KeyError, RuntimeError, ValueError) as error:
            return self._fallback(positions, fallback_actions, started, str(error))

    def _prepare_step(self, step_index: int, defender_count: int) -> None:
        if self._defender_count != defender_count:
            self.reset()
            self._defender_count = defender_count
            self._inbox = [dict() for _ in range(defender_count)]
        elif step_index == 0 and self._last_step >= 0:
            self.reset()
            self._defender_count = defender_count
            self._inbox = [dict() for _ in range(defender_count)]
        elif step_index < self._last_step:
            raise ValueError("step_index must be non-decreasing within an episode")
        self._last_step = step_index

    def _validated_previous_sequence(self, value: np.ndarray | None, defender_count: int) -> np.ndarray | None:
        if value is None:
            return None
        sequence = np.asarray(value, dtype=np.float64)
        expected = (self.config.horizon_steps, defender_count, 3)
        if sequence.shape != expected or not np.isfinite(sequence).all():
            return None
        return _clip_rows(sequence, self.config.max_speed_mps)

    def _initial_sequences(
        self,
        observation: dict[str, Any],
        scenarios: ScenarioTrajectorySet,
        positions: np.ndarray,
        previous: np.ndarray | None,
    ) -> np.ndarray:
        if previous is not None:
            return previous.copy()
        sequences = []
        for agent_id, position in enumerate(positions):
            local = self._local_candidate_sequences(observation, scenarios, agent_id, position, {})
            if not local:
                raise RuntimeError("unable to initialize local action sequence")
            sequences.append(local[0])
        return np.stack(sequences, axis=1)

    def _broadcast_sequences(
        self,
        step_index: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        sequences: np.ndarray,
    ) -> None:
        for sender in range(positions.shape[0]):
            self._broadcast_one(step_index, positions, velocities, sender, sequences[:, sender])

    def _broadcast_one(
        self,
        step_index: int,
        positions: np.ndarray,
        velocities: np.ndarray,
        sender: int,
        sequence: np.ndarray,
    ) -> None:
        if self.distributed.communication_mode == "none":
            return
        if step_index % self.distributed.communication_interval_steps != 0:
            return
        delay = 0 if self.distributed.communication_mode == "ideal" else self.distributed.message_delay_steps
        for receiver in range(positions.shape[0]):
            if receiver == sender:
                continue
            self._stats["messages_attempted"] += 1
            if (
                self.distributed.communication_mode == "dropout"
                and self._dropout(sender, receiver, step_index)
            ):
                self._stats["messages_dropped"] += 1
                continue
            message = _PlannerMessage(
                sender=sender,
                receiver=receiver,
                sent_step=step_index,
                delivery_step=step_index + int(delay),
                position=np.asarray(positions[sender], dtype=np.float64).copy(),
                velocity=np.asarray(velocities[sender], dtype=np.float64).copy(),
                action_sequence=np.asarray(sequence, dtype=np.float64).copy(),
            )
            self._pending.append(message)
            self._stats["messages_sent"] += 1
            self._stats["message_bytes_sent"] += self._message_bytes(message)

    def _dropout(self, sender: int, receiver: int, step_index: int) -> bool:
        # A deterministic hash keeps communication ablations reproducible and
        # independent from the simulator's target/observation RNG stream.
        value = (7919 * (step_index + 1) + 104729 * (sender + 1) + 15485863 * (receiver + 1)) % 1000003
        return float(value) / 1000003.0 < float(self.distributed.message_dropout_probability)

    def _deliver_messages(self, step_index: int) -> None:
        remaining: list[_PlannerMessage] = []
        for message in self._pending:
            if message.delivery_step > step_index:
                remaining.append(message)
                continue
            current = self._inbox[message.receiver].get(message.sender)
            if current is None or message.sent_step >= current.sent_step:
                self._inbox[message.receiver][message.sender] = message
                self._stats["messages_received"] += 1
                self._stats["message_bytes_received"] += self._message_bytes(message)
        self._pending = remaining

    def _known_peer_messages(self, receiver: int, step_index: int) -> dict[int, _PlannerMessage]:
        known: dict[int, _PlannerMessage] = {}
        for sender, message in self._inbox[receiver].items():
            age = max(step_index - int(message.sent_step), 0)
            if age <= self.distributed.max_message_age_steps:
                known[int(sender)] = message
                self._stats["age_samples"].append(age)
        return known

    def _message_bytes(self, message: _PlannerMessage) -> int:
        scalar_count = int(message.position.size + message.velocity.size + message.action_sequence.size)
        return scalar_count * int(self.distributed.bytes_per_float)

    def _local_candidate_sequences(
        self,
        observation: dict[str, Any],
        scenarios: ScenarioTrajectorySet,
        agent_id: int,
        own_position: np.ndarray,
        known: dict[int, _PlannerMessage],
    ) -> list[np.ndarray]:
        paths = np.asarray(scenarios.trajectories, dtype=np.float64)
        reference = np.average(paths, axis=0, weights=scenarios.normalized_weights)
        selected_paths = [reference]
        selected_paths.extend(list(paths[: self.distributed.max_local_candidate_paths]))
        team_positions = {agent_id: np.asarray(own_position, dtype=np.float64)}
        team_positions.update({peer: message.position for peer, message in known.items()})
        first_target = reference[0]
        interceptor = min(team_positions, key=lambda peer: float(np.linalg.norm(team_positions[peer] - first_target)))
        _belief_position, target_velocity = belief_reference(
            observation,
            self.config,
            anchor_index=agent_id,
        )
        result: list[np.ndarray] = []
        if _qdr_execution_aware(observation):
            # Keep explicit low-motion recovery candidates in the finite set.
            # They are important when every target-tracking candidate enters
            # an obstacle or an inter-agent conflict during the delayed
            # suffix.  This does not cancel the immutable queue prefix.
            current_velocity = np.asarray(observation["defender_velocities"], dtype=np.float64)[agent_id]
            result.append(np.zeros((self.config.horizon_steps, 3), dtype=np.float64))
            result.append(
                np.broadcast_to(
                    current_velocity,
                    (self.config.horizon_steps, 3),
                ).copy()
            )
        for target_path in selected_paths:
            for perimeter_scale in self.config.perimeter_scales:
                current = np.asarray(own_position, dtype=np.float64).copy()
                actions: list[np.ndarray] = []
                perimeter = self.config.role_perimeter_m * float(perimeter_scale)
                for timestep in range(self.config.horizon_steps):
                    if timestep >= self.config.control_horizon_steps and actions:
                        action = actions[-1].copy()
                    else:
                        target = target_path[timestep]
                        if timestep > 0:
                            target_velocity_step = (
                                target_path[timestep] - target_path[timestep - 1]
                            ) / self.config.dt_seconds
                        else:
                            target_velocity_step = target_velocity
                        if agent_id == interceptor:
                            target_point = target
                        else:
                            direction = TETRAHEDRON_DIRECTIONS[agent_id % len(TETRAHEDRON_DIRECTIONS)]
                            target_point = target + direction * perimeter
                        action = self.config.slot_gain * (target_point - current)
                        action += self.config.target_velocity_gain * target_velocity_step
                        action = _clip_rows(action[None, :], self.config.max_speed_mps)[0]
                    actions.append(action)
                    current = current + action * self.config.dt_seconds
                result.append(np.stack(actions, axis=0))
        return result

    def _select_local_sequence(
        self,
        observation: dict[str, Any],
        scenarios: ScenarioTrajectorySet,
        agent_id: int,
        local_candidates: list[np.ndarray],
        known: dict[int, _PlannerMessage],
        peer_sequences: dict[int, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        costs = self._local_scenario_cost_matrix(
            observation,
            scenarios,
            agent_id,
            np.stack(local_candidates, axis=0),
            known,
            peer_sequences,
        )
        if self.config.risk_mode == "expected":
            objectives = costs @ scenarios.normalized_weights
        elif self.config.risk_mode == "worst_case":
            objectives = np.max(costs, axis=1)
        else:
            objectives = np.asarray(
                [
                    aggregate_scenario_costs(
                        row,
                        scenarios.normalized_weights,
                        self.config.risk_mode,
                        self.config.cvar_alpha,
                    )
                    for row in costs
                ],
                dtype=np.float64,
            )
        if not np.isfinite(objectives).all():
            raise FloatingPointError("local objective contains non-finite values")
        selected = int(np.argmin(objectives))
        return local_candidates[selected], costs[selected]

    def _local_scenario_cost_matrix(
        self,
        observation: dict[str, Any],
        scenarios: ScenarioTrajectorySet,
        agent_id: int,
        own_actions: np.ndarray,
        known: dict[int, _PlannerMessage],
        peer_sequences: dict[int, np.ndarray],
        *,
        include_reachability: bool = True,
        include_escape_gap: bool = True,
        include_fc_dbf: bool = True,
    ) -> np.ndarray:
        """Evaluate local action candidates against all target scenarios in one pass.

        The previous implementation nested Python loops over local candidates,
        target scenarios and horizon steps.  Keeping candidates in a leading
        batch dimension preserves the local information contract while making
        the expensive finite-shooting arithmetic run in NumPy kernels.
        """

        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        actions = np.asarray(own_actions, dtype=np.float64)
        if actions.ndim == 2:
            actions = actions[None, ...]
        if (
            actions.ndim != 3
            or actions.shape[1] != self.config.horizon_steps
            or actions.shape[2] != 3
        ):
            raise ValueError("own_actions must have shape [candidates, horizon, 3].")
        actions = _clip_rows(actions, self.config.max_speed_mps)

        lower = np.asarray(observation.get("world_lower_bounds", [-np.inf] * 3), dtype=np.float64)
        upper = np.asarray(observation.get("world_upper_bounds", [np.inf] * 3), dtype=np.float64)
        qdr_execution_aware = _qdr_execution_aware(observation)
        # QDR evaluates a command suffix over the full public horizon.  The
        # suffix gate therefore uses every public obstacle, while ordinary
        # distributed DN-MPC retains its local-obstacle information boundary.
        obstacles = (
            list(observation.get("obstacles", []))
            if qdr_execution_aware
            else self._local_obstacles(observation, positions[agent_id])
        )
        paths = np.asarray(scenarios.trajectories, dtype=np.float64)
        weights = scenarios.normalized_weights
        candidate_count = paths.shape[0]
        qdr_candidate_violation = np.zeros(actions.shape[0], dtype=np.float64)
        if qdr_execution_aware:
            current, executed_velocity_paths = _rollout_local_action_candidates(
                observation,
                actions,
                agent_id,
                self.config.dt_seconds,
            )
        else:
            current = positions[agent_id][None, None, :] + np.cumsum(
                actions * self.config.dt_seconds,
                axis=1,
            )
            executed_velocity_paths = actions
        distances = np.linalg.norm(
            current[:, None, :, :] - paths[None, :, :, :],
            axis=-1,
        )
        result = self.config.weight_distance * distances.sum(axis=2)
        result += self.config.weight_capture_hinge * np.maximum(
            distances - self.config.capture_radius_m,
            0.0,
        ).__pow__(2).sum(axis=2)
        result += self.config.weight_terminal_distance * distances[:, :, -1]

        target_reference = np.average(paths[:, 0], axis=0, weights=weights)
        team_positions = {agent_id: positions[agent_id]}
        team_positions.update({peer: message.position for peer, message in known.items()})
        interceptor = min(
            team_positions,
            key=lambda peer: float(np.linalg.norm(team_positions[peer] - target_reference)),
        )
        if agent_id != interceptor:
            direction = TETRAHEDRON_DIRECTIONS[agent_id % len(TETRAHEDRON_DIRECTIONS)]
            target_points = paths + direction[None, None, :] * self.config.role_perimeter_m
            formation = np.abs(
                np.linalg.norm(current[:, None, :, :] - target_points[None, :, :, :], axis=-1)
                - self.config.role_perimeter_m
            )
            result += self.config.weight_formation * formation.sum(axis=2)

        result += self.config.weight_control * np.sum(actions * actions, axis=(1, 2))[:, None]
        previous = np.concatenate(
            [
                np.broadcast_to(
                    np.asarray(observation["defender_velocities"], dtype=np.float64)[agent_id],
                    (actions.shape[0], 1, 3),
                ),
                actions[:, :-1, :],
            ],
            axis=1,
        )
        change = actions - previous
        result += self.config.weight_control_change * np.sum(change * change, axis=(1, 2))[:, None]
        result += self.config.weight_relative_speed * np.linalg.norm(change, axis=-1).sum(axis=1)[:, None]

        if self.config.reachability_normalized_cost_enabled and include_reachability:
            if self.config.reachability_cost_mode == "formation_slot":
                # A local best response can use the cooperative slot objective
                # only when it has a current public message for every peer.
                # With missing/delayed peers, omitting this term is explicit;
                # the final team score below still evaluates the full team
                # rollout against the same formation objective.
                expected_peers = set(range(positions.shape[0])) - {int(agent_id)}
                if set(known) >= expected_peers:
                    team_position_paths = [current]
                    team_velocity_paths = [executed_velocity_paths]
                    for peer in range(positions.shape[0]):
                        if peer == agent_id:
                            continue
                        message = known[peer]
                        peer_actions = peer_sequences.get(peer)
                        peer_position, peer_velocity = _rollout_peer_action_path(
                            observation,
                            message,
                            peer_actions,
                            self.config.dt_seconds,
                            self.config.horizon_steps,
                            self.config.max_speed_mps,
                        )
                        team_position_paths.append(
                            np.broadcast_to(peer_position, current.shape)
                        )
                        team_velocity_paths.append(
                            np.broadcast_to(peer_velocity, actions.shape)
                        )
                    formation_cost, _best_slack, _assignments, _arrival_times = formation_slot_reachability_cost(
                        np.stack(team_position_paths, axis=2),
                        np.stack(team_velocity_paths, axis=2),
                        paths,
                        slot_radius_m=float(
                            self.config.role_perimeter_m
                            if self.config.reachability_slot_radius_m is None
                            else self.config.reachability_slot_radius_m
                        ),
                        dt_seconds=self.config.dt_seconds,
                        max_speed_mps=self.config.max_speed_mps,
                        max_acceleration_mps2=self.config.reachability_max_acceleration_mps2,
                        time_margin_s=self.config.reachability_time_margin_s,
                        time_scale_s=self.config.reachability_time_scale_s,
                        target_tube_radius_m=scenarios.conformal_radius_by_step_m,
                        activation_slack_s=self.config.reachability_activation_slack_s,
                    )
                    # Every agent receives one equal share. Summing the local
                    # costs therefore preserves one cooperative RNIC term.
                    result += (
                        self.config.weight_reachability
                        * formation_cost
                        / max(positions.shape[0], 1)
                    )
            else:
                reachability_cost, _best_slack, _arrival_times = reachability_normalized_interception_cost(
                    current[:, :, None, :],
                    actions[:, :, None, :],
                    paths,
                    dt_seconds=self.config.dt_seconds,
                    max_speed_mps=self.config.max_speed_mps,
                    max_acceleration_mps2=self.config.reachability_max_acceleration_mps2,
                    time_margin_s=self.config.reachability_time_margin_s,
                    time_scale_s=self.config.reachability_time_scale_s,
                    target_tube_radius_m=scenarios.conformal_radius_by_step_m,
                    activation_slack_s=self.config.reachability_activation_slack_s,
                )
                result += self.config.weight_reachability * reachability_cost

        if (
            include_escape_gap
            and self.config.escape_gap_cost_enabled
            and self.config.weight_escape_gap > 0.0
            and len(known) == positions.shape[0] - 1
        ):
            # A local best response uses only the latest message from every
            # peer.  If any peer is missing or expired, the local objective
            # omits the topology term rather than fabricating a global view.
            team_position_paths = [current]
            for peer in range(positions.shape[0]):
                if peer == agent_id:
                    continue
                message = known[peer]
                peer_actions = peer_sequences.get(peer)
                peer_positions, _peer_velocity = _rollout_peer_action_path(
                    observation,
                    message,
                    peer_actions,
                    self.config.dt_seconds,
                    self.config.horizon_steps,
                    self.config.max_speed_mps,
                )
                team_position_paths.append(np.broadcast_to(peer_positions, current.shape))
            gap_metrics = escape_gap_metrics(
                np.stack(team_position_paths, axis=2),
                paths,
                gap_safe_rad=self.config.escape_gap_safe_rad,
                escape_gap_safe_rad=self.config.escape_gap_escape_safe_rad,
                max_gap_weight=self.config.escape_gap_max_weight,
                escape_gap_weight=self.config.escape_gap_direction_weight,
                horizon_discount=self.config.escape_gap_horizon_discount,
            )
            result += self.config.weight_escape_gap * gap_metrics["cost"]

        if include_fc_dbf and self.config.fc_dbf_enabled:
            result = self._apply_fc_dbf_local_gate(
                observation,
                scenarios,
                agent_id,
                actions,
                known,
                peer_sequences,
                result,
            )

        for peer, message in known.items():
            peer_actions = peer_sequences.get(peer)
            peer_positions, _peer_velocity = _rollout_peer_action_path(
                observation,
                message,
                peer_actions,
                self.config.dt_seconds,
                self.config.horizon_steps,
                self.config.max_speed_mps,
            )
            inter_agent = np.maximum(
                self.config.minimum_inter_agent_distance_m
                - np.linalg.norm(current - peer_positions[None, :, :], axis=-1),
                0.0,
            )
            result += self.config.weight_inter_agent * np.sum(inter_agent * inter_agent, axis=1)[:, None]
            if qdr_execution_aware:
                qdr_candidate_violation = np.maximum(
                    qdr_candidate_violation,
                    np.max(inter_agent, axis=1),
                )

        for obstacle in obstacles:
            shape = str(obstacle.get("shape", "cylinder"))
            center_xy = np.asarray(obstacle["center_xy"], dtype=np.float64)
            height = float(obstacle["height"])
            if shape == "cylinder":
                radial = np.linalg.norm(current[..., :2] - center_xy, axis=-1) - float(obstacle["radius"])
                vertical = np.maximum.reduce(
                    (-current[..., 2], current[..., 2] - height, np.zeros(current.shape[:2]))
                )
                clearance = np.where(
                    vertical == 0.0,
                    radial,
                    np.where(radial <= 0.0, vertical, np.hypot(radial, vertical)),
                )
            else:
                half = obstacle.get("half_extents_xy")
                if half is None:
                    half = [float(obstacle["radius"]), float(obstacle["radius"])]
                center = np.array([center_xy[0], center_xy[1], height * 0.5], dtype=np.float64)
                half_extent = np.array([float(half[0]), float(half[1]), height * 0.5], dtype=np.float64)
                signed = np.abs(current - center) - half_extent
                outside = np.maximum(signed, 0.0)
                outside_norm = np.linalg.norm(outside, axis=-1)
                clearance = np.where(
                    outside_norm > 0.0,
                    outside_norm,
                    -np.max(-signed, axis=-1),
                )
            violation = np.maximum(
                self.config.safety_margin_m - (clearance - self.config.drone_radius_m),
                0.0,
            )
            result += self.config.weight_obstacle * np.sum(violation * violation, axis=1)[:, None]
            if qdr_execution_aware:
                qdr_candidate_violation = np.maximum(
                    qdr_candidate_violation,
                    np.max(violation, axis=1),
                )

        boundary = np.maximum(
            np.max(np.maximum(lower[None, None, :] - current, 0.0), axis=2),
            np.max(np.maximum(current - upper[None, None, :], 0.0), axis=2),
        )
        result += self.config.weight_boundary * np.sum(boundary * boundary, axis=1)[:, None]
        if qdr_execution_aware:
            qdr_candidate_violation = np.maximum(
                qdr_candidate_violation,
                np.max(boundary, axis=1),
            )
            self._stats["qdr_suffix_gate_checks"] += 1
            feasible = qdr_candidate_violation <= 1.0e-9
            rejected = int(np.sum(~feasible))
            self._stats["qdr_suffix_gate_rejected_candidates"] += rejected
            if np.any(feasible):
                result += np.where(
                    feasible[:, None],
                    0.0,
                    1.0e6 + 1.0e5 * qdr_candidate_violation[:, None],
                )
            else:
                self._stats["qdr_suffix_gate_exhausted"] += 1
                result += 1.0e5 * qdr_candidate_violation[:, None]
        if result.shape != (actions.shape[0], candidate_count) or not np.isfinite(result).all():
            raise FloatingPointError("local scenario cost contains non-finite values")
        return result

    def _apply_fc_dbf_local_gate(
        self,
        observation: dict[str, Any],
        scenarios: ScenarioTrajectorySet,
        agent_id: int,
        actions: np.ndarray,
        known: dict[int, _PlannerMessage],
        peer_sequences: dict[int, np.ndarray],
        base_costs: np.ndarray,
    ) -> np.ndarray:
        """Apply FC-DBF only when the local agent has a complete peer view."""

        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        expected_peers = set(range(positions.shape[0])) - {int(agent_id)}
        if set(known) < expected_peers:
            self._stats["fc_dbf_incomplete_consensus"] += 1
            return base_costs

        actions = np.asarray(actions, dtype=np.float64)
        if _qdr_execution_aware(observation):
            own_path, own_velocity_path = _rollout_local_action_candidates(
                observation,
                actions,
                agent_id,
                self.config.dt_seconds,
            )
        else:
            own_path = positions[agent_id][None, None, :] + np.cumsum(
                actions * self.config.dt_seconds,
                axis=1,
            )
            own_velocity_path = actions
        path_count = actions.shape[0]
        team_position_paths = [own_path]
        team_velocity_paths = [own_velocity_path]
        for peer in range(positions.shape[0]):
            if peer == agent_id:
                continue
            message = known[peer]
            peer_actions = peer_sequences.get(peer)
            peer_path, peer_velocity = _rollout_peer_action_path(
                observation,
                message,
                peer_actions,
                self.config.dt_seconds,
                self.config.horizon_steps,
                self.config.max_speed_mps,
            )
            team_position_paths.append(np.broadcast_to(peer_path, own_path.shape))
            team_velocity_paths.append(np.broadcast_to(peer_velocity, actions.shape))

        team_position_array = np.stack(team_position_paths, axis=2)
        team_velocity_array = np.stack(team_velocity_paths, axis=2)
        token_metrics = feasible_consensus_slot_gate(
            team_position_array[:1],
            team_velocity_array[:1],
            np.asarray(scenarios.trajectories, dtype=np.float64),
            slot_radius_m=float(
                self.config.role_perimeter_m
                if self.config.fc_dbf_slot_radius_m is None
                else self.config.fc_dbf_slot_radius_m
            ),
            dt_seconds=self.config.dt_seconds,
            max_speed_mps=self.config.max_speed_mps,
            max_acceleration_mps2=self.config.reachability_max_acceleration_mps2,
            slot_tolerance_m=self.config.fc_dbf_slot_tolerance_m,
            min_slot_slack_s=self.config.fc_dbf_min_slot_slack_s,
            min_progress_m=self.config.fc_dbf_min_progress_m,
            gate_horizon_steps=self.config.fc_dbf_gate_horizon_steps,
            max_assignment_switch_rate=self.config.fc_dbf_max_assignment_switch_rate,
            switch_penalty=self.config.fc_dbf_switch_penalty,
            time_scale_s=self.config.reachability_time_scale_s,
            previous_assignment=(
                self._fc_dbf_previous_assignment
                if self.config.fc_dbf_hold_previous_slot
                else None
            ),
        )
        slot_assignments = np.asarray(token_metrics["assignment_paths"], dtype=np.int64)[0]
        metrics = evaluate_fixed_consensus_slots(
            team_position_array,
            team_velocity_array,
            np.asarray(scenarios.trajectories, dtype=np.float64),
            slot_assignments,
            slot_radius_m=float(
                self.config.role_perimeter_m
                if self.config.fc_dbf_slot_radius_m is None
                else self.config.fc_dbf_slot_radius_m
            ),
            dt_seconds=self.config.dt_seconds,
            max_speed_mps=self.config.max_speed_mps,
            max_acceleration_mps2=self.config.reachability_max_acceleration_mps2,
            slot_tolerance_m=self.config.fc_dbf_slot_tolerance_m,
            min_slot_slack_s=self.config.fc_dbf_min_slot_slack_s,
            min_progress_m=self.config.fc_dbf_min_progress_m,
            gate_horizon_steps=self.config.fc_dbf_gate_horizon_steps,
            max_assignment_switch_rate=self.config.fc_dbf_max_assignment_switch_rate,
            switch_penalty=self.config.fc_dbf_switch_penalty,
            time_scale_s=self.config.reachability_time_scale_s,
            previous_assignment=(
                self._fc_dbf_previous_assignment
                if self.config.fc_dbf_hold_previous_slot
                else None
            ),
        )
        feasible = np.asarray(metrics["feasible"], dtype=bool)
        row_feasible = np.all(feasible, axis=1)
        self._stats["fc_dbf_local_checks"] += int(path_count)
        if not np.any(row_feasible):
            self._stats["fc_dbf_gate_exhausted"] += 1
            return base_costs
        adjusted = base_costs + self.config.fc_dbf_cost_weight * np.asarray(
            metrics["cost"],
            dtype=np.float64,
        )
        adjusted[~row_feasible, :] += 1.0e6
        return adjusted

    def _local_scenario_costs(
        self,
        observation: dict[str, Any],
        scenarios: ScenarioTrajectorySet,
        agent_id: int,
        own_actions: np.ndarray,
        known: dict[int, _PlannerMessage],
        peer_sequences: dict[int, np.ndarray],
        *,
        include_reachability: bool = True,
        include_escape_gap: bool = True,
        include_fc_dbf: bool = True,
    ) -> np.ndarray:
        return self._local_scenario_cost_matrix(
            observation,
            scenarios,
            agent_id,
            np.asarray(own_actions, dtype=np.float64)[None, ...],
            known,
            peer_sequences,
            include_reachability=include_reachability,
            include_escape_gap=include_escape_gap,
            include_fc_dbf=include_fc_dbf,
        )[0]

    def _local_obstacles(self, observation: dict[str, Any], own_position: np.ndarray) -> list[dict[str, Any]]:
        obstacle_range = float(self.distributed.local_obstacle_range_m)
        if _qdr_execution_aware(observation):
            # A queued command is not evaluated at the current instant.  The
            # queue-aware local objective therefore needs to see obstacles
            # that can enter the agent's reachable swept volume during the
            # shooting horizon, while the ordinary distributed contract keeps
            # its original local-obstacle range.
            obstacle_range += float(self.config.max_speed_mps) * float(self.config.horizon_steps) * float(
                self.config.dt_seconds
            )
        result = []
        for obstacle in observation.get("obstacles", []):
            center_xy = np.asarray(obstacle["center_xy"], dtype=np.float64)
            radius = float(obstacle.get("radius", 0.0))
            distance = float(np.linalg.norm(own_position[:2] - center_xy) - radius)
            if distance <= obstacle_range:
                result.append(obstacle)
        return result

    def _team_scenario_costs(
        self,
        observation: dict[str, Any],
        scenarios: ScenarioTrajectorySet,
        sequences: np.ndarray,
        step_index: int,
    ) -> np.ndarray:
        total = np.zeros(scenarios.candidate_count, dtype=np.float64)
        for agent_id in range(sequences.shape[1]):
            known = self._known_peer_messages(agent_id, step_index)
            peer_sequences = {
                peer_id: message.action_sequence
                for peer_id, message in known.items()
                if message.action_sequence.shape == sequences[:, peer_id].shape
            }
            total += self._local_scenario_costs(
                observation,
                scenarios,
                agent_id,
                sequences[:, agent_id],
                known,
                peer_sequences,
                include_reachability=(self.config.reachability_cost_mode != "formation_slot"),
                include_escape_gap=False,
                include_fc_dbf=False,
            )
        if (
            self.config.reachability_normalized_cost_enabled
            and self.config.reachability_cost_mode == "formation_slot"
        ):
            position_paths, velocity_paths = _rollout_full_action_sequence(
                observation,
                np.asarray(sequences, dtype=np.float64),
                self.config.dt_seconds,
            )
            formation_cost, _best_slack, _assignments, _arrival_times = formation_slot_reachability_cost(
                position_paths[None, :, :, :],
                velocity_paths[None, :, :, :],
                np.asarray(scenarios.trajectories, dtype=np.float64),
                slot_radius_m=float(
                    self.config.role_perimeter_m
                    if self.config.reachability_slot_radius_m is None
                    else self.config.reachability_slot_radius_m
                ),
                dt_seconds=self.config.dt_seconds,
                max_speed_mps=self.config.max_speed_mps,
                max_acceleration_mps2=self.config.reachability_max_acceleration_mps2,
                time_margin_s=self.config.reachability_time_margin_s,
                time_scale_s=self.config.reachability_time_scale_s,
                target_tube_radius_m=scenarios.conformal_radius_by_step_m,
                activation_slack_s=self.config.reachability_activation_slack_s,
            )
            total += self.config.weight_reachability * formation_cost[0]
        if self.config.escape_gap_cost_enabled and self.config.weight_escape_gap > 0.0:
            position_paths, _velocity_paths = _rollout_full_action_sequence(
                observation,
                np.asarray(sequences, dtype=np.float64),
                self.config.dt_seconds,
            )
            gap_metrics = escape_gap_metrics(
                position_paths[None, :, :, :],
                np.asarray(scenarios.trajectories, dtype=np.float64),
                gap_safe_rad=self.config.escape_gap_safe_rad,
                escape_gap_safe_rad=self.config.escape_gap_escape_safe_rad,
                max_gap_weight=self.config.escape_gap_max_weight,
                escape_gap_weight=self.config.escape_gap_direction_weight,
                horizon_discount=self.config.escape_gap_horizon_discount,
            )
            # The cooperative topology term is added exactly once to the team
            # score; local best responses omit it to avoid D-fold counting.
            total += self.config.weight_escape_gap * gap_metrics["cost"][0]
        if self.config.fc_dbf_enabled:
            position_path, velocity_path = _rollout_full_action_sequence(
                observation,
                np.asarray(sequences, dtype=np.float64),
                self.config.dt_seconds,
            )
            position_paths = position_path[None, :, :, :]
            action_paths = velocity_path[None, :, :, :]
            fc_metrics = feasible_consensus_slot_gate(
                position_paths,
                action_paths,
                np.asarray(scenarios.trajectories, dtype=np.float64),
                slot_radius_m=float(
                    self.config.role_perimeter_m
                    if self.config.fc_dbf_slot_radius_m is None
                    else self.config.fc_dbf_slot_radius_m
                ),
                dt_seconds=self.config.dt_seconds,
                max_speed_mps=self.config.max_speed_mps,
                max_acceleration_mps2=self.config.reachability_max_acceleration_mps2,
                slot_tolerance_m=self.config.fc_dbf_slot_tolerance_m,
                min_slot_slack_s=self.config.fc_dbf_min_slot_slack_s,
                min_progress_m=self.config.fc_dbf_min_progress_m,
                gate_horizon_steps=self.config.fc_dbf_gate_horizon_steps,
                max_assignment_switch_rate=self.config.fc_dbf_max_assignment_switch_rate,
                switch_penalty=self.config.fc_dbf_switch_penalty,
                time_scale_s=self.config.reachability_time_scale_s,
                previous_assignment=(
                    self._fc_dbf_previous_assignment
                    if self.config.fc_dbf_hold_previous_slot
                    else None
                ),
            )
            feasible = np.asarray(fc_metrics["feasible"], dtype=bool)
            gate_exhausted = bool(not np.any(feasible))
            self._fc_dbf_last_metrics = fc_metrics
            self._fc_dbf_last_gate_exhausted = gate_exhausted
            if not gate_exhausted:
                total += self.config.fc_dbf_cost_weight * np.asarray(fc_metrics["cost"], dtype=np.float64)[0]
        return total

    def _escape_gap_summary(
        self,
        observation: dict[str, Any],
        sequences: np.ndarray,
        target_paths: np.ndarray,
        target_weights: np.ndarray,
    ) -> dict[str, float]:
        position_paths, _velocity_paths = _rollout_full_action_sequence(
            observation,
            np.asarray(sequences, dtype=np.float64),
            self.config.dt_seconds,
        )
        metrics = escape_gap_metrics(
            position_paths[None, :, :, :],
            target_paths,
            gap_safe_rad=self.config.escape_gap_safe_rad,
            escape_gap_safe_rad=self.config.escape_gap_escape_safe_rad,
            max_gap_weight=self.config.escape_gap_max_weight,
            escape_gap_weight=self.config.escape_gap_direction_weight,
            horizon_discount=self.config.escape_gap_horizon_discount,
        )
        weights = np.asarray(target_weights, dtype=np.float64)
        weights = weights / max(float(weights.sum()), 1.0e-12)
        return {
            "escape_gap_cost": float(np.dot(metrics["cost"][0], weights)),
            "escape_gap_max_rad": float(np.dot(metrics["max_gap_rad"][0], weights)),
            "escape_gap_escape_rad": float(np.dot(metrics["escape_gap_rad"][0], weights)),
            "escape_gap_coverage_ratio": float(np.dot(metrics["coverage_ratio"][0], weights)),
            "escape_gap_violation_rate": float(
                np.dot(metrics["max_gap_violation_rate"][0], weights)
            ),
        }

    def _diagnostics(
        self,
        *,
        status: str,
        iterations: int,
        converged: bool,
        local_solver_failures: int,
        scenario_costs: np.ndarray,
        objective: float,
        expected: float,
        worst: float,
        cvar: float,
        latency_ms: float,
        max_delta: float,
        fallback_reason: str | None,
        qdr_suffix_gate_active: bool = False,
        qdr_suffix_gate_exhausted: bool = False,
        qdr_suffix_gate_rejected_candidates: int = 0,
        escape_gap_cost: float = float("nan"),
        escape_gap_max_rad: float = float("nan"),
        escape_gap_escape_rad: float = float("nan"),
        escape_gap_coverage_ratio: float = float("nan"),
        escape_gap_violation_rate: float = float("nan"),
        fc_dbf_enabled: bool = False,
        fc_dbf_feasible: bool = False,
        fc_dbf_min_slot_slack_s: float = float("nan"),
        fc_dbf_max_slot_error_m: float = float("nan"),
        fc_dbf_mean_slot_error_m: float = float("nan"),
        fc_dbf_mean_slot_progress_m: float = float("nan"),
        fc_dbf_assignment_switch_rate: float = float("nan"),
        fc_dbf_feasible_rate: float = float("nan"),
        fc_dbf_gate_exhausted: bool = False,
        fc_dbf_cost: float = float("nan"),
        belief_fusion_enabled: float = 0.0,
        belief_fusion_effective_sample_size: float = float("nan"),
        belief_fusion_weight_entropy: float = float("nan"),
        belief_fusion_mean_age_steps: float = float("nan"),
        belief_fusion_max_age_steps: float = float("nan"),
        belief_fusion_mean_covariance_trace_m2: float = float("nan"),
        belief_fusion_fallback_used: float = 0.0,
        belief_fusion_fallback_reason: str = "disabled",
    ) -> DistributedDNMPCDiagnostics:
        ages = np.asarray(self._stats.get("age_samples", []), dtype=np.float64)
        return DistributedDNMPCDiagnostics(
            status=status,
            communication_mode=self.distributed.communication_mode,
            iterations=int(iterations),
            converged=bool(converged),
            local_solver_failures=int(local_solver_failures),
            messages_attempted=int(self._stats.get("messages_attempted", 0)),
            messages_sent=int(self._stats.get("messages_sent", 0)),
            messages_received=int(self._stats.get("messages_received", 0)),
            messages_dropped=int(self._stats.get("messages_dropped", 0)),
            message_bytes_sent=int(self._stats.get("message_bytes_sent", 0)),
            message_bytes_received=int(self._stats.get("message_bytes_received", 0)),
            max_message_age_steps=int(np.max(ages)) if ages.size else 0,
            mean_message_age_steps=float(np.mean(ages)) if ages.size else 0.0,
            scenario_costs=tuple(float(value) for value in scenario_costs),
            objective_value=float(objective),
            expected_cost=float(expected),
            worst_case_cost=float(worst),
            cvar_cost=float(cvar),
            latency_ms=float(latency_ms),
            max_action_delta_mps=float(max_delta),
            fallback_reason=fallback_reason,
            qdr_suffix_gate_active=bool(qdr_suffix_gate_active),
            qdr_suffix_gate_exhausted=bool(qdr_suffix_gate_exhausted),
            qdr_suffix_gate_rejected_candidates=int(qdr_suffix_gate_rejected_candidates),
            escape_gap_cost=float(escape_gap_cost),
            escape_gap_max_rad=float(escape_gap_max_rad),
            escape_gap_escape_rad=float(escape_gap_escape_rad),
            escape_gap_coverage_ratio=float(escape_gap_coverage_ratio),
            escape_gap_violation_rate=float(escape_gap_violation_rate),
            fc_dbf_enabled=bool(self.config.fc_dbf_enabled),
            fc_dbf_feasible=bool(fc_dbf_feasible),
            fc_dbf_min_slot_slack_s=float(fc_dbf_min_slot_slack_s),
            fc_dbf_max_slot_error_m=float(fc_dbf_max_slot_error_m),
            fc_dbf_mean_slot_error_m=float(fc_dbf_mean_slot_error_m),
            fc_dbf_mean_slot_progress_m=float(fc_dbf_mean_slot_progress_m),
            fc_dbf_assignment_switch_rate=float(fc_dbf_assignment_switch_rate),
            fc_dbf_feasible_rate=float(fc_dbf_feasible_rate),
            fc_dbf_gate_exhausted=bool(fc_dbf_gate_exhausted),
            fc_dbf_cost=float(fc_dbf_cost),
            belief_fusion_enabled=float(belief_fusion_enabled),
            belief_fusion_effective_sample_size=float(belief_fusion_effective_sample_size),
            belief_fusion_weight_entropy=float(belief_fusion_weight_entropy),
            belief_fusion_mean_age_steps=float(belief_fusion_mean_age_steps),
            belief_fusion_max_age_steps=float(belief_fusion_max_age_steps),
            belief_fusion_mean_covariance_trace_m2=float(belief_fusion_mean_covariance_trace_m2),
            belief_fusion_fallback_used=float(belief_fusion_fallback_used),
            belief_fusion_fallback_reason=str(belief_fusion_fallback_reason),
        )

    def _fallback(
        self,
        positions: np.ndarray,
        fallback_actions: np.ndarray | None,
        started: float,
        reason: str,
    ) -> DistributedDNMPCPlan:
        if fallback_actions is None:
            actions = np.zeros_like(positions)
        else:
            actions = np.asarray(fallback_actions, dtype=np.float64)
            if actions.shape != positions.shape or not np.isfinite(actions).all():
                actions = np.zeros_like(positions)
        actions = _clip_rows(actions, self.config.max_speed_mps)
        sequence = np.repeat(actions[None, :, :], self.config.horizon_steps, axis=0)
        diagnostics = self._diagnostics(
            status="fallback",
            iterations=0,
            converged=False,
            local_solver_failures=0,
            scenario_costs=np.zeros(0, dtype=np.float64),
            objective=float("nan"),
            expected=float("nan"),
            worst=float("nan"),
            cvar=float("nan"),
            latency_ms=(perf_counter() - started) * 1000.0,
            max_delta=0.0,
            fallback_reason=reason,
        )
        return DistributedDNMPCPlan(actions=actions, action_sequence=sequence, diagnostics=diagnostics)


__all__ = [
    "DistributedDNMPCConfig",
    "DistributedDNMPCDiagnostics",
    "DistributedDNMPCPlan",
    "DistributedMinimaxDNMPC",
]
