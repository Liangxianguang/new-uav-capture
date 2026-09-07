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

from .minimax_mpc import MinimaxMPCConfig, ScenarioTrajectorySet, aggregate_scenario_costs
from .pursuit_env import TETRAHEDRON_DIRECTIONS, _unit


CommunicationMode = Literal["none", "ideal", "delayed", "dropout"]
_COMMUNICATION_MODES = {"none", "ideal", "delayed", "dropout"}


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
        }
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
        belief_velocities = np.asarray(
            observation.get("target_belief_velocities", np.zeros((len(team_positions), 3))),
            dtype=np.float64,
        )
        target_velocity = (
            belief_velocities[agent_id]
            if belief_velocities.ndim == 2 and agent_id < belief_velocities.shape[0]
            else np.zeros(3, dtype=np.float64)
        )
        result: list[np.ndarray] = []
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
        costs = np.stack(
            [
                self._local_scenario_costs(
                    observation,
                    scenarios,
                    agent_id,
                    candidate,
                    known,
                    peer_sequences,
                )
                for candidate in local_candidates
            ],
            axis=0,
        )
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

    def _local_scenario_costs(
        self,
        observation: dict[str, Any],
        scenarios: ScenarioTrajectorySet,
        agent_id: int,
        own_actions: np.ndarray,
        known: dict[int, _PlannerMessage],
        peer_sequences: dict[int, np.ndarray],
    ) -> np.ndarray:
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        own_position = positions[agent_id].copy()
        own_actions = _clip_rows(np.asarray(own_actions, dtype=np.float64), self.config.max_speed_mps)
        lower = np.asarray(observation.get("world_lower_bounds", [-np.inf] * 3), dtype=np.float64)
        upper = np.asarray(observation.get("world_upper_bounds", [np.inf] * 3), dtype=np.float64)
        obstacles = self._local_obstacles(observation, own_position)
        paths = np.asarray(scenarios.trajectories, dtype=np.float64)
        weights = scenarios.normalized_weights
        target_reference = np.average(paths[:, 0], axis=0, weights=weights)
        team_positions = {agent_id: own_position}
        team_positions.update({peer: message.position for peer, message in known.items()})
        interceptor = min(team_positions, key=lambda peer: float(np.linalg.norm(team_positions[peer] - target_reference)))
        result = np.zeros(paths.shape[0], dtype=np.float64)
        for candidate_index, target_path in enumerate(paths):
            current = own_position.copy()
            peers = {peer: message.position.copy() for peer, message in known.items()}
            for timestep in range(self.config.horizon_steps):
                action = own_actions[timestep]
                current += action * self.config.dt_seconds
                for peer, message in known.items():
                    peer_action = peer_sequences.get(peer)
                    if peer_action is None:
                        peers[peer] += message.velocity * self.config.dt_seconds
                    else:
                        peers[peer] += _clip_rows(peer_action[timestep][None, :], self.config.max_speed_mps)[0] * self.config.dt_seconds
                target = target_path[timestep]
                distance = float(np.linalg.norm(current - target))
                result[candidate_index] += self.config.weight_distance * distance
                result[candidate_index] += self.config.weight_capture_hinge * max(
                    distance - self.config.capture_radius_m,
                    0.0,
                ) ** 2
                if timestep == self.config.horizon_steps - 1:
                    result[candidate_index] += self.config.weight_terminal_distance * distance
                if agent_id != interceptor:
                    direction = TETRAHEDRON_DIRECTIONS[agent_id % len(TETRAHEDRON_DIRECTIONS)]
                    target_point = target + direction * self.config.role_perimeter_m
                    result[candidate_index] += self.config.weight_formation * abs(
                        float(np.linalg.norm(current - target_point)) - self.config.role_perimeter_m
                    )
                result[candidate_index] += self.config.weight_control * float(np.sum(action * action))
                if timestep == 0:
                    previous = np.asarray(observation["defender_velocities"], dtype=np.float64)[agent_id]
                else:
                    previous = own_actions[timestep - 1]
                change = action - previous
                result[candidate_index] += self.config.weight_control_change * float(np.sum(change * change))
                result[candidate_index] += self.config.weight_relative_speed * float(np.linalg.norm(change))
                for obstacle in obstacles:
                    clearance = _obstacle_clearance(current, obstacle)
                    violation = max(
                        self.config.safety_margin_m - (clearance - self.config.drone_radius_m),
                        0.0,
                    )
                    result[candidate_index] += self.config.weight_obstacle * violation * violation
                boundary = max(
                    float(np.max(np.maximum(lower - current, 0.0))),
                    float(np.max(np.maximum(current - upper, 0.0))),
                )
                result[candidate_index] += self.config.weight_boundary * boundary * boundary
                for peer_position in peers.values():
                    inter_agent = max(
                        self.config.minimum_inter_agent_distance_m
                        - float(np.linalg.norm(current - peer_position)),
                        0.0,
                    )
                    result[candidate_index] += self.config.weight_inter_agent * inter_agent * inter_agent
        if not np.isfinite(result).all():
            raise FloatingPointError("local scenario cost contains non-finite values")
        return result

    def _local_obstacles(self, observation: dict[str, Any], own_position: np.ndarray) -> list[dict[str, Any]]:
        result = []
        for obstacle in observation.get("obstacles", []):
            center_xy = np.asarray(obstacle["center_xy"], dtype=np.float64)
            radius = float(obstacle.get("radius", 0.0))
            distance = float(np.linalg.norm(own_position[:2] - center_xy) - radius)
            if distance <= self.distributed.local_obstacle_range_m:
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
            )
        return total

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
