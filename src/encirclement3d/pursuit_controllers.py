"""Rule and safety baselines for capture-radius pursuit-evasion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .pursuit_env import CaptureRadiusPursuit3DEnv, TETRAHEDRON_DIRECTIONS, _unit


@dataclass(frozen=True)
class PursuitSafetyDiagnostics:
    action_correction_norm: float
    minimum_barrier_value: float
    queue_preview_unsafe: bool = False


class PursuitCBFSafetyFilter:
    """A local-information CBF projection for obstacles, teammates, and bounds.

    The target is deliberately excluded from the barrier constraints because
    approaching it is the task objective. The filter sees the same obstacle and
    teammate data as the decentralized pursuit controllers.
    """

    def __init__(self, env: CaptureRadiusPursuit3DEnv, *, projection_iterations: int = 4) -> None:
        self.env = env
        self.gamma = float(env.task.get("cbf_gamma", 0.25))
        self.margin = float(env.pursuit["safety_margin"])
        self.projection_iterations = max(1, int(projection_iterations))

    def filter(
        self,
        desired_actions: np.ndarray,
        observation: dict[str, Any],
        *,
        max_speed: float | None = None,
    ) -> tuple[np.ndarray, PursuitSafetyDiagnostics]:
        command_limit = (
            float(self.env.agents["defender_max_speed"])
            if max_speed is None
            else float(max_speed)
        )
        if command_limit <= 0.0:
            raise ValueError("max_speed must be positive.")
        desired = self.env._clip_rows(
            np.asarray(desired_actions, dtype=np.float64),
            command_limit,
        )
        safe = desired.copy()
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        radius = float(self.env.agents["drone_radius"])
        barriers: list[float] = []
        queue_preview_unsafe = False

        # A delayed command is already committed before this filter runs.  A
        # current-state-only projection can therefore approve a command whose
        # queued predecessor drives a defender into a boundary or obstacle.
        # Check the reachable queue prefix as a conservative empirical guard.
        execution = observation.get("execution", {})
        queued = execution.get("action_queue", []) if isinstance(execution, dict) else []
        future_positions = positions.copy()
        future_commands = [np.asarray(item, dtype=np.float64) for item in queued]
        future_commands.append(safe.copy())
        future_horizon = max(len(future_commands) - 1, 0)
        for command in future_commands[:future_horizon]:
            future_positions = future_positions + command * float(self.env.dt)
            for index, position in enumerate(future_positions):
                for obstacle in self.env.obstacles:
                    clearance, _normal = self.env._cylinder_clearance_and_normal(position, obstacle)
                    barriers.append(float(clearance - radius - self.margin))
                barriers.extend(
                    [
                        float(position[axis] - self.env.lower[axis] - radius - self.margin)
                        for axis in range(3)
                    ]
                )
                barriers.extend(
                    [
                        float(self.env.upper[axis] - position[axis] - radius - self.margin)
                        for axis in range(3)
                    ]
                )
            if future_horizon and min(barriers[-max(self.env.n_defenders * (2 * len(self.env.obstacles) + 6), 1):]) < 0.0:
                # The committed prefix is unsafe, but a full stop can create
                # timeouts.  Keep the flag for diagnostics and let the
                # projected-position constraints below steer the next command
                # back toward a feasible state.
                queue_preview_unsafe = True
                break

        # The newly appended command will act after the committed queue.  Use
        # that reachable state for the projection constraints; projecting at
        # the current state leaves the delay window uncontrolled.
        constraint_positions = future_positions if future_horizon else positions

        for _ in range(self.projection_iterations):
            for index, position in enumerate(constraint_positions):
                for obstacle in self.env.obstacles:
                    clearance, normal = self.env._cylinder_clearance_and_normal(position, obstacle)
                    barrier = clearance - radius - self.margin
                    lower_bound = -self.gamma * barrier / self.env.dt
                    projection = float(np.sum(normal * safe[index]))
                    if projection < lower_bound:
                        safe[index] += (lower_bound - projection) * normal
                    barriers.append(barrier)
                for axis in range(3):
                    lower_barrier = position[axis] - self.env.lower[axis] - radius - self.margin
                    upper_barrier = self.env.upper[axis] - position[axis] - radius - self.margin
                    if safe[index, axis] < -self.gamma * lower_barrier / self.env.dt:
                        safe[index, axis] = -self.gamma * lower_barrier / self.env.dt
                    if safe[index, axis] > self.gamma * upper_barrier / self.env.dt:
                        safe[index, axis] = self.gamma * upper_barrier / self.env.dt
                    barriers.extend([lower_barrier, upper_barrier])
            for first in range(self.env.n_defenders):
                for second in range(first + 1, self.env.n_defenders):
                    delta = constraint_positions[first] - constraint_positions[second]
                    distance = float(np.linalg.norm(delta))
                    normal = _unit(delta, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
                    barrier = distance - (2.0 * radius + self.margin)
                    lower_bound = -self.gamma * barrier / self.env.dt
                    relative_projection = float(np.sum(normal * (safe[first] - safe[second])))
                    if relative_projection < lower_bound:
                        correction = 0.5 * (lower_bound - relative_projection) * normal
                        safe[first] += correction
                        safe[second] -= correction
                    barriers.append(barrier)
            safe = self.env._clip_rows(safe, command_limit)

        return safe, PursuitSafetyDiagnostics(
            action_correction_norm=float(np.mean(np.linalg.norm(safe - desired, axis=1))),
            minimum_barrier_value=float(min(barriers)) if barriers else float("inf"),
            queue_preview_unsafe=queue_preview_unsafe,
        )


class _PursuitController:
    def __init__(self, env: CaptureRadiusPursuit3DEnv) -> None:
        self.env = env
        self.max_speed = float(env.agents["defender_max_speed"])
        self.gain = float(env.task.get("slot_tracking_gain", 4.0))
        self.obstacle_distance = float(env.pursuit["controller_obstacle_avoidance_distance"])
        self.obstacle_gain = float(env.pursuit["controller_obstacle_avoidance_gain"])
        self.inter_agent_distance = float(env.pursuit["controller_inter_agent_distance"])
        self.inter_agent_gain = float(env.pursuit["controller_inter_agent_gain"])

    def _avoidance(self, desired: np.ndarray, observation: dict[str, Any]) -> np.ndarray:
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        corrected = np.asarray(desired, dtype=np.float64).copy()
        for index, position in enumerate(positions):
            for obstacle in self.env.obstacles:
                clearance, normal = self.env._cylinder_clearance_and_normal(position, obstacle)
                if clearance < self.obstacle_distance:
                    corrected[index] += normal * (self.obstacle_distance - clearance) * self.obstacle_gain
            for other_index, other_position in enumerate(positions):
                if index == other_index:
                    continue
                delta = position - other_position
                distance = float(np.linalg.norm(delta))
                if distance < self.inter_agent_distance:
                    corrected[index] += (
                        _unit(delta, fallback=TETRAHEDRON_DIRECTIONS[index])
                        * (self.inter_agent_distance - distance)
                        * self.inter_agent_gain
                    )
        return self.env._clip_rows(corrected, self.max_speed)

    @staticmethod
    def _team_prediction(observation: dict[str, Any], horizon_seconds: float) -> tuple[np.ndarray, np.ndarray]:
        beliefs = np.asarray(observation["target_belief_positions"], dtype=np.float64)
        velocities = np.asarray(observation["target_belief_velocities"], dtype=np.float64)
        ages = np.asarray(observation["message_age_steps"], dtype=np.float64)
        weights = 1.0 / (1.0 + ages)
        weights /= np.sum(weights)
        position = np.sum(beliefs * weights[:, None], axis=0)
        velocity = np.sum(velocities * weights[:, None], axis=0)
        return position + horizon_seconds * velocity, velocity


class PurePursuitController(_PursuitController):
    """Greedy baseline that pursues each defender's own target belief."""

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        beliefs = np.asarray(observation["target_belief_positions"], dtype=np.float64)
        desired = self.gain * (beliefs - positions)
        return self._avoidance(desired, observation)


class PredictionPursuitController(_PursuitController):
    """Constant-velocity target predictor followed by local pursuit."""

    def __init__(self, env: CaptureRadiusPursuit3DEnv, horizon_seconds: float = 0.45) -> None:
        super().__init__(env)
        self.horizon_seconds = float(horizon_seconds)

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        beliefs = np.asarray(observation["target_belief_positions"], dtype=np.float64)
        velocities = np.asarray(observation["target_belief_velocities"], dtype=np.float64)
        predicted = beliefs + self.horizon_seconds * velocities
        desired = self.gain * (predicted - positions) + velocities
        return self._avoidance(desired, observation)


class DynamicEncirclementController(_PursuitController):
    """Use delayed observations to divide agents into one interceptor and blockers."""

    def __init__(self, env: CaptureRadiusPursuit3DEnv, horizon_seconds: float = 0.55) -> None:
        super().__init__(env)
        self.horizon_seconds = float(horizon_seconds)
        self.interceptor_id: int | None = None

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        target, target_velocity = self._team_prediction(observation, self.horizon_seconds)
        distances = np.linalg.norm(positions - target[None, :], axis=1)
        if self.interceptor_id is None:
            self.interceptor_id = int(np.argmin(distances))
        interceptor = self.interceptor_id
        perimeter = float(
            np.clip(
                0.55 * np.median(distances),
                float(self.env.pursuit["capture_radius"]) + 0.35,
                2.6,
            )
        )
        desired = np.zeros_like(positions)
        for index, position in enumerate(positions):
            if index == interceptor:
                target_point = target
            else:
                target_point = target + TETRAHEDRON_DIRECTIONS[index] * perimeter
            desired[index] = self.gain * (target_point - position) + target_velocity
        return self._avoidance(desired, observation)


class FixedRoleEncirclementController(DynamicEncirclementController):
    """Development-only encirclement teacher with a fixed interceptor role.

    ``DynamicEncirclementController`` chooses the nearest defender as the
    interceptor, which is useful for a rule baseline but gives a shared actor
    no stable role label to imitate. This subclass keeps the role assignment
    fixed so it can be paired with ``policy_role_slot_features`` during a
    development warm-up. It is not part of the formal RL comparison.
    """

    def __init__(
        self,
        env: CaptureRadiusPursuit3DEnv,
        horizon_seconds: float = 0.55,
        interceptor_id: int = 0,
    ) -> None:
        super().__init__(env, horizon_seconds=horizon_seconds)
        if not 0 <= int(interceptor_id) < env.n_defenders:
            raise ValueError("interceptor_id must identify an existing defender.")
        self.interceptor_id = int(interceptor_id)


class PublicBeliefRouteIntentController(_PursuitController):
    """Route-aware public-belief expert for obstacle-crossing demonstrations.

    This teacher is deliberately restricted to the same information that is
    available to a decentralized policy: delayed/noisy target beliefs,
    observed obstacle geometry, teammate positions, and the execution queue.
    It does not read ``env.target_position`` or any other simulator truth.

    At a low frequency it evaluates lower/upper horizontal bypasses and an
    overhead bypass when the obstacle heights permit one.  The selected route
    is held for a short interval, and each defender follows a conservative
    route waypoint before switching to a loose capture formation.  The final
    command is rate-limited so the resulting demonstrations are physically
    plausible and suitable for behavior cloning.
    """

    ROUTE_NAMES = ("left_bypass", "right_bypass", "upper_bypass")

    def __init__(
        self,
        env: CaptureRadiusPursuit3DEnv,
        horizon_seconds: float = 0.75,
        replan_interval_steps: int = 8,
        min_hold_steps: int = 6,
        grid_step: float = 0.75,
        route_margin: float = 0.85,
        require_bypass_route: bool = False,
        interceptor_id: int | None = None,
    ) -> None:
        super().__init__(env)
        if replan_interval_steps <= 0 or min_hold_steps <= 0:
            raise ValueError("Route replanning and route hold intervals must be positive.")
        self.horizon_seconds = float(horizon_seconds)
        self.replan_interval_steps = int(replan_interval_steps)
        self.min_hold_steps = int(min_hold_steps)
        self.grid_step = float(grid_step)
        self.route_margin = float(route_margin)
        self.require_bypass_route = bool(require_bypass_route)
        if interceptor_id is not None and not 0 <= int(interceptor_id) < env.n_defenders:
            raise ValueError("interceptor_id must identify an existing defender.")
        self.fixed_interceptor_id = None if interceptor_id is None else int(interceptor_id)
        self.route_name = "direct"
        self.route_started_step = -10**9
        self.route_paths: list[list[np.ndarray]] = []
        self.belief_blind = False
        self.last_command = np.zeros((env.n_defenders, 3), dtype=np.float64)
        self.last_delta = np.zeros_like(self.last_command)

    @staticmethod
    def _half_extents(obstacle: Any) -> tuple[float, float]:
        extents = getattr(obstacle, "half_extents_xy", None)
        if extents is None:
            return float(obstacle.radius), float(obstacle.radius)
        values = np.asarray(extents, dtype=np.float64)
        return float(values[0]), float(values[1])

    def _route_anchor(self, route_name: str, target: np.ndarray) -> np.ndarray | None:
        if not self.env.obstacles:
            return None
        x_center = float(np.mean([obstacle.center_xy[0] for obstacle in self.env.obstacles]))
        y_extent = max(
            abs(float(obstacle.center_xy[1])) + self._half_extents(obstacle)[1]
            for obstacle in self.env.obstacles
        )
        if route_name == "left_bypass":
            y = -y_extent - self.route_margin
            if y < float(self.env.lower[1]) + 0.6:
                return None
            return np.array([x_center, y, float(target[2])], dtype=np.float64)
        if route_name == "right_bypass":
            y = y_extent + self.route_margin
            if y > float(self.env.upper[1]) - 0.6:
                return None
            return np.array([x_center, y, float(target[2])], dtype=np.float64)
        if route_name == "upper_bypass":
            height = max(float(obstacle.height) for obstacle in self.env.obstacles)
            z = height + self.route_margin + float(self.env.agents["drone_radius"])
            if z > float(self.env.upper[2]) - 0.6:
                return None
            return np.array([x_center, float(target[1]), z], dtype=np.float64)
        raise ValueError(f"Unknown route name: {route_name}")

    def _segment_is_clear(self, first: np.ndarray, second: np.ndarray) -> bool:
        distance = float(np.linalg.norm(second - first))
        samples = max(int(np.ceil(distance / 0.20)), 1)
        required = float(self.env.pursuit["safety_margin"]) + float(self.env.agents["drone_radius"])
        for fraction in np.linspace(0.0, 1.0, samples + 1):
            point = first + float(fraction) * (second - first)
            if any(self.env._obstacle_clearance(point, obstacle) < required for obstacle in self.env.obstacles):
                return False
        return True

    def _build_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        route_name: str,
    ) -> list[np.ndarray] | None:
        if route_name == "direct" or not self.env.obstacles:
            return [start.copy(), goal.copy()] if self._segment_is_clear(start, goal) else None
        anchor = self._route_anchor(route_name, goal)
        if anchor is None:
            return None
        if route_name == "upper_bypass":
            if not self._segment_is_clear(start, anchor) or not self._segment_is_clear(anchor, goal):
                return None
            return [start.copy(), anchor, goal.copy()]
        # A full grid A* is retained by the scene validator, but is too costly
        # to invoke at every teacher replanning step.  The teacher instead
        # uses a public-geometry corridor around the complete obstacle union
        # and verifies every segment with the exact 3-D clearance routine.
        x_low = min(float(obstacle.center_xy[0]) - self._half_extents(obstacle)[0] for obstacle in self.env.obstacles)
        x_high = max(float(obstacle.center_xy[0]) + self._half_extents(obstacle)[0] for obstacle in self.env.obstacles)
        transition_x = x_high + 1.0 if goal[0] >= start[0] else x_low - 1.0
        corridor = np.array([transition_x, anchor[1], anchor[2]], dtype=np.float64)
        entry = np.array([start[0], anchor[1], anchor[2]], dtype=np.float64)
        exit_point = np.array([goal[0], anchor[1], anchor[2]], dtype=np.float64)
        path = [start.copy(), entry, corridor, exit_point, goal.copy()]
        if any(not self._segment_is_clear(first, second) for first, second in zip(path, path[1:])):
            return None
        return path

    def _candidate_routes(self, target: np.ndarray, target_velocity: np.ndarray) -> tuple[str, dict[str, Any]]:
        goal = np.clip(
            target + self.horizon_seconds * target_velocity,
            self.env.lower + 0.6,
            self.env.upper - 0.6,
        )
        candidates: dict[str, Any] = {}
        route_candidates = self.ROUTE_NAMES if self.require_bypass_route else ("direct", *self.ROUTE_NAMES)
        for route_name in route_candidates:
            paths = [self._build_path(position, goal, route_name) for position in self.env.defender_positions]
            feasible = [path is not None for path in paths]
            if not any(feasible):
                continue
            lengths = [
                float(np.sum(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1)))
                if path is not None and len(path) > 1
                else float("inf")
                for path in paths
            ]
            usable_lengths = [length for length in lengths if np.isfinite(length)]
            if not usable_lengths:
                continue
            # Prefer routes that are feasible for the whole team and are
            # short, with a small directional prior from the public belief.
            infeasible_penalty = float(self.max_speed * 2.0) * (len(paths) - len(usable_lengths))
            direction_prior = 0.0
            if route_name == "left_bypass":
                direction_prior = 0.35 * max(float(target_velocity[1]), 0.0)
            elif route_name == "right_bypass":
                direction_prior = 0.35 * max(float(-target_velocity[1]), 0.0)
            score = float(np.mean(usable_lengths) + 0.35 * np.max(usable_lengths) + infeasible_penalty - direction_prior)
            candidates[route_name] = {"paths": paths, "score": score, "goal": goal.copy()}
        if not candidates:
            return "direct", {"paths": [[position.copy(), goal.copy()] for position in self.env.defender_positions], "score": float("inf"), "goal": goal}
        previous = candidates.get(self.route_name)
        if previous is not None and self.env.step_count - self.route_started_step < self.min_hold_steps:
            return self.route_name, previous
        selected = min(candidates, key=lambda name: candidates[name]["score"])
        if selected != self.route_name and self.route_name in candidates:
            # Hysteresis avoids route flapping when two exits are nearly tied.
            if candidates[selected]["score"] > candidates[self.route_name]["score"] - 0.5:
                selected = self.route_name
        return selected, candidates[selected]

    def _public_target_estimate(self, observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Return a public-belief target estimate with a protocol-neutral prior.

        At the beginning of crossing scenes the target is intentionally out of
        detection range, so every public belief can be empty.  Sending a
        teacher toward the world floor in that case would encode a spurious
        ``under-the-obstacle`` shortcut.  A neutral rally prior at the team
        altitude lets the defenders occupy the central corridor until a real
        delayed/noisy observation arrives; it uses no hidden target state.
        """
        beliefs = np.asarray(observation["target_belief_positions"], dtype=np.float64)
        confidence = np.asarray(observation["target_observation_confidence"], dtype=np.float64)
        if float(np.max(confidence, initial=0.0)) < 1e-6 and float(np.max(np.linalg.norm(beliefs, axis=1), initial=0.0)) < 1e-6:
            self.belief_blind = True
            positions = np.asarray(observation["defender_positions"], dtype=np.float64)
            rally = np.array(
                [
                    0.0,
                    float(np.mean(positions[:, 1])),
                    float(np.clip(np.mean(positions[:, 2]), self.env.lower[2] + 0.8, self.env.upper[2] - 0.8)),
                ],
                dtype=np.float64,
            )
            return rally, np.zeros(3, dtype=np.float64)
        self.belief_blind = False
        return self._team_prediction(observation, self.horizon_seconds)

    def _blind_rally_routes(self) -> tuple[str, dict[str, Any]]:
        """Plan only to a visible bypass anchor before target detection."""
        candidates: dict[str, Any] = {}
        for route_name in self.ROUTE_NAMES:
            anchor = self._route_anchor(route_name, np.array([0.0, 0.0, np.mean(self.env.defender_positions[:, 2])]))
            if anchor is None:
                continue
            paths = [self._build_path(position, anchor, route_name) for position in self.env.defender_positions]
            lengths = [
                float(np.sum(np.linalg.norm(np.diff(np.asarray(path), axis=0), axis=1)))
                if path is not None and len(path) > 1
                else float("inf")
                for path in paths
            ]
            feasible = [length for length in lengths if np.isfinite(length)]
            if len(feasible) < max(2, self.env.n_defenders - 1):
                continue
            candidates[route_name] = {
                "paths": paths,
                "score": float(np.mean(feasible) + 0.35 * np.max(feasible)),
                "goal": anchor.copy(),
            }
        if not candidates:
            goal = np.array([0.0, 0.0, np.mean(self.env.defender_positions[:, 2])], dtype=np.float64)
            return "direct", {
                "paths": [[position.copy(), goal.copy()] for position in self.env.defender_positions],
                "score": float("inf"),
                "goal": goal,
            }
        if self.route_name in candidates and self.env.step_count - self.route_started_step < self.min_hold_steps:
            return self.route_name, candidates[self.route_name]
        selected = min(candidates, key=lambda name: candidates[name]["score"])
        return selected, candidates[selected]

    def _next_waypoint(self, path: list[np.ndarray], position: np.ndarray) -> np.ndarray:
        for point in path[1:]:
            if float(np.linalg.norm(point - position)) > 0.55:
                return point
        return path[-1]

    def route_features(self, observation: dict[str, Any]) -> np.ndarray:
        """Return a compact public route-intent hint for a low-level actor.

        The feature contains the next waypoint relative to each defender, a
        one-hot route intent, and a belief-blind indicator.  It is generated
        from delayed/noisy public beliefs and observed obstacle geometry only;
        keeping it explicit prevents the action regressor from averaging
        incompatible left/right bypass actions.
        """
        target, target_velocity = self._public_target_estimate(observation)
        if not self.route_paths or self.env.step_count % self.replan_interval_steps == 0:
            selected, details = (
                self._blind_rally_routes()
                if self.belief_blind
                else self._candidate_routes(target, target_velocity)
            )
            if selected != self.route_name:
                self.route_started_step = int(self.env.step_count)
            self.route_name = selected
            self.route_paths = details["paths"]
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        route_index = {name: index for index, name in enumerate(("left_bypass", "right_bypass", "upper_bypass"))}
        one_hot = np.zeros(3, dtype=np.float32)
        if self.route_name in route_index:
            one_hot[route_index[self.route_name]] = 1.0
        values: list[np.ndarray] = []
        for index, position in enumerate(positions):
            path = self.route_paths[index]
            waypoint = self._next_waypoint(path, position) if path is not None else target
            values.append(
                np.concatenate(
                    [
                        ((waypoint - position) / float(self.env.world["half_extent_xy"])).astype(np.float32),
                        one_hot,
                        np.array([float(self.belief_blind)], dtype=np.float32),
                    ]
                )
            )
        return np.stack(values).astype(np.float32)

    def _smooth(self, desired: np.ndarray) -> np.ndarray:
        dt = float(self.env.dt)
        max_delta = float(self.env.agents["defender_max_acceleration"]) * dt
        max_jerk_delta = float(self.env.pursuit.get("teacher_max_jerk_mps3", 12.0)) * dt * dt
        max_turn = float(self.env.pursuit.get("teacher_max_turn_rate_rad_s", 1.05)) * dt
        clipped = self.env._clip_rows(desired, self.max_speed)
        output = np.empty_like(clipped)
        for index, command in enumerate(clipped):
            previous = self.last_command[index]
            delta = command - previous
            norm = float(np.linalg.norm(delta))
            if norm > max_delta:
                delta *= max_delta / norm
            jerk_delta = delta - self.last_delta[index]
            jerk_norm = float(np.linalg.norm(jerk_delta))
            if jerk_norm > max_jerk_delta:
                delta += jerk_delta * (max_jerk_delta / jerk_norm - 1.0)
            candidate = previous + delta
            previous_norm = float(np.linalg.norm(previous))
            candidate_norm = float(np.linalg.norm(candidate))
            if previous_norm > 1e-6 and candidate_norm > 1e-6:
                cosine = float(np.clip(np.dot(previous, candidate) / (previous_norm * candidate_norm), -1.0, 1.0))
                angle = float(np.arccos(cosine))
                if angle > max_turn:
                    blend = max_turn / angle
                    direction = _unit((1.0 - blend) * previous / previous_norm + blend * candidate / candidate_norm)
                    candidate = direction * candidate_norm
            output[index] = candidate
        self.last_delta = output - self.last_command
        self.last_command = output.copy()
        return self.env._clip_rows(output, self.max_speed)

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        target, target_velocity = self._public_target_estimate(observation)
        if not self.route_paths or self.env.step_count % self.replan_interval_steps == 0:
            selected, details = (
                self._blind_rally_routes()
                if self.belief_blind
                else self._candidate_routes(target, target_velocity)
            )
            if selected != self.route_name:
                self.route_started_step = int(self.env.step_count)
            self.route_name = selected
            self.route_paths = details["paths"]
        positions = np.asarray(observation["defender_positions"], dtype=np.float64)
        distances = np.linalg.norm(positions - target[None, :], axis=1)
        interceptor = (
            int(np.argmin(distances))
            if self.fixed_interceptor_id is None
            else self.fixed_interceptor_id
        )
        perimeter = float(np.clip(0.55 * np.median(distances), self.env.pursuit["capture_radius"] + 0.3, 2.4))
        desired = np.zeros_like(positions)
        for index, position in enumerate(positions):
            if float(distances[index]) < 3.0:
                goal = target if index == interceptor else target + TETRAHEDRON_DIRECTIONS[index] * perimeter
            else:
                path = self.route_paths[index]
                goal = self._next_waypoint(path, position) if path is not None else target
            desired[index] = self.gain * (goal - position) + target_velocity
        return self._avoidance(self._smooth(desired), observation)


class SafetyFilteredPursuitController:
    """Wrap a pursuit policy with the local-information CBF filter."""

    def __init__(self, controller: _PursuitController) -> None:
        self.controller = controller
        self.safety_filter = PursuitCBFSafetyFilter(controller.env)
        self.last_diagnostics = PursuitSafetyDiagnostics(0.0, float("inf"))

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        desired = self.controller.act(observation)
        safe, self.last_diagnostics = self.safety_filter.filter(desired, observation)
        return safe
