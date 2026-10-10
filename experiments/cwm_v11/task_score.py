"""Piecewise differentiable FULL frozen K=1 local cost, not a new objective.

Only the target-dependent distance/hinge/formation terms are reimplemented.
Every target-independent term is retained from the unmodified original engine.
Unsupported QDR/reachability/escape/FC objectives fail closed, never disappear.
Role selection is discrete: gradients are local within a fixed role region.
"""
import numpy as np
import torch


class FrozenLocalScore:
    def __init__(self, call, actions):
        from encirclement3d.distributed_dn_mpc import _clip_rows, _qdr_execution_aware
        from encirclement3d.minimax_mpc import ScenarioTrajectorySet, TETRAHEDRON_DIRECTIONS
        self.call, self.config = call, call["planner"].config
        config, observation = self.config, call["observation"]
        if (config.reachability_normalized_cost_enabled or config.escape_gap_cost_enabled
                or config.fc_dbf_enabled or observation.get("qdr_execution_aware", False)
                or _qdr_execution_aware(observation)):
            raise ValueError("Unsupported target-dependent/gated objective; do not omit it")
        if config.horizon_steps != 8 or config.risk_mode not in ("worst_case", "expected", "cvar"):
            raise ValueError("Frozen K=1 horizon/risk contract mismatch")
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 3 or actions.shape[1:] != (8, 3) or not np.isfinite(actions).all():
            raise ValueError("Finite local candidate shape mismatch")
        self.actions = _clip_rows(actions, config.max_speed_mps)
        self.agent = int(call["agent"])
        self.current = torch.from_numpy(observation["defender_positions"][self.agent][None, None]
                                       + np.cumsum(self.actions * config.dt_seconds, axis=1))
        # Preserve original insertion order, including exact-distance ties.
        positions = {self.agent: observation["defender_positions"][self.agent]}
        positions.update({int(p): message.position for p, message in call["known"].items()})
        self.peer_ids = torch.tensor(list(positions), dtype=torch.int64)
        self.peer_positions = torch.from_numpy(np.asarray(list(positions.values()), dtype=np.float64))
        self.direction = torch.from_numpy(TETRAHEDRON_DIRECTIONS[self.agent % len(TETRAHEDRON_DIRECTIONS)].copy())
        zero = torch.zeros((len(actions), 8, 3), dtype=torch.float64)
        original = call["planner"]._local_scenario_cost_matrix(
            observation, ScenarioTrajectorySet(np.zeros((1, 8, 3)), np.ones(1), dynamics_status="raw"),
            self.agent, self.actions, call["known"], call["peer_sequences"])[:, 0]
        self.offset = torch.from_numpy(original) - self.target_terms(zero).detach()
        if not torch.isfinite(self.offset).all():
            raise ValueError("Nonfinite original target-independent cost")

    def roles(self, paths):
        distances = torch.linalg.vector_norm(self.peer_positions[None] - paths[:, 0, None], dim=-1)
        return self.peer_ids[torch.argmin(distances, dim=1)]

    def role_gap(self, paths):
        distances = torch.sort(torch.linalg.vector_norm(self.peer_positions[None] - paths[:, 0, None], dim=-1), dim=1).values
        return distances[:, 1] - distances[:, 0] if len(self.peer_ids) > 1 else torch.full((len(paths),), float("inf"))

    def target_terms(self, paths):
        if paths.shape != self.current.shape or paths.dtype != torch.float64 or not torch.isfinite(paths).all():
            raise ValueError("Finite float64 one-path-per-candidate contract mismatch")
        config = self.config
        distances = torch.linalg.vector_norm(self.current - paths, dim=-1)
        result = config.weight_distance * distances.sum(1)
        result = result + config.weight_capture_hinge * torch.relu(distances - config.capture_radius_m).square().sum(1)
        result = result + config.weight_terminal_distance * distances[:, -1]
        target_points = paths + self.direction[None, None] * config.role_perimeter_m
        formation = (torch.linalg.vector_norm(self.current - target_points, dim=-1) - config.role_perimeter_m).abs()
        return result + config.weight_formation * formation.sum(1) * (self.roles(paths) != self.agent)

    def __call__(self, paths):
        return self.target_terms(paths) + self.offset


def original_scores(call, actions, paths):
    """Independent original full-engine evaluation, one K=1 path per action."""
    from encirclement3d.minimax_mpc import ScenarioTrajectorySet, aggregate_scenario_costs
    planner = call["planner"]
    if np.array_equal(paths, np.repeat(paths[:1], len(paths), axis=0)):
        scenarios = ScenarioTrajectorySet(paths[:1], np.ones(1), dynamics_status="raw")
        matrix = planner._local_scenario_cost_matrix(call["observation"], scenarios, call["agent"], actions,
                                                    call["known"], call["peer_sequences"])
        return np.asarray([aggregate_scenario_costs(r, scenarios.normalized_weights, planner.config.risk_mode,
                                                  planner.config.cvar_alpha) for r in matrix])
    result = []
    for action, path in zip(actions, paths):
        scenarios = ScenarioTrajectorySet(path[None], np.ones(1), dynamics_status="raw")
        row = planner._local_scenario_cost_matrix(call["observation"], scenarios, call["agent"], action[None],
                                                 call["known"], call["peer_sequences"])[0]
        result.append(aggregate_scenario_costs(row, scenarios.normalized_weights, planner.config.risk_mode, planner.config.cvar_alpha))
    return np.asarray(result)


def task_effect_loss(score, response, target, anchor_target, valid, anchor_valid):
    """Full-support paired cost-effect supervision, anchored in observed truth.

    Does NOT reward low predicted cost or backpropagate into the motion head.
    Both predicted and label effects use the SAME observed reference path;
    values have a gradient-norm normalization into approximate squared meters.
    Terminal-incomplete calls must never enter this auxiliary full-horizon loss.
    """
    if response.shape != target.shape or anchor_target.shape != (8, 3):
        raise ValueError("Paired effect label shapes mismatch")
    if valid.shape != response.shape[:2] or anchor_valid.shape != (8,) or not bool(valid.all()) or not bool(anchor_valid.all()):
        raise ValueError("Task-effect loss requires complete common terminal support")
    reference = anchor_target.detach()[None].repeat(len(response), 1, 1).double().requires_grad_(True)
    reference_cost = score(reference)
    gradient = torch.autograd.grad(reference_cost.sum(), reference, create_graph=False)[0].detach()
    scale = gradient.square().sum((1, 2)).clamp_min(1.)
    label_effect = (score(target.detach().double()) - reference_cost.detach()).detach()
    predicted_effect = score(reference.detach() + response.double()) - reference_cost.detach()
    return ((predicted_effect - label_effect).square() / scale).mean()
