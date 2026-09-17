"""Train and validate a DN-MPC-teacher recurrent residual pursuit actor.

This calibration-only experiment keeps the Phase78 obstacle-avoidance-first
target contract.  The public-belief route controller is the cheap base action;
distributed delayed DN-MPC supplies teacher labels, and DAgger rounds collect
labels on states visited by the current residual actor.  No locked-test scene
is read by this script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.distributed_dn_mpc import DistributedDNMPCConfig, DistributedMinimaxDNMPC  # noqa: E402
from encirclement3d.minimax_mpc import MinimaxMPCConfig, make_belief_candidate_set  # noqa: E402
from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.pursuit_controllers import (  # noqa: E402
    PublicBeliefRouteIntentController,
    PursuitCBFSafetyFilter,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.queue_aware_rollout import (  # noqa: E402
    prepare_queue_aware_observation,
    shift_scenario_trajectory_set,
)
from encirclement3d.residual_actor import RecurrentResidualActor  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    prepare_showcase_episode,
    scenario_from_metadata,
)

from evaluate_minimax_mpc import _planning_observation, _shift_warm_start_sequence  # noqa: E402
from evaluate_s4_closed_loop import config_for_phase73_spec, read_scenes  # noqa: E402


@dataclass
class Trajectory:
    local: np.ndarray
    base: np.ndarray
    target: np.ndarray
    recovery: np.ndarray
    metadata: dict[str, Any]


@dataclass
class TeacherOutput:
    base_action: np.ndarray
    teacher_action: np.ndarray
    route_features: np.ndarray
    recovery: bool
    planner_status: str
    planner_latency_ms: float
    base_cbf_correction_norm: float = 0.0
    teacher_cbf_correction_norm: float = 0.0
    teacher_base_gap_norm: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--torch-threads",
        type=int,
        help="Optional deterministic PyTorch intra/inter-op CPU thread count.",
    )
    parser.add_argument("--demo-episodes", type=int)
    parser.add_argument("--dagger-rounds", type=int)
    parser.add_argument("--episodes-per-round", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument("--allow-small", action="store_true", help="Allow a tiny smoke run; never use for Phase79 results.")
    return parser.parse_args()


def load_document(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    document = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Phase79 config must be a mapping.")
    environment_path = Path(str(document["environment_config"]))
    if not environment_path.is_absolute():
        environment_path = path.parent / environment_path
    environment = yaml.safe_load(environment_path.resolve().read_text(encoding="utf-8"))
    if not isinstance(environment, dict):
        raise ValueError("environment_config must contain a mapping.")
    overrides = document.get("environment_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("environment_overrides must be a mapping.")

    def merge(base: dict[str, Any], update: dict[str, Any]) -> None:
        for key, value in update.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)

    merge(environment, overrides)
    settings = dict(document.get("experiment", {}))
    return document, environment, settings


def select_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return torch.device(name)


def resolve_config_path(value: str | Path, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = (config_path.parent / path, PROJECT_ROOT / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return (config_path.parent / path).resolve()


def load_records(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records = read_scenes(path.resolve(), limit)
    if not records:
        raise ValueError("The calibration scene file is empty.")
    if any(bool(item.get("target_crossing_required", False)) for item in records):
        raise ValueError("Phase79 refuses target-crossing scenes.")
    if any("locked" in str(item.get("scene_block", "")).lower() for item in records):
        raise ValueError("Phase79 refuses locked-test records.")
    return records


def planner_config(path: Path) -> MinimaxMPCConfig:
    document = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("planner"), dict):
        raise ValueError("MPC config must contain planner.")
    return MinimaxMPCConfig.from_mapping(dict(document["planner"]))


def build_environment(
    environment_config: dict[str, Any],
    record: dict[str, Any],
    *,
    max_steps: int | None,
) -> tuple[CaptureRadiusPursuit3DEnv, Any, dict[str, Any]]:
    config = copy.deepcopy(environment_config)
    pursuit = config.setdefault("task", {}).setdefault("pursuit", {})
    pursuit.update(copy.deepcopy(record["pursuit_overrides"]))
    pursuit["target_motion_mode"] = str(record.get("target_motion_mode", "adaptive_maneuvering"))
    execution = config.setdefault("dynamics", {}).setdefault("execution", {})
    record_execution = copy.deepcopy(record.get("execution", {}))
    if "command_noise_std_mps" in record_execution and "command_noise_std" not in record_execution:
        record_execution["command_noise_std"] = record_execution.pop("command_noise_std_mps")
    execution.update(record_execution)
    config.setdefault("experiments", [{"obstacle_count": int(record.get("obstacle_count", 3)), "target_speed_scale": float(record["target_speed_scale"])}])
    config["world"]["max_steps"] = int(max_steps if max_steps is not None else config["world"].get("max_steps", 250))
    scenario = scenario_from_metadata(record["scenario"])
    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=int(record.get("obstacle_count", len(scenario.obstacles))),
        target_speed_scale=float(record["target_speed_scale"]),
    )
    observation = prepare_showcase_episode(
        env,
        scenario,
        seed=int(record["episode_seed"]),
        record_history=True,
        validate_scenario=False,
    )
    return env, observation, scenario


class DNMPCTeacher:
    """Public-belief delayed distributed DN-MPC teacher with route fallback."""

    def __init__(self, env: CaptureRadiusPursuit3DEnv, config: MinimaxMPCConfig, settings: dict[str, Any]) -> None:
        self.env = env
        self.config = config
        self.settings = settings
        self.route = PublicBeliefRouteIntentController(
            env,
            horizon_seconds=float(settings.get("teacher_horizon_seconds", 0.75)),
            replan_interval_steps=int(settings.get("teacher_replan_interval_steps", 8)),
            min_hold_steps=int(settings.get("teacher_min_hold_steps", 6)),
            grid_step=float(settings.get("teacher_grid_step_m", 0.75)),
            route_margin=float(settings.get("teacher_route_margin_m", 0.85)),
        )
        self.safety = PursuitCBFSafetyFilter(env)
        distributed = DistributedDNMPCConfig(
            communication_mode="delayed",
            max_iterations=int(settings.get("dnmpc_max_iterations", 3)),
            communication_interval_steps=int(settings.get("dnmpc_communication_interval_steps", 1)),
            message_delay_steps=int(settings.get("dnmpc_message_delay_steps", 1)),
            max_message_age_steps=int(settings.get("dnmpc_max_message_age_steps", 8)),
            local_timeout_ms=float(settings.get("dnmpc_local_timeout_ms", 50.0)),
        )
        self.planner = DistributedMinimaxDNMPC(config, distributed)
        self.previous_sequence: np.ndarray | None = None
        self.cached_sequence: np.ndarray | None = None
        self.last_route_features: np.ndarray | None = None

    def output(self, observation: dict[str, Any]) -> TeacherOutput:
        route_features = self.route.route_features(observation)
        base_action = _route_base_action(self.route, self.safety, observation, self.settings)
        replan_interval = int(self.settings.get("dnmpc_replan_interval_steps", 4))
        should_replan = self.cached_sequence is None or self.env.step_count % replan_interval == 0
        planner_status = "held_previous_plan"
        planner_latency_ms = 0.0
        if should_replan:
            candidates = make_belief_candidate_set(
                observation,
                horizon_steps=int(self.config.horizon_steps),
                dt_seconds=float(self.config.dt_seconds),
                max_speed_mps=float(self.env.agents["target_max_speed"]),
                candidate_count=int(self.settings.get("dnmpc_candidate_count", 8)),
                belief_fusion_config=self.config,
            )
            planning_observation, qdr_state = prepare_queue_aware_observation(
                _planning_observation(self.env, observation),
                dt_seconds=float(self.config.dt_seconds),
            )
            planning_scenarios = shift_scenario_trajectory_set(
                candidates,
                offset_steps=int(qdr_state.queue_length),
                horizon_steps=int(self.config.horizon_steps),
                dt_seconds=float(self.config.dt_seconds),
                max_speed_mps=float(self.config.max_speed_mps),
            )
            started = time.perf_counter()
            plan = self.planner.plan(
                planning_observation,
                planning_scenarios,
                step_index=int(self.env.step_count),
                previous_action_sequence=self.previous_sequence,
                fallback_actions=base_action,
            )
            planner_latency_ms = (time.perf_counter() - started) * 1000.0
            self.cached_sequence = np.asarray(plan.action_sequence, dtype=np.float64)
            self.previous_sequence = _shift_warm_start_sequence(self.cached_sequence)
            planner_status = str(plan.diagnostics.status)
        if self.cached_sequence is None:
            raise RuntimeError("DN-MPC teacher did not produce an action sequence.")
        teacher_action = np.asarray(self.cached_sequence[0], dtype=np.float64).copy()
        self.cached_sequence = _shift_warm_start_sequence(self.cached_sequence)
        teacher_action, safety_diag = self.safety.filter(teacher_action, observation)
        residual_norm = float(np.mean(np.linalg.norm(teacher_action - base_action, axis=1)))
        base_cbf_correction = float(getattr(self.route, "last_base_cbf_correction_norm", 0.0))
        recovery = bool(
            residual_norm >= float(self.settings.get("recovery_action_gap_m", 0.35))
            or safety_diag.action_correction_norm >= float(self.settings.get("recovery_cbf_correction_mps", 0.20))
            or planner_status not in {"success", "held_previous_plan"}
        )
        self.last_route_features = route_features
        return TeacherOutput(
            base_action=np.asarray(base_action, dtype=np.float32),
            teacher_action=np.asarray(teacher_action, dtype=np.float32),
            route_features=np.asarray(route_features, dtype=np.float32),
            recovery=recovery,
            planner_status=planner_status,
            planner_latency_ms=float(planner_latency_ms),
            base_cbf_correction_norm=base_cbf_correction,
            teacher_cbf_correction_norm=float(safety_diag.action_correction_norm),
            teacher_base_gap_norm=residual_norm,
        )

    def base_output(self, observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Return the cheap public-belief route base without invoking DN-MPC."""

        route_features = self.route.route_features(observation)
        base_action = _route_base_action(self.route, self.safety, observation, self.settings)
        return np.asarray(base_action, dtype=np.float32), np.asarray(route_features, dtype=np.float32)


def _route_base_action(
    route: PublicBeliefRouteIntentController,
    safety: PursuitCBFSafetyFilter,
    observation: dict[str, Any],
    settings: dict[str, Any],
) -> np.ndarray:
    """Return the exact route-base contract used by training and deployment.

    Phase78 bootstrap actions were produced after the local safety projection.
    A residual actor must therefore receive the same projected base during
    DAgger and evaluation; otherwise a zero residual is trained against one
    action distribution and deployed on another one.
    """

    raw_action = route.act(observation)
    if not bool(settings.get("filtered_route_base", True)):
        route.last_base_cbf_correction_norm = 0.0
        return np.asarray(raw_action, dtype=np.float32)
    filtered_action, diagnostics = safety.filter(raw_action, observation)
    route.last_base_cbf_correction_norm = float(diagnostics.action_correction_norm)
    return np.asarray(filtered_action, dtype=np.float32)


def actor_action(
    actor: RecurrentResidualActor,
    local: np.ndarray,
    base_action: np.ndarray,
    hidden: torch.Tensor,
    action_scale: float,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    with torch.no_grad():
        action, hidden, _residual = actor.step(
            torch.as_tensor(local, dtype=torch.float32, device=device),
            torch.as_tensor(base_action, dtype=torch.float32, device=device),
            hidden,
            action_scale,
        )
    return action.cpu().numpy().astype(np.float32), hidden


def _evaluation_rate(rows: list[dict[str, Any]], name: str) -> float:
    return float(np.mean([bool(row.get(name, False)) for row in rows])) if rows else 0.0


def _evaluation_percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        f"p{percentile}": float(np.percentile(np.asarray(values, dtype=np.float64), percentile))
        for percentile in (50, 95, 99)
    }


def _write_evaluation_progress(
    path: Path,
    *,
    target_episodes: int,
    rows: list[dict[str, Any]],
    route_latencies: list[float],
    actor_latencies: list[float],
    safety_latencies: list[float],
    total_latencies: list[float],
) -> None:
    """Write an atomic, calibration-only progress snapshot for long evaluations."""

    payload = {
        "experiment_name": "phase79_dagger_residual_policy_evaluation_progress",
        "evaluation_split": "development_validation_only",
        "locked_test_used": False,
        "episodes_target": int(target_episodes),
        "episodes_completed": len(rows),
        "complete": bool(len(rows) == target_episodes),
        "safe_capture_rate": _evaluation_rate(rows, "safe_capture_success"),
        "collision_rate": _evaluation_rate(rows, "collision"),
        "boundary_violation_rate": _evaluation_rate(rows, "boundary_violation"),
        "timeout_rate": _evaluation_rate(rows, "timeout"),
        "latency_ms": {
            "route_intent": _evaluation_percentiles(route_latencies),
            "actor": _evaluation_percentiles(actor_latencies),
            "safety": _evaluation_percentiles(safety_latencies),
            "total": _evaluation_percentiles(total_latencies),
        },
        "last_episode": rows[-1] if rows else None,
    }
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def rollout(
    environment_config: dict[str, Any],
    record: dict[str, Any],
    planner: MinimaxMPCConfig,
    settings: dict[str, Any],
    *,
    actor: RecurrentResidualActor | None,
    device: torch.device,
    max_steps: int | None,
    use_actor: bool,
    episode_seed: int,
) -> tuple[Trajectory, dict[str, Any]]:
    record = copy.deepcopy(record)
    record["episode_seed"] = int(episode_seed)
    env, observation, _scenario = build_environment(environment_config, record, max_steps=max_steps)
    teacher = DNMPCTeacher(env, planner, settings)
    action_scale = float(env.agents["defender_max_speed"])
    hidden = None if actor is None else actor.initial_hidden(env.n_defenders, device=device)
    local_frames: list[np.ndarray] = []
    base_frames: list[np.ndarray] = []
    target_frames: list[np.ndarray] = []
    recovery_frames: list[bool] = []
    planner_statuses: list[str] = []
    planner_latencies: list[float] = []
    while True:
        teacher_output = teacher.output(observation)
        local = np.concatenate(
            [policy_observations(env, observation), teacher_output.route_features], axis=1
        ).astype(np.float32)
        local_frames.append(local.copy())
        base_frames.append(teacher_output.base_action.copy())
        target_frames.append(teacher_output.teacher_action.copy())
        recovery_frames.append(bool(teacher_output.recovery))
        planner_statuses.append(teacher_output.planner_status)
        planner_latencies.append(float(teacher_output.planner_latency_ms))
        if use_actor:
            if actor is None or hidden is None:
                raise RuntimeError("use_actor requires actor and hidden state.")
            action, hidden = actor_action(
                actor,
                local,
                teacher_output.base_action,
                hidden,
                action_scale,
                device,
            )
        else:
            action = teacher_output.teacher_action
        observation, _reward, terminated, truncated, info = env.step(action, record_history=True)
        if terminated or truncated:
            break
    local_array = np.stack(local_frames, axis=0)
    trajectory = Trajectory(
        local=local_array,
        base=np.stack(base_frames, axis=0),
        target=np.stack(target_frames, axis=0),
        recovery=np.asarray(recovery_frames, dtype=bool),
        metadata={
            "episode_index": int(record["episode_index"]),
            "episode_seed": int(episode_seed),
            "steps": int(env.step_count),
            "planner_status_counts": {
                status: planner_statuses.count(status) for status in sorted(set(planner_statuses))
            },
            "recovery_steps": int(np.sum(recovery_frames)),
            "planner_latency_ms_p50": float(np.percentile(planner_latencies, 50)) if planner_latencies else 0.0,
            "planner_latency_ms_p95": float(np.percentile(planner_latencies, 95)) if planner_latencies else 0.0,
            "termination_reason": str(info["termination_reason"]),
            "safe_capture_success": bool(info["safe_capture_success"]),
            "capture_event": bool(info["capture_event"]),
            "collision": bool(info["collision"]),
            "boundary_violation": bool(info.get("boundary_violation", info["world_violation_steps"] > 0)),
            "target_boundary_violation": bool(
                info.get("target_boundary_violation", info.get("target_world_violation_steps", 0) > 0)
            ),
            "defender_boundary_violation": bool(
                info.get("defender_boundary_violation", info.get("defender_world_violation_steps", 0) > 0)
            ),
            "target_invalid_episode": bool(info.get("target_invalid_episode", False)),
            "task_valid_for_policy_evaluation": bool(info.get("task_valid_for_policy_evaluation", True)),
            "first_target_boundary_violation_step": info.get("first_target_boundary_violation_step"),
            "first_defender_boundary_violation_step": info.get("first_defender_boundary_violation_step"),
            "timeout": bool(info["termination_reason"] == "timeout"),
        },
    )
    return trajectory, trajectory.metadata


def build_sequences(
    trajectories: list[Trajectory],
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    local_sequences: list[np.ndarray] = []
    base_sequences: list[np.ndarray] = []
    target_sequences: list[np.ndarray] = []
    reset_sequences: list[np.ndarray] = []
    weight_sequences: list[np.ndarray] = []
    for trajectory in trajectories:
        for start in range(0, trajectory.local.shape[0], sequence_length):
            stop = min(start + sequence_length, trajectory.local.shape[0])
            size = stop - start
            local = np.repeat(trajectory.local[-1:, :, :], sequence_length, axis=0)
            base = np.repeat(trajectory.base[-1:, :, :], sequence_length, axis=0)
            target = np.repeat(trajectory.target[-1:, :, :], sequence_length, axis=0)
            recovery = np.repeat(trajectory.recovery[-1:], sequence_length, axis=0)
            local[:size] = trajectory.local[start:stop]
            base[:size] = trajectory.base[start:stop]
            target[:size] = trajectory.target[start:stop]
            recovery[:size] = trajectory.recovery[start:stop]
            valid = np.zeros(sequence_length, dtype=np.float32)
            valid[:size] = 1.0
            reset = np.zeros(sequence_length, dtype=np.float32)
            reset[0] = 1.0
            weights = valid * np.where(recovery, 3.0, 1.0).astype(np.float32)
            local_sequences.append(local)
            base_sequences.append(base)
            target_sequences.append(target)
            reset_sequences.append(reset)
            weight_sequences.append(weights)
    if not local_sequences:
        raise ValueError("No trajectory frames were collected.")
    return tuple(
        np.stack(values, axis=0).astype(np.float32)
        for values in (local_sequences, base_sequences, target_sequences, reset_sequences, weight_sequences)
    )


def load_bootstrap_trajectories(paths: list[Path]) -> tuple[list[Trajectory], int]:
    """Reuse retained Phase78 accepted demonstrations as one 192-demo pool."""

    trajectories: list[Trajectory] = []
    accepted_episode_count = 0
    for path in paths:
        with np.load(path.resolve()) as archive:
            required = {"local_observations", "actions"}
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(f"Bootstrap dataset is missing: {', '.join(missing)}: {path}")
            local = np.asarray(archive["local_observations"], dtype=np.float32)
            actions = np.asarray(archive["actions"], dtype=np.float32)
        if local.ndim != 4 or actions.shape != (*local.shape[:3], 3):
            raise ValueError(f"Bootstrap dataset has incompatible shapes: {path}")
        manifest_path = path.with_name("expert_dataset_manifest.json")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            accepted_episode_count += int(manifest.get("accepted_episodes", len(local)))
        else:
            accepted_episode_count += int(local.shape[0])
        for sequence_local, sequence_actions in zip(local, actions, strict=True):
            length = int(sequence_local.shape[0])
            trajectories.append(
                Trajectory(
                    local=sequence_local.copy(),
                    base=sequence_actions.copy(),
                    target=sequence_actions.copy(),
                    recovery=np.zeros(length, dtype=bool),
                    metadata={"source": str(path.resolve()), "bootstrap": True},
                )
            )
    if accepted_episode_count < 192:
        raise ValueError(
            "Bootstrap demonstrations must contain at least 192 accepted episodes; "
            f"found {accepted_episode_count}."
        )
    return trajectories, accepted_episode_count


def teacher_demo_is_accepted(metadata: dict[str, Any]) -> bool:
    """Return whether one teacher rollout is safe enough for bootstrap data."""

    return bool(
        metadata.get("safe_capture_success", False)
        and not metadata.get("collision", False)
        and not metadata.get("boundary_violation", False)
        and not metadata.get("target_invalid_episode", False)
        and metadata.get("task_valid_for_policy_evaluation", True)
        and not metadata.get("timeout", False)
    )


def train_actor(
    actor: RecurrentResidualActor,
    trajectories: list[Trajectory],
    settings: dict[str, Any],
    device: torch.device,
    action_scale: float,
) -> list[dict[str, float]]:
    sequence_length = int(settings.get("sequence_length", 32))
    batch_size = int(settings.get("sequence_batch_size", 16))
    local, base, target, resets, weights = build_sequences(trajectories, sequence_length)
    local_tensor = torch.as_tensor(local, device=device)
    base_tensor = torch.as_tensor(base, device=device)
    target_tensor = torch.as_tensor(target, device=device)
    reset_tensor = torch.as_tensor(resets, device=device)
    weight_tensor = torch.as_tensor(weights, device=device)
    optimizer = torch.optim.Adam(actor.actor_parameters(), lr=float(settings.get("learning_rate", 5.0e-4)))
    rng = np.random.default_rng(int(settings["seed"]))
    history: list[dict[str, float]] = []
    for epoch in range(int(settings.get("epochs", 25))):
        actor.train()
        permutation = rng.permutation(local.shape[0])
        losses: list[float] = []
        for start in range(0, local.shape[0], batch_size):
            indices = permutation[start : start + batch_size]
            selected_local = local_tensor[indices]
            selected_base = base_tensor[indices]
            selected_target = target_tensor[indices]
            selected_reset = reset_tensor[indices]
            selected_weight = weight_tensor[indices]
            hidden = actor.initial_hidden(
                int(selected_local.shape[2]), batch_size=int(selected_local.shape[0]), device=device
            )
            predicted, _residual = actor.sequence(
                selected_local,
                selected_base,
                hidden,
                selected_reset,
                action_scale,
            )
            error = torch.nn.functional.smooth_l1_loss(predicted, selected_target, reduction="none").mean(dim=-1)
            weighted = error * selected_weight[:, :, None]
            loss = weighted.sum() / torch.clamp(selected_weight.sum() * selected_local.shape[2], min=1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.actor_parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses))})
    actor.eval()
    return history


def evaluate_policy(
    environment_config: dict[str, Any],
    records: list[dict[str, Any]],
    planner: MinimaxMPCConfig,
    settings: dict[str, Any],
    actor: RecurrentResidualActor,
    device: torch.device,
    max_steps: int | None,
    progress_path: Path | None = None,
    progress_interval: int = 10,
) -> dict[str, Any]:
    if progress_interval <= 0:
        raise ValueError("progress_interval must be positive")
    rows: list[dict[str, Any]] = []
    route_latencies: list[float] = []
    actor_latencies: list[float] = []
    safety_latencies: list[float] = []
    total_latencies: list[float] = []
    if progress_path is not None:
        _write_evaluation_progress(
            progress_path,
            target_episodes=len(records),
            rows=rows,
            route_latencies=route_latencies,
            actor_latencies=actor_latencies,
            safety_latencies=safety_latencies,
            total_latencies=total_latencies,
        )
    for index, record in enumerate(records):
        evaluation_record = copy.deepcopy(record)
        evaluation_record["episode_seed"] = int(record["episode_seed"]) + 910_000 + index
        env, observation, _scenario = build_environment(environment_config, evaluation_record, max_steps=max_steps)
        route = PublicBeliefRouteIntentController(
            env,
            horizon_seconds=float(settings.get("teacher_horizon_seconds", 0.75)),
            replan_interval_steps=int(settings.get("teacher_replan_interval_steps", 8)),
            min_hold_steps=int(settings.get("teacher_min_hold_steps", 6)),
            grid_step=float(settings.get("teacher_grid_step_m", 0.75)),
            route_margin=float(settings.get("teacher_route_margin_m", 0.85)),
        )
        safety = PursuitCBFSafetyFilter(env)
        hidden = actor.initial_hidden(env.n_defenders, device=device)
        while True:
            route_started = time.perf_counter()
            route_features = route.route_features(observation)
            base_action = _route_base_action(route, safety, observation, settings)
            route_ms = (time.perf_counter() - route_started) * 1000.0
            local = np.concatenate(
                [policy_observations(env, observation), route_features], axis=1
            ).astype(np.float32)
            actor_started = time.perf_counter()
            action, hidden = actor_action(
                actor,
                local,
                base_action,
                hidden,
                float(env.agents["defender_max_speed"]),
                device,
            )
            actor_ms = (time.perf_counter() - actor_started) * 1000.0
            safety_started = time.perf_counter()
            action, _diagnostics = safety.filter(action, observation)
            safety_ms = (time.perf_counter() - safety_started) * 1000.0
            total_ms = route_ms + actor_ms + safety_ms
            route_latencies.append(route_ms)
            actor_latencies.append(actor_ms)
            safety_latencies.append(safety_ms)
            total_latencies.append(total_ms)
            observation, _reward, terminated, truncated, info = env.step(action, record_history=True)
            if terminated or truncated:
                rows.append(
                    {
                        "episode_index": int(record["episode_index"]),
                        "episode_seed": int(evaluation_record["episode_seed"]),
                        "safe_capture_success": bool(info["safe_capture_success"]),
                        "capture_event": bool(info["capture_event"]),
                        "collision": bool(info["collision"]),
                        "boundary_violation": bool(
                            info.get("boundary_violation", info["world_violation_steps"] > 0)
                        ),
                        "target_boundary_violation": bool(
                            info.get("target_boundary_violation", info.get("target_world_violation_steps", 0) > 0)
                        ),
                        "defender_boundary_violation": bool(
                            info.get("defender_boundary_violation", info.get("defender_world_violation_steps", 0) > 0)
                        ),
                        "target_invalid_episode": bool(info.get("target_invalid_episode", False)),
                        "task_valid_for_policy_evaluation": bool(
                            info.get("task_valid_for_policy_evaluation", True)
                        ),
                        "first_target_boundary_violation_step": info.get("first_target_boundary_violation_step"),
                        "first_defender_boundary_violation_step": info.get("first_defender_boundary_violation_step"),
                        "timeout": bool(info["termination_reason"] == "timeout"),
                        "termination_reason": str(info["termination_reason"]),
                        "steps": int(env.step_count),
                        "min_clearance_m": float(info["min_clearance_so_far"]),
                    }
                )
                if progress_path is not None and (
                    len(rows) == 1
                    or len(rows) % progress_interval == 0
                    or len(rows) == len(records)
                ):
                    _write_evaluation_progress(
                        progress_path,
                        target_episodes=len(records),
                        rows=rows,
                        route_latencies=route_latencies,
                        actor_latencies=actor_latencies,
                        safety_latencies=safety_latencies,
                        total_latencies=total_latencies,
                    )
                    print(
                        json.dumps(
                            {
                                "phase": "evaluation_progress",
                                "completed": len(rows),
                                "total": len(records),
                            }
                        ),
                        flush=True,
                    )
                break
    def rate(name: str) -> float:
        return _evaluation_rate(rows, name)

    def percentiles(values: list[float]) -> dict[str, float]:
        return _evaluation_percentiles(values)

    return {
        "experiment_name": "phase79_dagger_residual_nominal_validation",
        "evaluation_split": "development_validation_only",
        "locked_test_used": False,
        "episodes": len(rows),
        "safe_capture_rate": rate("safe_capture_success"),
        "capture_event_rate": rate("capture_event"),
        "collision_rate": rate("collision"),
        "boundary_violation_rate": rate("boundary_violation"),
        "target_invalid_episode_rate": rate("target_invalid_episode"),
        "target_boundary_violation_rate": rate("target_boundary_violation"),
        "defender_boundary_violation_rate": rate("defender_boundary_violation"),
        "timeout_rate": rate("timeout"),
        "mean_capture_time_seconds": float(np.mean([row["steps"] for row in rows])) * float(settings.get("dt_seconds", 0.1)) if rows else 0.0,
        "mean_min_clearance_m": float(np.mean([row["min_clearance_m"] for row in rows])) if rows else 0.0,
        "termination_reason_counts": {
            reason: sum(row["termination_reason"] == reason for row in rows)
            for reason in sorted({row["termination_reason"] for row in rows})
        },
        "latency_ms": {
            "route_intent": percentiles(route_latencies),
            "actor": percentiles(actor_latencies),
            "safety": percentiles(safety_latencies),
            "total": percentiles(total_latencies),
        },
        "local_cbf_is_empirical_filter_only": True,
        "formal_robust_cbf_qp_claim": False,
        "rows": rows,
    }


def checkpoint_payload(actor: RecurrentResidualActor, settings: dict[str, Any], action_scale: float, seed: int) -> dict[str, Any]:
    return {
        "state_dict": actor.state_dict(),
        "local_observation_dim": actor.local_observation_dim,
        "action_dim": actor.action_dim,
        "recurrent_hidden_dim": actor.hidden_dim,
        "residual_scale": actor.residual_scale,
        "action_scale": float(action_scale),
        "seed": int(seed),
        "algorithm": "dnmpc_teacher_dagger_recurrent_residual_actor",
        "actor_recurrent": True,
        "target_contract": "phase78_obstacle_avoidance_first",
        "local_cbf_is_empirical_filter_only": True,
        "formal_robust_cbf_qp_claim": False,
        "settings": settings,
    }


def main() -> None:
    args = parse_args()
    document, environment, settings = load_document(args.config)
    seed = int(args.seed if args.seed is not None else settings.get("seed", 791501))
    settings["seed"] = seed
    settings["training_scene_file"] = document["training_scene_file"]
    settings["evaluation_scene_file"] = document["evaluation_scene_file"]
    settings["mpc_config"] = document["mpc_config"]
    settings["bootstrap_expert_datasets"] = document.get("bootstrap_expert_datasets", [])
    device = select_device(args.device)
    torch_threads = args.torch_threads
    if torch_threads is None and settings.get("torch_threads") is not None:
        torch_threads = int(settings["torch_threads"])
    if torch_threads is not None:
        if torch_threads <= 0:
            raise ValueError("torch-threads must be positive")
        torch.set_num_threads(torch_threads)
        torch.set_num_interop_threads(torch_threads)
        settings["torch_threads"] = int(torch_threads)
    if args.mode == "evaluate":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required in evaluate mode.")
        evaluation_path = resolve_config_path(settings["evaluation_scene_file"], args.config)
        records = load_records(evaluation_path, args.evaluation_episodes)
        checkpoint = torch.load(args.checkpoint.resolve(), map_location=device, weights_only=True)
        actor = RecurrentResidualActor(
            int(checkpoint["local_observation_dim"]),
            action_dim=int(checkpoint.get("action_dim", 3)),
            hidden_dim=int(checkpoint["recurrent_hidden_dim"]),
            residual_scale=float(checkpoint["residual_scale"]),
        ).to(device)
        actor.load_state_dict(checkpoint["state_dict"], strict=True)
        args.output.mkdir(parents=True, exist_ok=True)
        result = evaluate_policy(
            environment,
            records,
            planner_config(resolve_config_path(settings["mpc_config"], args.config)),
            settings,
            actor,
            device,
            args.max_steps,
            progress_path=args.output / "evaluation_progress.json",
            progress_interval=int(settings.get("evaluation_progress_interval", 10)),
        )
        args.output.joinpath("evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    output = args.output
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    demo_episodes = int(args.demo_episodes if args.demo_episodes is not None else settings.get("demo_episodes", 192))
    dagger_rounds = int(args.dagger_rounds if args.dagger_rounds is not None else settings.get("dagger_rounds", 2))
    episodes_per_round = int(args.episodes_per_round if args.episodes_per_round is not None else settings.get("episodes_per_round", 24))
    if (demo_episodes < 192 and not args.allow_small) or dagger_rounds < 1 or episodes_per_round <= 0:
        raise ValueError("Phase79 requires at least 192 demonstrations and at least one DAgger round.")
    scene_path = resolve_config_path(settings["training_scene_file"], args.config)
    records = load_records(scene_path, None)
    planner = planner_config(resolve_config_path(settings["mpc_config"], args.config))
    trajectories: list[Trajectory] = []
    metadata: dict[str, Any] = {
        "experiment_name": str(document.get("experiment_name", "phase79_dagger_residual")),
        "target_contract": "phase78_obstacle_avoidance_first",
        "locked_test_used": False,
        "initial_demo_target": demo_episodes,
        "dagger_rounds": dagger_rounds,
        "episodes_per_round": episodes_per_round,
        "source_scene_file": str(scene_path.resolve()),
        "source_scene_sha256": hashlib.sha256(scene_path.resolve().read_bytes()).hexdigest(),
        "rounds": [],
    }
    probe_env, probe_observation, _probe_scenario = build_environment(environment, records[0], max_steps=args.max_steps)
    local_dim = int(policy_observations(probe_env, probe_observation).shape[-1] + 7)
    action_scale = float(probe_env.agents["defender_max_speed"])
    actor = RecurrentResidualActor(
        local_dim,
        hidden_dim=int(settings.get("hidden_dim", 128)),
        residual_scale=float(settings.get("residual_scale_mps", 2.5)),
    ).to(device)
    bootstrap_paths = [resolve_config_path(value, args.config) for value in settings["bootstrap_expert_datasets"]]
    if bootstrap_paths:
        trajectories, bootstrap_count = load_bootstrap_trajectories(bootstrap_paths)
        if bootstrap_count < demo_episodes:
            raise ValueError(
                f"Bootstrap pool has {bootstrap_count} accepted demonstrations, below requested {demo_episodes}."
            )
        metadata["initial_demo_source"] = "retained_phase78_accepted_expert_datasets"
        metadata["initial_demo_episodes"] = int(bootstrap_count)
        metadata["bootstrap_dataset_paths"] = [str(path) for path in bootstrap_paths]
        metadata["rounds"].append(
            {"name": "initial_retained_phase78_demonstrations", "episodes": int(bootstrap_count)}
        )
        print(json.dumps({"phase": "initial_demos", "source": "retained", "completed": bootstrap_count}), flush=True)
    else:
        quality_gate = bool(settings.get("quality_gate_initial_demos", False))
        max_attempts = int(
            settings.get("max_initial_demo_attempts", max(demo_episodes, demo_episodes * 3))
        )
        if max_attempts < demo_episodes:
            raise ValueError("max_initial_demo_attempts must cover demo_episodes")
        attempts = 0
        rejected = 0
        rejection_reasons: dict[str, int] = {}
        while len(trajectories) < demo_episodes:
            if attempts >= max_attempts:
                raise RuntimeError(
                    "teacher quality gate could not collect the requested demonstrations: "
                    f"accepted={len(trajectories)}, requested={demo_episodes}, attempts={attempts}, "
                    f"max_attempts={max_attempts}"
                )
            record = records[attempts % len(records)]
            trajectory, episode_meta = rollout(
                environment,
                record,
                planner,
                settings,
                actor=None,
                device=device,
                max_steps=args.max_steps,
                use_actor=False,
                episode_seed=seed + attempts,
            )
            attempts += 1
            accepted = teacher_demo_is_accepted(episode_meta)
            if accepted or not quality_gate:
                trajectories.append(trajectory)
            else:
                rejected += 1
                reason = str(episode_meta.get("termination_reason", "unknown"))
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            if attempts % 24 == 0 or len(trajectories) == demo_episodes:
                print(
                    json.dumps(
                        {
                            "phase": "initial_demos",
                            "accepted": len(trajectories),
                            "requested": demo_episodes,
                            "attempts": attempts,
                            "rejected": rejected,
                            "quality_gate": quality_gate,
                            "last": episode_meta,
                        }
                    ),
                    flush=True,
                )
        metadata["initial_demo_attempts"] = int(attempts)
        metadata["initial_demo_rejected"] = int(rejected)
        metadata["initial_demo_rejection_reasons"] = dict(sorted(rejection_reasons.items()))
        metadata["initial_demo_quality_gate"] = quality_gate
        metadata["initial_demo_source"] = "new_dnmpc_teacher_rollouts"
        metadata["initial_demo_episodes"] = demo_episodes
        metadata["rounds"].append(
            {
                "name": "initial_dnmpc_teacher",
                "episodes": demo_episodes,
                "attempts": int(attempts),
                "rejected": int(rejected),
                "quality_gate": quality_gate,
            }
        )
    training_history: list[dict[str, Any]] = []
    for round_index in range(dagger_rounds):
        history = train_actor(actor, trajectories, settings, device, action_scale)
        training_history.append({"round": round_index, "dataset_episodes": len(trajectories), "history": history})
        round_rows: list[dict[str, Any]] = []
        for index in range(episodes_per_round):
            record = records[(demo_episodes + round_index * episodes_per_round + index) % len(records)]
            trajectory, episode_meta = rollout(
                environment,
                record,
                planner,
                settings,
                actor=actor,
                device=device,
                max_steps=args.max_steps,
                use_actor=True,
                episode_seed=seed + demo_episodes + round_index * episodes_per_round + index,
            )
            trajectories.append(trajectory)
            round_rows.append(episode_meta)
        metadata["rounds"].append(
            {
                "name": f"dagger_recovery_round_{round_index + 1}",
                "episodes": episodes_per_round,
                "recovery_steps": int(sum(int(row["recovery_steps"]) for row in round_rows)),
                "termination_reason_counts": {
                    reason: sum(row["termination_reason"] == reason for row in round_rows)
                    for reason in sorted({row["termination_reason"] for row in round_rows})
                },
            }
        )
        print(json.dumps({"phase": "dagger_round", "round": round_index + 1, "completed": episodes_per_round, "summary": metadata["rounds"][-1]}), flush=True)
    final_history = train_actor(actor, trajectories, settings, device, action_scale)
    # Bootstrap archives are stored as padded 32-step chunks, so
    # ``len(trajectories)`` is a chunk count rather than a conceptual episode
    # count.  Keep the manifest episode statistic interpretable and reproducible.
    metadata["total_trajectory_episodes"] = int(
        metadata["initial_demo_episodes"]
        + sum(
            int(item["episodes"])
            for item in metadata["rounds"]
            if str(item.get("name", "")).startswith("dagger_")
        )
    )
    metadata["total_frames"] = int(sum(trajectory.local.shape[0] for trajectory in trajectories))
    metadata["total_recovery_frames"] = int(sum(np.sum(trajectory.recovery) for trajectory in trajectories))
    output.joinpath("manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output.joinpath("training.json").write_text(json.dumps(training_history + [{"round": "final", "history": final_history}], indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "dagger_recovery_dataset.npz",
        **{
            "local_observations": np.concatenate([item.local for item in trajectories], axis=0),
            "base_actions": np.concatenate([item.base for item in trajectories], axis=0),
            "teacher_actions": np.concatenate([item.target for item in trajectories], axis=0),
            "recovery_masks": np.concatenate([item.recovery for item in trajectories], axis=0).astype(np.float32),
        },
    )
    torch.save(checkpoint_payload(actor, settings, action_scale, seed), output / "checkpoint.pt")
    eval_records = load_records(resolve_config_path(settings["evaluation_scene_file"], args.config), args.evaluation_episodes)
    result = evaluate_policy(
        environment,
        eval_records,
        planner,
        settings,
        actor,
        device,
        args.max_steps,
        progress_path=output / "evaluation_progress.json",
        progress_interval=int(settings.get("evaluation_progress_interval", 10)),
    )
    result["checkpoint_sha256"] = hashlib.sha256((output / "checkpoint.pt").read_bytes()).hexdigest()
    output.joinpath("evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    output.joinpath("config.yaml").write_text(yaml.safe_dump({"document": document, "settings": settings}, sort_keys=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
