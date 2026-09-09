"""Evaluate centralized scenario risk-sensitive MPC on fixed episode seeds.

The planner receives only policy-safe observations and either a deterministic
belief baseline or a frozen prediction checkpoint. Target truth is used only
inside the environment for episode termination metrics.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if os.name == "nt":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import yaml

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - depends on environment extras
    SummaryWriter = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def require_summary_writer() -> Any:
    if SummaryWriter is None:
        raise RuntimeError("TensorBoard logging requires the optional 'tensorboard' package.")
    return SummaryWriter

from encirclement3d.minimax_mpc import (  # noqa: E402
    MinimaxMPCConfig,
    MinimaxMPCDiagnostics,
    ScenarioMinimaxMPC,
    ScenarioTrajectorySet,
    aggregate_scenario_costs,
    evaluate_candidate_capture_distances,
    make_belief_candidate_set,
)
from encirclement3d.distributed_dn_mpc import (  # noqa: E402
    DistributedDNMPCConfig,
    DistributedMinimaxDNMPC,
)
from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.prediction import (  # noqa: E402
    ConditionalDiffusionTrajectoryPredictor,
    HistoryTargetPredictor,
    TrajectoryNormalizer,
    project_candidate_trajectories,
)
from encirclement3d.pursuit_controllers import (  # noqa: E402
    DynamicEncirclementController,
    PursuitCBFSafetyFilter,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.safety_certificate import check_one_step_safety  # noqa: E402
from encirclement3d.safety_qp import RobustCBFQPConfig, RobustCBFQPFilter  # noqa: E402
from encirclement3d.showcase import prepare_showcase_episode  # noqa: E402


DEFAULT_ENVIRONMENT_CONFIG = PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml"
DEFAULT_MPC_CONFIG = PROJECT_ROOT / "configs" / "innovation_mpc.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_ENVIRONMENT_CONFIG)
    parser.add_argument("--mpc-config", type=Path, default=DEFAULT_MPC_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--candidate-source", choices=("checkpoint", "belief"), default="checkpoint")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "dynamic_encirclement",
            "expected",
            "worst_case",
            "cvar",
            "distributed_ideal",
            "distributed_delayed",
            "distributed_dropout",
            "distributed_none",
        ),
    )
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--obstacle-count", type=int)
    parser.add_argument("--target-speed-scale", type=float)
    parser.add_argument("--target-motion-mode", choices=("flee_persistence", "random_turn", "s_curve", "burst", "boundary_escape"))
    parser.add_argument("--max-steps", type=int, help="Optional smoke-run horizon override.")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--projection-iterations", type=int)
    parser.add_argument(
        "--prediction-refresh-interval-steps",
        type=int,
        help="Refresh learned prediction every N control steps; 1 preserves per-step sampling.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--safety-layer",
        choices=("local_cbf", "robust_cbf_qp"),
        default="local_cbf",
        help="Safety filter used after planning. robust_cbf_qp is velocity-level and conditional.",
    )
    parser.add_argument(
        "--safety-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "innovation_safety.yaml",
        help="Safety configuration used by --safety-layer robust_cbf_qp.",
    )
    parser.add_argument("--without-local-cbf", action="store_true")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML document must be a mapping: {path}")
    return value


def source_hashes(mpc_config_path: Path | None = None) -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "scripts" / "evaluate_minimax_mpc.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "minimax_mpc.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "distributed_dn_mpc.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "prediction.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_controllers.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_env.py",
        PROJECT_ROOT / "configs" / "innovation_mpc.yaml",
    )
    hashes = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    if mpc_config_path is not None:
        path = mpc_config_path.resolve()
        hashes[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def add_safety_source_hashes(hashes: dict[str, str], safety_config_path: Path) -> None:
    paths = (
        PROJECT_ROOT / "src" / "encirclement3d" / "safety_qp.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "safety_certificate.py",
        safety_config_path.resolve(),
    )
    for path in paths:
        hashes[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()


def weighted_belief_velocity(observation: dict[str, Any]) -> np.ndarray:
    velocities = np.asarray(observation["target_belief_velocities"], dtype=np.float64)
    confidences = np.asarray(observation.get("target_observation_confidence", np.ones(len(velocities))), dtype=np.float64)
    ages = np.asarray(observation.get("message_age_steps", np.zeros(len(velocities))), dtype=np.float64)
    weights = np.maximum(confidences, 1e-3) / (1.0 + np.maximum(ages, 0.0))
    weights /= max(float(weights.sum()), 1e-12)
    return np.sum(velocities * weights[:, None], axis=0)


def model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, str, TrajectoryNormalizer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path.resolve(), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Prediction checkpoint must contain a mapping.")
    model_kind = str(checkpoint.get("model_kind", ""))
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Prediction checkpoint is missing model_config.")
    if model_kind == "gru":
        model: torch.nn.Module = HistoryTargetPredictor(**model_config)
    elif model_kind == "diffusion":
        model = ConditionalDiffusionTrajectoryPredictor(**model_config)
    else:
        raise ValueError(f"Unsupported prediction model kind: {model_kind}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    normalizer_config = checkpoint.get("target_normalizer")
    if not isinstance(normalizer_config, dict):
        raise ValueError("Prediction checkpoint is missing target_normalizer.")
    normalizer = TrajectoryNormalizer(
        center=np.asarray(normalizer_config["center"], dtype=np.float32),
        scale=np.asarray(normalizer_config["scale"], dtype=np.float32),
        kind=str(normalizer_config.get("kind", "checkpoint")),
    )
    return model, model_kind, normalizer, checkpoint


@dataclass
class PredictionRuntime:
    """Stateful frozen predictor adapter for one environment episode."""

    env: CaptureRadiusPursuit3DEnv
    source: str
    device: torch.device
    model: torch.nn.Module | None = None
    model_kind: str | None = None
    normalizer: TrajectoryNormalizer | None = None
    num_samples: int = 8
    sampling_steps: int = 8
    sampling_seed: int = 745102
    projection_iterations: int = 4
    refresh_interval_steps: int = 1
    history_length: int = 16
    history: list[np.ndarray] | None = None
    step_index: int = 0
    cached_scenarios: ScenarioTrajectorySet | None = None
    cached_age_steps: int = 0
    refresh_count: int = 0

    def reset(self) -> None:
        self.history = []
        self.step_index = 0
        self.cached_scenarios = None
        self.cached_age_steps = 0
        self.refresh_count = 0

    def predict(
        self,
        observation: dict[str, Any],
        planner_horizon: int,
    ) -> tuple[ScenarioTrajectorySet, float, bool, int]:
        started = time.perf_counter()
        if self.refresh_interval_steps <= 0:
            raise ValueError("refresh_interval_steps must be positive.")
        refresh = self.cached_scenarios is None or self.step_index % self.refresh_interval_steps == 0
        if not refresh:
            assert self.history is not None
            frame = policy_observations(self.env, observation).astype(np.float32)
            self.history.append(frame.copy())
            if len(self.history) > self.history_length:
                self.history.pop(0)
            self.step_index += 1
            self.cached_age_steps += 1
            return self.cached_scenarios, 0.0, False, self.cached_age_steps
        if self.source == "belief":
            result = make_belief_candidate_set(
                observation,
                horizon_steps=planner_horizon,
                dt_seconds=self.env.dt,
                max_speed_mps=float(self.env.agents["target_max_speed"]),
                candidate_count=self.num_samples,
            )
            self.cached_scenarios = result
            self.cached_age_steps = 0
            self.refresh_count += 1
            self.step_index += 1
            return result, (time.perf_counter() - started) * 1000.0, True, 0
        if self.model is None or self.normalizer is None or self.model_kind is None:
            raise RuntimeError("checkpoint prediction runtime is not initialized")
        frame = policy_observations(self.env, observation).astype(np.float32)
        if self.history is None:
            self.reset()
        assert self.history is not None
        self.history.append(frame.copy())
        if len(self.history) > self.history_length:
            self.history.pop(0)
        padded = [self.history[0]] * (self.history_length - len(self.history)) + self.history
        window = np.stack(padded, axis=0).reshape(self.history_length, -1)
        inputs = torch.as_tensor(window[None], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.model_kind == "gru":
                mean, _log_variance = self.model(inputs)
                raw_displacements = self.normalizer.denormalize(mean[:, None])
            else:
                generator = torch.Generator(device=self.device.type).manual_seed(
                    int(self.sampling_seed + self.step_index)
                )
                candidate_set = self.model.sample_set(
                    inputs,
                    num_samples=self.num_samples,
                    sampling_steps=self.sampling_steps,
                    generator=generator,
                )
                raw_displacements = self.normalizer.denormalize(candidate_set.trajectories)
        reference = self._belief_reference(observation)
        reference_velocity = weighted_belief_velocity(observation)
        projected_displacements = project_candidate_trajectories(
            raw_displacements,
            torch.as_tensor(reference[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(reference_velocity[None], dtype=torch.float32, device=self.device),
            float(self.env.dt),
            float(self.env.agents["target_max_speed"]),
            float(self.env.agents["target_max_acceleration"]),
            torch.as_tensor(self.env.lower[None], dtype=torch.float32, device=self.device),
            torch.as_tensor(self.env.upper[None], dtype=torch.float32, device=self.device),
            self.projection_iterations,
        )
        trajectories = projected_displacements[0].detach().cpu().numpy() + reference[None, None, :]
        result = ScenarioTrajectorySet(
            trajectories=trajectories,
            weights=np.ones(trajectories.shape[0], dtype=np.float64),
            score_kind="uniform_uncalibrated",
            dynamics_status="projected",
        )
        self.cached_scenarios = result
        self.cached_age_steps = 0
        self.refresh_count += 1
        self.step_index += 1
        return result, (time.perf_counter() - started) * 1000.0, True, 0

    @staticmethod
    def _belief_reference(observation: dict[str, Any]) -> np.ndarray:
        beliefs = np.asarray(observation["target_belief_positions"], dtype=np.float64)
        confidences = np.asarray(observation.get("target_observation_confidence", np.ones(len(beliefs))), dtype=np.float64)
        ages = np.asarray(observation.get("message_age_steps", np.zeros(len(beliefs))), dtype=np.float64)
        weights = np.maximum(confidences, 1e-3) / (1.0 + np.maximum(ages, 0.0))
        weights /= max(float(weights.sum()), 1e-12)
        return np.sum(beliefs * weights[:, None], axis=0)


def _default_diagnostics(status: str = "not_used") -> MinimaxMPCDiagnostics:
    return MinimaxMPCDiagnostics(
        status=status,
        risk_mode="none",
        selected_sequence_index=-1,
        candidate_sequence_count=0,
        scenario_costs=tuple(),
        objective_value=float("nan"),
        expected_cost=float("nan"),
        worst_case_cost=float("nan"),
        cvar_cost=float("nan"),
        latency_ms=0.0,
        max_rollout_constraint_violation=0.0,
    )


def _planning_observation(env: CaptureRadiusPursuit3DEnv, observation: dict[str, Any]) -> dict[str, Any]:
    value = dict(observation)
    # Bounds are public environment geometry, not target truth. Keeping them
    # attached to the planner observation makes the rollout penalty explicit.
    value["world_lower_bounds"] = env.lower.copy()
    value["world_upper_bounds"] = env.upper.copy()
    return value


def _shift_warm_start_sequence(sequence: np.ndarray) -> np.ndarray:
    """Advance a prior receding-horizon sequence by one executed control step."""

    value = np.asarray(sequence, dtype=np.float64)
    if value.ndim != 3 or value.shape[0] < 1:
        raise ValueError("warm-start sequence must have shape [horizon, defenders, 3]")
    if value.shape[0] == 1:
        return value.copy()
    return np.concatenate([value[1:], value[-1:]], axis=0)


def run_episode(
    config: dict[str, Any],
    *,
    seed: int,
    method: str,
    planner_config: MinimaxMPCConfig,
    candidate_source: str,
    checkpoint_data: tuple[torch.nn.Module, str, TrajectoryNormalizer, dict[str, Any]] | None,
    device: torch.device,
    num_samples: int,
    sampling_steps: int,
    sampling_seed: int,
    projection_iterations: int,
    use_local_cbf: bool,
    safety_layer: str | None = None,
    robust_safety_config: RobustCBFQPConfig | None = None,
    prediction_refresh_interval_steps: int = 1,
    distributed_config: DistributedDNMPCConfig | None = None,
    scenario: Any | None = None,
    validate_scenario: bool = True,
    record_history: bool = False,
    trajectory_path: Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=int(config["experiments"][0]["obstacle_count"]),
        target_speed_scale=float(config["experiments"][0]["target_speed_scale"]),
    )
    if scenario is None:
        observation = env.reset(seed=seed, record_history=record_history)
    else:
        observation = prepare_showcase_episode(
            env,
            scenario,
            seed=seed,
            record_history=record_history,
            validate_scenario=validate_scenario,
        )
    fallback_controller = DynamicEncirclementController(env)
    resolved_safety_layer = safety_layer or ("local_cbf" if use_local_cbf else "none")
    if resolved_safety_layer not in {"none", "local_cbf", "robust_cbf_qp"}:
        raise ValueError(f"Unsupported safety layer: {resolved_safety_layer}")
    safety_filter = PursuitCBFSafetyFilter(env) if resolved_safety_layer == "local_cbf" else None
    robust_safety_filter = (
        RobustCBFQPFilter(env, robust_safety_config)
        if resolved_safety_layer == "robust_cbf_qp"
        else None
    )
    if resolved_safety_layer == "robust_cbf_qp" and robust_safety_config is None:
        raise ValueError("robust_safety_config is required for the robust_cbf_qp safety layer")
    planner = ScenarioMinimaxMPC(planner_config)
    distributed_planner: DistributedMinimaxDNMPC | None = None
    previous_distributed_sequence: np.ndarray | None = None
    distributed_methods = {
        "distributed_ideal": "ideal",
        "distributed_delayed": "delayed",
        "distributed_dropout": "dropout",
        "distributed_none": "none",
    }
    if method in distributed_methods:
        if distributed_config is None:
            distributed_config = DistributedDNMPCConfig(
                communication_mode=distributed_methods[method],
            )
        elif distributed_config.communication_mode != distributed_methods[method]:
            raise ValueError(
                f"distributed config mode {distributed_config.communication_mode!r} "
                f"does not match method {method!r}"
            )
        distributed_planner = DistributedMinimaxDNMPC(
            MinimaxMPCConfig(**{**planner_config.__dict__, "risk_mode": "worst_case"}),
            distributed_config,
        )
    runtime = None
    if method != "dynamic_encirclement":
        if candidate_source == "checkpoint" and checkpoint_data is None:
            raise ValueError("--checkpoint is required for checkpoint candidate source.")
        runtime = PredictionRuntime(
            env=env,
            source=candidate_source,
            device=device,
            model=None if checkpoint_data is None else checkpoint_data[0],
            model_kind=None if checkpoint_data is None else checkpoint_data[1],
            normalizer=None if checkpoint_data is None else checkpoint_data[2],
            num_samples=num_samples,
            sampling_steps=sampling_steps,
            sampling_seed=sampling_seed,
            projection_iterations=projection_iterations,
            refresh_interval_steps=prediction_refresh_interval_steps,
            history_length=16,
        )
        runtime.reset()

    path_length = np.zeros(env.n_defenders, dtype=np.float64)
    previous_positions = env.defender_positions.copy()
    step_rows: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    while True:
        control_started = time.perf_counter()
        fallback_actions = fallback_controller.act(observation)
        if method == "dynamic_encirclement":
            nominal_actions = fallback_actions
            planner_diagnostics = _default_diagnostics()
            predictor_latency_ms = 0.0
            candidate_distance_metrics = {
                "terminal_distances_m": np.empty(0, dtype=np.float64),
                "minimum_distances_m": np.empty(0, dtype=np.float64),
            }
            candidate_weights = np.empty(0, dtype=np.float64)
        else:
            assert runtime is not None
            scenarios, predictor_latency_ms, prediction_refreshed, prediction_age_steps = runtime.predict(
                observation,
                planner_config.horizon_steps,
            )
            planner_config_for_method = MinimaxMPCConfig(
                **{
                    **planner_config.__dict__,
                    "risk_mode": method if method in {"expected", "worst_case", "cvar"} else "worst_case",
                }
            )
            planning_observation = _planning_observation(env, observation)
            if distributed_planner is not None:
                plan = distributed_planner.plan(
                    planning_observation,
                    scenarios,
                    step_index=env.step_count,
                    previous_action_sequence=previous_distributed_sequence,
                    fallback_actions=fallback_actions,
                )
                previous_distributed_sequence = _shift_warm_start_sequence(plan.action_sequence)
            else:
                planner = ScenarioMinimaxMPC(planner_config_for_method)
                plan = planner.plan(
                    planning_observation,
                    scenarios,
                    fallback_actions=fallback_actions,
                )
            nominal_actions = plan.actions
            planner_diagnostics = plan.diagnostics
            candidate_distance_metrics = evaluate_candidate_capture_distances(
                planning_observation,
                plan.action_sequence,
                scenarios.truncate(planner_config.horizon_steps),
                dt_seconds=planner_config.dt_seconds,
                max_speed_mps=planner_config.max_speed_mps,
            )
            candidate_weights = scenarios.normalized_weights
        candidate_minimum_distances = candidate_distance_metrics["minimum_distances_m"]
        candidate_terminal_distances = candidate_distance_metrics["terminal_distances_m"]
        if candidate_minimum_distances.size:
            candidate_expected_minimum = aggregate_scenario_costs(
                candidate_minimum_distances,
                candidate_weights,
                "expected",
                planner_config.cvar_alpha,
            )
            candidate_worst_minimum = aggregate_scenario_costs(
                candidate_minimum_distances,
                candidate_weights,
                "worst_case",
                planner_config.cvar_alpha,
            )
            candidate_cvar_minimum = aggregate_scenario_costs(
                candidate_minimum_distances,
                candidate_weights,
                "cvar",
                planner_config.cvar_alpha,
            )
            candidate_expected_terminal = aggregate_scenario_costs(
                candidate_terminal_distances,
                candidate_weights,
                "expected",
                planner_config.cvar_alpha,
            )
            candidate_worst_terminal = aggregate_scenario_costs(
                candidate_terminal_distances,
                candidate_weights,
                "worst_case",
                planner_config.cvar_alpha,
            )
        else:
            candidate_expected_minimum = float("nan")
            candidate_worst_minimum = float("nan")
            candidate_cvar_minimum = float("nan")
            candidate_expected_terminal = float("nan")
            candidate_worst_terminal = float("nan")
        if safety_filter is None:
            if robust_safety_filter is None:
                safe_actions = np.asarray(nominal_actions, dtype=np.float64)
                cbf_correction = 0.0
                safety_latency_ms = 0.0
                safety_diagnostics: Any = None
            else:
                safety_started = time.perf_counter()
                safe_actions, safety_diagnostics = robust_safety_filter.filter(nominal_actions, observation)
                safety_latency_ms = (time.perf_counter() - safety_started) * 1000.0
                cbf_correction = float(safety_diagnostics.action_correction_norm)
        else:
            safety_started = time.perf_counter()
            safe_actions, cbf_diagnostics = safety_filter.filter(nominal_actions, observation)
            cbf_correction = float(cbf_diagnostics.action_correction_norm)
            safety_latency_ms = (time.perf_counter() - safety_started) * 1000.0
            safety_diagnostics = cbf_diagnostics
        certificate_observation = dict(observation)
        certificate_observation.setdefault("world_lower_bounds", np.asarray(env.lower, dtype=np.float64))
        certificate_observation.setdefault("world_upper_bounds", np.asarray(env.upper, dtype=np.float64))
        certificate_observation.setdefault("obstacles", list(getattr(env, "obstacles", ())))
        independent_certificate = check_one_step_safety(
            certificate_observation,
            np.asarray(safe_actions, dtype=np.float64),
            dt=float(env.dt),
            drone_radius=float(env.agents["drone_radius"]),
            max_speed_mps=float(env.agents["defender_max_speed"]),
            max_acceleration_mps2=float(env.agents["defender_max_acceleration"]),
            safety_margin_m=float(env.pursuit["safety_margin"]),
            robust_margin_m=(
                float(robust_safety_config.robust_margin_m)
                if resolved_safety_layer == "robust_cbf_qp" and robust_safety_config is not None
                else 0.0
            ),
            action_change_limit_mps=(
                None if robust_safety_config is None else robust_safety_config.action_change_limit_mps
            ),
            enforce_action_change=resolved_safety_layer == "robust_cbf_qp",
        )
        safe_actions = env._clip_rows(safe_actions, float(env.agents["defender_max_speed"]))
        total_control_latency_ms = (time.perf_counter() - control_started) * 1000.0
        observation, _reward, terminated, truncated, final_info = env.step(
            safe_actions,
            record_history=record_history,
        )
        path_length += np.linalg.norm(env.defender_positions - previous_positions, axis=1)
        previous_positions = env.defender_positions.copy()
        planner_status = str(planner_diagnostics.status)
        step_rows.append(
            {
                "step": float(env.step_count),
                "predictor_latency_ms": float(predictor_latency_ms),
                "prediction_refreshed": 1.0 if method != "dynamic_encirclement" and prediction_refreshed else 0.0,
                "prediction_age_steps": float(prediction_age_steps if method != "dynamic_encirclement" else 0.0),
                "planner_latency_ms": float(planner_diagnostics.latency_ms),
                "planner_status": 1.0 if planner_status == "success" else 0.0,
                "planner_fallback": 1.0 if planner_status in {"fallback", "partial_fallback"} else 0.0,
                "planner_valid": 1.0 if planner_status in {"success", "not_converged", "partial_fallback"} else 0.0,
                "planner_converged": 1.0 if bool(getattr(planner_diagnostics, "converged", False)) else 0.0,
                "planner_local_solver_failures": float(getattr(planner_diagnostics, "local_solver_failures", 0)),
                "messages_attempted": float(getattr(planner_diagnostics, "messages_attempted", 0)),
                "messages_sent": float(getattr(planner_diagnostics, "messages_sent", 0)),
                "messages_received": float(getattr(planner_diagnostics, "messages_received", 0)),
                "messages_dropped": float(getattr(planner_diagnostics, "messages_dropped", 0)),
                "message_bytes_sent": float(getattr(planner_diagnostics, "message_bytes_sent", 0)),
                "message_bytes_received": float(getattr(planner_diagnostics, "message_bytes_received", 0)),
                "max_message_age_steps": float(getattr(planner_diagnostics, "max_message_age_steps", 0)),
                "mean_message_age_steps": float(getattr(planner_diagnostics, "mean_message_age_steps", 0.0)),
                "distributed_iterations": float(getattr(planner_diagnostics, "iterations", 0)),
                "distributed_action_delta_mps": float(getattr(planner_diagnostics, "max_action_delta_mps", 0.0)),
                "objective_value": float(planner_diagnostics.objective_value),
                "worst_case_cost": float(planner_diagnostics.worst_case_cost),
                "cbf_action_correction_norm": float(cbf_correction),
                "safety_latency_ms": float(safety_latency_ms),
                "safety_layer": resolved_safety_layer,
                "safety_independent_certificate_valid": bool(independent_certificate.valid),
                "safety_independent_certificate_status": str(independent_certificate.status),
                "safety_independent_current_state_safe": bool(independent_certificate.current_state_safe),
                "safety_independent_next_state_safe": bool(independent_certificate.next_state_safe),
                "safety_independent_current_min_barrier_m": float(independent_certificate.current_min_barrier_m),
                "safety_independent_next_min_barrier_m": float(independent_certificate.next_min_barrier_m),
                "safety_independent_max_action_change_mps": float(
                    independent_certificate.maximum_action_change_mps
                ),
                "safety_independent_violations": list(independent_certificate.violations),
                "safety_status": (
                    None if safety_diagnostics is None else str(getattr(safety_diagnostics, "status", "unknown"))
                ),
                "safety_solver_message": (
                    None if safety_diagnostics is None else str(getattr(safety_diagnostics, "solver_message", ""))
                ),
                "safety_failure_category": (
                    None
                    if safety_diagnostics is None
                    else str(getattr(safety_diagnostics, "failure_category", "none"))
                ),
                "safety_fallback_reason": (
                    None
                    if safety_diagnostics is None or getattr(safety_diagnostics, "fallback_reason", None) is None
                    else str(getattr(safety_diagnostics, "fallback_reason"))
                ),
                "safety_precondition_valid": (
                    None
                    if safety_diagnostics is None or not hasattr(safety_diagnostics, "precondition_valid")
                    else bool(getattr(safety_diagnostics, "precondition_valid"))
                ),
                "safety_recovery_action_used": (
                    None
                    if safety_diagnostics is None or not hasattr(safety_diagnostics, "recovery_action_used")
                    else bool(getattr(safety_diagnostics, "recovery_action_used"))
                ),
                "safety_solver_success": (
                    None
                    if safety_diagnostics is None or not hasattr(safety_diagnostics, "solver_success")
                    else bool(getattr(safety_diagnostics, "solver_success"))
                ),
                "safety_certificate_valid": (
                    None
                    if safety_diagnostics is None or not hasattr(safety_diagnostics, "certificate_valid")
                    else bool(getattr(safety_diagnostics, "certificate_valid"))
                ),
                "safety_fallback_used": (
                    None
                    if safety_diagnostics is None or not hasattr(safety_diagnostics, "fallback_used")
                    else bool(getattr(safety_diagnostics, "fallback_used"))
                ),
                "safety_abort_required": (
                    None
                    if safety_diagnostics is None or not hasattr(safety_diagnostics, "abort_required")
                    else bool(getattr(safety_diagnostics, "abort_required"))
                ),
                "safety_minimum_barrier_m": (
                    None
                    if safety_diagnostics is None or not hasattr(safety_diagnostics, "minimum_barrier_value_m")
                    else float(getattr(safety_diagnostics, "minimum_barrier_value_m"))
                ),
                "safety_maximum_constraint_violation_m": (
                    None
                    if safety_diagnostics is None or not hasattr(safety_diagnostics, "maximum_constraint_violation")
                    else float(getattr(safety_diagnostics, "maximum_constraint_violation"))
                ),
                "safety_maximum_slack_m": (
                    None
                    if safety_diagnostics is None or not hasattr(safety_diagnostics, "maximum_safety_slack_m")
                    else float(getattr(safety_diagnostics, "maximum_safety_slack_m"))
                ),
                "safety_active_constraint_count": (
                    None
                    if safety_diagnostics is None or not hasattr(safety_diagnostics, "active_constraint_count")
                    else int(getattr(safety_diagnostics, "active_constraint_count"))
                ),
                "safety_constraint_count": (
                    None
                    if safety_diagnostics is None or not hasattr(safety_diagnostics, "constraint_count")
                    else int(getattr(safety_diagnostics, "constraint_count"))
                ),
                "safety_recoverability_status": (
                    None
                    if safety_diagnostics is None or not hasattr(safety_diagnostics, "recoverability_status")
                    else str(getattr(safety_diagnostics, "recoverability_status"))
                ),
                "safety_prefix_admissible": (
                    None
                    if safety_diagnostics is None or not hasattr(safety_diagnostics, "prefix_admissible")
                    else bool(getattr(safety_diagnostics, "prefix_admissible"))
                ),
                "safety_immutable_prefix_horizon_steps": (
                    None
                    if safety_diagnostics is None
                    or not hasattr(safety_diagnostics, "immutable_prefix_horizon_steps")
                    else int(getattr(safety_diagnostics, "immutable_prefix_horizon_steps"))
                ),
                "safety_immutable_prefix_min_robust_barrier_m": (
                    None
                    if safety_diagnostics is None
                    or not hasattr(safety_diagnostics, "immutable_prefix_min_robust_barrier_m")
                    else float(getattr(safety_diagnostics, "immutable_prefix_min_robust_barrier_m"))
                ),
                "total_control_latency_ms": float(total_control_latency_ms),
                "nearest_target_distance": float(final_info["nearest_target_distance"]),
                "candidate_count": float(candidate_minimum_distances.size),
                "candidate_expected_minimum_distance_m": float(candidate_expected_minimum),
                "candidate_worst_minimum_distance_m": float(candidate_worst_minimum),
                "candidate_cvar_minimum_distance_m": float(candidate_cvar_minimum),
                "candidate_expected_terminal_distance_m": float(candidate_expected_terminal),
                "candidate_worst_terminal_distance_m": float(candidate_worst_terminal),
                "candidate_minimum_distances_m": candidate_minimum_distances.tolist(),
                "candidate_terminal_distances_m": candidate_terminal_distances.tolist(),
                "candidate_scenario_costs": list(planner_diagnostics.scenario_costs),
            }
        )
        if terminated or truncated:
            break
    summary = {
        "method": method,
        "seed": int(seed),
        "safe_capture_success": bool(final_info.get("safe_capture_success", False)),
        "capture_event": bool(final_info.get("capture_event", False)),
        "collision": bool(final_info.get("collision", False)),
        "boundary_violation": bool(int(final_info.get("world_violation_steps", 0)) > 0),
        "timeout": str(final_info.get("termination_reason")) == "timeout",
        "termination_reason": str(final_info.get("termination_reason")),
        "capture_time_seconds": final_info.get("capture_time_seconds"),
        "min_clearance_m": float(final_info.get("min_clearance_so_far", final_info.get("min_clearance", np.nan))),
        "mean_defender_path_length_m": float(np.mean(path_length)),
        "total_defender_path_length_m": float(np.sum(path_length)),
        "mean_visible_fraction": float(final_info.get("target_visible_fraction", np.nan)),
        "mean_observation_age_steps": float(final_info.get("mean_observation_age_steps", np.nan)),
        "mean_planner_latency_ms": float(np.nanmean([row["planner_latency_ms"] for row in step_rows])),
        "mean_predictor_latency_ms": float(np.nanmean([row["predictor_latency_ms"] for row in step_rows])),
        "prediction_refresh_rate": float(np.mean([row["prediction_refreshed"] for row in step_rows])),
        "mean_prediction_age_steps": float(np.mean([row["prediction_age_steps"] for row in step_rows])),
        "max_prediction_age_steps": int(max(row["prediction_age_steps"] for row in step_rows)),
        "mean_safety_latency_ms": float(np.nanmean([row["safety_latency_ms"] for row in step_rows])),
        "safety_solver_success_rate": _diagnostic_rate(step_rows, "safety_solver_success"),
        "safety_certificate_valid_rate": _diagnostic_rate(step_rows, "safety_certificate_valid"),
        "safety_fallback_rate": _diagnostic_rate(step_rows, "safety_fallback_used"),
        "safety_abort_required_rate": _diagnostic_rate(step_rows, "safety_abort_required"),
        "minimum_safety_barrier_m": _diagnostic_min(step_rows, "safety_minimum_barrier_m"),
        "maximum_safety_constraint_violation_m": _diagnostic_max(
            step_rows, "safety_maximum_constraint_violation_m"
        ),
        "safety_precondition_valid_rate": _diagnostic_rate(step_rows, "safety_precondition_valid"),
        "safety_recovery_action_rate": _diagnostic_rate(step_rows, "safety_recovery_action_used"),
        "safety_maximum_slack_m": _diagnostic_max(step_rows, "safety_maximum_slack_m"),
        "safety_mean_active_constraint_count": _diagnostic_mean(step_rows, "safety_active_constraint_count"),
        "safety_mean_constraint_count": _diagnostic_mean(step_rows, "safety_constraint_count"),
        "safety_failure_category_counts": _diagnostic_category_counts(step_rows, "safety_failure_category"),
        "safety_fallback_reason_counts": _diagnostic_category_counts(step_rows, "safety_fallback_reason"),
        "safety_independent_certificate_valid_rate": _diagnostic_rate(
            step_rows, "safety_independent_certificate_valid"
        ),
        "safety_independent_current_state_safe_rate": _diagnostic_rate(
            step_rows, "safety_independent_current_state_safe"
        ),
        "safety_independent_next_state_safe_rate": _diagnostic_rate(
            step_rows, "safety_independent_next_state_safe"
        ),
        "safety_independent_current_min_barrier_m": _diagnostic_min(
            step_rows, "safety_independent_current_min_barrier_m"
        ),
        "safety_independent_next_min_barrier_m": _diagnostic_min(
            step_rows, "safety_independent_next_min_barrier_m"
        ),
        "safety_independent_violation_counts": _diagnostic_violation_counts(
            step_rows, "safety_independent_violations"
        ),
        "mean_total_control_latency_ms": float(
            np.nanmean([row["total_control_latency_ms"] for row in step_rows])
        ),
        "planner_fallback_count": int(sum(row["planner_fallback"] for row in step_rows)),
        "planner_success_count": int(sum(row["planner_status"] for row in step_rows)),
        "planner_valid_count": int(sum(row["planner_valid"] for row in step_rows)),
        "planner_converged_count": int(sum(row["planner_converged"] for row in step_rows)),
        "planner_local_solver_failures": int(sum(row["planner_local_solver_failures"] for row in step_rows)),
        "planner_step_count": len(step_rows),
        "messages_attempted": int(sum(row["messages_attempted"] for row in step_rows)),
        "messages_sent": int(sum(row["messages_sent"] for row in step_rows)),
        "messages_received": int(sum(row["messages_received"] for row in step_rows)),
        "messages_dropped": int(sum(row["messages_dropped"] for row in step_rows)),
        "message_bytes_sent": int(sum(row["message_bytes_sent"] for row in step_rows)),
        "message_bytes_received": int(sum(row["message_bytes_received"] for row in step_rows)),
        "max_message_age_steps": int(max(row["max_message_age_steps"] for row in step_rows)),
        "mean_message_age_steps": finite_mean([row["mean_message_age_steps"] for row in step_rows]),
        "mean_distributed_iterations": finite_mean([row["distributed_iterations"] for row in step_rows]),
        "mean_distributed_action_delta_mps": finite_mean([row["distributed_action_delta_mps"] for row in step_rows]),
        "mean_cbf_action_correction_norm": float(
            np.nanmean([row["cbf_action_correction_norm"] for row in step_rows])
        ),
        "mean_candidate_worst_minimum_distance_m": float(
            finite_mean([row["candidate_worst_minimum_distance_m"] for row in step_rows])
        ),
        "worst_step_candidate_worst_minimum_distance_m": float(
            finite_max([row["candidate_worst_minimum_distance_m"] for row in step_rows])
        ),
        "mean_candidate_cvar_minimum_distance_m": float(
            finite_mean([row["candidate_cvar_minimum_distance_m"] for row in step_rows])
        ),
        "mean_candidate_expected_minimum_distance_m": float(
            finite_mean([row["candidate_expected_minimum_distance_m"] for row in step_rows])
        ),
        "candidate_distance_step_count": int(
            sum(float(row["candidate_count"]) > 0.0 for row in step_rows)
        ),
    }
    if trajectory_path is not None:
        save_episode_trajectory(env, trajectory_path)
    return summary, step_rows


def save_episode_trajectory(env: CaptureRadiusPursuit3DEnv, output_path: Path) -> None:
    """Persist a replay-only trajectory without changing evaluation artifacts."""

    if not env.history:
        raise ValueError("Trajectory history is empty; call run_episode with record_history=True.")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    defenders = np.asarray([frame["defender_positions"] for frame in env.history], dtype=np.float64)
    target = np.asarray([frame["target_position"] for frame in env.history], dtype=np.float64)
    centers = np.asarray([item.center_xy for item in env.obstacles], dtype=np.float64)
    radii = np.asarray([item.radius for item in env.obstacles], dtype=np.float64)
    heights = np.asarray([item.height for item in env.obstacles], dtype=np.float64)
    shapes = np.asarray([str(item.shape) for item in env.obstacles], dtype="U16")
    half_extents = np.asarray(
        [
            np.array([item.radius, item.radius], dtype=np.float64)
            if item.half_extents_xy is None
            else np.asarray(item.half_extents_xy, dtype=np.float64)
            for item in env.obstacles
        ],
        dtype=np.float64,
    )
    np.savez_compressed(
        output_path,
        defender_positions=defenders,
        target_positions=target,
        obstacle_centers_xy=centers,
        obstacle_radii=radii,
        obstacle_heights=heights,
        obstacle_shapes=shapes,
        obstacle_half_extents_xy=half_extents,
        world_half_extent=float(np.max(np.abs(env.lower[:2]))),
        world_height=float(env.upper[2]),
        capture_radius=float(env.pursuit["capture_radius"]),
        dt_seconds=float(env.dt),
    )


def percentile(values: list[float], quantile: float) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.percentile(finite, quantile)) if finite.size else float("nan")


def _diagnostic_values(step_rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in step_rows if row.get(key) is not None and np.isfinite(float(row[key]))]


def _diagnostic_rate(step_rows: list[dict[str, Any]], key: str) -> float:
    values = _diagnostic_values(step_rows, key)
    return float(np.mean(values)) if values else float("nan")


def _diagnostic_min(step_rows: list[dict[str, Any]], key: str) -> float:
    values = _diagnostic_values(step_rows, key)
    return float(np.min(values)) if values else float("nan")


def _diagnostic_max(step_rows: list[dict[str, Any]], key: str) -> float:
    values = _diagnostic_values(step_rows, key)
    return float(np.max(values)) if values else float("nan")


def _diagnostic_mean(step_rows: list[dict[str, Any]], key: str) -> float:
    values = _diagnostic_values(step_rows, key)
    return float(np.mean(values)) if values else float("nan")


def _diagnostic_category_counts(step_rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in step_rows:
        value = row.get(key)
        if value is None or str(value) in {"", "none", "None"}:
            continue
        label = str(value)
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _diagnostic_violation_counts(step_rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in step_rows:
        for value in row.get(key) or []:
            label = str(value)
            counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def finite_mean(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.mean(finite)) if finite.size else float("nan")


def finite_max(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return float(np.max(finite)) if finite.size else float("nan")


def summarize_rows(rows: list[dict[str, Any]], step_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize empty planner evaluation.")
    capture_times = [float(row["capture_time_seconds"]) for row in rows if row["capture_time_seconds"] is not None]
    planner_latencies = [float(row["planner_latency_ms"]) for row in step_rows]
    predictor_latencies = [float(row["predictor_latency_ms"]) for row in step_rows]
    safety_latencies = [float(row["safety_latency_ms"]) for row in step_rows]
    total_control_latencies = [float(row["total_control_latency_ms"]) for row in step_rows]
    success_steps = sum(float(row["planner_success_count"]) for row in rows)
    planner_steps = sum(float(row["planner_step_count"]) for row in rows)
    fallback_steps = sum(float(row["planner_fallback_count"]) for row in rows)
    return {
        "episodes": len(rows),
        "safe_capture_rate": float(np.mean([bool(row["safe_capture_success"]) for row in rows])),
        "capture_rate": float(np.mean([bool(row["capture_event"]) for row in rows])),
        "collision_rate": float(np.mean([bool(row["collision"]) for row in rows])),
        "boundary_violation_rate": float(np.mean([bool(row["boundary_violation"]) for row in rows])),
        "timeout_rate": float(np.mean([bool(row["timeout"]) for row in rows])),
        "mean_capture_time_seconds": float(np.mean(capture_times)) if capture_times else None,
        "median_capture_time_seconds": float(np.median(capture_times)) if capture_times else None,
        "mean_min_clearance_m": float(np.nanmean([float(row["min_clearance_m"]) for row in rows])),
        "worst_min_clearance_m": float(np.nanmin([float(row["min_clearance_m"]) for row in rows])),
        "mean_defender_path_length_m": float(np.mean([row["mean_defender_path_length_m"] for row in rows])),
        "mean_observation_age_steps": float(np.nanmean([row["mean_observation_age_steps"] for row in rows])),
        "mean_candidate_worst_minimum_distance_m": finite_mean(
            [row["mean_candidate_worst_minimum_distance_m"] for row in rows]
        ),
        "worst_candidate_minimum_distance_m": finite_max(
            [row["worst_step_candidate_worst_minimum_distance_m"] for row in rows]
        ),
        "mean_candidate_cvar_minimum_distance_m": finite_mean(
            [row["mean_candidate_cvar_minimum_distance_m"] for row in rows]
        ),
        "solver_success_rate": float(success_steps / max(planner_steps, 1.0)),
        "valid_plan_rate": float(sum(float(row["planner_valid_count"]) for row in rows) / max(planner_steps, 1.0)),
        "effective_plan_rate": float(success_steps / max(planner_steps, 1.0)),
        "convergence_rate": float(
            sum(float(row["planner_converged_count"]) for row in rows) / max(planner_steps, 1.0)
        ),
        "local_solver_failure_rate": float(
            sum(float(row["planner_local_solver_failures"]) for row in rows)
            / max(planner_steps, 1.0)
        ),
        "messages_attempted": int(sum(row["messages_attempted"] for row in rows)),
        "messages_sent": int(sum(row["messages_sent"] for row in rows)),
        "messages_received": int(sum(row["messages_received"] for row in rows)),
        "messages_dropped": int(sum(row["messages_dropped"] for row in rows)),
        "message_bytes_sent": int(sum(row["message_bytes_sent"] for row in rows)),
        "message_bytes_received": int(sum(row["message_bytes_received"] for row in rows)),
        "max_message_age_steps": int(max(row["max_message_age_steps"] for row in rows)),
        "mean_message_age_steps": finite_mean([row["mean_message_age_steps"] for row in rows]),
        "mean_distributed_iterations": finite_mean([row["mean_distributed_iterations"] for row in rows]),
        "mean_distributed_action_delta_mps": finite_mean(
            [row["mean_distributed_action_delta_mps"] for row in rows]
        ),
        "fallback_rate": float(fallback_steps / max(planner_steps, 1.0)),
        "planner_latency_ms": {
            "p50": percentile(planner_latencies, 50),
            "p95": percentile(planner_latencies, 95),
            "p99": percentile(planner_latencies, 99),
        },
        "predictor_latency_ms": {
            "p50": percentile(predictor_latencies, 50),
            "p95": percentile(predictor_latencies, 95),
            "p99": percentile(predictor_latencies, 99),
        },
        "prediction_refresh_rate": finite_mean([row["prediction_refresh_rate"] for row in rows]),
        "mean_prediction_age_steps": finite_mean([row["mean_prediction_age_steps"] for row in rows]),
        "max_prediction_age_steps": int(max(row["max_prediction_age_steps"] for row in rows)),
        "safety_latency_ms": {
            "p50": percentile(safety_latencies, 50),
            "p95": percentile(safety_latencies, 95),
            "p99": percentile(safety_latencies, 99),
        },
        "safety_solver_success_rate": _diagnostic_rate(step_rows, "safety_solver_success"),
        "safety_certificate_valid_rate": _diagnostic_rate(step_rows, "safety_certificate_valid"),
        "safety_fallback_rate": _diagnostic_rate(step_rows, "safety_fallback_used"),
        "safety_abort_required_rate": _diagnostic_rate(step_rows, "safety_abort_required"),
        "minimum_safety_barrier_m": _diagnostic_min(step_rows, "safety_minimum_barrier_m"),
        "maximum_safety_constraint_violation_m": _diagnostic_max(
            step_rows, "safety_maximum_constraint_violation_m"
        ),
        "safety_precondition_valid_rate": _diagnostic_rate(step_rows, "safety_precondition_valid"),
        "safety_recovery_action_rate": _diagnostic_rate(step_rows, "safety_recovery_action_used"),
        "safety_maximum_slack_m": _diagnostic_max(step_rows, "safety_maximum_slack_m"),
        "safety_mean_active_constraint_count": _diagnostic_mean(step_rows, "safety_active_constraint_count"),
        "safety_mean_constraint_count": _diagnostic_mean(step_rows, "safety_constraint_count"),
        "safety_failure_category_counts": _diagnostic_category_counts(step_rows, "safety_failure_category"),
        "safety_fallback_reason_counts": _diagnostic_category_counts(step_rows, "safety_fallback_reason"),
        "safety_independent_certificate_valid_rate": _diagnostic_rate(
            step_rows, "safety_independent_certificate_valid"
        ),
        "safety_independent_current_state_safe_rate": _diagnostic_rate(
            step_rows, "safety_independent_current_state_safe"
        ),
        "safety_independent_next_state_safe_rate": _diagnostic_rate(
            step_rows, "safety_independent_next_state_safe"
        ),
        "safety_independent_current_min_barrier_m": _diagnostic_min(
            step_rows, "safety_independent_current_min_barrier_m"
        ),
        "safety_independent_next_min_barrier_m": _diagnostic_min(
            step_rows, "safety_independent_next_min_barrier_m"
        ),
        "safety_independent_violation_counts": _diagnostic_violation_counts(
            step_rows, "safety_independent_violations"
        ),
        "total_control_latency_ms": {
            "p50": percentile(total_control_latencies, 50),
            "p95": percentile(total_control_latencies, 95),
            "p99": percentile(total_control_latencies, 99),
        },
        "step_latency_samples": len(step_rows),
    }


def main() -> None:
    args = parse_args()
    mpc_document = load_yaml(args.mpc_config)
    planner_mapping = dict(mpc_document.get("planner", {}))
    distributed_mapping = dict(mpc_document.get("distributed", {}))
    evaluation_mapping = dict(mpc_document.get("evaluation", {}))
    prediction_mapping = dict(mpc_document.get("prediction", {}))
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    episodes = int(args.episodes if args.episodes is not None else evaluation_mapping.get("episodes", 8))
    seed_start = int(args.seed_start if args.seed_start is not None else evaluation_mapping.get("seed_start", 648301))
    obstacle_count = int(args.obstacle_count if args.obstacle_count is not None else evaluation_mapping.get("obstacle_count", 3))
    target_speed_scale = float(
        args.target_speed_scale
        if args.target_speed_scale is not None
        else evaluation_mapping.get("target_speed_scale", 0.55)
    )
    target_motion_mode = str(
        args.target_motion_mode
        if args.target_motion_mode is not None
        else evaluation_mapping.get("target_motion_mode", "flee_persistence")
    )
    methods = list(args.methods or evaluation_mapping.get("methods", ["dynamic_encirclement", "expected", "worst_case", "cvar"]))
    if episodes <= 0 or obstacle_count < 0 or target_speed_scale <= 0.0:
        raise ValueError("episodes must be positive, obstacle_count non-negative, and target_speed_scale positive.")
    planner_config = MinimaxMPCConfig.from_mapping(planner_mapping)
    device = select_device(args.device)
    num_samples = int(args.num_samples if args.num_samples is not None else prediction_mapping.get("num_samples", 8))
    sampling_steps = int(args.sampling_steps if args.sampling_steps is not None else prediction_mapping.get("sampling_steps", 8))
    sampling_seed = int(args.sampling_seed if args.sampling_seed is not None else prediction_mapping.get("sampling_seed", 745102))
    projection_iterations = int(
        args.projection_iterations
        if args.projection_iterations is not None
        else prediction_mapping.get("projection_iterations", 4)
    )
    prediction_refresh_interval_steps = int(
        args.prediction_refresh_interval_steps
        if args.prediction_refresh_interval_steps is not None
        else prediction_mapping.get("refresh_interval_steps", 1)
    )
    if (
        num_samples <= 0
        or sampling_steps <= 0
        or projection_iterations <= 0
        or prediction_refresh_interval_steps <= 0
    ):
        raise ValueError("prediction sampling, projection and refresh settings must be positive.")
    if args.candidate_source == "checkpoint" and not args.checkpoint:
        raise ValueError("--checkpoint is required when --candidate-source=checkpoint.")
    checkpoint_data = model_from_checkpoint(args.checkpoint, device) if args.checkpoint else None
    if checkpoint_data is not None:
        model_config = checkpoint_data[3].get("model_config", {})
        if int(model_config.get("horizon_count", 0)) < planner_config.horizon_steps:
            raise ValueError("Prediction checkpoint horizon is shorter than planner horizon.")

    base_config = load_yaml(args.environment_config)
    base_config = copy.deepcopy(base_config)
    base_config.setdefault("task", {}).setdefault("pursuit", {})["target_motion_mode"] = target_motion_mode
    base_config["experiments"] = [
        {
            "name": "phase3_scenario_mpc",
            "episodes": 1,
            "obstacle_count": obstacle_count,
            "target_speed_scale": target_speed_scale,
        }
    ]
    if args.max_steps is not None:
        if args.max_steps <= 0:
            raise ValueError("max-steps must be positive when supplied.")
        base_config["world"]["max_steps"] = int(args.max_steps)
    if args.safety_layer == "robust_cbf_qp" and args.without_local_cbf:
        raise ValueError("--without-local-cbf cannot be combined with --safety-layer robust_cbf_qp")
    use_local_cbf = args.safety_layer == "local_cbf" and not args.without_local_cbf
    robust_safety_config: RobustCBFQPConfig | None = None
    if args.safety_layer == "robust_cbf_qp":
        safety_document = load_yaml(args.safety_config)
        safety_mapping = dict(safety_document.get("safety", {}))
        safety_probe = CaptureRadiusPursuit3DEnv(
            copy.deepcopy(base_config),
            obstacle_count=obstacle_count,
            target_speed_scale=target_speed_scale,
        )
        safety_mapping.setdefault("max_speed_mps", float(safety_probe.agents["defender_max_speed"]))
        safety_mapping.setdefault("max_acceleration_mps2", float(safety_probe.agents["defender_max_acceleration"]))
        safety_mapping.setdefault("safety_margin_m", float(safety_probe.pursuit["safety_margin"]))
        robust_safety_config = RobustCBFQPConfig.from_mapping(safety_mapping)
    serialized_arguments = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    serialized_arguments["output_dir"] = str(output)
    hashes = source_hashes(args.mpc_config)
    if args.safety_layer == "robust_cbf_qp":
        add_safety_source_hashes(hashes, args.safety_config)
    run_config = {
        "arguments": serialized_arguments,
        "environment_config": str(args.environment_config.resolve()),
        "mpc_config": str(args.mpc_config.resolve()),
        "planner": planner_config.__dict__,
        "distributed": distributed_mapping,
        "evaluation": {
            "episodes": episodes,
            "seed_start": seed_start,
            "obstacle_count": obstacle_count,
            "target_speed_scale": target_speed_scale,
            "target_motion_mode": target_motion_mode,
            "max_steps": args.max_steps,
            "methods": methods,
            "candidate_source": args.candidate_source,
            "checkpoint": None if args.checkpoint is None else str(args.checkpoint.resolve()),
            "device": str(device),
            "use_local_cbf": use_local_cbf,
            "safety_layer": args.safety_layer if use_local_cbf or args.safety_layer == "robust_cbf_qp" else "none",
            "safety_config": str(args.safety_config.resolve()) if args.safety_layer == "robust_cbf_qp" else None,
            "prediction_refresh_interval_steps": prediction_refresh_interval_steps,
        },
        "source_hashes": hashes,
    }
    output.joinpath("config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    all_summaries: dict[str, Any] = {}
    for method in methods:
        method_output = output / method
        method_output.mkdir(parents=True, exist_ok=True)
        method_run_config = {
            **run_config,
            "evaluation": {**run_config["evaluation"], "method": method},
        }
        method_output.joinpath("config.yaml").write_text(
            yaml.safe_dump(method_run_config, sort_keys=False), encoding="utf-8"
        )
        rows: list[dict[str, Any]] = []
        all_step_rows: list[dict[str, float]] = []
        with (
            method_output.joinpath("episodes.jsonl").open("w", encoding="utf-8") as episode_file,
            method_output.joinpath("steps.jsonl").open("w", encoding="utf-8") as step_file,
        ):
            for episode_index in range(episodes):
                seed = seed_start + episode_index
                distributed_config = None
                distributed_modes = {
                    "distributed_ideal": "ideal",
                    "distributed_delayed": "delayed",
                    "distributed_dropout": "dropout",
                    "distributed_none": "none",
                }
                if method in distributed_modes:
                    distributed_config = DistributedDNMPCConfig.from_mapping(
                        {
                            **distributed_mapping,
                            "communication_mode": distributed_modes[method],
                        }
                    )
                row, step_rows = run_episode(
                    base_config,
                    seed=seed,
                    method=method,
                    planner_config=planner_config,
                    candidate_source=args.candidate_source,
                    checkpoint_data=checkpoint_data,
                    device=device,
                    num_samples=num_samples,
                    sampling_steps=sampling_steps,
                    sampling_seed=sampling_seed + episode_index * 1000,
                    projection_iterations=projection_iterations,
                    use_local_cbf=use_local_cbf,
                    safety_layer=(args.safety_layer if not args.without_local_cbf else "none"),
                    robust_safety_config=robust_safety_config,
                    prediction_refresh_interval_steps=prediction_refresh_interval_steps,
                    distributed_config=distributed_config,
                )
                rows.append(row)
                all_step_rows.extend(step_rows)
                episode_file.write(json.dumps(row, allow_nan=True) + "\n")
                for step_row in step_rows:
                    step_file.write(
                        json.dumps(
                            {
                                "method": method,
                                "episode_index": episode_index,
                                "episode_seed": seed,
                                **step_row,
                            },
                            allow_nan=True,
                        )
                        + "\n"
                    )
        summary = summarize_rows(rows, all_step_rows)
        method_output.joinpath("summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
        all_summaries[method] = summary
        with require_summary_writer()(log_dir=str(method_output / "tensorboard"), flush_secs=5) as writer:
            writer.add_text("Evaluation/config", yaml.safe_dump(run_config, sort_keys=False), 0)
            writer.add_text("Evaluation/source_hashes", json.dumps(run_config["source_hashes"], indent=2), 0)
            for episode_index, row in enumerate(rows):
                for key in (
                    "safe_capture_success",
                    "capture_event",
                    "collision",
                    "boundary_violation",
                    "timeout",
                    "mean_min_clearance_m",
                    "mean_planner_latency_ms",
                    "mean_predictor_latency_ms",
                    "prediction_refresh_rate",
                    "mean_prediction_age_steps",
                    "max_prediction_age_steps",
                    "mean_safety_latency_ms",
                    "mean_total_control_latency_ms",
                    "safety_solver_success_rate",
                    "safety_certificate_valid_rate",
                    "safety_fallback_rate",
                    "safety_abort_required_rate",
                    "safety_precondition_valid_rate",
                    "safety_recovery_action_rate",
                    "safety_maximum_slack_m",
                    "safety_mean_active_constraint_count",
                    "safety_mean_constraint_count",
                    "safety_independent_certificate_valid_rate",
                    "safety_independent_current_state_safe_rate",
                    "safety_independent_next_state_safe_rate",
                    "safety_independent_current_min_barrier_m",
                    "safety_independent_next_min_barrier_m",
                    "minimum_safety_barrier_m",
                    "maximum_safety_constraint_violation_m",
                    "planner_fallback_count",
                    "planner_success_count",
                    "planner_valid_count",
                    "planner_converged_count",
                    "planner_local_solver_failures",
                    "messages_attempted",
                    "messages_sent",
                    "messages_received",
                    "messages_dropped",
                    "message_bytes_sent",
                    "max_message_age_steps",
                    "mean_distributed_iterations",
                    "mean_distributed_action_delta_mps",
                    "mean_candidate_worst_minimum_distance_m",
                    "mean_candidate_cvar_minimum_distance_m",
                ):
                    value = row.get(key)
                    if value is not None and np.isfinite(float(value)):
                        writer.add_scalar(f"Episode/{key}", float(value), episode_index)
            for key, value in summary.items():
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    writer.add_scalar(f"Summary/{key}", float(value), 0)
            writer.add_text(
                "Summary/SafetyFailureCategoryCounts",
                json.dumps(summary.get("safety_failure_category_counts", {}), sort_keys=True),
                0,
            )
            writer.add_text(
                "Summary/SafetyFallbackReasonCounts",
                json.dumps(summary.get("safety_fallback_reason_counts", {}), sort_keys=True),
                0,
            )
            writer.add_text(
                "Summary/SafetyIndependentViolationCounts",
                json.dumps(summary.get("safety_independent_violation_counts", {}), sort_keys=True),
                0,
            )
            writer.add_scalar("Summary/PlannerLatency/p50_ms", summary["planner_latency_ms"]["p50"], 0)
            writer.add_scalar("Summary/PlannerLatency/p95_ms", summary["planner_latency_ms"]["p95"], 0)
            writer.add_scalar("Summary/PlannerLatency/p99_ms", summary["planner_latency_ms"]["p99"], 0)
            writer.add_scalar("Summary/PredictorLatency/p50_ms", summary["predictor_latency_ms"]["p50"], 0)
            writer.add_scalar("Summary/PredictorLatency/p95_ms", summary["predictor_latency_ms"]["p95"], 0)
            writer.add_scalar("Summary/PredictorLatency/p99_ms", summary["predictor_latency_ms"]["p99"], 0)
            writer.add_scalar("Summary/SafetyLatency/p50_ms", summary["safety_latency_ms"]["p50"], 0)
            writer.add_scalar("Summary/SafetyLatency/p95_ms", summary["safety_latency_ms"]["p95"], 0)
            writer.add_scalar("Summary/SafetyLatency/p99_ms", summary["safety_latency_ms"]["p99"], 0)
            writer.add_scalar("Summary/TotalControlLatency/p50_ms", summary["total_control_latency_ms"]["p50"], 0)
            writer.add_scalar("Summary/TotalControlLatency/p95_ms", summary["total_control_latency_ms"]["p95"], 0)
            writer.add_scalar("Summary/TotalControlLatency/p99_ms", summary["total_control_latency_ms"]["p99"], 0)
            writer.add_hparams(
                {
                    "risk_mode": method if method in {"expected", "worst_case", "cvar"} else "worst_case",
                    "distributed_method": int(method.startswith("distributed_")),
                    "horizon_steps": planner_config.horizon_steps,
                    "control_horizon_steps": planner_config.control_horizon_steps,
                    "candidate_source": args.candidate_source,
                    "num_samples": num_samples,
                    "sampling_steps": sampling_steps,
                    "use_local_cbf": int(use_local_cbf),
                    "safety_layer": args.safety_layer if use_local_cbf or args.safety_layer == "robust_cbf_qp" else "none",
                    "prediction_refresh_interval_steps": prediction_refresh_interval_steps,
                },
                {
                    "hparam/safe_capture_rate": float(summary["safe_capture_rate"]),
                    "hparam/solver_success_rate": float(summary["solver_success_rate"]),
                    "hparam/planner_latency_p95_ms": float(summary["planner_latency_ms"]["p95"]),
                },
            )
    result = {
        "config": run_config,
        "methods": all_summaries,
        "decision": "diagnostic_only",
        "decision_reason": "P4 communication robustness is diagnostic until the formal S3 gate is reviewed.",
    }
    output.joinpath("summary.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
