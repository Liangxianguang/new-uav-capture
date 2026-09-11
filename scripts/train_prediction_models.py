"""Train TensorBoard-tracked GRU and SSM-conditioned diffusion predictors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

# Windows environments that package both Intel and LLVM OpenMP runtimes can
# initialize incompatible duplicate runtimes before PyTorch is imported.
if os.name == "nt":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.prediction import (  # noqa: E402
    CandidateTrajectorySet,
    ConditionalDiffusionTrajectoryPredictor,
    HistoryTargetPredictor,
    OfficialS4ConditionalDiffusionTrajectoryPredictor,
    S4ConditionalDiffusionTrajectoryPredictor,
    TrajectoryNormalizer,
    assess_candidate_feasibility,
    candidate_energy_score,
    conformal_coverage,
    conformal_nonconformity,
    conformal_radius,
    deterministic_mse,
    gaussian_nll,
    prediction_metrics,
)
from encirclement3d.trajectory_dataset import (  # noqa: E402
    PredictionDataset,
    load_prediction_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("gru", "diffusion", "s4_diffusion", "official_s4_diffusion"),
        required=True,
    )
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument(
        "--official-s4-root",
        type=Path,
        help="Upstream state-spaces/s4 source root required by official_s4_diffusion.",
    )
    parser.add_argument("--official-s4-state-dim", type=int, default=16)
    parser.add_argument("--official-s4-rank", type=int, default=1)
    parser.add_argument(
        "--action-conditioning",
        choices=("none", "history", "future", "both"),
        default="both",
        help="Use explicit action history, future planned-action conditions, both, or neither.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--target-normalization",
        choices=("train_split_standardize", "fixed_scale"),
        default="train_split_standardize",
        help="Default fits a per-horizon coordinate normalizer on training targets only.",
    )
    parser.add_argument(
        "--target-scale",
        type=float,
        default=10.0,
        help="Legacy scalar normalization used only with --target-normalization fixed_scale.",
    )
    parser.add_argument("--normalizer-minimum-scale", type=float, default=1e-3)
    parser.add_argument(
        "--evaluation-sampling-seed",
        type=int,
        default=745102,
        help="Seed for repeatable diffusion validation noise; independent from training seed.",
    )
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--latency-repeats", type=int, default=32)
    parser.add_argument(
        "--candidate-max-speed",
        type=float,
        help="Physical target speed limit in m/s; default is derived from dataset metadata.",
    )
    parser.add_argument(
        "--candidate-max-acceleration",
        type=float,
        help="Physical target acceleration limit in m/s^2; default is derived from dataset metadata.",
    )
    parser.add_argument(
        "--candidate-minimum-clearance",
        type=float,
        help="Obstacle clearance in m; default is the recorded drone radius when available.",
    )
    parser.add_argument("--seed", type=int, default=745101)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def flatten_inputs(
    dataset: PredictionDataset,
    action_conditioning: str = "both",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if action_conditioning not in {"none", "history", "future", "both"}:
        raise ValueError("action_conditioning must be one of none/history/future/both.")
    inputs = torch.as_tensor(dataset.history_observations, dtype=torch.float32)
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
    use_history = action_conditioning in {"history", "both"}
    use_future = action_conditioning in {"future", "both"}
    if use_history:
        if dataset.history_action_features is None:
            raise ValueError("The dataset does not provide history_action_features.")
        history_actions = torch.as_tensor(dataset.history_action_features, dtype=torch.float32)
        inputs = torch.cat([inputs, history_actions], dim=-1)
    targets = torch.as_tensor(dataset.future_target_displacements, dtype=torch.float32)
    if use_future:
        if dataset.future_action_conditions is None:
            raise ValueError("The dataset does not provide future_action_conditions.")
        action_conditions = torch.as_tensor(dataset.future_action_conditions, dtype=torch.float32)
    else:
        action_conditions = torch.zeros(
            targets.shape[0], targets.shape[1], 0, dtype=torch.float32
        )
    return inputs, targets, action_conditions


@dataclass(frozen=True)
class CandidateConstraints:
    max_speed: float | None
    max_acceleration: float | None
    minimum_clearance: float
    provenance: dict[str, str]

    def as_dict(self) -> dict[str, object]:
        return {
            "max_speed_mps": self.max_speed,
            "max_acceleration_mps2": self.max_acceleration,
            "minimum_obstacle_clearance_m": self.minimum_clearance,
            "provenance": self.provenance,
        }


def load_dataset_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path.resolve().parent / "metadata.json"
    if not metadata_path.is_file():
        return {}
    loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Dataset metadata must be a mapping: {metadata_path}")
    return loaded


def derive_candidate_constraints(args: argparse.Namespace) -> CandidateConstraints:
    metadata = load_dataset_metadata(args.train_dataset)
    config = metadata.get("config", {})
    agents = config.get("agents", {}) if isinstance(config, dict) else {}
    derived_speed = None
    if isinstance(agents.get("target_max_speed"), (int, float)):
        # target_speed_scale changes the nominal behavior, while the agent
        # configuration is the hard physical speed limit used by feasibility.
        derived_speed = float(agents["target_max_speed"])
    derived_acceleration = (
        float(agents["target_max_acceleration"])
        if isinstance(agents.get("target_max_acceleration"), (int, float))
        else None
    )
    derived_clearance = (
        float(agents["drone_radius"])
        if isinstance(agents.get("drone_radius"), (int, float))
        else 0.0
    )
    return CandidateConstraints(
        max_speed=(float(args.candidate_max_speed) if args.candidate_max_speed is not None else derived_speed),
        max_acceleration=(
            float(args.candidate_max_acceleration)
            if args.candidate_max_acceleration is not None
            else derived_acceleration
        ),
        minimum_clearance=(
            float(args.candidate_minimum_clearance)
            if args.candidate_minimum_clearance is not None
            else derived_clearance
        ),
        provenance={
            "max_speed": "cli" if args.candidate_max_speed is not None else "train_dataset_metadata",
            "max_acceleration": (
                "cli" if args.candidate_max_acceleration is not None else "train_dataset_metadata"
            ),
            "minimum_clearance": (
                "cli" if args.candidate_minimum_clearance is not None else "train_dataset_metadata"
            ),
        },
    )


def seeded_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device.type).manual_seed(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def source_hashes() -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "scripts" / "train_prediction_models.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "prediction.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "trajectory_dataset.py",
    )
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def build_model(
    args: argparse.Namespace,
    input_dim: int,
    horizon_count: int,
    action_condition_dim: int,
    history_length: int,
) -> torch.nn.Module:
    if args.model == "gru":
        return HistoryTargetPredictor(
            input_dim=input_dim,
            horizon_count=horizon_count,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            action_condition_dim=action_condition_dim,
        )
    if args.model == "official_s4_diffusion":
        if args.official_s4_root is None:
            raise ValueError("--official-s4-root is required for official_s4_diffusion.")
        return OfficialS4ConditionalDiffusionTrajectoryPredictor(
            input_dim=input_dim,
            horizon_count=horizon_count,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            diffusion_steps=args.diffusion_steps,
            action_condition_dim=action_condition_dim,
            history_length=history_length,
            official_s4_root=args.official_s4_root,
            state_dim=args.official_s4_state_dim,
            rank=args.official_s4_rank,
        )
    predictor_class = S4ConditionalDiffusionTrajectoryPredictor if args.model == "s4_diffusion" else ConditionalDiffusionTrajectoryPredictor
    return predictor_class(
        input_dim=input_dim,
        horizon_count=horizon_count,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        diffusion_steps=args.diffusion_steps,
        action_condition_dim=action_condition_dim,
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_kind: str,
    normalizer: TrajectoryNormalizer,
) -> float:
    model.train()
    losses: list[float] = []
    for inputs, targets, action_conditions in loader:
        inputs = inputs.to(device)
        targets = normalizer.normalize(targets.to(device))
        action_condition = action_conditions.to(device) if action_conditions.shape[-1] > 0 else None
        if model_kind == "gru":
            mean, log_variance = model(inputs, action_condition)
            loss = gaussian_nll(mean, log_variance, targets)
        else:
            loss = model.diffusion_loss(inputs, targets, action_condition=action_condition)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {model_kind} training loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def sample_physical_candidates(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    sample_indices: np.ndarray,
    device: torch.device,
    model_kind: str,
    normalizer: TrajectoryNormalizer,
    num_samples: int,
    sampling_steps: int,
    sampling_seed: int,
    action_conditions: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    indices = torch.as_tensor(sample_indices, dtype=torch.long)
    selected_inputs = inputs.index_select(0, indices).to(device)
    selected_actions = action_conditions.index_select(0, indices).to(device)
    action_condition = selected_actions if selected_actions.shape[-1] > 0 else None
    if model_kind == "gru":
        mean, _log_variance = model(selected_inputs, action_condition)
        return normalizer.denormalize(mean[:, None])
    candidate_set = model.sample_set(
        selected_inputs,
        num_samples=num_samples,
        sampling_steps=sampling_steps,
        generator=seeded_generator(device, sampling_seed),
        action_condition=action_condition,
    )
    return normalizer.denormalize(candidate_set.trajectories)


@torch.no_grad()
def evaluate_subset(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    dataset: PredictionDataset,
    sample_indices: np.ndarray,
    device: torch.device,
    model_kind: str,
    normalizer: TrajectoryNormalizer,
    num_samples: int,
    sampling_steps: int,
    evaluation_sampling_seed: int,
    constraints: CandidateConstraints,
    action_conditions: torch.Tensor,
    calibration_radius: float | None = None,
) -> dict[str, float]:
    if sample_indices.ndim != 1 or sample_indices.size == 0:
        raise ValueError("sample_indices must select at least one validation sample.")
    model.eval()
    indices = torch.as_tensor(sample_indices, dtype=torch.long)
    inputs = inputs.index_select(0, indices).to(device)
    targets = targets.index_select(0, indices).to(device)
    selected_actions = action_conditions.index_select(0, indices).to(device)
    action_condition = selected_actions if selected_actions.shape[-1] > 0 else None
    normalized_targets = normalizer.normalize(targets)
    if model_kind == "gru":
        mean, log_variance = model(inputs, action_condition)
        loss = gaussian_nll(mean, log_variance, normalized_targets)
        candidate_set = CandidateTrajectorySet.uniform(normalizer.denormalize(mean[:, None]))
        deterministic = deterministic_mse(mean, normalized_targets)
    else:
        loss = model.diffusion_loss(
            inputs,
            normalized_targets,
            generator=seeded_generator(device, evaluation_sampling_seed),
            action_condition=action_condition,
        )
        candidate_set = model.sample_set(
            inputs,
            num_samples=num_samples,
            sampling_steps=sampling_steps,
            generator=seeded_generator(device, evaluation_sampling_seed + 1),
            action_condition=action_condition,
        )
        candidate_set = CandidateTrajectorySet(
            trajectories=normalizer.denormalize(candidate_set.trajectories),
            logits=candidate_set.logits,
            score_kind=candidate_set.score_kind,
        )
        deterministic = torch.tensor(float("nan"), device=device)
    metrics = prediction_metrics(candidate_set.trajectories, targets)
    metrics["energy_score"] = candidate_energy_score(candidate_set.trajectories, targets)
    constant_velocity = (
        torch.as_tensor(dataset.reference_velocities[sample_indices], device=device)[:, None, :]
        * float(dataset.dt_seconds)
        * torch.arange(1, targets.shape[1] + 1, device=device, dtype=targets.dtype)[None, :, None]
    )
    baseline = prediction_metrics(constant_velocity[:, None], targets)
    result = {
        "loss": float(loss.detach().cpu()),
        "deterministic_mse": float(deterministic.detach().cpu()),
        **{f"model_{key}": value for key, value in metrics.items()},
        **{f"constant_velocity_{key}": value for key, value in baseline.items()},
    }
    if dataset.has_geometry_context and constraints.max_speed is not None:
        feasibility = assess_candidate_feasibility(
            candidate_set.trajectories,
            torch.as_tensor(dataset.reference_positions[sample_indices], device=device),
            torch.as_tensor(dataset.reference_velocities[sample_indices], device=device),
            dataset.dt_seconds,
            torch.as_tensor(dataset.world_lower_bounds[sample_indices], device=device),
            torch.as_tensor(dataset.world_upper_bounds[sample_indices], device=device),
            constraints.max_speed,
            constraints.max_acceleration,
            torch.as_tensor(dataset.obstacle_centers_xy[sample_indices], device=device),
            torch.as_tensor(dataset.obstacle_radii[sample_indices], device=device),
            torch.as_tensor(dataset.obstacle_heights[sample_indices], device=device),
            torch.as_tensor(dataset.obstacle_half_extents_xy[sample_indices], device=device),
            torch.as_tensor(dataset.obstacle_shape_codes[sample_indices], device=device),
            constraints.minimum_clearance,
        )
        result.update(feasibility.metrics())
        result["candidate_feasibility_available"] = 1.0
    else:
        result["candidate_feasibility_available"] = 0.0
    if calibration_radius is not None:
        result.update(conformal_coverage(candidate_set.trajectories, targets, calibration_radius))
        result["calibration_radius_m"] = float(calibration_radius)
    return result


def validation_calibration_indices(dataset: PredictionDataset) -> tuple[np.ndarray, np.ndarray]:
    """Split validation episodes, rather than overlapping windows, for calibration."""

    episode_seeds = np.unique(dataset.episode_seeds)
    if episode_seeds.size < 2:
        raise ValueError("Validation calibration requires at least two distinct episodes.")
    calibration_count = max(1, episode_seeds.size // 2)
    calibration_seeds = episode_seeds[:calibration_count]
    evaluation_seeds = episode_seeds[calibration_count:]
    calibration = np.flatnonzero(np.isin(dataset.episode_seeds, calibration_seeds)).astype(np.int64)
    evaluation = np.flatnonzero(np.isin(dataset.episode_seeds, evaluation_seeds)).astype(np.int64)
    return calibration, evaluation


@torch.no_grad()
def measure_single_sample_latency(
    model: torch.nn.Module,
    input_sample: torch.Tensor,
    device: torch.device,
    model_kind: str,
    action_condition: torch.Tensor | None,
    num_samples: int,
    sampling_steps: int,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("latency-warmup must be non-negative and latency-repeats must be positive.")
    model.eval()
    inputs = input_sample[None].to(device)
    action = None if action_condition is None else action_condition[None].to(device)

    def run() -> None:
        if model_kind == "gru":
            model(inputs, action)
        else:
            model.sample_set(inputs, num_samples=num_samples, sampling_steps=sampling_steps, action_condition=action)

    for _ in range(warmup):
        run()
    synchronize(device)
    latencies_ms: list[float] = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        run()
        synchronize(device)
        latencies_ms.append((time.perf_counter() - started) * 1_000.0)
    return {
        "single_sample_latency_p50_ms": float(np.percentile(latencies_ms, 50)),
        "single_sample_latency_p95_ms": float(np.percentile(latencies_ms, 95)),
        "single_sample_latency_p99_ms": float(np.percentile(latencies_ms, 99)),
    }


def write_metadata(
    output: Path,
    args: argparse.Namespace,
    train_dataset: PredictionDataset,
    validation_dataset: PredictionDataset,
    input_dim: int,
    device: torch.device,
    normalizer: TrajectoryNormalizer,
    constraints: CandidateConstraints,
    action_conditioning: str,
    action_condition_dim: int,
) -> None:
    serializable_arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metadata = {
        "model": args.model,
        "model_backend": (
            "official_state_spaces_s4_dplr"
            if args.model == "official_s4_diffusion"
            else
            "s4_dplr_dense_reference"
            if args.model == "s4_diffusion"
            else "portable_diagonal_ssm"
            if args.model == "diffusion"
            else "gru_gaussian"
        ),
        "arguments": serializable_arguments,
        "train_dataset": str(args.train_dataset.resolve()),
        "validation_dataset": str(args.validation_dataset.resolve()),
        "train_samples": train_dataset.sample_count,
        "validation_samples": validation_dataset.sample_count,
        "input_dim": input_dim,
        "action_conditioning": action_conditioning,
        "action_condition_dim": action_condition_dim,
        "history_action_features": (
            None
            if train_dataset.history_action_features is None
            else int(train_dataset.history_action_features.shape[-1])
        ),
        "future_action_conditions": (
            None
            if train_dataset.future_action_conditions is None
            else int(train_dataset.future_action_conditions.shape[-1])
        ),
        "horizon_count": train_dataset.horizon_steps,
        "dt_seconds": float(train_dataset.dt_seconds),
        "target_normalizer": normalizer.as_dict(),
        "candidate_feasibility_constraints": constraints.as_dict(),
        "validation_geometry_context_available": validation_dataset.has_geometry_context,
        "candidate_score_kind": "uniform_uncalibrated",
        "confidence_calibration": {
            "method": "split_conformal_full_trajectory_max_distance",
            "coverage_target": 0.90,
            "calibration_split": "first half of validation episodes",
            "evaluation_split": "second half of validation episodes",
        },
        "device": str(device),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": version("torch"),
        "tensorboard": version("tensorboard"),
        "runtime_workaround": {
            "windows_openmp_duplicate_runtime": os.name == "nt",
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "KMP_DUPLICATE_LIB_OK": os.environ.get("KMP_DUPLICATE_LIB_OK"),
        },
        "source_hashes": source_hashes(),
    }
    output.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output.joinpath("config.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0 or args.num_layers <= 0:
        raise ValueError("epochs, batch-size, hidden-dim, and num-layers must be positive.")
    if (
        args.learning_rate <= 0.0
        or args.diffusion_steps <= 0
        or args.sampling_steps <= 0
        or args.num_samples <= 0
        or args.normalizer_minimum_scale <= 0.0
    ):
        raise ValueError("Learning rate, diffusion settings, and normalizer-minimum-scale must be positive.")
    if args.target_normalization == "fixed_scale" and args.target_scale <= 0.0:
        raise ValueError("target-scale must be positive with fixed_scale normalization.")
    for name in ("candidate_max_speed", "candidate_max_acceleration"):
        value = getattr(args, name)
        if value is not None and value <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive when supplied.")
    if args.candidate_minimum_clearance is not None and args.candidate_minimum_clearance < 0.0:
        raise ValueError("candidate-minimum-clearance must be non-negative when supplied.")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    train_dataset = load_prediction_dataset(str(args.train_dataset.resolve()))
    validation_dataset = load_prediction_dataset(str(args.validation_dataset.resolve()))
    if (
        train_dataset.history_length != validation_dataset.history_length
        or train_dataset.horizon_steps != validation_dataset.horizon_steps
        or train_dataset.defender_count != validation_dataset.defender_count
        or train_dataset.feature_dim != validation_dataset.feature_dim
        or not np.isclose(train_dataset.dt_seconds, validation_dataset.dt_seconds)
    ):
        raise ValueError("Train and validation prediction datasets have incompatible shapes.")
    train_inputs, train_targets, train_action_conditions = flatten_inputs(
        train_dataset, args.action_conditioning
    )
    validation_inputs, validation_targets, validation_action_conditions = flatten_inputs(
        validation_dataset, args.action_conditioning
    )
    input_dim = int(train_inputs.shape[-1])
    normalizer = (
        TrajectoryNormalizer.fit(train_targets.numpy(), args.normalizer_minimum_scale)
        if args.target_normalization == "train_split_standardize"
        else TrajectoryNormalizer.fixed(train_dataset.horizon_steps, args.target_scale)
    )
    constraints = derive_candidate_constraints(args)
    action_condition_dim = int(train_action_conditions.shape[-1])
    if int(validation_action_conditions.shape[-1]) != action_condition_dim:
        raise ValueError("Train and validation action-condition dimensions differ.")
    model = build_model(
        args,
        input_dim,
        train_dataset.horizon_steps,
        action_condition_dim,
        train_dataset.history_length,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets, train_action_conditions),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    write_metadata(
        output,
        args,
        train_dataset,
        validation_dataset,
        input_dim,
        device,
        normalizer,
        constraints,
        action_conditioning=args.action_conditioning,
        action_condition_dim=action_condition_dim,
    )
    writer = SummaryWriter(log_dir=str(output / "tensorboard"), flush_secs=10)
    writer.add_text(
        "Config/arguments",
        yaml.safe_dump(
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            sort_keys=False,
        ),
        0,
    )
    writer.add_text("Config/target_normalizer", yaml.safe_dump(normalizer.as_dict(), sort_keys=False), 0)
    writer.add_text(
        "Config/candidate_feasibility_constraints",
        yaml.safe_dump(constraints.as_dict(), sort_keys=False),
        0,
    )
    writer.add_text(
        "Config/confidence_calibration",
        yaml.safe_dump(
            {
                "method": "split_conformal_full_trajectory_max_distance",
                "coverage_target": 0.90,
                "calibration_split": "first half of validation episodes",
                "evaluation_split": "second half of validation episodes",
            },
            sort_keys=False,
        ),
        0,
    )
    writer.add_text(
        "Config/dataset_contract",
        yaml.safe_dump(
            {
                "inputs": "policy-safe observations",
                "labels": "future target truth",
                "dt_seconds": float(train_dataset.dt_seconds),
            },
            sort_keys=False,
        ),
        0,
    )
    writer.add_text("Runtime/openmp", json.dumps({
        "windows_duplicate_runtime_workaround": os.name == "nt",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "KMP_DUPLICATE_LIB_OK": os.environ.get("KMP_DUPLICATE_LIB_OK"),
    }), 0)
    writer.add_text(
        "Model/backend",
        (
            "s4_dplr_dense_reference"
            if args.model == "s4_diffusion"
            else "portable_diagonal_ssm"
            if args.model == "diffusion"
            else "gru_gaussian"
        ),
        0,
    )
    writer.add_histogram("Data/train_targets_physical_m", train_targets.reshape(-1), 0)
    writer.add_histogram(
        "Data/train_targets_normalized",
        normalizer.normalize(train_targets).reshape(-1),
        0,
    )
    writer.add_histogram("Data/validation_targets_physical_m", validation_targets.reshape(-1), 0)
    writer.add_histogram(
        "Data/validation_targets_normalized",
        normalizer.normalize(validation_targets).reshape(-1),
        0,
    )
    history: list[dict[str, Any]] = []
    validation_indices = np.arange(validation_dataset.sample_count, dtype=np.int64)
    calibration_indices, calibrated_evaluation_indices = validation_calibration_indices(validation_dataset)
    validation_modes = sorted(set(validation_dataset.target_motion_modes.tolist()))
    try:
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args.model,
                normalizer,
            )
            validation = evaluate_subset(
                model,
                validation_inputs,
                validation_targets,
                validation_dataset,
                validation_indices,
                device,
                args.model,
                normalizer,
                args.num_samples,
                args.sampling_steps,
                args.evaluation_sampling_seed,
                constraints,
                validation_action_conditions,
            )
            calibration_candidates = sample_physical_candidates(
                model,
                validation_inputs,
                calibration_indices,
                device,
                args.model,
                normalizer,
                args.num_samples,
                args.sampling_steps,
                args.evaluation_sampling_seed + 2,
                validation_action_conditions,
            )
            calibration_targets = validation_targets.index_select(
                0, torch.as_tensor(calibration_indices, dtype=torch.long)
            ).to(device)
            radius = conformal_radius(
                conformal_nonconformity(calibration_candidates, calibration_targets),
                coverage=0.90,
            )
            calibrated_validation = evaluate_subset(
                model,
                validation_inputs,
                validation_targets,
                validation_dataset,
                calibrated_evaluation_indices,
                device,
                args.model,
                normalizer,
                args.num_samples,
                args.sampling_steps,
                args.evaluation_sampling_seed + 3,
                constraints,
                validation_action_conditions,
                calibration_radius=radius,
            )
            calibrated_record = {f"calibrated_{key}": value for key, value in calibrated_validation.items()}
            for key, value in calibrated_record.items():
                if np.isfinite(value):
                    writer.add_scalar(f"Calibration/{key}", value, epoch)
            per_mode: dict[str, dict[str, float]] = {}
            for mode_index, mode in enumerate(validation_modes):
                mode_indices = np.flatnonzero(validation_dataset.target_motion_modes == mode).astype(np.int64)
                mode_metrics = evaluate_subset(
                    model,
                    validation_inputs,
                    validation_targets,
                    validation_dataset,
                    mode_indices,
                    device,
                    args.model,
                    normalizer,
                    args.num_samples,
                    args.sampling_steps,
                    args.evaluation_sampling_seed + 10_000 * (mode_index + 1),
                    constraints,
                    validation_action_conditions,
                )
                per_mode[mode] = mode_metrics
                for key, value in mode_metrics.items():
                    if np.isfinite(value):
                        writer.add_scalar(f"ValidationByMode/{mode}/{key}", value, epoch)
            latency = measure_single_sample_latency(
                model,
                validation_inputs[0],
                device,
                args.model,
                (
                    validation_action_conditions[0]
                    if validation_action_conditions.shape[-1] > 0
                    else None
                ),
                args.num_samples,
                args.sampling_steps,
                args.latency_warmup,
                args.latency_repeats,
            )
            elapsed = time.perf_counter() - started
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                **validation,
                **calibrated_record,
                **latency,
                "epoch_seconds": elapsed,
                "per_target_motion_mode": per_mode,
            }
            history.append(record)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/validation", validation["loss"], epoch)
            for key, value in validation.items():
                if key != "loss" and np.isfinite(value):
                    writer.add_scalar(f"Metrics/{key}", value, epoch)
            writer.add_scalar("Timing/epoch_seconds", elapsed, epoch)
            for key, value in latency.items():
                writer.add_scalar(f"Timing/{key}", value, epoch)
            writer.add_scalar("Optimization/learning_rate", args.learning_rate, epoch)
            if epoch == 1 or epoch % 5 == 0:
                for name, parameter in model.named_parameters():
                    writer.add_histogram(f"Parameters/{name}", parameter.detach().cpu(), epoch)
            writer.flush()
    finally:
        writer.close()
    final = history[-1]
    with SummaryWriter(log_dir=str(output / "tensorboard_hparams")) as hparam_writer:
        hparam_writer.add_hparams(
            {
                "model": args.model,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "learning_rate": args.learning_rate,
                "target_normalization": args.target_normalization,
                "legacy_target_scale": args.target_scale if args.target_normalization == "fixed_scale" else 0.0,
                "diffusion_steps": args.diffusion_steps,
                "sampling_steps": args.sampling_steps,
                "num_samples": args.num_samples,
                "action_conditioning": args.action_conditioning,
            },
            {
                "hparam/final_loss": float(final["loss"]),
                "hparam/model_min_ade": float(final["model_min_ade"]),
                "hparam/model_min_fde": float(final["model_min_fde"]),
                "hparam/constant_velocity_min_ade": float(final["constant_velocity_min_ade"]),
                "hparam/single_sample_latency_p95_ms": float(final["single_sample_latency_p95_ms"]),
                "hparam/candidate_feasible_fraction": float(final.get("candidate_feasible_fraction", float("nan"))),
                "hparam/calibrated_coverage_full_trajectory": float(
                    final.get("calibrated_coverage_full_trajectory", float("nan"))
                ),
            },
        )
    output.joinpath("training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = {
        "model_kind": args.model,
        "model_backend": (
            "official_state_spaces_s4_dplr"
            if args.model == "official_s4_diffusion"
            else
            "s4_dplr_dense_reference"
            if args.model == "s4_diffusion"
            else "portable_diagonal_ssm"
            if args.model == "diffusion"
            else "gru_gaussian"
        ),
        "model_config": (
            model.model_config
            if args.model in {"diffusion", "s4_diffusion", "official_s4_diffusion"}
            else {
                "input_dim": input_dim,
                "horizon_count": train_dataset.horizon_steps,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "action_condition_dim": action_condition_dim,
            }
        ),
        "target_normalizer": normalizer.as_dict(),
        "candidate_feasibility_constraints": constraints.as_dict(),
        "dt_seconds": float(train_dataset.dt_seconds),
        "seed": args.seed,
        "state_dict": model.state_dict(),
        "final_metrics": final,
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    print(json.dumps({"output": str(output), "device": str(device), "final": final}, indent=2))


if __name__ == "__main__":
    main()
