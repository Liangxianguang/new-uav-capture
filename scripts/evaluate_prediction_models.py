"""Evaluate a frozen prediction checkpoint on the locked test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

if os.name == "nt":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import yaml
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
    prediction_metrics,
    project_candidate_trajectories,
)
from encirclement3d.trajectory_dataset import PredictionDataset, load_prediction_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--locked-test-dataset", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Evaluation artifact directory. It may be separate from the training output.",
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        help="Read-only training artifact directory. Defaults to --output for legacy evaluations.",
    )
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--sampling-seed", type=int, default=745102)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--projection-iterations", type=int, default=4)
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Use the second half of validation episodes for selection metrics without reading locked-test data.",
    )
    parser.add_argument(
        "--official-s4-root",
        type=Path,
        help="Override the upstream state-spaces/s4 source root stored in an official checkpoint.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def source_hashes() -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "scripts" / "evaluate_prediction_models.py",
        PROJECT_ROOT / "scripts" / "train_prediction_models.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "prediction.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "trajectory_dataset.py",
    )
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def flatten_inputs(
    dataset: PredictionDataset,
    action_conditioning: str = "none",
) -> tuple[torch.Tensor, torch.Tensor]:
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
        if history_actions.ndim != 3 or history_actions.shape[:2] != inputs.shape[:2]:
            raise ValueError("history_action_features is incompatible with history_observations.")
        inputs = torch.cat([inputs, history_actions], dim=-1)
    if use_future:
        if dataset.future_action_conditions is None:
            raise ValueError("The dataset does not provide future_action_conditions.")
        action_conditions = torch.as_tensor(dataset.future_action_conditions, dtype=torch.float32)
        if action_conditions.ndim != 3 or action_conditions.shape[0] != inputs.shape[0]:
            raise ValueError("future_action_conditions is incompatible with the dataset.")
    else:
        action_conditions = torch.zeros(
            inputs.shape[0], dataset.horizon_steps, 0, dtype=torch.float32
        )
    return inputs, action_conditions


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    official_s4_root: Path | None = None,
) -> tuple[torch.nn.Module, str, TrajectoryNormalizer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Prediction checkpoint must contain a mapping.")
    model_kind = str(checkpoint["model_kind"])
    model_config = dict(checkpoint["model_config"])
    state_dict = checkpoint.get("state_dict", {})
    if "diffusion_steps" not in model_config and "betas" in state_dict:
        model_config["diffusion_steps"] = int(state_dict["betas"].shape[0])
    if model_kind == "s4_diffusion":
        if "state_dim" not in model_config and "encoder.log_real_eigenvalues" in state_dict:
            model_config["state_dim"] = int(state_dict["encoder.log_real_eigenvalues"].shape[-1])
        if "rank" not in model_config and "encoder.p_real" in state_dict:
            model_config["rank"] = int(state_dict["encoder.p_real"].shape[-1])
    if model_kind == "official_s4_diffusion":
        if official_s4_root is not None:
            model_config["official_s4_root"] = str(official_s4_root.resolve())
        if "state_dim" not in model_config and "encoder.kernels.0.A_real" in state_dict:
            model_config["state_dim"] = int(state_dict["encoder.kernels.0.A_real"].shape[-1] * 2)
        if "rank" not in model_config and "encoder.kernels.0.P" in state_dict:
            model_config["rank"] = int(state_dict["encoder.kernels.0.P"].shape[0])
        model_config.pop("official_s4_source_hashes", None)
    if model_kind == "gru":
        model = HistoryTargetPredictor(**model_config)
    elif model_kind == "diffusion":
        model = ConditionalDiffusionTrajectoryPredictor(**model_config)
    elif model_kind == "s4_diffusion":
        model = S4ConditionalDiffusionTrajectoryPredictor(**model_config)
    elif model_kind == "official_s4_diffusion":
        model = OfficialS4ConditionalDiffusionTrajectoryPredictor(**model_config)
    else:
        raise ValueError(f"Unsupported checkpoint model kind: {model_kind}")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    normalizer_config = checkpoint["target_normalizer"]
    normalizer = TrajectoryNormalizer(
        center=np.asarray(normalizer_config["center"], dtype=np.float32),
        scale=np.asarray(normalizer_config["scale"], dtype=np.float32),
        kind=str(normalizer_config.get("kind", "checkpoint")),
    )
    return model, model_kind, normalizer, checkpoint


def sample_candidates(
    model: torch.nn.Module,
    model_kind: str,
    normalizer: TrajectoryNormalizer,
    inputs: torch.Tensor,
    indices: np.ndarray,
    device: torch.device,
    num_samples: int,
    sampling_steps: int,
    seed: int,
    batch_size: int,
    action_conditions: torch.Tensor,
) -> torch.Tensor:
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("indices must select at least one sample.")
    if batch_size <= 0:
        raise ValueError("batch-size must be positive.")
    generator = torch.Generator(device=device.type).manual_seed(seed)
    chunks: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, indices.size, batch_size):
            selected = torch.as_tensor(indices[start : start + batch_size], dtype=torch.long)
            batch_inputs = inputs.index_select(0, selected).to(device)
            selected_actions = action_conditions.index_select(0, selected).to(device)
            action_condition = selected_actions if selected_actions.shape[-1] > 0 else None
            if model_kind == "gru":
                mean, _log_variance = model(batch_inputs, action_condition)
                candidates = mean[:, None]
            else:
                candidates = model.sample_set(
                    batch_inputs,
                    num_samples=num_samples,
                    sampling_steps=sampling_steps,
                    generator=generator,
                    action_condition=action_condition,
                ).trajectories
            chunks.append(normalizer.denormalize(candidates).cpu())
    return torch.cat(chunks, dim=0)


def validation_calibration_indices(dataset: PredictionDataset) -> tuple[np.ndarray, np.ndarray]:
    episode_seeds = np.unique(dataset.episode_seeds)
    if episode_seeds.size < 2:
        raise ValueError("Validation calibration requires at least two episodes.")
    count = max(1, episode_seeds.size // 2)
    calibration = np.flatnonzero(np.isin(dataset.episode_seeds, episode_seeds[:count])).astype(np.int64)
    evaluation = np.flatnonzero(np.isin(dataset.episode_seeds, episode_seeds[count:])).astype(np.int64)
    return calibration, evaluation


def candidate_constraints(checkpoint: dict[str, Any]) -> dict[str, Any]:
    values = checkpoint.get("candidate_feasibility_constraints", {})
    return {
        "max_speed": values.get("max_speed_mps"),
        "max_acceleration": values.get("max_acceleration_mps2"),
        "minimum_clearance": float(values.get("minimum_obstacle_clearance_m", 0.0)),
    }


def assess(
    candidates: torch.Tensor,
    dataset: PredictionDataset,
    indices: np.ndarray,
    constraints: dict[str, Any],
) -> dict[str, float]:
    if not dataset.has_geometry_context or constraints["max_speed"] is None:
        return {"candidate_feasibility_available": 0.0}
    selected = torch.as_tensor(indices, dtype=torch.long)
    feasibility = assess_candidate_feasibility(
        candidates,
        torch.as_tensor(dataset.reference_positions, dtype=torch.float32).index_select(0, selected),
        torch.as_tensor(dataset.reference_velocities, dtype=torch.float32).index_select(0, selected),
        dataset.dt_seconds,
        torch.as_tensor(dataset.world_lower_bounds, dtype=torch.float32).index_select(0, selected),
        torch.as_tensor(dataset.world_upper_bounds, dtype=torch.float32).index_select(0, selected),
        float(constraints["max_speed"]),
        None if constraints["max_acceleration"] is None else float(constraints["max_acceleration"]),
        torch.as_tensor(dataset.obstacle_centers_xy, dtype=torch.float32).index_select(0, selected),
        torch.as_tensor(dataset.obstacle_radii, dtype=torch.float32).index_select(0, selected),
        torch.as_tensor(dataset.obstacle_heights, dtype=torch.float32).index_select(0, selected),
        torch.as_tensor(dataset.obstacle_half_extents_xy, dtype=torch.float32).index_select(0, selected),
        torch.as_tensor(dataset.obstacle_shape_codes, dtype=torch.int8).index_select(0, selected),
        float(constraints["minimum_clearance"]),
    )
    return feasibility.metrics()


def metrics(
    candidates: torch.Tensor,
    dataset: PredictionDataset,
    indices: np.ndarray,
    constraints: dict[str, Any],
    radius: float,
) -> dict[str, float]:
    selected = torch.as_tensor(indices, dtype=torch.long)
    targets = torch.as_tensor(dataset.future_target_displacements, dtype=torch.float32).index_select(0, selected)
    result = prediction_metrics(candidates, targets)
    result["energy_score"] = candidate_energy_score(candidates, targets)
    result.update(assess(candidates, dataset, indices, constraints))
    coverage = conformal_coverage(candidates, targets, radius)
    result.update(coverage)
    result["conformal_full_trajectory_coverage_error"] = abs(
        float(coverage["coverage_full_trajectory"]) - 0.90
    )
    result.update(branch_metrics(candidates, dataset, indices))
    result["calibration_radius_m"] = float(radius)
    return result


def branch_metrics(
    candidates: torch.Tensor,
    dataset: PredictionDataset,
    indices: np.ndarray,
) -> dict[str, float]:
    """Measure inference after a hidden target branch decision is made.

    Diffusion candidates have uniform uncalibrated weights, so the vote and
    coverage values below are not probability-calibration metrics.
    """

    if not dataset.has_branch_labels:
        return {"branch_metrics_available": 0.0}
    selected = torch.as_tensor(indices, dtype=torch.long)
    signs = torch.as_tensor(dataset.target_branch_signs, dtype=torch.int8).index_select(0, selected)
    decision_steps = torch.as_tensor(
        dataset.target_branch_decision_steps, dtype=torch.long
    ).index_select(0, selected)
    timesteps = torch.as_tensor(dataset.timesteps, dtype=torch.long).index_select(0, selected)
    eligible = timesteps >= decision_steps
    eligible_count = int(eligible.sum().item())
    result: dict[str, float] = {
        "branch_metrics_available": 1.0,
        "branch_eligible_sample_count": float(eligible_count),
        "branch_eligible_fraction": float(eligible.float().mean().item()),
    }
    if eligible_count == 0:
        result.update(
            {
                "branch_top1_accuracy": float("nan"),
                "branch_uniform_vote_accuracy": float("nan"),
                "branch_any_candidate_coverage": float("nan"),
                "branch_bimodal_candidate_fraction": float("nan"),
            }
        )
        return result
    reference_y = torch.as_tensor(dataset.reference_positions, dtype=torch.float32).index_select(0, selected)[:, 1]
    final_y = candidates[:, :, -1, 1] + reference_y[:, None]
    candidate_signs = torch.where(final_y >= 0.0, 1, -1).to(torch.int8)
    eligible_signs = signs[eligible]
    eligible_candidates = candidate_signs[eligible]
    top1 = eligible_candidates[:, 0]
    vote = torch.where(eligible_candidates.sum(dim=1) >= 0, 1, -1).to(torch.int8)
    matches = eligible_candidates == eligible_signs[:, None]
    result.update(
        {
            "branch_top1_accuracy": float((top1 == eligible_signs).float().mean().item()),
            "branch_uniform_vote_accuracy": float((vote == eligible_signs).float().mean().item()),
            "branch_any_candidate_coverage": float(matches.any(dim=1).float().mean().item()),
            "branch_bimodal_candidate_fraction": float(
                ((eligible_candidates.min(dim=1).values < 0) & (eligible_candidates.max(dim=1).values > 0))
                .float()
                .mean()
                .item()
            ),
        }
    )
    return result


def mode_metrics(
    candidates: torch.Tensor,
    dataset: PredictionDataset,
    indices: np.ndarray,
    constraints: dict[str, Any],
    radius: float,
) -> dict[str, dict[str, float]]:
    modes = sorted(set(dataset.target_motion_modes[indices].tolist()))
    result: dict[str, dict[str, float]] = {}
    for mode in modes:
        mode_indices = indices[dataset.target_motion_modes[indices] == mode]
        local_positions = {int(value): offset for offset, value in enumerate(indices.tolist())}
        local_indices = np.asarray([local_positions[int(value)] for value in mode_indices], dtype=np.int64)
        result[mode] = metrics(candidates[torch.as_tensor(local_indices)], dataset, mode_indices, constraints, radius)
    return result


def measure_latency(
    model: torch.nn.Module,
    model_kind: str,
    normalizer: TrajectoryNormalizer,
    inputs: torch.Tensor,
    device: torch.device,
    num_samples: int,
    sampling_steps: int,
    action_conditions: torch.Tensor,
) -> dict[str, float]:
    sample = inputs[:1].to(device)
    sample_actions = action_conditions[:1].to(device)
    action_condition = sample_actions if sample_actions.shape[-1] > 0 else None
    generator = torch.Generator(device=device.type).manual_seed(99173)

    def run() -> None:
        with torch.no_grad():
            if model_kind == "gru":
                mean, _ = model(sample, action_condition)
                normalizer.denormalize(mean[:, None])
            else:
                model.sample_set(
                    sample,
                    num_samples=num_samples,
                    sampling_steps=sampling_steps,
                    generator=generator,
                    action_condition=action_condition,
                )

    for _ in range(5):
        run()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    values: list[float] = []
    for _ in range(32):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        run()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        values.append((time.perf_counter() - started) * 1000.0)
    return {
        "single_sample_latency_p50_ms": float(np.percentile(values, 50)),
        "single_sample_latency_p95_ms": float(np.percentile(values, 95)),
        "single_sample_latency_p99_ms": float(np.percentile(values, 99)),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.projection_iterations <= 0:
        raise ValueError("batch-size and projection-iterations must be positive.")
    if not args.validation_only and args.locked_test_dataset is None:
        raise ValueError("--locked-test-dataset is required unless --validation-only is used.")
    if args.validation_only and args.locked_test_dataset is not None:
        raise ValueError("--validation-only must not be combined with --locked-test-dataset.")
    output = args.output.resolve()
    training_output = (args.training_output or output).resolve()
    if not training_output.joinpath("metadata.json").is_file():
        raise FileNotFoundError(f"Training metadata is missing: {training_output / 'metadata.json'}")
    if output != training_output and output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty evaluation output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    validation = load_prediction_dataset(str(args.validation_dataset.resolve()))
    locked_test = None if args.validation_only else load_prediction_dataset(str(args.locked_test_dataset.resolve()))
    model, model_kind, normalizer, checkpoint = load_model(
        args.checkpoint.resolve(), device, args.official_s4_root
    )
    model_args = json.loads(json.dumps(checkpoint.get("final_metrics", {}), allow_nan=True))
    train_config = json.loads(training_output.joinpath("metadata.json").read_text(encoding="utf-8"))
    configured_args = train_config.get("arguments", {})
    action_conditioning = str(configured_args.get("action_conditioning", "none"))
    validation_inputs, validation_action_conditions = flatten_inputs(validation, action_conditioning)
    evaluation_dataset = validation if args.validation_only else locked_test
    assert evaluation_dataset is not None
    test_inputs, test_action_conditions = flatten_inputs(evaluation_dataset, action_conditioning)
    model_input_dim = int(checkpoint["model_config"]["input_dim"])
    if validation_inputs.shape[-1] != model_input_dim or test_inputs.shape[-1] != model_input_dim:
        raise ValueError(
            "Dataset input width does not match the checkpoint: "
            f"model={model_input_dim}, validation={validation_inputs.shape[-1]}, "
            f"evaluation={test_inputs.shape[-1]}."
        )
    model_action_dim = int(checkpoint["model_config"].get("action_condition_dim", 0))
    if validation_action_conditions.shape[-1] != model_action_dim:
        raise ValueError(
            "Dataset action-condition width does not match the checkpoint: "
            f"model={model_action_dim}, validation={validation_action_conditions.shape[-1]}."
        )
    if test_action_conditions.shape[-1] != model_action_dim:
        raise ValueError(
            "Dataset action-condition width does not match the checkpoint: "
            f"model={model_action_dim}, evaluation={test_action_conditions.shape[-1]}."
        )
    num_samples = int(args.num_samples if args.num_samples is not None else configured_args.get("num_samples", 8))
    sampling_steps = int(
        args.sampling_steps if args.sampling_steps is not None else configured_args.get("sampling_steps", 8)
    )
    constraints = candidate_constraints(checkpoint)
    calibration_indices, validation_evaluation_indices = validation_calibration_indices(validation)
    validation_calibration_candidates = sample_candidates(
        model,
        model_kind,
        normalizer,
        validation_inputs,
        calibration_indices,
        device,
        num_samples,
        sampling_steps,
        args.sampling_seed + 2,
        args.batch_size,
        validation_action_conditions,
    )
    calibration_targets = torch.as_tensor(validation.future_target_displacements).index_select(
        0, torch.as_tensor(calibration_indices, dtype=torch.long)
    )
    raw_radius = conformal_radius(
        conformal_nonconformity(validation_calibration_candidates, calibration_targets), coverage=0.90
    )
    projected_calibration_candidates = project_candidate_trajectories(
        validation_calibration_candidates,
        torch.as_tensor(validation.reference_positions).index_select(0, torch.as_tensor(calibration_indices)),
        torch.as_tensor(validation.reference_velocities).index_select(0, torch.as_tensor(calibration_indices)),
        validation.dt_seconds,
        float(constraints["max_speed"]),
        None if constraints["max_acceleration"] is None else float(constraints["max_acceleration"]),
        torch.as_tensor(validation.world_lower_bounds).index_select(0, torch.as_tensor(calibration_indices)),
        torch.as_tensor(validation.world_upper_bounds).index_select(0, torch.as_tensor(calibration_indices)),
        args.projection_iterations,
    )
    projected_radius = conformal_radius(
        conformal_nonconformity(projected_calibration_candidates, calibration_targets), coverage=0.90
    )
    test_indices = (
        validation_evaluation_indices
        if args.validation_only
        else np.arange(evaluation_dataset.sample_count, dtype=np.int64)
    )
    raw_candidates = sample_candidates(
        model,
        model_kind,
        normalizer,
        test_inputs,
        test_indices,
        device,
        num_samples,
        sampling_steps,
        args.sampling_seed,
        args.batch_size,
        test_action_conditions,
    )
    projected_candidates = project_candidate_trajectories(
        raw_candidates,
        torch.as_tensor(evaluation_dataset.reference_positions).index_select(
            0, torch.as_tensor(test_indices, dtype=torch.long)
        ),
        torch.as_tensor(evaluation_dataset.reference_velocities).index_select(
            0, torch.as_tensor(test_indices, dtype=torch.long)
        ),
        evaluation_dataset.dt_seconds,
        float(constraints["max_speed"]),
        None if constraints["max_acceleration"] is None else float(constraints["max_acceleration"]),
        torch.as_tensor(evaluation_dataset.world_lower_bounds).index_select(
            0, torch.as_tensor(test_indices, dtype=torch.long)
        ),
        torch.as_tensor(evaluation_dataset.world_upper_bounds).index_select(
            0, torch.as_tensor(test_indices, dtype=torch.long)
        ),
        args.projection_iterations,
    )
    raw_metrics = metrics(raw_candidates, evaluation_dataset, test_indices, constraints, raw_radius)
    projected_metrics = metrics(projected_candidates, evaluation_dataset, test_indices, constraints, projected_radius)
    raw_modes = mode_metrics(raw_candidates, evaluation_dataset, test_indices, constraints, raw_radius)
    projected_modes = mode_metrics(projected_candidates, evaluation_dataset, test_indices, constraints, projected_radius)
    latency = measure_latency(
        model,
        model_kind,
        normalizer,
        test_inputs,
        device,
        num_samples,
        sampling_steps,
        test_action_conditions,
    )
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "training_output": str(training_output),
        "evaluation_output": str(output),
        "validation_dataset": str(args.validation_dataset.resolve()),
        "evaluation_split": "validation_second_half_episode_seeds" if args.validation_only else "locked_test",
        "locked_test_dataset": None if args.validation_only else str(args.locked_test_dataset.resolve()),
        "model_kind": model_kind,
        "model_backend": checkpoint.get("model_backend"),
        "action_conditioning": action_conditioning,
        "history_action_feature_dim": int(
            validation.history_action_features.shape[-1]
            if validation.history_action_features is not None
            else 0
        ),
        "future_action_condition_dim": int(validation_action_conditions.shape[-1]),
        "device": str(device),
        "sampling_seed": args.sampling_seed,
        "num_samples": num_samples,
        "sampling_steps": sampling_steps,
        "projection_iterations": args.projection_iterations,
        "calibration": {
            "source": "validation_first_half_episode_seeds_only",
            "raw_radius_m": raw_radius,
            "projected_radius_m": projected_radius,
            "calibration_sample_count": int(calibration_indices.size),
        },
        "raw": raw_metrics,
        "projected": projected_metrics,
        "raw_by_target_motion_mode": raw_modes,
        "projected_by_target_motion_mode": projected_modes,
        "latency": latency,
        "source_hashes": source_hashes(),
        "checkpoint_final_metrics": model_args,
    }
    artifact_stem = "validation_selection_metrics" if args.validation_only else "locked_test_metrics"
    config_stem = "validation_selection_config" if args.validation_only else "locked_test_config"
    output.joinpath(f"{artifact_stem}.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    output.joinpath(f"{config_stem}.yaml").write_text(
        yaml.safe_dump({key: value for key, value in result.items() if key not in {"raw_by_target_motion_mode", "projected_by_target_motion_mode"}}, sort_keys=False),
        encoding="utf-8",
    )
    with SummaryWriter(log_dir=str(output / "tensorboard_locked_test"), flush_secs=10) as writer:
        writer.add_text("Evaluation/config", yaml.safe_dump(result["calibration"], sort_keys=False), 0)
        for prefix, values in (("Raw", raw_metrics), ("Projected", projected_metrics)):
            for key, value in values.items():
                if np.isfinite(value):
                    writer.add_scalar(f"LockedTest/{prefix}/{key}", value, 0)
        for key, value in latency.items():
            writer.add_scalar(f"LockedTest/Latency/{key}", value, 0)
        for mode, values in raw_modes.items():
            for key, value in values.items():
                if np.isfinite(value):
                    writer.add_scalar(f"LockedTest/RawByMode/{mode}/{key}", value, 0)
        for mode, values in projected_modes.items():
            for key, value in values.items():
                if np.isfinite(value):
                    writer.add_scalar(f"LockedTest/ProjectedByMode/{mode}/{key}", value, 0)
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
