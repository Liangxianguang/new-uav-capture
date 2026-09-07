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
    parser.add_argument("--locked-test-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Existing training output directory.")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--sampling-seed", type=int, default=745102)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--projection-iterations", type=int, default=4)
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


def flatten_inputs(dataset: PredictionDataset) -> torch.Tensor:
    inputs = torch.as_tensor(dataset.history_observations, dtype=torch.float32)
    return inputs.reshape(inputs.shape[0], inputs.shape[1], -1)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, str, TrajectoryNormalizer, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Prediction checkpoint must contain a mapping.")
    model_kind = str(checkpoint["model_kind"])
    model_config = checkpoint["model_config"]
    if model_kind == "gru":
        model = HistoryTargetPredictor(**model_config)
    elif model_kind == "diffusion":
        model = ConditionalDiffusionTrajectoryPredictor(**model_config)
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
            if model_kind == "gru":
                mean, _log_variance = model(batch_inputs)
                candidates = mean[:, None]
            else:
                candidates = model.sample_set(
                    batch_inputs,
                    num_samples=num_samples,
                    sampling_steps=sampling_steps,
                    generator=generator,
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
    result.update(conformal_coverage(candidates, targets, radius))
    result["calibration_radius_m"] = float(radius)
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
) -> dict[str, float]:
    sample = inputs[:1].to(device)
    generator = torch.Generator(device=device.type).manual_seed(99173)

    def run() -> None:
        with torch.no_grad():
            if model_kind == "gru":
                mean, _ = model(sample)
                normalizer.denormalize(mean[:, None])
            else:
                model.sample_set(sample, num_samples=num_samples, sampling_steps=sampling_steps, generator=generator)

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
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    validation = load_prediction_dataset(str(args.validation_dataset.resolve()))
    locked_test = load_prediction_dataset(str(args.locked_test_dataset.resolve()))
    validation_inputs = flatten_inputs(validation)
    test_inputs = flatten_inputs(locked_test)
    model, model_kind, normalizer, checkpoint = load_model(args.checkpoint.resolve(), device)
    model_args = json.loads(json.dumps(checkpoint.get("final_metrics", {}), allow_nan=True))
    train_config = json.loads(output.joinpath("metadata.json").read_text(encoding="utf-8"))
    configured_args = train_config.get("arguments", {})
    num_samples = int(args.num_samples if args.num_samples is not None else configured_args.get("num_samples", 8))
    sampling_steps = int(
        args.sampling_steps if args.sampling_steps is not None else configured_args.get("sampling_steps", 8)
    )
    constraints = candidate_constraints(checkpoint)
    calibration_indices, _evaluation_indices = validation_calibration_indices(validation)
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
    test_indices = np.arange(locked_test.sample_count, dtype=np.int64)
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
    )
    projected_candidates = project_candidate_trajectories(
        raw_candidates,
        torch.as_tensor(locked_test.reference_positions),
        torch.as_tensor(locked_test.reference_velocities),
        locked_test.dt_seconds,
        float(constraints["max_speed"]),
        None if constraints["max_acceleration"] is None else float(constraints["max_acceleration"]),
        torch.as_tensor(locked_test.world_lower_bounds),
        torch.as_tensor(locked_test.world_upper_bounds),
        args.projection_iterations,
    )
    raw_metrics = metrics(raw_candidates, locked_test, test_indices, constraints, raw_radius)
    projected_metrics = metrics(projected_candidates, locked_test, test_indices, constraints, projected_radius)
    raw_modes = mode_metrics(raw_candidates, locked_test, test_indices, constraints, raw_radius)
    projected_modes = mode_metrics(projected_candidates, locked_test, test_indices, constraints, projected_radius)
    latency = measure_latency(model, model_kind, normalizer, test_inputs, device, num_samples, sampling_steps)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_dataset": str(args.validation_dataset.resolve()),
        "locked_test_dataset": str(args.locked_test_dataset.resolve()),
        "model_kind": model_kind,
        "model_backend": checkpoint.get("model_backend"),
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
    output.joinpath("locked_test_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    output.joinpath("locked_test_config.yaml").write_text(
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
