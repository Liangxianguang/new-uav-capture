"""Evaluate an observation-only constant-velocity prediction baseline.

The baseline advances the public team-belief velocity stored in a predictor
dataset.  It deliberately never reads a target-truth velocity, so it is a fair
offline comparator for action-conditioned learned predictors.
"""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.prediction import (  # noqa: E402
    conformal_nonconformity,
    conformal_radius,
    project_candidate_trajectories,
)
from encirclement3d.trajectory_dataset import PredictionDataset, load_prediction_dataset  # noqa: E402
from evaluate_prediction_models import (  # noqa: E402
    branch_metrics,
    metrics,
    validation_calibration_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--locked-test-dataset", type=Path)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projection-iterations", type=int, default=4)
    parser.add_argument("--latency-warmup", type=int, default=5)
    parser.add_argument("--latency-repeats", type=int, default=32)
    parser.add_argument("--validation-only", action="store_true")
    return parser.parse_args()


def source_hashes() -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "scripts" / "evaluate_constant_velocity_baseline.py",
        PROJECT_ROOT / "scripts" / "evaluate_prediction_models.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "prediction.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "trajectory_dataset.py",
    )
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def load_constraints(path: Path) -> dict[str, float]:
    document = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Environment config must be a mapping.")
    agents = document.get("agents", {})
    if not isinstance(agents, dict):
        raise ValueError("Environment config must contain an agents mapping.")
    values = {
        "max_speed": float(agents["target_max_speed"]),
        "max_acceleration": float(agents["target_max_acceleration"]),
        "minimum_clearance": float(agents["drone_radius"]),
    }
    if not np.isfinite(list(values.values())).all() or any(value <= 0.0 for value in values.values()):
        raise ValueError("Target constraints and drone radius must be positive and finite.")
    return values


def constant_velocity_candidates(dataset: PredictionDataset, indices: np.ndarray) -> torch.Tensor:
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("indices must select at least one sample.")
    selected = torch.as_tensor(indices, dtype=torch.long)
    velocities = torch.as_tensor(dataset.reference_velocities, dtype=torch.float32).index_select(0, selected)
    times = torch.arange(1, dataset.horizon_steps + 1, dtype=torch.float32)[None, :, None]
    return (velocities[:, None, :] * float(dataset.dt_seconds) * times)[:, None, :, :]


def project(
    candidates: torch.Tensor,
    dataset: PredictionDataset,
    indices: np.ndarray,
    constraints: dict[str, float],
    iterations: int,
) -> torch.Tensor:
    if not dataset.has_geometry_context:
        raise ValueError("Constant-velocity evaluation requires geometry context.")
    selected = torch.as_tensor(indices, dtype=torch.long)
    return project_candidate_trajectories(
        candidates,
        torch.as_tensor(dataset.reference_positions, dtype=torch.float32).index_select(0, selected),
        torch.as_tensor(dataset.reference_velocities, dtype=torch.float32).index_select(0, selected),
        dataset.dt_seconds,
        constraints["max_speed"],
        constraints["max_acceleration"],
        torch.as_tensor(dataset.world_lower_bounds, dtype=torch.float32).index_select(0, selected),
        torch.as_tensor(dataset.world_upper_bounds, dtype=torch.float32).index_select(0, selected),
        iterations,
    )


def measure_latency(dataset: PredictionDataset, repeats: int, warmup: int) -> dict[str, float]:
    if repeats <= 0 or warmup < 0:
        raise ValueError("latency-repeats must be positive and latency-warmup non-negative.")
    indices = np.asarray([0], dtype=np.int64)
    for _ in range(warmup):
        constant_velocity_candidates(dataset, indices)
    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        constant_velocity_candidates(dataset, indices)
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "single_sample_latency_p50_ms": float(np.percentile(timings, 50)),
        "single_sample_latency_p95_ms": float(np.percentile(timings, 95)),
        "single_sample_latency_p99_ms": float(np.percentile(timings, 99)),
    }


def evaluate(
    dataset: PredictionDataset,
    indices: np.ndarray,
    constraints: dict[str, float],
    raw_radius: float,
    projected_radius: float,
    iterations: int,
) -> tuple[dict[str, float], dict[str, float]]:
    raw = constant_velocity_candidates(dataset, indices)
    projected = project(raw, dataset, indices, constraints, iterations)
    return (
        metrics(raw, dataset, indices, constraints, raw_radius),
        metrics(projected, dataset, indices, constraints, projected_radius),
    )


def main() -> None:
    args = parse_args()
    if args.projection_iterations <= 0:
        raise ValueError("projection-iterations must be positive.")
    if not args.validation_only and args.locked_test_dataset is None:
        raise ValueError("--locked-test-dataset is required unless --validation-only is used.")
    if args.validation_only and args.locked_test_dataset is not None:
        raise ValueError("--validation-only must not be combined with --locked-test-dataset.")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    validation = load_prediction_dataset(str(args.validation_dataset.resolve()))
    evaluation_dataset = (
        validation
        if args.validation_only
        else load_prediction_dataset(str(args.locked_test_dataset.resolve()))
    )
    constraints = load_constraints(args.environment_config)
    calibration_indices, validation_evaluation_indices = validation_calibration_indices(validation)
    calibration_candidates = constant_velocity_candidates(validation, calibration_indices)
    calibration_targets = torch.as_tensor(validation.future_target_displacements).index_select(
        0, torch.as_tensor(calibration_indices, dtype=torch.long)
    )
    raw_radius = conformal_radius(conformal_nonconformity(calibration_candidates, calibration_targets), coverage=0.90)
    projected_calibration = project(
        calibration_candidates, validation, calibration_indices, constraints, args.projection_iterations
    )
    projected_radius = conformal_radius(
        conformal_nonconformity(projected_calibration, calibration_targets), coverage=0.90
    )
    evaluation_indices = (
        validation_evaluation_indices
        if args.validation_only
        else np.arange(evaluation_dataset.sample_count, dtype=np.int64)
    )
    raw_metrics, projected_metrics = evaluate(
        evaluation_dataset,
        evaluation_indices,
        constraints,
        raw_radius,
        projected_radius,
        args.projection_iterations,
    )
    result: dict[str, Any] = {
        "model_kind": "constant_velocity_observation_only",
        "model_backend": "public_team_belief_velocity",
        "evaluation_split": "validation_second_half_episode_seeds" if args.validation_only else "locked_test",
        "validation_dataset": str(args.validation_dataset.resolve()),
        "locked_test_dataset": None if args.validation_only else str(args.locked_test_dataset.resolve()),
        "environment_config": str(args.environment_config.resolve()),
        "baseline_contract": {
            "position_origin": "public team belief reference stored in dataset.reference_positions",
            "velocity": "public team belief velocity stored in dataset.reference_velocities",
            "target_truth_velocity_used": False,
        },
        "candidate_feasibility_constraints": constraints,
        "num_samples": 1,
        "projection_iterations": args.projection_iterations,
        "calibration": {
            "source": "validation_first_half_episode_seeds_only",
            "raw_radius_m": raw_radius,
            "projected_radius_m": projected_radius,
            "calibration_sample_count": int(calibration_indices.size),
        },
        "raw": raw_metrics,
        "projected": projected_metrics,
        "latency": measure_latency(validation, args.latency_repeats, args.latency_warmup),
        "source_hashes": source_hashes(),
    }
    stem = "validation_selection_metrics" if args.validation_only else "locked_test_metrics"
    output.joinpath(f"{stem}.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
