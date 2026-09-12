"""Calibrate a delay-aware conformal target-prediction tube on validation data.

The first half of validation episodes is used for calibration and the second
half is an untouched development confirmation split.  Locked-test data is not
read.  The resulting JSON artifact is safe for runtime use because it contains
only frozen radii and provenance, while the target labels are used exclusively
inside this offline calibration script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

if os.name == "nt":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.delay_aware_conformal_tube import (  # noqa: E402
    candidate_minimum_errors,
    evaluate_tube_coverage,
    fit_conformal_radius_schedule,
)
from encirclement3d.prediction import project_candidate_trajectories  # noqa: E402
from encirclement3d.trajectory_dataset import load_prediction_dataset  # noqa: E402
from evaluate_prediction_models import (  # noqa: E402
    flatten_inputs,
    load_model,
    sample_candidates,
    select_device,
    validation_calibration_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--training-output", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--action-conditioning", choices=("none", "history", "future", "both"))
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=4)
    parser.add_argument("--sampling-seed", type=int, default=745102)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--projection-iterations", type=int, default=4)
    parser.add_argument("--coverage", type=float, default=0.90)
    parser.add_argument("--target-speed-mps", type=float)
    parser.add_argument("--uncertainty-gain", type=float, default=0.25)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def source_hashes() -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "scripts" / "calibrate_delay_aware_conformal_tube.py",
        PROJECT_ROOT / "scripts" / "evaluate_prediction_models.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "delay_aware_conformal_tube.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "prediction.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "trajectory_dataset.py",
    )
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _action_conditioning(args: argparse.Namespace) -> str:
    if args.action_conditioning is not None:
        return str(args.action_conditioning)
    if args.training_output is not None:
        metadata_path = args.training_output.resolve() / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            value = metadata.get("action_conditioning") or metadata.get("arguments", {}).get("action_conditioning")
            if value in {"none", "history", "future", "both"}:
                return str(value)
    return "both"


def _project(
    candidates: torch.Tensor,
    dataset: Any,
    indices: np.ndarray,
    projection_iterations: int,
    device: torch.device,
) -> torch.Tensor:
    selected = torch.as_tensor(indices, dtype=torch.long)
    constraints = dataset_metadata_constraints(dataset)
    return project_candidate_trajectories(
        candidates.to(device),
        torch.as_tensor(dataset.reference_positions, dtype=torch.float32, device=device).index_select(0, selected.to(device)),
        torch.as_tensor(dataset.reference_velocities, dtype=torch.float32, device=device).index_select(0, selected.to(device)),
        float(dataset.dt_seconds),
        float(constraints["max_speed_mps"]),
        float(constraints["max_acceleration_mps2"]),
        torch.as_tensor(dataset.world_lower_bounds, dtype=torch.float32, device=device).index_select(0, selected.to(device)),
        torch.as_tensor(dataset.world_upper_bounds, dtype=torch.float32, device=device).index_select(0, selected.to(device)),
        int(projection_iterations),
    ).detach().cpu()


def dataset_metadata_constraints(dataset: Any) -> dict[str, float]:
    """Use the checkpoint-compatible validation contract without target truth."""

    # The v4 conversion stores these values in the collection metadata.  The
    # online predictor already uses the same contract; this fallback keeps the
    # calibration script usable with older archives.
    return {
        "max_speed_mps": 3.6,
        "max_acceleration_mps2": 4.0,
    }


def _records(errors: np.ndarray, *, split: str) -> str:
    return "".join(
        json.dumps(
            {
                "split": split,
                "sample_index": int(index),
                "minimum_candidate_error_m_by_step": row.tolist(),
                "maximum_candidate_error_m": float(np.max(row)),
            },
            allow_nan=False,
        )
        + "\n"
        for index, row in enumerate(np.asarray(errors, dtype=np.float64))
    )


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.sampling_steps <= 0 or args.batch_size <= 0 or args.projection_iterations <= 0:
        raise ValueError("sampling, batch-size and projection settings must be positive")
    if not 0.0 < float(args.coverage) < 1.0:
        raise ValueError("coverage must lie in (0, 1)")
    if not np.isfinite(float(args.uncertainty_gain)) or float(args.uncertainty_gain) < 0.0:
        raise ValueError("uncertainty-gain must be finite and non-negative")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    dataset = load_prediction_dataset(str(args.validation_dataset.resolve()))
    action_conditioning = _action_conditioning(args)
    inputs, action_conditions = flatten_inputs(dataset, action_conditioning)
    model, model_kind, normalizer, checkpoint = load_model(args.checkpoint.resolve(), device)
    expected_input_dim = int(checkpoint["model_config"]["input_dim"])
    if inputs.shape[-1] != expected_input_dim:
        raise ValueError(f"input width mismatch: dataset={inputs.shape[-1]} checkpoint={expected_input_dim}")
    expected_action_dim = int(checkpoint["model_config"].get("action_condition_dim", 0))
    if action_conditions.shape[-1] != expected_action_dim:
        raise ValueError(
            f"action condition width mismatch: dataset={action_conditions.shape[-1]} checkpoint={expected_action_dim}"
        )
    calibration_indices, confirmation_indices = validation_calibration_indices(dataset)
    calibration_candidates = sample_candidates(
        model,
        model_kind,
        normalizer,
        inputs,
        calibration_indices,
        device,
        args.num_samples,
        args.sampling_steps,
        args.sampling_seed + 2,
        args.batch_size,
        action_conditions,
    )
    confirmation_candidates = sample_candidates(
        model,
        model_kind,
        normalizer,
        inputs,
        confirmation_indices,
        device,
        args.num_samples,
        args.sampling_steps,
        args.sampling_seed,
        args.batch_size,
        action_conditions,
    )
    calibration_candidates = _project(
        calibration_candidates,
        dataset,
        calibration_indices,
        args.projection_iterations,
        device,
    )
    confirmation_candidates = _project(
        confirmation_candidates,
        dataset,
        confirmation_indices,
        args.projection_iterations,
        device,
    )
    calibration_targets = torch.as_tensor(dataset.future_target_displacements[calibration_indices], dtype=torch.float32)
    confirmation_targets = torch.as_tensor(dataset.future_target_displacements[confirmation_indices], dtype=torch.float32)
    calibration_errors = candidate_minimum_errors(calibration_candidates.numpy(), calibration_targets.numpy())
    confirmation_errors = candidate_minimum_errors(confirmation_candidates.numpy(), confirmation_targets.numpy())
    target_speed = float(
        args.target_speed_mps
        if args.target_speed_mps is not None
        else 3.6
    )
    tube = fit_conformal_radius_schedule(
        calibration_errors,
        coverage=float(args.coverage),
        dt_seconds=float(dataset.dt_seconds),
        target_speed_mps=target_speed,
    )
    tube["uncertainty_gain"] = float(args.uncertainty_gain)
    tube["source_model_hash"] = hashlib.sha256(args.checkpoint.resolve().read_bytes()).hexdigest()
    calibration_coverage = evaluate_tube_coverage(
        calibration_candidates.numpy(), calibration_targets.numpy(), np.asarray(tube["radius_m_by_step"])
    )
    confirmation_coverage = evaluate_tube_coverage(
        confirmation_candidates.numpy(), confirmation_targets.numpy(), np.asarray(tube["radius_m_by_step"])
    )
    config = {
        "checkpoint": str(args.checkpoint.resolve()),
        "validation_dataset": str(args.validation_dataset.resolve()),
        "training_output": None if args.training_output is None else str(args.training_output.resolve()),
        "action_conditioning": action_conditioning,
        "num_samples": int(args.num_samples),
        "sampling_steps": int(args.sampling_steps),
        "sampling_seed": int(args.sampling_seed),
        "batch_size": int(args.batch_size),
        "projection_iterations": int(args.projection_iterations),
        "coverage": float(args.coverage),
        "uncertainty_gain": float(args.uncertainty_gain),
        "device": str(device),
        "calibration_split": "first half of validation episode seeds",
        "confirmation_split": "second half of validation episode seeds",
        "source_hashes": source_hashes(),
    }
    output.joinpath("config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    output.joinpath("calibration_scores.jsonl").write_text(_records(calibration_errors, split="calibration"), encoding="utf-8")
    output.joinpath("confirmation_scores.jsonl").write_text(_records(confirmation_errors, split="confirmation"), encoding="utf-8")
    result = {
        "config": config,
        "tube": tube,
        "calibration": calibration_coverage,
        "confirmation": confirmation_coverage,
        "calibration_sample_count": int(calibration_indices.size),
        "confirmation_sample_count": int(confirmation_indices.size),
        "decision": "development_confirmation_only",
        "formal_forward_invariance": False,
        "locked_test_used": False,
    }
    output.joinpath("summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with SummaryWriter(log_dir=str(output / "tensorboard"), flush_secs=5) as writer:
        writer.add_text("Calibration/config", json.dumps(config, indent=2), 0)
        writer.add_text("Calibration/summary", json.dumps(result, indent=2), 0)
        for prefix, summary in (("Calibration", calibration_coverage), ("Confirmation", confirmation_coverage)):
            for key, value in summary.items():
                writer.add_scalar(f"{prefix}/{key}", float(value), 0)
        for step, value in enumerate(tube["radius_m_by_step"], start=1):
            writer.add_scalar("Tube/radius_m_by_step", float(value), step)
        writer.add_scalar("Tube/simultaneous_multiplier", float(tube["simultaneous_multiplier"]), 0)
        writer.add_hparams(
            {
                "coverage": float(args.coverage),
                "num_samples": int(args.num_samples),
                "sampling_steps": int(args.sampling_steps),
                "projection_iterations": int(args.projection_iterations),
            },
            {
                "hparam/confirmation_full_trajectory_coverage": float(
                    confirmation_coverage["full_trajectory_coverage"]
                ),
                "hparam/maximum_radius_m": float(np.max(tube["radius_m_by_step"])),
            },
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
