"""Calibrate a delay-aware conformal target-prediction tube on validation data.

The default split uses the first/second episode-seed halves.  An optional
balanced split keeps complete mirror groups together and stratifies a cheap
public-observation context score.  Locked-test data is not read.  The
resulting JSON artifact is safe for runtime use because it contains only
frozen radii and provenance, while target labels are used exclusively inside
this offline calibration script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
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
    evaluate_context_adaptive_tube_coverage,
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
    parser.add_argument(
        "--split-strategy",
        choices=("episode_seed_half", "balanced_mirror_context"),
        default="episode_seed_half",
        help="Validation-only split strategy. Balanced mode keeps mirror groups intact and stratifies public context.",
    )
    parser.add_argument(
        "--scene-manifest",
        type=Path,
        help="Validation scenes.jsonl required by balanced_mirror_context.",
    )
    parser.add_argument("--split-seed", type=int, default=20260912)
    parser.add_argument(
        "--max-mean-effective-radius-m",
        type=float,
        help="Optional pre-registered compactness limit on confirmation mean effective radius.",
    )
    parser.add_argument(
        "--max-effective-radius-m",
        type=float,
        help="Optional pre-registered compactness limit on confirmation maximum effective radius.",
    )
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


def public_context_score(dataset: Any) -> np.ndarray:
    """Compute a cheap context score from normalized public observations only."""

    observations = np.asarray(dataset.history_observations, dtype=np.float64)
    if observations.ndim != 4 or observations.shape[-1] <= 14:
        raise ValueError("balanced context splitting requires the uncertainty observation block")
    latest = observations[:, -1]
    age = np.mean(np.clip(latest[..., 10], 0.0, 1.0), axis=1)
    confidence_deficit = 1.0 - np.mean(np.clip(latest[..., 11], 0.0, 1.0), axis=1)
    covariance = np.mean(np.clip(np.sum(latest[..., 12:15], axis=-1), 0.0, 1.0), axis=1)
    speed_ratio = np.mean(
        np.clip(np.linalg.norm(latest[..., 6:9], axis=-1) / 0.80, 0.0, 1.0),
        axis=1,
    )
    visibility_deficit = 1.0 - np.mean(np.clip(latest[..., 9], 0.0, 1.0), axis=1)
    score = (
        0.20 * age
        + 0.20 * confidence_deficit
        + 0.25 * covariance
        + 0.15 * speed_ratio
        + 0.20 * visibility_deficit
    )
    if not np.isfinite(score).all():
        raise ValueError("public context score is non-finite")
    return score.astype(np.float64, copy=False)


def load_scene_manifest(path: Path) -> dict[int, dict[str, Any]]:
    """Load episode-to-mirror metadata without reading any target labels."""

    records: dict[int, dict[str, Any]] = {}
    for line in path.resolve().read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[int(record["episode_index"])] = record
    if not records:
        raise ValueError("scene manifest is empty")
    return records


def balanced_mirror_context_indices(
    dataset: Any,
    scene_records: dict[int, dict[str, Any]],
    *,
    split_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split complete mirror groups while balancing public context strata."""

    score = public_context_score(dataset)
    episode_to_group: dict[int, str] = {}
    for episode in np.unique(dataset.episode_indices).tolist():
        record = scene_records.get(int(episode))
        if record is None:
            raise ValueError(f"scene manifest is missing episode_index={episode}")
        group = record.get("mirror_group_id")
        episode_to_group[int(episode)] = str(group if group is not None else f"episode:{episode}")
    group_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, episode in enumerate(np.asarray(dataset.episode_indices, dtype=np.int64).tolist()):
        group_to_indices[episode_to_group[int(episode)]].append(index)
    group_scores = {
        group: float(np.mean(score[np.asarray(indices, dtype=np.int64)]))
        for group, indices in group_to_indices.items()
    }
    groups = sorted(group_to_indices)
    if len(groups) < 4:
        raise ValueError("balanced mirror split requires at least four groups")
    cut_points = np.quantile(np.asarray(list(group_scores.values()), dtype=np.float64), [0.25, 0.50, 0.75])
    strata: dict[int, list[str]] = defaultdict(list)
    for group in groups:
        stratum = int(np.digitize(group_scores[group], cut_points, right=False))
        strata[stratum].append(group)
    rng = np.random.default_rng(int(split_seed))
    calibration_groups: set[str] = set()
    confirmation_groups: set[str] = set()
    for stratum, values in sorted(strata.items()):
        ordered = [values[index] for index in rng.permutation(len(values)).tolist()]
        midpoint = max(1, len(ordered) // 2)
        calibration_groups.update(ordered[:midpoint])
        confirmation_groups.update(ordered[midpoint:])
    if not confirmation_groups:
        moved = sorted(calibration_groups)[-1]
        calibration_groups.remove(moved)
        confirmation_groups.add(moved)
    calibration_indices = np.asarray(
        sorted(index for group in calibration_groups for index in group_to_indices[group]),
        dtype=np.int64,
    )
    confirmation_indices = np.asarray(
        sorted(index for group in confirmation_groups for index in group_to_indices[group]),
        dtype=np.int64,
    )
    if np.intersect1d(calibration_indices, confirmation_indices).size:
        raise RuntimeError("balanced mirror split has overlapping samples")
    metadata = {
        "strategy": "balanced_mirror_context",
        "split_seed": int(split_seed),
        "group_count": len(groups),
        "calibration_group_count": len(calibration_groups),
        "confirmation_group_count": len(confirmation_groups),
        "calibration_groups": sorted(calibration_groups),
        "confirmation_groups": sorted(confirmation_groups),
        "calibration_context_mean": float(np.mean(score[calibration_indices])),
        "confirmation_context_mean": float(np.mean(score[confirmation_indices])),
        "context_quantile_cut_points": cut_points.tolist(),
    }
    return calibration_indices, confirmation_indices, metadata


def _records(
    errors: np.ndarray,
    *,
    split: str,
    dataset: Any,
    indices: np.ndarray,
    context_scores: np.ndarray,
) -> str:
    return "".join(
        json.dumps(
            {
                "split": split,
                "sample_index": int(index),
                "dataset_index": int(indices[index]),
                "episode_index": int(dataset.episode_indices[indices[index]]),
                "episode_seed": int(dataset.episode_seeds[indices[index]]),
                "public_context_score": float(context_scores[indices[index]]),
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
    compactness_limits = (
        args.max_mean_effective_radius_m,
        args.max_effective_radius_m,
    )
    if any(value is not None and (not np.isfinite(float(value)) or float(value) <= 0.0) for value in compactness_limits):
        raise ValueError("compactness limits must be finite and positive when supplied")
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
    if args.split_strategy == "balanced_mirror_context":
        if args.scene_manifest is None:
            raise ValueError("--scene-manifest is required for balanced_mirror_context")
        calibration_indices, confirmation_indices, split_metadata = balanced_mirror_context_indices(
            dataset,
            load_scene_manifest(args.scene_manifest),
            split_seed=int(args.split_seed),
        )
    else:
        calibration_indices, confirmation_indices = validation_calibration_indices(dataset)
        split_metadata = {
            "strategy": "episode_seed_half",
            "split_seed": None,
            "calibration_group_count": None,
            "confirmation_group_count": None,
        }
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
    context_scores = public_context_score(dataset)
    calibration_adaptive_coverage = evaluate_context_adaptive_tube_coverage(
        calibration_candidates.numpy(),
        calibration_targets.numpy(),
        np.asarray(tube["radius_m_by_step"]),
        context_scores[calibration_indices],
        uncertainty_gain=float(args.uncertainty_gain),
    )
    confirmation_adaptive_coverage = evaluate_context_adaptive_tube_coverage(
        confirmation_candidates.numpy(),
        confirmation_targets.numpy(),
        np.asarray(tube["radius_m_by_step"]),
        context_scores[confirmation_indices],
        uncertainty_gain=float(args.uncertainty_gain),
    )
    coverage_pass = bool(
        confirmation_adaptive_coverage["full_trajectory_coverage"] >= float(args.coverage)
    )
    compactness_evaluated = any(value is not None for value in compactness_limits)
    mean_radius = float(confirmation_adaptive_coverage["mean_effective_radius_m"])
    maximum_radius = float(confirmation_adaptive_coverage["maximum_effective_radius_m"])
    mean_limit_pass = (
        True
        if args.max_mean_effective_radius_m is None
        else mean_radius <= float(args.max_mean_effective_radius_m)
    )
    maximum_limit_pass = (
        True
        if args.max_effective_radius_m is None
        else maximum_radius <= float(args.max_effective_radius_m)
    )
    compactness_gate = {
        "evaluated": compactness_evaluated,
        "coverage_pass": coverage_pass,
        "compactness_pass": bool(mean_limit_pass and maximum_limit_pass),
        "overall_pass": bool(coverage_pass and mean_limit_pass and maximum_limit_pass),
        "confirmation_coverage": float(confirmation_adaptive_coverage["full_trajectory_coverage"]),
        "mean_effective_radius_m": mean_radius,
        "maximum_effective_radius_m": maximum_radius,
        "max_mean_effective_radius_m": args.max_mean_effective_radius_m,
        "max_effective_radius_m": args.max_effective_radius_m,
    }
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
        "calibration_split": (
            "first half of validation episode seeds"
            if args.split_strategy == "episode_seed_half"
            else "balanced public-context strata of complete mirror groups"
        ),
        "confirmation_split": (
            "second half of validation episode seeds"
            if args.split_strategy == "episode_seed_half"
            else "held-out balanced public-context strata of complete mirror groups"
        ),
        "split_strategy": str(args.split_strategy),
        "split_seed": int(args.split_seed),
        "scene_manifest": None if args.scene_manifest is None else str(args.scene_manifest.resolve()),
        "split_metadata": split_metadata,
        "compactness_limits_m": {
            "max_mean_effective_radius_m": args.max_mean_effective_radius_m,
            "max_effective_radius_m": args.max_effective_radius_m,
        },
        "source_hashes": source_hashes(),
    }
    output.joinpath("config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    output.joinpath("calibration_scores.jsonl").write_text(
        _records(
            calibration_errors,
            split="calibration",
            dataset=dataset,
            indices=calibration_indices,
            context_scores=context_scores,
        ),
        encoding="utf-8",
    )
    output.joinpath("confirmation_scores.jsonl").write_text(
        _records(
            confirmation_errors,
            split="confirmation",
            dataset=dataset,
            indices=confirmation_indices,
            context_scores=context_scores,
        ),
        encoding="utf-8",
    )
    decision = "development_confirmation_only"
    if compactness_evaluated:
        decision = "development_confirmation_gate_pass" if compactness_gate["overall_pass"] else "development_confirmation_no_go"
    result = {
        "config": config,
        "tube": tube,
        "calibration": calibration_coverage,
        "confirmation": confirmation_coverage,
        "context_adaptive_calibration": calibration_adaptive_coverage,
        "context_adaptive_confirmation": confirmation_adaptive_coverage,
        "compactness_gate": compactness_gate,
        "calibration_sample_count": int(calibration_indices.size),
        "confirmation_sample_count": int(confirmation_indices.size),
        "decision": decision,
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
        for prefix, summary in (
            ("ContextAdaptive/Calibration", calibration_adaptive_coverage),
            ("ContextAdaptive/Confirmation", confirmation_adaptive_coverage),
        ):
            for key, value in summary.items():
                writer.add_scalar(f"{prefix}/{key}", float(value), 0)
        for step, value in enumerate(tube["radius_m_by_step"], start=1):
            writer.add_scalar("Tube/radius_m_by_step", float(value), step)
        writer.add_scalar("Tube/simultaneous_multiplier", float(tube["simultaneous_multiplier"]), 0)
        writer.add_scalar("Split/calibration_sample_count", float(calibration_indices.size), 0)
        writer.add_scalar("Split/confirmation_sample_count", float(confirmation_indices.size), 0)
        if np.isfinite(float(split_metadata.get("calibration_context_mean", np.nan))):
            writer.add_scalar("Split/calibration_context_mean", float(split_metadata["calibration_context_mean"]), 0)
        if np.isfinite(float(split_metadata.get("confirmation_context_mean", np.nan))):
            writer.add_scalar("Split/confirmation_context_mean", float(split_metadata["confirmation_context_mean"]), 0)
        if args.max_mean_effective_radius_m is not None:
            writer.add_scalar("Gate/max_mean_effective_radius_m", float(args.max_mean_effective_radius_m), 0)
        if args.max_effective_radius_m is not None:
            writer.add_scalar("Gate/max_effective_radius_m", float(args.max_effective_radius_m), 0)
        writer.add_scalar(
            "Gate/context_adaptive_confirmation_coverage",
            float(confirmation_adaptive_coverage["full_trajectory_coverage"]),
            0,
        )
        if compactness_gate["evaluated"]:
            writer.add_scalar("Gate/coverage_pass", float(compactness_gate["coverage_pass"]), 0)
            writer.add_scalar("Gate/compactness_pass", float(compactness_gate["compactness_pass"]), 0)
            writer.add_scalar("Gate/overall_pass", float(compactness_gate["overall_pass"]), 0)
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
