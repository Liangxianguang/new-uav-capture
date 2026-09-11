"""Aggregate validation-only Phase 16 predictor-selection artifacts.

Unlike the locked-test aggregator, this command accepts only artifacts emitted
by ``evaluate_prediction_models.py --validation-only``. It validates that
calibration and selection used disjoint validation episode sets and that no
locked-test location appears in either the evaluation or training metadata.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


METRICS = (
    "raw.min_fde",
    "projected.min_ade",
    "projected.min_fde",
    "projected.energy_score",
    "projected.coverage_full_trajectory",
    "projected.conformal_full_trajectory_coverage_error",
    "projected.candidate_feasible_fraction",
    "projected.candidate_obstacle_clear_fraction",
    "projected.branch_top1_accuracy",
    "projected.branch_uniform_vote_accuracy",
    "projected.branch_any_candidate_coverage",
    "projected.branch_bimodal_candidate_fraction",
    "latency.single_sample_latency_p50_ms",
    "latency.single_sample_latency_p95_ms",
    "latency.single_sample_latency_p99_ms",
)
SEED_PATTERN = re.compile(r"_seed(?P<seed>[0-9]+)$")
VALIDATION_SPLIT = "validation_second_half_episode_seeds"
CALIBRATION_SOURCE = "validation_first_half_episode_seeds_only"


@dataclass(frozen=True)
class SelectionRun:
    group: str
    seed: int
    path: Path
    document: dict[str, Any]
    training_metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        metavar="NAME=GLOB",
        help="One condition/model family and its validation-result directories.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-group", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260911)
    return parser.parse_args()


def parse_group(specification: str) -> tuple[str, str]:
    if "=" not in specification:
        raise ValueError("--group must use NAME=GLOB format.")
    name, pattern = (part.strip() for part in specification.split("=", 1))
    if not name or not pattern:
        raise ValueError("--group requires both a name and a glob pattern.")
    return name, pattern


def load_mapping(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return loaded


def contains_locked_test(value: object) -> bool:
    return "locked_test" in str(value).lower().replace("\\", "/") or "locked-test" in str(value).lower()


def nested_value(document: dict[str, Any], dotted_path: str) -> float:
    value: Any = document
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Missing metric {dotted_path!r}.")
        value = value[part]
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"Metric {dotted_path!r} is not finite.")
    return numeric


def read_seed(path: Path) -> int:
    match = SEED_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Result directory must end with _seed<integer>: {path}")
    return int(match.group("seed"))


def validate_training_metadata(metadata: dict[str, Any], path: Path) -> None:
    for key in ("train_dataset", "validation_dataset"):
        if contains_locked_test(metadata.get(key)):
            raise ValueError(f"Training metadata at {path} references locked-test data in {key}.")
    arguments = metadata.get("arguments")
    if not isinstance(arguments, dict):
        raise ValueError(f"Training metadata has no arguments mapping: {path}")
    if any(contains_locked_test(value) for value in arguments.values()):
        raise ValueError(f"Training metadata at {path} references locked-test data in arguments.")


def load_run(group: str, path: Path) -> SelectionRun:
    artifact_path = path / "validation_selection_metrics.json"
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Missing validation selection artifact: {artifact_path}")
    document = load_mapping(artifact_path)
    if document.get("evaluation_split") != VALIDATION_SPLIT:
        raise ValueError(f"Artifact {artifact_path} is not a validation-only selection evaluation.")
    if document.get("locked_test_dataset") is not None:
        raise ValueError(f"Artifact {artifact_path} references a locked-test dataset.")
    calibration = document.get("calibration")
    if not isinstance(calibration, dict) or calibration.get("source") != CALIBRATION_SOURCE:
        raise ValueError(f"Artifact {artifact_path} does not use the required validation-only calibration split.")
    training_output = Path(str(document.get("training_output", ""))).resolve()
    metadata_path = training_output / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing training metadata for selection artifact: {metadata_path}")
    training_metadata = load_mapping(metadata_path)
    validate_training_metadata(training_metadata, metadata_path)
    seed = read_seed(path)
    arguments = training_metadata["arguments"]
    if int(arguments.get("seed", -1)) != seed:
        raise ValueError(f"Training seed and result-directory seed disagree at {path}.")
    for metric in METRICS:
        nested_value(document, metric)
    return SelectionRun(group, seed, path, document, training_metadata)


def load_groups(specifications: list[str]) -> dict[str, list[SelectionRun]]:
    groups: dict[str, list[SelectionRun]] = {}
    for specification in specifications:
        name, pattern = parse_group(specification)
        if name in groups:
            raise ValueError(f"Duplicate group: {name}")
        paths = [Path(item).resolve() for item in sorted(glob.glob(pattern)) if Path(item).is_dir()]
        if not paths:
            raise FileNotFoundError(f"No result directories matched {pattern!r}")
        runs = [load_run(name, path) for path in paths]
        seeds = [run.seed for run in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Group {name!r} contains duplicate training seeds.")
        groups[name] = sorted(runs, key=lambda run: run.seed)
    return groups


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0 or samples <= 0:
        raise ValueError("Bootstrap requires a non-empty one-dimensional sample and positive count.")
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_group(runs: list[SelectionRun], rng: np.random.Generator, samples: int) -> dict[str, Any]:
    first = runs[0]
    model_fields = ("model", "model_backend", "action_conditioning")
    for run in runs[1:]:
        for field in model_fields:
            if run.training_metadata.get(field) != first.training_metadata.get(field):
                raise ValueError(f"Group {first.group!r} mixes {field!r} at {run.path}.")
    result: dict[str, Any] = {
        "training_seeds": [run.seed for run in runs],
        "seed_count": len(runs),
        "model": first.training_metadata.get("model"),
        "model_backend": first.training_metadata.get("model_backend"),
        "action_conditioning": first.training_metadata.get("action_conditioning"),
        "num_samples": int(first.document["num_samples"]),
        "sampling_steps": int(first.document["sampling_steps"]),
        "projection_iterations": int(first.document["projection_iterations"]),
        "metrics": {},
    }
    for metric in METRICS:
        values = np.asarray([nested_value(run.document, metric) for run in runs], dtype=np.float64)
        low, high = bootstrap_mean_ci(values, rng, samples)
        result["metrics"][metric] = {
            "mean": float(values.mean()),
            "std_across_seeds": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "bootstrap_95_ci": [low, high],
        }
    return result


def paired_summary(
    reference: list[SelectionRun],
    candidate: list[SelectionRun],
    rng: np.random.Generator,
    samples: int,
) -> dict[str, Any]:
    reference_by_seed = {run.seed: run for run in reference}
    candidate_by_seed = {run.seed: run for run in candidate}
    shared = sorted(set(reference_by_seed).intersection(candidate_by_seed))
    if not shared:
        raise ValueError("Compared groups do not share a training seed.")
    result: dict[str, Any] = {"shared_training_seeds": shared, "metrics": {}}
    for metric in METRICS:
        deltas = np.asarray(
            [
                nested_value(candidate_by_seed[seed].document, metric)
                - nested_value(reference_by_seed[seed].document, metric)
                for seed in shared
            ],
            dtype=np.float64,
        )
        low, high = bootstrap_mean_ci(deltas, rng, samples)
        result["metrics"][metric] = {
            "mean_delta_candidate_minus_reference": float(deltas.mean()),
            "std_delta": float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0,
            "paired_bootstrap_95_ci": [low, high],
        }
    return result


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 16 Validation-Only Predictor Selection",
        "",
        "All rows use the second half of validation episode seeds for selection and the first half for split-conformal calibration. Locked-test data are not read.",
        "",
        "| Group | Model backend | Action condition | K | Projected minFDE | Projected minADE | Coverage error | Feasible | Branch coverage | Bimodal | p95 (ms) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, summary in payload["groups"].items():
        metrics = summary["metrics"]
        mean = lambda name: float(metrics[name]["mean"])
        lines.append(
            f"| {group} | {summary['model_backend']} | {summary['action_conditioning']} | {summary['num_samples']} "
            f"| {mean('projected.min_fde'):.4f} | {mean('projected.min_ade'):.4f} "
            f"| {mean('projected.conformal_full_trajectory_coverage_error'):.2%} "
            f"| {mean('projected.candidate_feasible_fraction'):.2%} "
            f"| {mean('projected.branch_any_candidate_coverage'):.2%} "
            f"| {mean('projected.branch_bimodal_candidate_fraction'):.2%} "
            f"| {mean('latency.single_sample_latency_p95_ms'):.3f} |"
        )
    lines.extend(["", "## Paired Differences", "", "Negative error deltas favour the candidate. Candidate branch metrics are not probabilities because diffusion candidates have uniform uncalibrated weights.", ""])
    for group, summary in payload["paired_vs_reference"].items():
        metrics = summary["metrics"]
        fde = metrics["projected.min_fde"]
        feasible = metrics["projected.candidate_feasible_fraction"]
        lines.append(
            f"- {group} vs {payload['reference_group']}: projected minFDE delta "
            f"{fde['mean_delta_candidate_minus_reference']:.4f} "
            f"[{fde['paired_bootstrap_95_ci'][0]:.4f}, {fde['paired_bootstrap_95_ci'][1]:.4f}], "
            f"feasibility delta {feasible['mean_delta_candidate_minus_reference']:.2%}."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    groups = load_groups(args.group)
    if args.reference_group not in groups:
        raise ValueError(f"Reference group {args.reference_group!r} was not supplied.")
    rng = np.random.default_rng(args.bootstrap_seed)
    summaries = {
        group: summarize_group(runs, rng, args.bootstrap_samples)
        for group, runs in groups.items()
    }
    paired = {
        group: paired_summary(groups[args.reference_group], runs, rng, args.bootstrap_samples)
        for group, runs in groups.items()
        if group != args.reference_group
    }
    payload = {
        "selection_split": VALIDATION_SPLIT,
        "calibration_split": CALIBRATION_SOURCE,
        "reference_group": args.reference_group,
        "groups": summaries,
        "paired_vs_reference": paired,
        "metrics": list(METRICS),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_unit": "matched training seed",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
