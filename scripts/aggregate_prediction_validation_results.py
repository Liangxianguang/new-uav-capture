"""Aggregate validation-only prediction ablations across matched seeds.

This script intentionally consumes training metadata and validation history
only. It refuses artifacts that mention a locked-test dataset, so action
conditioning and model-selection decisions cannot silently use the final test
split.
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
    "loss",
    "model_min_ade",
    "model_min_fde",
    "model_ade_top1",
    "model_fde_top1",
    "model_energy_score",
    "model_candidate_spread",
    "candidate_speed_feasible_fraction",
    "candidate_acceleration_feasible_fraction",
    "candidate_obstacle_clear_fraction",
    "candidate_feasible_fraction",
    "calibrated_coverage_full_trajectory",
    "single_sample_latency_p95_ms",
)
SEED_PATTERN = re.compile(r"_seed(?P<seed>[0-9]+)$")


@dataclass(frozen=True)
class ValidationRun:
    condition: str
    seed: int
    path: Path
    metadata: dict[str, Any]
    final: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        action="append",
        required=True,
        metavar="NAME=GLOB",
        help="A condition and run-directory glob, for example both=...\\*_seed*.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-condition", default="both")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260911)
    return parser.parse_args()


def parse_condition(specification: str) -> tuple[str, str]:
    if "=" not in specification:
        raise ValueError("--condition must use NAME=GLOB format.")
    name, pattern = specification.split("=", 1)
    name = name.strip()
    pattern = pattern.strip()
    if not name or not pattern:
        raise ValueError("--condition requires both a name and a glob.")
    return name, pattern


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping: {path}")
    return value


def read_seed(path: Path) -> int:
    match = SEED_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Validation run directory must end with _seed<integer>: {path}")
    return int(match.group("seed"))


def reject_locked_test_references(metadata: dict[str, Any], path: Path) -> None:
    for key in ("train_dataset", "validation_dataset"):
        value = str(metadata.get(key, "")).lower().replace("\\", "/")
        if "locked_test" in value or "locked-test" in value:
            raise ValueError(f"Validation artifact {path} references locked-test data in {key}.")
    arguments = metadata.get("arguments", {})
    if isinstance(arguments, dict):
        for key, value in arguments.items():
            text = str(value).lower().replace("\\", "/")
            if "locked_test" in text or "locked-test" in text:
                raise ValueError(f"Validation artifact {path} references locked-test data in arguments.{key}.")


def load_run(condition: str, path: Path) -> ValidationRun:
    metadata_path = path / "metadata.json"
    training_path = path / "training.json"
    if not metadata_path.is_file() or not training_path.is_file():
        raise FileNotFoundError(f"Run {path} must contain metadata.json and training.json.")
    metadata = load_json(metadata_path)
    reject_locked_test_references(metadata, path)
    records = json.loads(training_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records or not isinstance(records[-1], dict):
        raise ValueError(f"Training history is empty or malformed: {training_path}")
    final = dict(records[-1])
    arguments = metadata.get("arguments", {})
    if not isinstance(arguments, dict):
        raise ValueError(f"Run arguments are malformed: {metadata_path}")
    if str(metadata.get("action_conditioning")) != condition:
        raise ValueError(
            f"Run {path} declares action_conditioning={metadata.get('action_conditioning')!r}, "
            f"but was supplied under {condition!r}."
        )
    if str(metadata.get("model")) != "gru":
        raise ValueError(f"This ablation aggregator expects GRU runs, got {metadata.get('model')!r} in {path}.")
    if int(final.get("epoch", 0)) != int(arguments.get("epochs", 0)):
        raise ValueError(f"Run {path} did not finish its configured epoch budget.")
    return ValidationRun(
        condition=condition,
        seed=int(arguments.get("seed", read_seed(path))),
        path=path,
        metadata=metadata,
        final=final,
    )


def load_groups(specifications: list[str]) -> dict[str, list[ValidationRun]]:
    groups: dict[str, list[ValidationRun]] = {}
    for specification in specifications:
        condition, pattern = parse_condition(specification)
        if condition in groups:
            raise ValueError(f"Duplicate condition: {condition}")
        paths = [Path(item).resolve() for item in sorted(glob.glob(pattern)) if Path(item).is_dir()]
        if not paths:
            raise FileNotFoundError(f"No validation runs matched {pattern!r}")
        runs = [load_run(condition, path) for path in paths]
        seeds = [run.seed for run in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Condition {condition!r} contains duplicate training seeds.")
        groups[condition] = sorted(runs, key=lambda run: run.seed)
    return groups


def finite_metric(run: ValidationRun, metric: str) -> float:
    value = run.final.get(metric)
    if value is None:
        raise KeyError(f"Metric {metric!r} is missing from {run.path / 'training.json'}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Metric {metric!r} is not finite in {run.path / 'training.json'}")
    return result


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Bootstrap values must be a non-empty vector.")
    if samples <= 0:
        raise ValueError("bootstrap-samples must be positive.")
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_runs(
    runs: list[ValidationRun],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    first = runs[0]
    arguments = first.metadata["arguments"]
    for run in runs[1:]:
        other = run.metadata["arguments"]
        for key in ("epochs", "batch_size", "hidden_dim", "num_layers", "learning_rate", "seed"):
            if key == "seed":
                continue
            if other.get(key) != arguments.get(key):
                raise ValueError(f"Condition {first.condition} mixes training configurations at {key!r}.")
    summary: dict[str, Any] = {
        "training_seeds": [run.seed for run in runs],
        "seed_count": len(runs),
        "epochs": int(arguments["epochs"]),
        "batch_size": int(arguments["batch_size"]),
        "hidden_dim": int(arguments["hidden_dim"]),
        "num_layers": int(arguments["num_layers"]),
        "metrics": {},
    }
    for metric in METRICS:
        values = np.asarray([finite_metric(run, metric) for run in runs], dtype=np.float64)
        low, high = bootstrap_mean_ci(values, rng, bootstrap_samples)
        summary["metrics"][metric] = {
            "mean": float(values.mean()),
            "std_across_seeds": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
            "max": float(values.max()),
            "bootstrap_95_ci": [low, high],
        }
    return summary


def paired_summary(
    reference: list[ValidationRun],
    candidate: list[ValidationRun],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    reference_by_seed = {run.seed: run for run in reference}
    candidate_by_seed = {run.seed: run for run in candidate}
    shared = sorted(set(reference_by_seed).intersection(candidate_by_seed))
    if not shared:
        raise ValueError("Compared conditions do not share any training seeds.")
    result: dict[str, Any] = {"shared_training_seeds": shared, "metrics": {}}
    for metric in METRICS:
        differences = np.asarray(
            [
                finite_metric(candidate_by_seed[seed], metric)
                - finite_metric(reference_by_seed[seed], metric)
                for seed in shared
            ],
            dtype=np.float64,
        )
        low, high = bootstrap_mean_ci(differences, rng, bootstrap_samples)
        result["metrics"][metric] = {
            "mean_delta_candidate_minus_reference": float(differences.mean()),
            "std_delta": float(differences.std(ddof=1)) if differences.size > 1 else 0.0,
            "paired_bootstrap_95_ci": [low, high],
        }
    return result


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 15 S4-v3 Validation Action-Conditioning Ablation",
        "",
        "This report uses validation histories only. Intervals are percentile 95% bootstrap CIs over matched training seeds; locked-test data are not read.",
        "",
        "| Condition | Seeds | Projected/physical minFDE | minADE | Energy score | Calibrated coverage | Predictor p95 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for condition, summary in payload["conditions"].items():
        metrics = summary["metrics"]
        def interval(name: str, percent: bool = False) -> str:
            value = metrics[name]["mean"]
            low, high = metrics[name]["bootstrap_95_ci"]
            if percent:
                return f"{value:.2%} [{low:.2%}, {high:.2%}]"
            return f"{value:.4f} [{low:.4f}, {high:.4f}]"
        lines.append(
            f"| {condition} | {summary['seed_count']} | {interval('model_min_fde')} "
            f"| {interval('model_min_ade')} | {interval('model_energy_score')} "
            f"| {interval('calibrated_coverage_full_trajectory', percent=True)} "
            f"| {metrics['single_sample_latency_p95_ms']['mean']:.3f} |"
        )
    lines.extend([
        "",
        "## Paired Differences",
        "",
        "Positive minFDE/minADE deltas favour the reference condition; negative deltas favour the candidate.",
        "",
        "| Candidate | minFDE delta | minADE delta | Coverage delta |",
        "| --- | ---: | ---: | ---: |",
    ])
    for condition, summary in payload["paired_vs_reference"].items():
        metrics = summary["metrics"]
        def delta(name: str, percent: bool = False) -> str:
            item = metrics[name]
            low, high = item["paired_bootstrap_95_ci"]
            if percent:
                return f"{item['mean_delta_candidate_minus_reference']:.2%} [{low:.2%}, {high:.2%}]"
            return f"{item['mean_delta_candidate_minus_reference']:.4f} [{low:.4f}, {high:.4f}]"
        lines.append(
            f"| {condition} | {delta('model_min_fde')} | {delta('model_min_ade')} "
            f"| {delta('calibrated_coverage_full_trajectory', percent=True)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    groups = load_groups(args.condition)
    if args.reference_condition not in groups:
        raise ValueError(f"Reference condition {args.reference_condition!r} was not supplied.")
    rng = np.random.default_rng(args.bootstrap_seed)
    conditions = {
        condition: summarize_runs(runs, rng, args.bootstrap_samples)
        for condition, runs in groups.items()
    }
    paired = {
        condition: paired_summary(
            groups[args.reference_condition], runs, rng, args.bootstrap_samples
        )
        for condition, runs in groups.items()
        if condition != args.reference_condition
    }
    payload = {
        "reference_condition": args.reference_condition,
        "selection_split": "validation_only",
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "matched training seed",
        },
        "conditions": conditions,
        "paired_vs_reference": paired,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
