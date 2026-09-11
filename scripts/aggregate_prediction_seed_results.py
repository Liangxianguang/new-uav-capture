"""Aggregate multi-seed locked-test prediction results with paired bootstrap CIs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_METRICS = (
    "raw.min_fde",
    "projected.min_fde",
    "projected.min_ade",
    "projected.coverage_full_trajectory",
    "projected.candidate_feasible_fraction",
    "latency.single_sample_latency_p95_ms",
)
SEED_PATTERN = re.compile(r"^(?P<model>.+)_seed(?P<seed>[0-9]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-model", default="gru")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260911)
    return parser.parse_args()


def nested_value(document: dict[str, Any], path: str) -> float:
    value: Any = document
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"Missing metric {path!r} in locked-test artifact.")
        value = value[part]
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"Metric {path!r} is not finite.")
    return result


def load_records(root: Path, metrics: tuple[str, ...]) -> dict[str, dict[int, dict[str, float]]]:
    grouped: dict[str, dict[int, dict[str, float]]] = {}
    for directory in sorted(root.iterdir()):
        if not directory.is_dir():
            continue
        match = SEED_PATTERN.match(directory.name)
        if match is None:
            continue
        artifact = directory / "locked_test_metrics.json"
        if not artifact.is_file():
            raise FileNotFoundError(f"Missing locked-test metrics: {artifact}")
        document = json.loads(artifact.read_text(encoding="utf-8"))
        model = match.group("model")
        seed = int(match.group("seed"))
        if seed in grouped.setdefault(model, {}):
            raise ValueError(f"Duplicate model/seed result: {model}/{seed}")
        grouped[model][seed] = {metric: nested_value(document, metric) for metric in metrics}
    if not grouped:
        raise ValueError(f"No model_seed directories found under {root}")
    return grouped


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Bootstrap values must be a non-empty vector.")
    if samples <= 0:
        raise ValueError("bootstrap-samples must be positive.")
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(
    grouped: dict[str, dict[int, dict[str, float]]],
    metrics: tuple[str, ...],
    reference_model: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if reference_model not in grouped:
        raise ValueError(f"Reference model {reference_model!r} is not present.")
    rng = np.random.default_rng(bootstrap_seed)
    summary: dict[str, Any] = {"models": {}, "paired_vs_reference": {}}
    for model, records in sorted(grouped.items()):
        model_summary: dict[str, Any] = {"seed_count": len(records), "seeds": sorted(records)}
        for metric in metrics:
            values = np.asarray([records[seed][metric] for seed in sorted(records)], dtype=np.float64)
            ci_low, ci_high = bootstrap_mean_ci(values, rng, bootstrap_samples)
            model_summary[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
                "bootstrap_95_ci": [ci_low, ci_high],
            }
        summary["models"][model] = model_summary

    reference_records = grouped[reference_model]
    for model, records in sorted(grouped.items()):
        if model == reference_model:
            continue
        shared_seeds = sorted(set(reference_records).intersection(records))
        if not shared_seeds:
            raise ValueError(f"No shared seeds between {reference_model!r} and {model!r}.")
        comparison: dict[str, Any] = {"seed_count": len(shared_seeds), "seeds": shared_seeds}
        for metric in metrics:
            differences = np.asarray(
                [records[seed][metric] - reference_records[seed][metric] for seed in shared_seeds],
                dtype=np.float64,
            )
            ci_low, ci_high = bootstrap_mean_ci(differences, rng, bootstrap_samples)
            comparison[metric] = {
                "mean_delta_model_minus_reference": float(differences.mean()),
                "std_delta": float(differences.std(ddof=1)) if differences.size > 1 else 0.0,
                "paired_bootstrap_95_ci": [ci_low, ci_high],
            }
        summary["paired_vs_reference"][model] = comparison
    return summary


def markdown_report(summary: dict[str, Any], metrics: tuple[str, ...], reference_model: str) -> str:
    lines = [
        "# S4-v3 Formal Multi-Seed Locked-Test Summary",
        "",
        f"Reference model for paired differences: `{reference_model}`.",
        "Bootstrap intervals are percentile 95% CIs over shared training seeds.",
        "",
        "| Model | Seeds | Projected minFDE | Projected minADE | Coverage | Feasible candidates | Predictor p95 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, values in summary["models"].items():
        def mean(metric: str) -> float:
            return float(values[metric]["mean"])

        lines.append(
            f"| {model} | {values['seed_count']} | {mean('projected.min_fde'):.4f} +/- {values['projected.min_fde']['std']:.4f} "
            f"| {mean('projected.min_ade'):.4f} +/- {values['projected.min_ade']['std']:.4f} "
            f"| {mean('projected.coverage_full_trajectory'):.2%} "
            f"| {mean('projected.candidate_feasible_fraction'):.2%} "
            f"| {mean('latency.single_sample_latency_p95_ms'):.3f} |"
        )
    lines.extend(["", "## Paired Differences vs GRU", "", "| Model | Projected minFDE delta | Coverage delta | Feasible-candidate delta |", "| --- | ---: | ---: | ---: |"])
    for model, values in summary["paired_vs_reference"].items():
        lines.append(
            f"| {model} | {values['projected.min_fde']['mean_delta_model_minus_reference']:.4f} "
            f"[{values['projected.min_fde']['paired_bootstrap_95_ci'][0]:.4f}, {values['projected.min_fde']['paired_bootstrap_95_ci'][1]:.4f}] "
            f"| {values['projected.coverage_full_trajectory']['mean_delta_model_minus_reference']:.2%} "
            f"| {values['projected.candidate_feasible_fraction']['mean_delta_model_minus_reference']:.2%} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    metrics = DEFAULT_METRICS
    grouped = load_records(root, metrics)
    summary = summarize(
        grouped,
        metrics,
        args.reference_model,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    payload = {
        "root": str(root),
        "reference_model": args.reference_model,
        "metrics": list(metrics),
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        **summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(
        markdown_report(payload, metrics, args.reference_model),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
