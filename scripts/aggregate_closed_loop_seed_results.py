"""Aggregate frozen-scene closed-loop results across predictor training seeds.

Each training seed is paired at the episode level before resampling. The
hierarchical bootstrap therefore reflects both predictor-seed variation and
the fixed locked-test scene block, while retaining the exact scene pairing
between models.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


OUTCOME_METRICS = (
    "safe_capture_success",
    "capture_event",
    "collision",
    "boundary_violation",
    "timeout",
    "capture_time_seconds",
    "min_clearance_m",
    "future_action_condition_available_rate",
    "mean_prediction_age_steps",
)
LATENCY_FIELDS = (
    "predictor_latency_ms",
    "planner_latency_ms",
    "safety_latency_ms",
    "total_control_latency_ms",
)
SEED_PATTERN = re.compile(r"_seed(?P<seed>[0-9]+)$")


@dataclass(frozen=True)
class RunArtifact:
    label: str
    seed: int
    path: Path
    scene_hash: str
    methods: dict[str, list[dict[str, Any]]]
    steps: dict[str, list[dict[str, Any]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        metavar="NAME=GLOB",
        help="One predictor family, for example gru=results\\phase15_*_gru_seed*.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-group", default="gru")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260911)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Artifact has no records: {path}")
    return records


def scene_file_hash(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen scene manifest: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_from_path(path: Path) -> int:
    match = SEED_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Run directory must end with _seed<integer>: {path}")
    return int(match.group("seed"))


def load_artifact(label: str, path: Path) -> RunArtifact:
    methods: dict[str, list[dict[str, Any]]] = {}
    steps: dict[str, list[dict[str, Any]]] = {}
    for method_directory in sorted(path.iterdir()):
        if not method_directory.is_dir():
            continue
        episode_path = method_directory / "episodes.jsonl"
        step_path = method_directory / "steps.jsonl"
        if not episode_path.is_file() and not step_path.is_file():
            continue
        episodes = read_jsonl(episode_path)
        method = method_directory.name
        episode_ids = [int(row["episode_index"]) for row in episodes]
        if len(set(episode_ids)) != len(episode_ids):
            raise ValueError(f"Duplicate episode indices in {episode_path}")
        methods[method] = sorted(episodes, key=lambda row: int(row["episode_index"]))
        steps[method] = read_jsonl(step_path)
    if not methods:
        raise ValueError(f"No method logs found under {path}")
    return RunArtifact(
        label=label,
        seed=seed_from_path(path),
        path=path,
        scene_hash=scene_file_hash(path / "scenes.jsonl"),
        methods=methods,
        steps=steps,
    )


def parse_group(specification: str) -> tuple[str, str]:
    if "=" not in specification:
        raise ValueError("--group must use NAME=GLOB format.")
    name, pattern = specification.split("=", 1)
    name = name.strip()
    pattern = pattern.strip()
    if not name or not pattern:
        raise ValueError("--group requires both a name and a glob pattern.")
    return name, pattern


def load_groups(specifications: list[str]) -> dict[str, list[RunArtifact]]:
    groups: dict[str, list[RunArtifact]] = {}
    for specification in specifications:
        label, pattern = parse_group(specification)
        if label in groups:
            raise ValueError(f"Duplicate group name: {label}")
        paths = [Path(item).resolve() for item in sorted(glob.glob(pattern)) if Path(item).is_dir()]
        if not paths:
            raise FileNotFoundError(f"No run directories matched {pattern!r}")
        artifacts = [load_artifact(label, path) for path in paths]
        seeds = [artifact.seed for artifact in artifacts]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate training seed in group {label!r}")
        scene_hashes = {artifact.scene_hash for artifact in artifacts}
        if len(scene_hashes) != 1:
            raise ValueError(f"Group {label!r} does not use one frozen scene manifest.")
        groups[label] = sorted(artifacts, key=lambda artifact: artifact.seed)
    all_hashes = {artifact.scene_hash for artifacts in groups.values() for artifact in artifacts}
    if len(all_hashes) != 1:
        raise ValueError("All compared groups must use the same frozen scene manifest.")
    return groups


def metric_value(row: dict[str, Any], metric: str) -> float:
    value = row.get(metric)
    if isinstance(value, bool):
        return float(value)
    if value is None:
        return float("nan")
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def aligned_metric_matrix(artifacts: list[RunArtifact], method: str, metric: str) -> tuple[list[int], np.ndarray]:
    records_by_seed: list[list[dict[str, Any]]] = []
    episode_ids: list[int] | None = None
    for artifact in artifacts:
        if method not in artifact.methods:
            raise ValueError(f"Method {method!r} is missing from {artifact.path}")
        records = artifact.methods[method]
        current_ids = [int(row["episode_index"]) for row in records]
        if episode_ids is None:
            episode_ids = current_ids
        elif episode_ids != current_ids:
            raise ValueError(f"Episode ordering differs for method {method!r} in {artifact.path}")
        records_by_seed.append(records)
    assert episode_ids is not None
    matrix = np.asarray(
        [[metric_value(row, metric) for row in records] for records in records_by_seed],
        dtype=np.float64,
    )
    return episode_ids, matrix


def hierarchical_bootstrap_ci(values: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Bootstrap values must have shape [seeds, episodes].")
    if samples <= 0:
        raise ValueError("bootstrap-samples must be positive.")
    finite_rows = np.any(np.isfinite(values), axis=1)
    values = values[finite_rows]
    if values.shape[0] == 0:
        return float("nan"), float("nan")
    seed_count, episode_count = values.shape
    seed_indices = rng.integers(0, seed_count, size=(samples, seed_count))
    episode_indices = rng.integers(0, episode_count, size=(samples, seed_count, episode_count))
    sampled = values[seed_indices[:, :, None], episode_indices]
    per_seed_counts = np.isfinite(sampled).sum(axis=2)
    per_seed = np.divide(
        np.nansum(sampled, axis=2),
        per_seed_counts,
        out=np.full(per_seed_counts.shape, np.nan, dtype=np.float64),
        where=per_seed_counts > 0,
    )
    seed_counts = np.isfinite(per_seed).sum(axis=1)
    means = np.divide(
        np.nansum(per_seed, axis=1),
        seed_counts,
        out=np.full(seed_counts.shape, np.nan, dtype=np.float64),
        where=seed_counts > 0,
    )
    finite_means = means[np.isfinite(means)]
    if finite_means.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(finite_means, 0.025)), float(np.quantile(finite_means, 0.975))


def metric_summary(matrix: np.ndarray, rng: np.random.Generator, bootstrap_samples: int) -> dict[str, Any]:
    per_seed_counts = np.isfinite(matrix).sum(axis=1)
    per_seed = np.divide(
        np.nansum(matrix, axis=1),
        per_seed_counts,
        out=np.full(per_seed_counts.shape, np.nan, dtype=np.float64),
        where=per_seed_counts > 0,
    )
    finite = per_seed[np.isfinite(per_seed)]
    mean = float(finite.mean()) if finite.size else float("nan")
    low, high = hierarchical_bootstrap_ci(matrix, rng, bootstrap_samples)
    return {
        "mean": mean,
        "std_across_training_seeds": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "bootstrap_95_ci": [low, high],
    }


def latency_summary(artifacts: list[RunArtifact], method: str) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for field in LATENCY_FIELDS:
        values = np.asarray(
            [metric_value(row, field) for artifact in artifacts for row in artifact.steps[method]],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        result[field] = {
            "samples": int(values.size),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
        }
    return result


def summarize_group(
    artifacts: list[RunArtifact],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    methods = sorted(set.intersection(*(set(artifact.methods) for artifact in artifacts)))
    if not methods:
        raise ValueError("No common methods are available across all runs in a group.")
    summary: dict[str, Any] = {}
    for method in methods:
        metrics: dict[str, Any] = {}
        episode_count: int | None = None
        for metric in OUTCOME_METRICS:
            episode_ids, matrix = aligned_metric_matrix(artifacts, method, metric)
            episode_count = len(episode_ids)
            metrics[metric] = metric_summary(matrix, rng, bootstrap_samples)
        summary[method] = {
            "training_seeds": [artifact.seed for artifact in artifacts],
            "seed_count": len(artifacts),
            "episodes_per_seed": episode_count,
            "episode_metrics": metrics,
            "pooled_step_latency_ms": latency_summary(artifacts, method),
        }
    return summary


def paired_comparison(
    reference: list[RunArtifact],
    candidate: list[RunArtifact],
    method: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    reference_by_seed = {artifact.seed: artifact for artifact in reference}
    candidate_by_seed = {artifact.seed: artifact for artifact in candidate}
    shared_seeds = sorted(set(reference_by_seed).intersection(candidate_by_seed))
    if not shared_seeds:
        raise ValueError("Compared predictor families have no matched training seeds.")
    aligned_reference = [reference_by_seed[seed] for seed in shared_seeds]
    aligned_candidate = [candidate_by_seed[seed] for seed in shared_seeds]
    metrics: dict[str, Any] = {}
    for metric in OUTCOME_METRICS:
        reference_ids, reference_matrix = aligned_metric_matrix(aligned_reference, method, metric)
        candidate_ids, candidate_matrix = aligned_metric_matrix(aligned_candidate, method, metric)
        if reference_ids != candidate_ids:
            raise ValueError(f"Paired comparison episode mismatch for {method!r}.")
        differences = candidate_matrix - reference_matrix
        difference_summary = metric_summary(differences, rng, bootstrap_samples)
        metrics[metric] = {
            "mean_delta_candidate_minus_reference": difference_summary["mean"],
            "paired_bootstrap_95_ci": list(difference_summary["bootstrap_95_ci"]),
        }
    return {"method": method, "shared_training_seeds": shared_seeds, "episode_metrics": metrics}


def render_interval(metric: dict[str, Any], percent: bool = False) -> str:
    value = float(metric["mean"])
    low, high = (float(part) for part in metric["bootstrap_95_ci"])
    if percent:
        return f"{value:.2%} [{low:.2%}, {high:.2%}]"
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 15 S4-v3 Action-Conditioned Closed-Loop Summary",
        "",
        "All runs use the same frozen 90-scene locked-test manifest. Confidence intervals are hierarchical 95% bootstrap intervals that resample matched predictor training seeds and episode indices.",
        "",
        "| Predictor family | Method | Safe capture | Collision | Timeout | Capture time (s) | Condition available | Total p95 (ms) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, methods in payload["groups"].items():
        for method, values in methods.items():
            metrics = values["episode_metrics"]
            total = values["pooled_step_latency_ms"]["total_control_latency_ms"]
            lines.append(
                f"| {group} | {method} | {render_interval(metrics['safe_capture_success'], percent=True)} "
                f"| {render_interval(metrics['collision'], percent=True)} "
                f"| {render_interval(metrics['timeout'], percent=True)} "
                f"| {render_interval(metrics['capture_time_seconds'])} "
                f"| {render_interval(metrics['future_action_condition_available_rate'], percent=True)} "
                f"| {total['p95']:.2f} |"
            )
    comparisons = payload.get("paired_vs_reference", {})
    if comparisons:
        lines.extend([
            "",
            "## Paired Differences Versus Reference",
            "",
            "A positive safe-capture delta favours the candidate family. These comparisons only include methods and training seeds present in both families.",
            "",
            "| Candidate family | Method | Safe-capture delta | Collision delta | Timeout delta |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for candidate, values in comparisons.items():
            for method, comparison in values.items():
                metrics = comparison["episode_metrics"]
                def delta(name: str) -> str:
                    item = metrics[name]
                    low, high = item["paired_bootstrap_95_ci"]
                    return f"{item['mean_delta_candidate_minus_reference']:.2%} [{low:.2%}, {high:.2%}]"
                lines.append(
                    f"| {candidate} | {method} | {delta('safe_capture_success')} "
                    f"| {delta('collision')} | {delta('timeout')} |"
                )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    groups = load_groups(args.group)
    if args.reference_group not in groups:
        raise ValueError(f"Reference group {args.reference_group!r} was not supplied.")
    rng = np.random.default_rng(args.bootstrap_seed)
    summaries = {
        label: summarize_group(artifacts, rng, args.bootstrap_samples)
        for label, artifacts in groups.items()
    }
    reference_methods = set(summaries[args.reference_group])
    paired: dict[str, Any] = {}
    for label, artifacts in groups.items():
        if label == args.reference_group:
            continue
        shared_methods = sorted(reference_methods.intersection(summaries[label]))
        paired[label] = {
            method: paired_comparison(
                groups[args.reference_group], artifacts, method, rng, args.bootstrap_samples
            )
            for method in shared_methods
        }
    payload = {
        "reference_group": args.reference_group,
        "scene_manifest_sha256": groups[args.reference_group][0].scene_hash,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "matched predictor training seed and locked-test episode",
        },
        "groups": summaries,
        "paired_vs_reference": paired,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
