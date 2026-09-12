"""Analyze whether nominal RNIC slack predicts closed-loop failures.

The analysis is post-hoc only: simulator outcomes are read from episode logs
and never fed back into planning. Each input is ``label=run-directory`` where
the directory contains ``episodes.jsonl``. The optional TensorBoard output
stores the exact analysis configuration and scalar summaries.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="RNIC run directory containing distributed_delayed/episodes.jsonl or episodes.jsonl.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path)
    return parser.parse_args()


def parse_run(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError("--run must use LABEL=DIR format")
    label, directory = specification.split("=", 1)
    if not label.strip() or not directory.strip():
        raise ValueError("--run requires a non-empty label and directory")
    return label.strip(), Path(directory.strip()).resolve()


def read_rows(directory: Path) -> list[dict[str, Any]]:
    episode_path = directory / "episodes.jsonl"
    if not episode_path.is_file():
        episode_path = directory / "distributed_delayed" / "episodes.jsonl"
    if not episode_path.is_file():
        raise FileNotFoundError(f"Missing episodes.jsonl under {directory}")
    rows = [json.loads(line) for line in episode_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty episode log: {episode_path}")
    return rows


def finite_float(row: dict[str, Any], key: str) -> float:
    try:
        value = float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def binary(row: dict[str, Any], key: str) -> float:
    return float(bool(row.get(key, False)))


def auc(scores: np.ndarray, outcomes: np.ndarray) -> float:
    """Compute tie-aware rank AUROC without a third-party dependency."""

    scores = np.asarray(scores, dtype=np.float64)
    outcomes = np.asarray(outcomes, dtype=np.float64)
    positive = outcomes > 0.5
    negative = ~positive
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        end = start + 1
        while end < sorted_scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    positive_rank_sum = float(ranks[positive].sum())
    positive_count = float(positive.sum())
    negative_count = float(negative.sum())
    return (positive_rank_sum - positive_count * (positive_count + 1.0) / 2.0) / (
        positive_count * negative_count
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if math.isfinite(finite_float(row, "rnic_minimum_best_slack_s"))]
    if not usable:
        raise ValueError("No finite rnic_minimum_best_slack_s values were found")
    slack = np.asarray([finite_float(row, "rnic_minimum_best_slack_s") for row in usable])
    risk_score = -slack
    collision = np.asarray([binary(row, "collision") for row in usable])
    unsafe = np.asarray([float(bool(row.get("collision", False) or row.get("boundary_violation", False))) for row in usable])
    order = np.argsort(risk_score, kind="mergesort")
    bins = np.array_split(order, min(4, len(order)))
    reliability = []
    for index, indices in enumerate(bins):
        reliability.append(
            {
                "bin": int(index + 1),
                "episodes": int(indices.size),
                "mean_risk_score": float(np.mean(risk_score[indices])),
                "unsafe_rate": float(np.mean(unsafe[indices])),
                "collision_rate": float(np.mean(collision[indices])),
            }
        )
    return {
        "episodes": len(rows),
        "usable_episodes": len(usable),
        "dropped_nonfinite": len(rows) - len(usable),
        "minimum_slack_s": float(np.min(slack)),
        "mean_slack_s": float(np.mean(slack)),
        "median_slack_s": float(np.median(slack)),
        "unsafe_rate": float(np.mean(unsafe)),
        "collision_rate": float(np.mean(collision)),
        "unsafe_auc": auc(risk_score, unsafe),
        "collision_auc": auc(risk_score, collision),
        "risk_reliability_bins": reliability,
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 17 RNIC Diagnostic Analysis",
        "",
        "Risk score is `-minimum_best_slack_s`; outcomes are post-hoc episode results and never enter control.",
        "",
        "| Group | Episodes | Mean slack (s) | Unsafe | Collision | Unsafe AUROC | Collision AUROC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, summary in payload["groups"].items():
        lines.append(
            f"| {label} | {summary['usable_episodes']} | {summary['mean_slack_s']:.3f} | "
            f"{summary['unsafe_rate']:.2%} | {summary['collision_rate']:.2%} | "
            f"{summary['unsafe_auc']:.3f} | {summary['collision_auc']:.3f} |"
        )
    pooled = payload["pooled"]
    lines.append(
        f"| pooled | {pooled['usable_episodes']} | {pooled['mean_slack_s']:.3f} | "
        f"{pooled['unsafe_rate']:.2%} | {pooled['collision_rate']:.2%} | "
        f"{pooled['unsafe_auc']:.3f} | {pooled['collision_auc']:.3f} |"
    )
    return "\n".join(lines) + "\n"


def log_tensorboard(payload: dict[str, Any], output: Path) -> None:
    if output is None:
        return
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("TensorBoard analysis logging requires tensorboard") from error
    output.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=str(output), flush_secs=5) as writer:
        writer.add_text("Analysis/config", json.dumps(payload["inputs"], indent=2), 0)
        for label, summary in payload["groups"].items():
            for key in (
                "minimum_slack_s",
                "mean_slack_s",
                "unsafe_rate",
                "collision_rate",
                "unsafe_auc",
                "collision_auc",
            ):
                value = summary[key]
                if math.isfinite(float(value)):
                    writer.add_scalar(f"RNIC/{label}/{key}", float(value), 0)
        pooled = payload["pooled"]
        for key in ("minimum_slack_s", "mean_slack_s", "unsafe_rate", "collision_rate", "unsafe_auc", "collision_auc"):
            value = pooled[key]
            if math.isfinite(float(value)):
                writer.add_scalar(f"RNIC/pooled/{key}", float(value), 0)


def main() -> None:
    args = parse_args()
    runs = [parse_run(specification) for specification in args.run]
    groups = {label: read_rows(directory) for label, directory in runs}
    pooled_rows = [row for rows in groups.values() for row in rows]
    payload = {
        "inputs": {"runs": {label: str(directory) for label, directory in runs}},
        "groups": {label: summarize(rows) for label, rows in groups.items()},
        "pooled": summarize(pooled_rows),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(payload), encoding="utf-8")
    log_tensorboard(payload, args.tensorboard_dir.resolve() if args.tensorboard_dir else None)
    print(markdown(payload), end="")


if __name__ == "__main__":
    main()
