"""Aggregate Phase79 recurrent-residual validation results across seeds.

The input directories must contain the standalone ``evaluation.json`` files
written by ``train_phase79_dagger_residual.py --mode evaluate``.  This helper
keeps the aggregate independent of generated ``results/`` artifacts, reports
episode-bootstrap intervals, and refuses locked-test inputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


SEED_PATTERN = re.compile(r"_seed(?P<seed>[0-9]+)(?:_|$)")
OUTCOME_FIELDS = (
    "safe_capture_success",
    "capture_event",
    "collision",
    "boundary_violation",
    "timeout",
)
PROMOTION_GATE = {
    "min_safe_capture_rate": 0.70,
    "max_collision_rate": 0.05,
    "max_boundary_violation_rate": 0.05,
    "max_timeout_rate": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=Path, help="Run directory with evaluation.json.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    return parser.parse_args()


def seed_from_path(path: Path) -> int:
    match = SEED_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"Run directory must contain _seed<integer>: {path}")
    return int(match.group("seed"))


def load_run(path: Path) -> tuple[int, dict[str, Any]]:
    path = path.resolve()
    evaluation_path = path / "evaluation.json"
    if not evaluation_path.is_file():
        raise FileNotFoundError(f"Missing evaluation.json: {evaluation_path}")
    result = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Evaluation result must be a JSON object: {evaluation_path}")
    if bool(result.get("locked_test_used", True)):
        raise ValueError(f"Phase79 aggregate refuses locked-test input: {evaluation_path}")
    rows = result.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Evaluation result has no episode rows: {evaluation_path}")
    return seed_from_path(path), result


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, samples: int) -> list[float]:
    if values.size == 0:
        return [0.0, 0.0]
    if samples <= 0:
        raise ValueError("bootstrap-samples must be positive.")
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def rate(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([bool(row[field]) for row in rows]))


def main() -> None:
    args = parse_args()
    runs = [load_run(path) for path in args.run]
    seeds = [seed for seed, _result in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Phase79 aggregate requires unique training seeds.")
    runs.sort(key=lambda item: item[0])
    episode_ids: list[int] | None = None
    pooled_rows: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    for seed, result in runs:
        rows = list(result["rows"])
        current_ids = [int(row["episode_index"]) for row in rows]
        if episode_ids is None:
            episode_ids = current_ids
        elif current_ids != episode_ids:
            raise ValueError("All Phase79 seeds must use the same ordered validation episodes.")
        for row in rows:
            if any(field not in row for field in OUTCOME_FIELDS):
                raise ValueError(f"Episode row is missing an outcome field for seed {seed}.")
        pooled_rows.extend(rows)
        per_seed.append(
            {
                "seed": seed,
                "run": str(Path(args.run[seeds.index(seed)]).resolve()),
                "episodes": len(rows),
                **{f"{field}_rate": rate(rows, field) for field in OUTCOME_FIELDS},
                "mean_capture_time_seconds": float(np.mean([float(row.get("steps", 0)) for row in rows]) * 0.1),
                "mean_min_clearance_m": float(np.mean([float(row["min_clearance_m"]) for row in rows])),
                "latency_ms": result.get("latency_ms", {}),
            }
        )

    rng = np.random.default_rng(args.bootstrap_seed)
    rates = {f"{field}_rate": rate(pooled_rows, field) for field in OUTCOME_FIELDS}
    intervals = {
        key: bootstrap_interval(
            np.asarray([float(bool(row[field])) for row in pooled_rows], dtype=np.float64),
            rng,
            args.bootstrap_samples,
        )
        for key, field in ((f"{field}_rate", field) for field in OUTCOME_FIELDS)
    }
    latency_quantiles: dict[str, dict[str, float]] = {}
    for component in ("route_intent", "actor", "safety", "total"):
        latency_quantiles[component] = {
            percentile: float(
                np.mean(
                    [
                        float(result.get("latency_ms", {}).get(component, {}).get(percentile, 0.0))
                        for _seed, result in runs
                    ]
                )
            )
            for percentile in ("p50", "p95", "p99")
        }

    promotion_by_seed = [
        bool(
            item["safe_capture_success_rate"] >= PROMOTION_GATE["min_safe_capture_rate"]
            and item["collision_rate"] <= PROMOTION_GATE["max_collision_rate"]
            and item["boundary_violation_rate"] <= PROMOTION_GATE["max_boundary_violation_rate"]
            and item["timeout_rate"] <= PROMOTION_GATE["max_timeout_rate"]
        )
        for item in per_seed
    ]
    aggregate = {
        "episodes": len(pooled_rows),
        "seed_count": len(runs),
        "weighted_rates": rates,
        "bootstrap_95_percent_intervals": intervals,
        "mean_capture_time_seconds": float(np.mean([float(row.get("steps", 0)) for row in pooled_rows]) * 0.1),
        "mean_min_clearance_m": float(np.mean([float(row["min_clearance_m"]) for row in pooled_rows])),
        "latency_ms_mean_across_seeds": latency_quantiles,
    }
    output = {
        "experiment_name": "phase79_dnmcp_teacher_dagger_residual_conservative",
        "evaluation_split": "development_validation_only",
        "locked_test_used": False,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "promotion_gate": PROMOTION_GATE,
        "promotion_pass_by_seed": promotion_by_seed,
        "promotion_pass_all_seeds": bool(all(promotion_by_seed)),
        "local_cbf_is_empirical_filter_only": True,
        "formal_robust_cbf_qp_claim": False,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
