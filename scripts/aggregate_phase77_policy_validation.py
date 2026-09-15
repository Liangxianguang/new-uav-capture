"""Aggregate fixed-pool Phase 77 policy validation summaries.

The script only consumes development/calibration summaries and refuses any
artifact marked as using locked-test.  It reports weighted rates and the mean
of per-seed latency quantiles; raw episode logs remain the source of truth for
paired or bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RATE_KEYS = (
    "safe_capture_rate",
    "capture_event_rate",
    "collision_rate",
    "boundary_violation_rate",
    "timeout_rate",
    "target_crossing_rate",
)

PROMOTION_GATE = {
    "min_safe_capture_rate": 0.70,
    "max_collision_rate": 0.05,
    "max_boundary_violation_rate": 0.05,
    "max_timeout_rate": 0.10,
    "min_target_crossing_rate": 0.95,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries: list[dict[str, Any]] = []
    for path in args.summaries:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if bool(summary.get("locked_test_used", True)):
            raise ValueError(f"Refusing locked-test summary: {path}")
        if summary.get("evaluation_split") != "development_validation_only":
            raise ValueError(f"Unexpected split in {path}: {summary.get('evaluation_split')}")
        summaries.append(summary)
    if not summaries:
        raise ValueError("At least one summary is required.")

    total_episodes = sum(int(item["episodes"]) for item in summaries)
    aggregate: dict[str, Any] = {
        "episodes": total_episodes,
        "seed_count": len(summaries),
        "weighted_rates": {},
        "mean_capture_time_seconds": sum(
            float(item["mean_capture_time_seconds"]) * int(item["episodes"])
            for item in summaries
        )
        / total_episodes,
        "mean_min_clearance_m": sum(
            float(item["mean_min_clearance_m"]) * int(item["episodes"])
            for item in summaries
        )
        / total_episodes,
        "latency_ms_mean_across_seeds": {},
    }
    for key in RATE_KEYS:
        aggregate["weighted_rates"][key] = sum(
            float(item[key]) * int(item["episodes"]) for item in summaries
        ) / total_episodes

    latency_names = ("route_intent", "actor", "safety", "total")
    for name in latency_names:
        aggregate["latency_ms_mean_across_seeds"][name] = {}
        for quantile in ("p50", "p95", "p99"):
            aggregate["latency_ms_mean_across_seeds"][name][quantile] = sum(
                float(item["latency_ms"][name][quantile]) for item in summaries
            ) / len(summaries)

    def passes_gate(item: dict[str, Any]) -> bool:
        return (
            float(item["safe_capture_rate"]) >= PROMOTION_GATE["min_safe_capture_rate"]
            and float(item["collision_rate"]) <= PROMOTION_GATE["max_collision_rate"]
            and float(item["boundary_violation_rate"]) <= PROMOTION_GATE["max_boundary_violation_rate"]
            and float(item["timeout_rate"]) <= PROMOTION_GATE["max_timeout_rate"]
            and float(item["target_crossing_rate"]) >= PROMOTION_GATE["min_target_crossing_rate"]
        )

    output = {
        "experiment_name": "phase77_nominal_policy_fixed_pool_aggregate",
        "evaluation_split": "development_validation_only",
        "locked_test_used": False,
        "source_summaries": [str(path.resolve()) for path in args.summaries],
        "per_seed": summaries,
        "aggregate": aggregate,
        "promotion_gate": PROMOTION_GATE,
        "promotion_pass_by_seed": [passes_gate(item) for item in summaries],
        "promotion_pass_all_seeds": all(passes_gate(item) for item in summaries),
        "local_cbf_is_empirical_filter_only": all(
            bool(item.get("local_cbf_is_empirical_filter_only", False)) for item in summaries
        ),
        "formal_robust_cbf_qp_claim": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
