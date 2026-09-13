"""Evaluate the pre-registered QDR liveness gate and log it to TensorBoard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate-json", type=Path, required=True)
    parser.add_argument("--on-summary", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--timeout-rate-max", type=float, default=0.05)
    parser.add_argument("--timeout-delta-vs-off-max", type=float, default=0.05)
    parser.add_argument("--max-exhaustion-streak-steps-max", type=float, default=24.0)
    return parser.parse_args()


def _metric(aggregate: dict[str, Any], path: list[str]) -> float:
    value: Any = aggregate
    for key in path:
        value = value[key]
    return float(value)


def evaluate(
    aggregate: dict[str, Any],
    summaries: list[dict[str, Any]],
    timeout_rate_max: float,
    timeout_delta_vs_off_max: float,
    max_exhaustion_streak_steps_max: float,
) -> dict[str, Any]:
    on_timeout_rate = _metric(
        aggregate,
        ["groups", "on", "distributed_delayed", "episode_metrics", "timeout", "mean"],
    )
    timeout_delta = _metric(
        aggregate,
        [
            "paired_vs_reference",
            "on",
            "distributed_delayed",
            "episode_metrics",
            "timeout",
            "mean_delta_candidate_minus_reference",
        ],
    )
    max_streaks = [
        float(summary["overall"]["qdr_suffix_gate_max_exhaustion_streak_steps"])
        for summary in summaries
    ]
    observed_max_streak = max(max_streaks)
    checks = {
        "timeout_rate": {
            "observed": on_timeout_rate,
            "limit": timeout_rate_max,
            "pass": on_timeout_rate <= timeout_rate_max,
        },
        "timeout_delta_vs_qdr_off": {
            "observed": timeout_delta,
            "limit": timeout_delta_vs_off_max,
            "pass": timeout_delta <= timeout_delta_vs_off_max,
        },
        "max_exhaustion_streak_steps": {
            "observed": observed_max_streak,
            "limit": max_exhaustion_streak_steps_max,
            "pass": observed_max_streak <= max_exhaustion_streak_steps_max,
        },
    }
    return {
        "gate": "qdr_liveness_confirmation",
        "status": "pass" if all(item["pass"] for item in checks.values()) else "no_go",
        "checks": checks,
        "training_seed_count": len(summaries),
    }


def main() -> None:
    args = parse_args()
    aggregate = json.loads(args.aggregate_json.read_text(encoding="utf-8"))
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in args.on_summary]
    result = evaluate(
        aggregate,
        summaries,
        args.timeout_rate_max,
        args.timeout_delta_vs_off_max,
        args.max_exhaustion_streak_steps_max,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(str(args.tensorboard_dir)) as writer:
        writer.add_text("Gate/config", json.dumps({
            "timeout_rate_max": args.timeout_rate_max,
            "timeout_delta_vs_off_max": args.timeout_delta_vs_off_max,
            "max_exhaustion_streak_steps_max": args.max_exhaustion_streak_steps_max,
        }, sort_keys=True), 0)
        for name, check in result["checks"].items():
            writer.add_scalar(f"Gate/{name}/observed", check["observed"], 0)
            writer.add_scalar(f"Gate/{name}/limit", check["limit"], 0)
            writer.add_scalar(f"Gate/{name}/pass", float(check["pass"]), 0)
        writer.add_scalar("Gate/overall_pass", float(result["status"] == "pass"), 0)
        writer.flush()
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
