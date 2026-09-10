"""Decompose queue-aware execution failures from preserved step JSONL logs.

This tool is diagnostic only. It does not reinterpret a failed certificate as
safe, and it does not estimate episode-level generalization from step rows.
It separates queue-prefix infeasibility, certificate failures, fallback use,
and actual post-state safety so later controller changes can be compared on
the same recorded trajectories.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_HORIZON_PATTERN = re.compile(r"step\[(\d+)\]")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else float("nan")


def _percentiles(values: Iterable[float]) -> dict[str, float]:
    finite = np.asarray([float(value) for value in values if np.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan")}
    return {
        "p50": float(np.percentile(finite, 50.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "p99": float(np.percentile(finite, 99.0)),
    }


def barrier_family(value: Any) -> str:
    """Map a detailed barrier label to a stable diagnostic family."""

    label = str(value or "")
    if not label:
        return "none"
    lowered = label.lower()
    if "boundary" in lowered:
        return "boundary"
    if "obstacle" in lowered:
        return "obstacle"
    if "inter_agent" in lowered:
        return "inter_agent"
    if "continuous" in lowered:
        return "continuous_contract"
    if "prefix" in lowered or "queue" in lowered:
        return "queue_prefix"
    if "action" in lowered or "speed" in lowered:
        return "action_limits"
    if "current_state" in lowered or "safe_set" in lowered:
        return "state_contract"
    return "other"


def barrier_horizon(value: Any) -> str:
    label = str(value or "")
    match = _HORIZON_PATTERN.search(label)
    return match.group(1) if match else "none"


def _counter_summary(counter: Counter[str], total: int) -> dict[str, dict[str, float | int]]:
    return {
        key: {"count": int(count), "rate": _rate(int(count), total)}
        for key, count in counter.most_common()
    }


def summarize_steps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    executed_rows = [row for row in rows if row.get("post_step_executed", True) is not False]
    executed_total = len(executed_rows)
    first_failed = Counter(str(row.get("first_failed_barrier") or "none") for row in rows)
    families = Counter(barrier_family(row.get("first_failed_barrier")) for row in rows)
    horizons = Counter(barrier_horizon(row.get("first_failed_barrier")) for row in rows)
    fallback_types = Counter(
        str(row.get("fallback_candidate_type") or "none")
        for row in rows
        if bool(row.get("fallback_used", False))
    )
    failure_categories = Counter(
        str(row.get("failure_category") or "none")
        for row in rows
        if row.get("failure_category") not in (None, "none")
    )
    execution_errors = [
        float(row["action_execution_error_norm_mps"])
        for row in rows
        if row.get("action_execution_error_norm_mps") is not None
    ]

    prefix_abort = sum(bool(row.get("abort_required", False)) for row in rows)
    prefix_admissible = sum(bool(row.get("prefix_admissible", False)) for row in rows)
    fallback = sum(bool(row.get("fallback_used", False)) for row in rows)
    fallback_certified = sum(bool(row.get("fallback_certificate_valid", False)) for row in rows)
    rollout_invalid = sum(not bool(row.get("execution_rollout_certificate_valid", False)) for row in rows)
    continuous_invalid = sum(not bool(row.get("continuous_segment_certificate_valid", False)) for row in rows)
    actual_unsafe = sum(
        not bool(row.get("actual_post_robust_state_safe", False)) for row in executed_rows
    )
    continuous_only_gap = sum(
        not bool(row.get("continuous_segment_certificate_valid", False))
        and bool(row.get("actual_post_robust_state_safe", False))
        for row in executed_rows
    )
    actual_unsafe_after_certificate = sum(
        bool(row.get("continuous_segment_certificate_valid", False))
        and not bool(row.get("actual_post_robust_state_safe", False))
        for row in executed_rows
    )
    unexecuted_abort = sum(
        row.get("post_step_executed") is False
        and bool(row.get("safety_abort_requested", False))
        for row in rows
    )

    return {
        "steps": total,
        "executed_steps": executed_total,
        "unexecuted_safety_abort": {
            "count": unexecuted_abort,
            "rate": _rate(unexecuted_abort, total),
        },
        "prefix_admissible": {"count": prefix_admissible, "rate": _rate(prefix_admissible, total)},
        "abort_required": {"count": prefix_abort, "rate": _rate(prefix_abort, total)},
        "fallback": {"count": fallback, "rate": _rate(fallback, total)},
        "fallback_certified": {"count": fallback_certified, "rate": _rate(fallback_certified, total)},
        "execution_rollout_invalid": {"count": rollout_invalid, "rate": _rate(rollout_invalid, total)},
        "continuous_segment_invalid": {"count": continuous_invalid, "rate": _rate(continuous_invalid, total)},
        "actual_post_robust_unsafe": {
            "count": actual_unsafe,
            "rate": _rate(actual_unsafe, executed_total),
        },
        "continuous_only_gap": {
            "count": continuous_only_gap,
            "rate": _rate(continuous_only_gap, executed_total),
        },
        "actual_unsafe_after_continuous_certificate": {
            "count": actual_unsafe_after_certificate,
            "rate": _rate(actual_unsafe_after_certificate, executed_total),
        },
        "failure_categories": _counter_summary(failure_categories, total),
        "first_failed_barriers": _counter_summary(first_failed, total),
        "barrier_families": _counter_summary(families, total),
        "barrier_horizons": _counter_summary(horizons, total),
        "fallback_candidate_types": _counter_summary(fallback_types, total),
        "execution_error_norm_mps": _percentiles(execution_errors),
    }


def discover_step_logs(input_dir: Path) -> list[Path]:
    return sorted(input_dir.resolve().glob("*/robust_cbf_qp/steps.jsonl"))


def analyze_directory(input_dir: Path) -> dict[str, Any]:
    logs = discover_step_logs(input_dir)
    if not logs:
        raise FileNotFoundError(f"No */robust_cbf_qp/steps.jsonl found under {input_dir}")
    variants: dict[str, Any] = {}
    for path in logs:
        variant = path.parent.parent.name
        rows = _read_jsonl(path)
        variants[variant] = {
            "steps_path": str(path),
            "summary": summarize_steps(rows),
        }
    return {
        "input_dir": str(input_dir.resolve()),
        "variants": variants,
        "diagnostic_only": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze_directory(args.input_dir)
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
