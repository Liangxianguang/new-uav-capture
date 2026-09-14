"""Aggregate the Phase 64 exhausted-candidate liveness repair.

The hard and soft QDR arms use different run directories but the same
evaluator method name.  This wrapper keeps the mirror-group bootstrap unit and
adds the pre-registered paired comparisons needed to decide whether the
planner-side intervention is worth a confirmation run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from aggregate_closed_loop_seed_results import (
    load_groups,
    paired_method_comparison,
    read_jsonl,
    summarize_group,
)


GROUP_METHODS = {
    "strong_delayed": "M0_current_state_delayed_mpc",
    "qdr_hard": "M6_qdr_synchronous_mpc",
    "qdr_soft_progress": "M6_qdr_normalized_soft_progress",
}
REPORT_ORDER = tuple(GROUP_METHODS)
LATENCY_ORDER = (
    "predictor_latency_ms",
    "planner_latency_ms",
    "qdr_or_tube_latency_ms",
    "safety_latency_ms",
    "total_control_latency_ms",
)


def _finite_mean(values: list[Any]) -> float:
    numeric: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            numeric.append(number)
    return float(np.mean(numeric)) if numeric else float("nan")


def _rows(artifacts: list[Any], method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        rows.extend(read_jsonl(artifact.path / method / "episodes.jsonl"))
    return rows


def _diagnostics(artifacts: list[Any], method: str) -> dict[str, Any]:
    rows = _rows(artifacts, method)
    policies = sorted({str(row.get("qdr_exhaustion_policy", "unknown")) for row in rows})
    return {
        "qdr_exhaustion_policies": policies,
        "soft_fallback_count_mean": _finite_mean(
            [row.get("qdr_exhaustion_soft_fallback_count") for row in rows]
        ),
        "soft_fallback_count_total": int(
            sum(float(row.get("qdr_exhaustion_soft_fallback_count", 0.0) or 0.0) for row in rows)
        ),
        "max_exhaustion_streak_steps": int(
            max(
                float(row.get("qdr_suffix_gate_max_exhaustion_streak_steps", 0.0) or 0.0)
                for row in rows
            )
        ),
        "episode_rows": len(rows),
    }


def _comparison(
    groups: dict[str, list[Any]],
    reference_group: str,
    candidate_group: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    return paired_method_comparison(
        groups[reference_group],
        groups[candidate_group],
        GROUP_METHODS[reference_group],
        GROUP_METHODS[candidate_group],
        rng,
        bootstrap_samples,
        "mirror_group",
    )


def _interval(item: dict[str, Any], percent: bool = False) -> str:
    mean = float(item["mean"])
    low, high = (float(value) for value in item["bootstrap_95_ci"])
    if percent:
        return f"{mean:.2%} [{low:.2%}, {high:.2%}]"
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def _delta(item: dict[str, Any]) -> str:
    mean = float(item["mean_delta_candidate_minus_reference"])
    low, high = (float(value) for value in item["paired_bootstrap_95_ci"])
    return f"{mean:+.2%} [{low:+.2%}, {high:+.2%}]"


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 64：QDR exhausted-candidate liveness repair",
        "",
        "This is a fresh development-calibration result. The intervention changes only the planner's exhausted-candidate ranking; queue authority, QDR time indexing, predictor checkpoint, scene manifest, local CBF, and execution contract remain fixed. It is not a locked-test result and does not provide a safety proof.",
        "",
        "## Outcome summary",
        "",
        "| Group | Policy | Safe capture | Collision | Boundary | Timeout | Total p50/p95/p99 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in REPORT_ORDER:
        method = GROUP_METHODS[group]
        values = payload["groups"][group][method]
        metrics = values["episode_metrics"]
        total = values["pooled_step_latency_ms"]["total_control_latency_ms"]
        policy = ",".join(payload["diagnostics"][group]["qdr_exhaustion_policies"])
        lines.append(
            f"| {group} | {policy} | {_interval(metrics['safe_capture_success'], True)} | "
            f"{_interval(metrics['collision'], True)} | {_interval(metrics['boundary_violation'], True)} | "
            f"{_interval(metrics['timeout'], True)} | "
            f"{total['p50']:.2f}/{total['p95']:.2f}/{total['p99']:.2f} |"
        )
    lines.extend([
        "",
        "## Paired comparisons",
        "",
        "| Candidate minus reference | Safe capture | Collision | Timeout |",
        "|---|---:|---:|---:|",
    ])
    for name, comparison in payload["paired_comparisons"].items():
        metrics = comparison["episode_metrics"]
        lines.append(
            f"| {name} | {_delta(metrics['safe_capture_success'])} | "
            f"{_delta(metrics['collision'])} | {_delta(metrics['timeout'])} |"
        )
    lines.extend(["", "## Latency components", "", "| Group | Component | p50 | p95 | p99 |", "|---|---|---:|---:|---:|"])
    for group in REPORT_ORDER:
        method = GROUP_METHODS[group]
        latency = payload["groups"][group][method]["pooled_step_latency_ms"]
        for component in LATENCY_ORDER:
            item = latency[component]
            lines.append(f"| {group} | {component} | {item['p50']:.3f} | {item['p95']:.3f} | {item['p99']:.3f} |")
    lines.extend([
        "",
        "## Exhaustion diagnostics",
        "",
        "| Group | Policy | Mean soft fallbacks/episode | Total soft fallbacks | Max streak |",
        "|---|---|---:|---:|---:|",
    ])
    for group in REPORT_ORDER:
        diagnostic = payload["diagnostics"][group]
        lines.append(
            f"| {group} | {','.join(diagnostic['qdr_exhaustion_policies'])} | "
            f"{diagnostic['soft_fallback_count_mean']:.2f} | "
            f"{diagnostic['soft_fallback_count_total']} | "
            f"{diagnostic['max_exhaustion_streak_steps']} |"
        )
    lines.extend([
        "",
        "## Decision rule",
        "",
        "A confirmation run is allowed only if the soft arm's paired safe-capture lower bound is no worse than -2 pp versus immutable hard QDR, its timeout upper bound is no worse than +5 pp, and its maximum exhaustion streak is at most 24 steps. These gates are evaluated once on this calibration artifact; no locked-test tuning is allowed.",
        "",
        "local CBF remains an empirical filter only. Robust CBF-QP/R-CLBF-QP remains diagnostic No-Go.",
        "",
    ])
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    groups = load_groups([
        f"strong_delayed={args.strong_pattern}",
        f"qdr_hard={args.hard_pattern}",
        f"qdr_soft_progress={args.soft_pattern}",
    ])
    rng = np.random.default_rng(args.bootstrap_seed)
    summaries = {
        group: summarize_group(artifacts, rng, args.bootstrap_samples, "mirror_group")
        for group, artifacts in groups.items()
    }
    comparisons = {
        "soft_progress_minus_hard": _comparison(groups, "qdr_hard", "qdr_soft_progress", rng, args.bootstrap_samples),
        "hard_minus_strong": _comparison(groups, "strong_delayed", "qdr_hard", rng, args.bootstrap_samples),
        "soft_progress_minus_strong": _comparison(groups, "strong_delayed", "qdr_soft_progress", rng, args.bootstrap_samples),
    }
    diagnostics = {
        group: _diagnostics(groups[group], GROUP_METHODS[group])
        for group in REPORT_ORDER
    }
    return {
        "experiment": "phase64_qdr_liveness_repair",
        "evaluation_split": "development_calibration",
        "scene_manifest_sha256": groups["strong_delayed"][0].scene_hash,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "matched predictor training seed and mirror_group",
        },
        "groups": summaries,
        "paired_comparisons": comparisons,
        "diagnostics": diagnostics,
        "gates": {
            "safe_capture_lower_bound_vs_hard": -0.02,
            "timeout_delta_upper_bound_vs_hard": 0.05,
            "maximum_exhaustion_streak_steps": 24,
        },
        "claim_boundary": {
            "stage": "development_calibration_only",
            "local_cbf": "empirical_filter_only",
            "robust_cbf_qp": "diagnostic_no_go",
            "locked_test_tuning": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strong-pattern", required=True)
    parser.add_argument("--hard-pattern", required=True)
    parser.add_argument("--soft-pattern", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    args = parser.parse_args()
    payload = build_payload(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
