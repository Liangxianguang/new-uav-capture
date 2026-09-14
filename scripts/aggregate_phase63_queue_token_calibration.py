"""Aggregate the Phase 63 fresh queue-token calibration.

The three compared groups use different evaluator method names, so the generic
seed aggregator cannot express every cross-method pair.  This wrapper keeps
the same mirror-group hierarchical bootstrap while adding explicit
strong-delayed vs QDR and immutable vs bounded-recovery comparisons.
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
    "qdr_immutable": "M6_qdr_synchronous_mpc",
    "qdr_bounded_replace": "M6_qdr_synchronous_mpc",
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
    numeric = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            numeric.append(number)
    return float(np.mean(numeric)) if numeric else float("nan")


def _full_episode_rows(group_artifacts: list[Any], method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in group_artifacts:
        rows.extend(read_jsonl(artifact.path / method / "episodes.jsonl"))
    return rows


def _merge_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(field, {})
        if not isinstance(value, dict):
            continue
        for key, count in value.items():
            counts[str(key)] = counts.get(str(key), 0) + int(count)
    return dict(sorted(counts.items()))


def _diagnostics(artifacts: list[Any], method: str) -> dict[str, Any]:
    rows = _full_episode_rows(artifacts, method)
    recovery_fields = (
        "qdr_prefix_recovery_request_rate",
        "qdr_prefix_recovery_apply_rate",
        "qdr_prefix_recovery_token_present_rate",
        "qdr_prefix_recovery_ack_accept_rate",
        "qdr_prefix_recovery_ack_apply_rate",
    )
    return {
        field: _finite_mean([row.get(field) for row in rows])
        for field in recovery_fields
    } | {
        "max_exhaustion_streak_steps": max(
            int(float(row.get("qdr_suffix_gate_max_exhaustion_streak_steps", 0.0)))
            for row in rows
        ),
        "ack_reason_counts": _merge_counts(rows, "qdr_prefix_recovery_ack_reason_counts"),
        "episode_rows": len(rows),
    }


def _comparison(
    groups: dict[str, list[Any]],
    reference_group: str,
    reference_method: str,
    candidate_group: str,
    candidate_method: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    return paired_method_comparison(
        groups[reference_group],
        groups[candidate_group],
        reference_method,
        candidate_method,
        rng,
        bootstrap_samples,
        "mirror_group",
    )


def _percent(item: dict[str, Any]) -> str:
    return f"{float(item['mean']):.2%} [{float(item['bootstrap_95_ci'][0]):.2%}, {float(item['bootstrap_95_ci'][1]):.2%}]"


def _delta(item: dict[str, Any]) -> str:
    return f"{float(item['mean_delta_candidate_minus_reference']):+.2%} [{float(item['paired_bootstrap_95_ci'][0]):+.2%}, {float(item['paired_bootstrap_95_ci'][1]):+.2%}]"


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 63 QueueToken Fresh Calibration",
        "",
        "This is a fresh development-calibration result on mirror-group-disjoint scenes. It is not a locked-test result. Intervals are hierarchical bootstrap 95% CIs over matched predictor seeds and mirror groups.",
        "",
        "## Outcome summary",
        "",
        "| Group | Method | Safe capture | Collision | Boundary | Timeout | Total p50/p95/p99 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for group in REPORT_ORDER:
        method = GROUP_METHODS[group]
        values = payload["groups"][group][method]
        metrics = values["episode_metrics"]
        total = values["pooled_step_latency_ms"]["total_control_latency_ms"]
        lines.append(
            f"| {group} | {method} | {_percent(metrics['safe_capture_success'])} | "
            f"{_percent(metrics['collision'])} | {_percent(metrics['boundary_violation'])} | "
            f"{_percent(metrics['timeout'])} | {total['p50']:.2f}/{total['p95']:.2f}/{total['p99']:.2f} |"
        )
    lines.extend(["", "## Paired comparisons", "", "| Candidate vs reference | Safe-capture delta | Collision delta | Timeout delta |", "|---|---:|---:|---:|"])
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
    lines.extend(["", "## Queue-contract diagnostics", "", "| Group | Recovery request | Applied | Token present | ACK accepted | ACK applied | Max exhaustion streak |", "|---|---:|---:|---:|---:|---:|---:|"])
    for group in REPORT_ORDER:
        diagnostic = payload["diagnostics"][group]
        lines.append(
            f"| {group} | {diagnostic['qdr_prefix_recovery_request_rate']:.2%} | "
            f"{diagnostic['qdr_prefix_recovery_apply_rate']:.2%} | "
            f"{diagnostic['qdr_prefix_recovery_token_present_rate']:.2%} | "
            f"{diagnostic['qdr_prefix_recovery_ack_accept_rate']:.2%} | "
            f"{diagnostic['qdr_prefix_recovery_ack_apply_rate']:.2%} | "
            f"{diagnostic['max_exhaustion_streak_steps']} |"
        )
    lines.extend([
        "",
        "## Decision",
        "",
        "The token-matched bounded replacement arm is not promoted: it has a negative paired safe-capture delta and a higher timeout rate than immutable QDR. The immutable arm provides a conditional safety-axis result relative to strong delayed-MPC, but its timeout and exhaustion-streak gates fail. QueueToken/ACK is therefore retained as an execution-consistency mechanism, not as evidence of improved liveness or a safety proof.",
        "",
        "local CBF remains an empirical filter only. Robust CBF-QP/R-CLBF-QP remains diagnostic No-Go. The locked diagnostic split was not read or tuned.",
        "",
    ])
    return "\n".join(lines)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    groups = load_groups(
        [
            f"strong_delayed={args.strong_pattern}",
            f"qdr_immutable={args.immutable_pattern}",
            f"qdr_bounded_replace={args.replace_pattern}",
        ]
    )
    rng = np.random.default_rng(args.bootstrap_seed)
    summaries = {
        group: summarize_group(artifacts, rng, args.bootstrap_samples, "mirror_group")
        for group, artifacts in groups.items()
    }
    comparisons = {
        "immutable_minus_strong": _comparison(
            groups, "strong_delayed", GROUP_METHODS["strong_delayed"],
            "qdr_immutable", GROUP_METHODS["qdr_immutable"], rng, args.bootstrap_samples,
        ),
        "bounded_replace_minus_strong": _comparison(
            groups, "strong_delayed", GROUP_METHODS["strong_delayed"],
            "qdr_bounded_replace", GROUP_METHODS["qdr_bounded_replace"], rng, args.bootstrap_samples,
        ),
        "bounded_replace_minus_immutable": _comparison(
            groups, "qdr_immutable", GROUP_METHODS["qdr_immutable"],
            "qdr_bounded_replace", GROUP_METHODS["qdr_bounded_replace"], rng, args.bootstrap_samples,
        ),
    }
    diagnostics = {
        group: _diagnostics(groups[group], GROUP_METHODS[group])
        for group in REPORT_ORDER
    }
    return {
        "experiment": "phase63_queue_token_calibration",
        "evaluation_split": "validation_selection",
        "scene_manifest_sha256": groups["strong_delayed"][0].scene_hash,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "matched predictor training seed and frozen-scene mirror_group",
        },
        "groups": summaries,
        "paired_comparisons": comparisons,
        "diagnostics": diagnostics,
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
    parser.add_argument("--immutable-pattern", required=True)
    parser.add_argument("--replace-pattern", required=True)
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
