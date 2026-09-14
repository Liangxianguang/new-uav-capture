"""Aggregate Phase 65 prefix/suffix/terminal QDR diagnostics.

The evaluator writes both methods into each seed directory.  This report
selects the strong delayed reference and immutable QDR method separately while
keeping the same frozen scenes, mirror-group bootstrap unit, and predictor
training seeds.
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
    summarize_group,
)


METHODS = {
    "strong_delayed": "M0_current_state_delayed_mpc",
    "immutable_qdr": "M6_qdr_synchronous_mpc",
}
LATENCY_ORDER = (
    "predictor_latency_ms",
    "planner_latency_ms",
    "qdr_or_tube_latency_ms",
    "safety_latency_ms",
    "total_control_latency_ms",
)


def _finite_mean(rows: list[dict[str, Any]], field: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(field, float("nan")))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def _finite_max(rows: list[dict[str, Any]], field: str) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = float(row.get(field, float("nan")))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            values.append(value)
    return float(np.max(values)) if values else float("nan")


def _merge_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    merged: dict[str, int] = {}
    for row in rows:
        value = row.get(field, {})
        if not isinstance(value, dict):
            continue
        for key, count in value.items():
            merged[str(key)] = merged.get(str(key), 0) + int(count)
    return dict(sorted(merged.items()))


def _rows(artifacts: list[Any], method: str, block: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        selected = artifact.methods[method]
        if block is None:
            rows.extend(selected)
            continue
        scenes = [json.loads(line) for line in (artifact.path / "scenes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        block_by_episode = {int(scene["episode_index"]): str(scene.get("scene_block", "unknown")) for scene in scenes}
        rows.extend(row for row in selected if block_by_episode.get(int(row["episode_index"])) == block)
    return rows


def _diagnostics(artifacts: list[Any], method: str, block: str | None = None) -> dict[str, Any]:
    rows = _rows(artifacts, method, block)
    return {
        "episodes": len(rows),
        "qdr_segment_status_counts": _merge_counts(rows, "qdr_segment_status_counts"),
        "qdr_terminal_any_candidate_feasible_rate": _finite_mean(
            rows, "qdr_terminal_any_candidate_feasible_rate"
        ),
        "qdr_terminal_all_candidate_feasible_rate": _finite_mean(
            rows, "qdr_terminal_all_candidate_feasible_rate"
        ),
        "qdr_segment_finite_progress_rate": _finite_mean(rows, "qdr_segment_finite_progress_rate"),
        "qdr_segment_best_progress_m": _finite_mean(rows, "qdr_segment_best_progress_m"),
        "qdr_segment_worst_progress_m": _finite_mean(rows, "qdr_segment_worst_progress_m"),
        "qdr_segment_best_terminal_distance_m": _finite_mean(
            rows, "qdr_segment_best_terminal_distance_m"
        ),
        "qdr_segment_worst_terminal_distance_m": _finite_mean(
            rows, "qdr_segment_worst_terminal_distance_m"
        ),
        "qdr_suffix_gate_max_exhaustion_streak_steps": _finite_max(
            rows, "qdr_suffix_gate_max_exhaustion_streak_steps"
        ),
    }


def _blocks(artifacts: list[Any]) -> list[str]:
    values: set[str] = set()
    for artifact in artifacts:
        for line in (artifact.path / "scenes.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                values.add(str(json.loads(line).get("scene_block", "unknown")))
    return sorted(values)


def _pct(value: float) -> str:
    return f"{value:.2%}" if np.isfinite(value) else "n/a"


def _json_safe(value: Any) -> Any:
    """Convert diagnostic NaN/Inf values to JSON null without hiding them."""

    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _interval(value: dict[str, Any]) -> str:
    return (
        f"{float(value['mean']):.2%}"
        f" [{float(value['bootstrap_95_ci'][0]):.2%}, {float(value['bootstrap_95_ci'][1]):.2%}]"
    )


def _delta(value: dict[str, Any]) -> str:
    return (
        f"{float(value['mean_delta_candidate_minus_reference']) * 100.0:+.2f} pp"
        f" [{float(value['paired_bootstrap_95_ci'][0]) * 100.0:+.2f},"
        f" {float(value['paired_bootstrap_95_ci'][1]) * 100.0:+.2f}] pp"
    )


def _num(value: float) -> str:
    return f"{value:.3f}" if np.isfinite(value) else "n/a"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    groups = load_groups([
        f"strong_delayed={args.runs_pattern}",
        f"immutable_qdr={args.runs_pattern}",
    ])
    rng = np.random.default_rng(args.bootstrap_seed)
    summaries = {
        group: summarize_group(artifacts, rng, args.bootstrap_samples, "mirror_group")
        for group, artifacts in groups.items()
    }
    comparisons = {
        "immutable_qdr_minus_strong_delayed": paired_method_comparison(
            groups["strong_delayed"],
            groups["immutable_qdr"],
            METHODS["strong_delayed"],
            METHODS["immutable_qdr"],
            rng,
            args.bootstrap_samples,
            "mirror_group",
        )
    }
    diagnostics: dict[str, Any] = {}
    for group, method in METHODS.items():
        diagnostics[group] = {
            "overall": _diagnostics(groups[group], method),
            "by_block": {
                block: _diagnostics(groups[group], method, block)
                for block in _blocks(groups[group])
            },
        }
    return {
        "experiment": "phase65_qdr_segment_liveness_audit",
        "evaluation_split": "development_calibration",
        "scene_manifest_sha256": groups["strong_delayed"][0].scene_hash,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "matched predictor training seed and mirror_group",
        },
        "methods": METHODS,
        "groups": summaries,
        "paired_comparisons": comparisons,
        "diagnostics": diagnostics,
        "claim_boundary": {
            "stage": "development_calibration_only",
            "public_candidate_space_terminal_audit": True,
            "local_cbf": "empirical_filter_only",
            "robust_cbf_qp": "diagnostic_no_go",
            "locked_test_tuning": False,
        },
    }


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 65：QDR prefix/suffix/terminal liveness audit",
        "",
        "本阶段在全新 tail-stress development calibration 上做 public-candidate-space 诊断。terminal feasibility 与 progress 只使用预测候选轨迹，不读取 simulator target，也不改变队列 authority；因此它们不是安全证明或真实目标捕获保证。",
        "",
        "## 主要闭环结果",
        "",
        "| 方法 | Safe capture | Collision | Boundary | Timeout | Total p50/p95/p99 (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, method in METHODS.items():
        summary = payload["groups"][group][method]
        metrics = summary["episode_metrics"]
        total = summary["pooled_step_latency_ms"]["total_control_latency_ms"]
        lines.append(
            f"| {group} | {_interval(metrics['safe_capture_success'])} | "
            f"{_interval(metrics['collision'])} | {_interval(metrics['boundary_violation'])} | "
            f"{_interval(metrics['timeout'])} | {total['p50']:.3f}/{total['p95']:.3f}/{total['p99']:.3f} |"
        )
    comparison = payload["paired_comparisons"]["immutable_qdr_minus_strong_delayed"]["episode_metrics"]
    lines.extend([
        "",
        "Immutable QDR 相对 strong delayed-MPC 的 paired delta：",
        "",
        f"- safe capture：`{_delta(comparison['safe_capture_success'])}`；",
        f"- collision：`{_delta(comparison['collision'])}`；",
        f"- timeout：`{_delta(comparison['timeout'])}`。",
        "",
        "## 五类延迟分位数",
        "",
        "单位为 ms；每项为 p50/p95/p99。",
        "",
        "| 方法 | predictor | planner | QDR/tube | safety | total |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for group, method in METHODS.items():
        latency = payload["groups"][group][method]["pooled_step_latency_ms"]
        values = [latency[name] for name in LATENCY_ORDER]
        lines.append(
            "| " + group + " | " + " | ".join(
                f"{item['p50']:.3f}/{item['p95']:.3f}/{item['p99']:.3f}" for item in values
            ) + " |"
        )
    lines.extend([
        "",
        "## Prefix/suffix/terminal 诊断",
        "",
        "| 方法 | block | finite progress | any terminal capture | all terminal capture | best progress (m) | best terminal distance (m) | max exhaustion streak |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for group in METHODS:
        for block, values in payload["diagnostics"][group]["by_block"].items():
            lines.append(
                f"| {group} | {block} | {_pct(values['qdr_segment_finite_progress_rate'])} | "
                f"{_pct(values['qdr_terminal_any_candidate_feasible_rate'])} | "
                f"{_pct(values['qdr_terminal_all_candidate_feasible_rate'])} | "
                f"{_num(values['qdr_segment_best_progress_m'])} | "
                f"{_num(values['qdr_segment_best_terminal_distance_m'])} | "
                f"{_num(values['qdr_suffix_gate_max_exhaustion_streak_steps'])} |"
            )
    lines.extend([
        "",
        "## 阶段判定",
        "",
        "本阶段的 segment audit 作为 failure-taxonomy 证据，不设为安全 gate；它只用于判断 timeout/exhaustion 是否由 prefix、suffix 或 terminal progress 缺失驱动。后续若要改 planner，必须在新的 calibration 上预注册并重新比较，不能在此数据上调参后访问 locked-test。",
        "",
        "local CBF 仍是经验过滤器；robust CBF-QP/R-CLBF-QP 仍为 diagnostic No-Go，不宣称安全证明。",
        "",
        "## 工件",
        "",
        "- fresh manifest：`results/phase65_qdr_segment_liveness_audit/manifest.json`；",
        "- calibration split：`results/phase65_qdr_segment_liveness_audit/development_calibration/scenes.jsonl`；",
        "- 三个 seed 的闭环日志包含 episode/step JSONL、config snapshot 与 TensorBoard event。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-pattern", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    args = parser.parse_args()
    payload = build_payload(args)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps(_json_safe(payload), indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
