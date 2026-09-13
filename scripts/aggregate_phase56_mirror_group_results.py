"""Aggregate Phase 56 baselines with mirror-group bootstrap and frozen gates.

This report is intentionally specific to Phase 56. It keeps upper/lower
mirror members together during bootstrap, reports every required runtime
component, and evaluates the pre-registered development gates without
changing any method configuration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from aggregate_closed_loop_seed_results import (
        OUTCOME_METRICS,
        RunArtifact,
        bootstrap_metric_matrix,
        load_groups,
        metric_summary,
        paired_method_comparison,
        read_jsonl,
        summarize_group,
    )
except ModuleNotFoundError:  # Importable from the repository root as well as CLI execution.
    from scripts.aggregate_closed_loop_seed_results import (
        OUTCOME_METRICS,
        RunArtifact,
        bootstrap_metric_matrix,
        load_groups,
        metric_summary,
        paired_method_comparison,
        read_jsonl,
        summarize_group,
    )


DIAGNOSTIC_METRICS = (
    "qdr_prefix_admissible_rate",
    "qdr_suffix_admissible_rate",
    "qdr_suffix_gate_exhaustion_rate",
    "qdr_suffix_gate_max_exhaustion_streak_steps",
    "mean_qdr_queue_length",
    "mean_qdr_first_controllable_step",
    "max_message_age_steps",
    "mean_message_age_steps",
    "messages_dropped",
    "planner_fallback_count",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        metavar="NAME=GLOB",
        help="Run group, normally phase56=results\\phase56_dev_calibration_seed*.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-method", default="B0_current_state_delayed_mpc")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260913)
    parser.add_argument(
        "--evaluation-split",
        choices=("validation_selection", "validation_confirmation", "locked_test", "ood_diagnostic"),
        required=True,
    )
    parser.add_argument("--report-title", required=True)
    return parser.parse_args()


def subset_artifact(artifact: RunArtifact, episode_ids: set[int]) -> RunArtifact:
    methods = {
        method: [row for row in rows if int(row["episode_index"]) in episode_ids]
        for method, rows in artifact.methods.items()
    }
    steps = {
        method: [row for row in rows if int(row["episode_index"]) in episode_ids]
        for method, rows in artifact.steps.items()
    }
    return RunArtifact(
        label=artifact.label,
        seed=artifact.seed,
        path=artifact.path,
        scene_hash=artifact.scene_hash,
        methods=methods,
        steps=steps,
    )


def scene_selectors(artifacts: list[RunArtifact]) -> dict[str, set[int]]:
    scenes = read_jsonl(artifacts[0].path / "scenes.jsonl")
    selectors: dict[str, set[int]] = {"all": set()}
    for scene in scenes:
        episode_id = int(scene["episode_index"])
        selectors["all"].add(episode_id)
        block = str(scene["scene_block"])
        selectors.setdefault(block, set()).add(episode_id)
        if block in {"delay_noise_grid", "delay_noise_factorial", "interaction_stress"}:
            overrides = scene["execution_overrides"]
            delay = int(overrides["action_delay_steps"])
            noise = float(overrides["command_noise_std"])
            if delay >= 6 or noise >= 0.08:
                selectors.setdefault("stress_delay_ge6_or_noise_ge008", set()).add(episode_id)
    return selectors


def subset_runs(artifacts: list[RunArtifact], episode_ids: set[int]) -> list[RunArtifact]:
    return [subset_artifact(artifact, episode_ids) for artifact in artifacts]


def diagnostic_summary(
    artifacts: list[RunArtifact],
    method: str,
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in DIAGNOSTIC_METRICS:
        _, matrix = bootstrap_metric_matrix(artifacts, method, metric, "mirror_group")
        result[metric] = metric_summary(matrix, rng, bootstrap_samples)
    return result


def method_names(artifacts: list[RunArtifact]) -> list[str]:
    common = set.intersection(*(set(artifact.methods) for artifact in artifacts))
    return sorted(common)


def gate_interval(comparison: dict[str, Any], metric: str) -> tuple[float, float, float]:
    item = comparison["episode_metrics"][metric]
    return (
        float(item["mean_delta_candidate_minus_reference"]),
        float(item["paired_bootstrap_95_ci"][0]),
        float(item["paired_bootstrap_95_ci"][1]),
    )


def compute_gates(
    all_comparisons: dict[str, Any],
    id_comparisons: dict[str, Any],
    stress_comparisons: dict[str, Any],
    id_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    def get(comparisons: dict[str, Any], method: str, metric: str) -> tuple[float, float, float]:
        return gate_interval(comparisons[method], metric)

    b1_id_safe = get(id_comparisons, "B1_qdr_mpc", "safe_capture_success")
    b1_id_collision = get(id_comparisons, "B1_qdr_mpc", "collision")
    b1_id_boundary = get(id_comparisons, "B1_qdr_mpc", "boundary_violation")
    b1_stress_collision = get(stress_comparisons, "B1_qdr_mpc", "collision")
    b1_timeout = get(all_comparisons, "B1_qdr_mpc", "timeout")
    b5_async_safe = get(id_comparisons, "B5_asynchronous_distributed_mpc", "safe_capture_success")
    b5_async_total_p95 = None
    b4_async_total_p95 = None
    # Runtime efficiency is evaluated from the summary outside this function.
    max_exhaustion = float(
        id_diagnostics["B1_qdr_mpc"]["qdr_suffix_gate_max_exhaustion_streak_steps"]["mean"]
    )

    id_noninferiority = (
        b1_id_safe[1] >= -0.02
        and (b1_id_collision[0] <= 0.0 or b1_id_collision[1] <= 0.0 <= b1_id_collision[2])
        and (b1_id_boundary[0] <= 0.0 or b1_id_boundary[1] <= 0.0 <= b1_id_boundary[2])
    )
    stress_safety = b1_stress_collision[0] <= -0.05 and b1_stress_collision[1] <= 0.0
    liveness = b1_timeout[2] <= 0.05 and max_exhaustion <= 24.0
    return {
        "id_noninferiority": {
            "passed": bool(id_noninferiority),
            "safe_capture_delta": b1_id_safe,
            "collision_delta": b1_id_collision,
            "boundary_delta": b1_id_boundary,
        },
        "stress_safety": {
            "passed": bool(stress_safety),
            "collision_delta": b1_stress_collision,
        },
        "liveness": {
            "passed": bool(liveness),
            "timeout_delta": b1_timeout,
            "mean_max_exhaustion_streak_steps": max_exhaustion,
        },
        "async_efficiency": {
            "passed": False,
            "status": "evaluated after total p95 summaries are available",
            "safe_capture_delta": b5_async_safe,
            "total_p95_reduction_fraction": None,
        },
        "promotion": {
            "passed": False,
            "status": "requires all four primary gates, including asynchronous efficiency",
        },
        "_unused": {"b5_async_total_p95": b5_async_total_p95, "b4_async_total_p95": b4_async_total_p95},
    }


def apply_runtime_gates(gates: dict[str, Any], summary: dict[str, Any], id_summary: dict[str, Any]) -> None:
    b5 = id_summary["B5_asynchronous_distributed_mpc"]["pooled_step_latency_ms"]["total_control_latency_ms"]
    b4 = id_summary["B4_synchronous_distributed_mpc"]["pooled_step_latency_ms"]["total_control_latency_ms"]
    reduction = 1.0 - float(b5["p95"]) / float(b4["p95"])
    b5_safe = gates["async_efficiency"]["safe_capture_delta"]
    passed = reduction >= 0.15 and b5_safe[1] >= -0.02
    gates["async_efficiency"].update(
        {
            "passed": bool(passed),
            "total_p95_reduction_fraction": reduction,
            "asynchronous_total_p95_ms": float(b5["p95"]),
            "synchronous_total_p95_ms": float(b4["p95"]),
        }
    )
    primary = (gates["id_noninferiority"]["passed"], gates["stress_safety"]["passed"], gates["liveness"]["passed"], passed)
    gates["promotion"] = {
        "passed": bool(all(primary)),
        "status": "QDR promoted as primary contribution" if all(primary) else "QDR remains conditional diagnostic",
    }
    gates.pop("_unused", None)


def render_interval(item: dict[str, Any], percent: bool = False) -> str:
    if item.get("mean") is None or any(x is None for x in item.get("bootstrap_95_ci", [])):
        return "n/a"
    value = float(item["mean"])
    low, high = (float(x) for x in item["bootstrap_95_ci"])
    if not all(np.isfinite([value, low, high])):
        return "n/a"
    if percent:
        return f"{value:.2%} [{low:.2%}, {high:.2%}]"
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def render_delta(item: dict[str, Any], percent: bool = True) -> str:
    if item.get("mean_delta_candidate_minus_reference") is None or any(
        x is None for x in item.get("paired_bootstrap_95_ci", [])
    ):
        return "n/a"
    value = float(item["mean_delta_candidate_minus_reference"])
    low, high = (float(x) for x in item["paired_bootstrap_95_ci"])
    if not all(np.isfinite([value, low, high])):
        return "n/a"
    if percent:
        return f"{value:.2%} [{low:.2%}, {high:.2%}]"
    return f"{value:.3f} [{low:.3f}, {high:.3f}]"


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['report_title']}",
        "",
        f"- Evaluation split: `{payload['evaluation_split']}`; scene manifest SHA-256: `{payload['scene_manifest_sha256']}`.",
        "- Bootstrap: matched predictor seeds and `mirror_group` units; upper/lower members are averaged within each group before resampling.",
        "- These are development/calibration results unless the split label says otherwise. No locked-test tuning is permitted.",
        "",
        "## Overall closed-loop results",
        "",
        "| Method | Safe capture | Collision | Boundary | Timeout | Capture time (s) | Clearance (m) | Total p50/p95/p99 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    overall = payload["summaries"]["all"]
    for method, values in overall.items():
        metrics = values["episode_metrics"]
        total = values["pooled_step_latency_ms"]["total_control_latency_ms"]
        lines.append(
            f"| {method} | {render_interval(metrics['safe_capture_success'], True)} | "
            f"{render_interval(metrics['collision'], True)} | {render_interval(metrics['boundary_violation'], True)} | "
            f"{render_interval(metrics['timeout'], True)} | {render_interval(metrics['capture_time_seconds'])} | "
            f"{render_interval(metrics['min_clearance_m'])} | {total['p50']:.2f}/{total['p95']:.2f}/{total['p99']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Runtime decomposition (p50 / p95 / p99 ms)",
            "",
            "| Method | Predictor | Planner | QDR/tube | Safety | Total |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method, values in overall.items():
        latency = values["pooled_step_latency_ms"]
        fmt = lambda name: f"{latency[name]['p50']:.2f}/{latency[name]['p95']:.2f}/{latency[name]['p99']:.2f}"
        lines.append(f"| {method} | {fmt('predictor_latency_ms')} | {fmt('planner_latency_ms')} | {fmt('qdr_or_tube_latency_ms')} | {fmt('safety_latency_ms')} | {fmt('total_control_latency_ms')} |")
    lines.extend(
        [
            "",
            "## QDR and communication diagnostics",
            "",
            "| Method | Prefix admissible | Suffix admissible | Exhaustion rate | Max exhaustion streak | Queue length | Max message age | Dropped messages |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for method, diagnostics in payload["diagnostics"]["all"].items():
        lines.append(
            f"| {method} | {render_interval(diagnostics['qdr_prefix_admissible_rate'], True)} | "
            f"{render_interval(diagnostics['qdr_suffix_admissible_rate'], True)} | "
            f"{render_interval(diagnostics['qdr_suffix_gate_exhaustion_rate'], True)} | "
            f"{render_interval(diagnostics['qdr_suffix_gate_max_exhaustion_streak_steps'])} | "
            f"{render_interval(diagnostics['mean_qdr_queue_length'])} | "
            f"{render_interval(diagnostics['max_message_age_steps'])} | "
            f"{render_interval(diagnostics['messages_dropped'])} |"
        )
    lines.extend(["", "## B1 QDR paired deltas versus B0", "", "| Scope | Safe capture | Collision | Timeout |", "| --- | ---: | ---: | ---: |"])
    for scope, comparisons in payload["paired_vs_b0"].items():
        comparison = comparisons["B1_qdr_mpc"]
        metrics = comparison["episode_metrics"]
        lines.append(f"| {scope} | {render_delta(metrics['safe_capture_success'])} | {render_delta(metrics['collision'])} | {render_delta(metrics['timeout'])} |")
    lines.extend(
        [
            "",
            "## Factorized paired contrasts",
            "",
            "These contrasts keep scenes and predictor seeds paired. They are evidence for the planned factor decomposition, not a claim that the implementation has isolated every causal pathway.",
            "",
            "| Scope | B1-B0 safe/collision | B5-B4 safe/collision | B6-B5 safe/collision | B6-B4 safe/collision |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for scope, pairs in payload["factor_contrasts"].items():
        def pair_text(key: str) -> str:
            metrics = pairs[key]["episode_metrics"]
            return f"{render_delta(metrics['safe_capture_success'])} / {render_delta(metrics['collision'])}"
        lines.append(
            f"| {scope} | {pair_text('B1_minus_B0')} | {pair_text('B5_minus_B4')} | "
            f"{pair_text('B6_minus_B5')} | {pair_text('B6_minus_B4')} |"
        )
    lines.extend(["", "## Pre-registered gate status", "", "| Gate | Status | Evidence |", "| --- | --- | --- |"])
    for name, gate in payload["gates"].items():
        if name == "promotion":
            evidence = gate["status"]
        elif name == "async_efficiency":
            evidence = f"p95 reduction={gate['total_p95_reduction_fraction']:.2%}; safe-capture CI lower={gate['safe_capture_delta'][1]:.2%}"
        elif name == "liveness":
            evidence = f"timeout CI upper={gate['timeout_delta'][2]:.2%}; mean max streak={gate['mean_max_exhaustion_streak_steps']:.2f}"
        elif name == "stress_safety":
            evidence = f"collision delta={render_tuple(gate['collision_delta'])}"
        else:
            evidence = f"safe-capture delta={render_tuple(gate['safe_capture_delta'])}"
        lines.append(f"| {name} | {'PASS' if gate['passed'] else 'FAIL'} | {evidence} |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "`local_cbf` is an empirical action filter. These results do not establish an R-CLBF-QP certificate, forward invariance, or a real-flight safety proof. A failed gate freezes the method as a diagnostic result; it does not authorize retuning on the locked split.",
            "",
        ]
    )
    return "\n".join(lines)


def render_tuple(values: tuple[float, float, float]) -> str:
    if not all(np.isfinite(values)):
        return "n/a"
    return f"{values[0]:.2%} [{values[1]:.2%}, {values[2]:.2%}]"


def json_safe(value: Any) -> Any:
    """Convert NumPy scalars and non-finite floats to JSON-safe values."""

    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (int, np.integer, bool)) or value is None or isinstance(value, str):
        return value
    return value


def main() -> None:
    args = parse_args()
    groups = load_groups(args.group)
    if len(groups) != 1:
        raise ValueError("Phase 56 mirror-group report expects exactly one run group.")
    label, artifacts = next(iter(groups.items()))
    methods = method_names(artifacts)
    if args.reference_method not in methods:
        raise ValueError(f"Reference method {args.reference_method!r} is missing.")
    selectors = scene_selectors(artifacts)
    summaries: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    factor_contrasts: dict[str, Any] = {}
    factor_specs = {
        "B1_minus_B0": ("B0_current_state_delayed_mpc", "B1_qdr_mpc"),
        "B5_minus_B4": ("B4_synchronous_distributed_mpc", "B5_asynchronous_distributed_mpc"),
        "B6_minus_B5": ("B5_asynchronous_distributed_mpc", "B6_qdr_asynchronous_mpc"),
        "B6_minus_B4": ("B4_synchronous_distributed_mpc", "B6_qdr_asynchronous_mpc"),
    }
    for index, (scope, episode_ids) in enumerate(selectors.items()):
        selected = subset_runs(artifacts, episode_ids)
        rng = np.random.default_rng(args.bootstrap_seed + index)
        summaries[scope] = summarize_group(selected, rng, args.bootstrap_samples, "mirror_group")
        diagnostics[scope] = {
            method: diagnostic_summary(selected, method, rng, args.bootstrap_samples)
            for method in methods
        }
        paired[scope] = {
            method: paired_method_comparison(
                selected,
                selected,
                args.reference_method,
                method,
                rng,
                args.bootstrap_samples,
                "mirror_group",
            )
            for method in methods
            if method != args.reference_method
        }
        factor_contrasts[scope] = {
            name: paired_method_comparison(
                selected,
                selected,
                reference_method,
                candidate_method,
                rng,
                args.bootstrap_samples,
                "mirror_group",
            )
            for name, (reference_method, candidate_method) in factor_specs.items()
        }
    id_scope = next((scope for scope in ("id_reference", "id_replication") if scope in selectors), None)
    if id_scope is None:
        raise ValueError("No ID reference scope is present in the scene manifest.")
    stress_scope = "stress_delay_ge6_or_noise_ge008"
    if stress_scope not in paired:
        raise ValueError("No declared delay/noise stress scope is present in the scene manifest.")
    gates = compute_gates(
        paired["all"],
        paired[id_scope],
        paired[stress_scope],
        diagnostics[id_scope],
    )
    apply_runtime_gates(gates, summaries["all"], summaries[id_scope])
    payload = {
        "schema_version": "phase56-mirror-group-aggregate-v1",
        "report_title": args.report_title,
        "evaluation_split": args.evaluation_split,
        "run_group": label,
        "scene_manifest_sha256": artifacts[0].scene_hash,
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.bootstrap_seed, "unit": "mirror_group"},
        "methods": methods,
        "scope_episode_counts": {scope: len(ids) for scope, ids in selectors.items()},
        "summaries": summaries,
        "diagnostics": diagnostics,
        "paired_vs_b0": paired,
        "factor_contrasts": factor_contrasts,
        "gates": gates,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe(payload)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    output.with_suffix(".md").write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps({"output": str(output), "promotion": gates["promotion"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
