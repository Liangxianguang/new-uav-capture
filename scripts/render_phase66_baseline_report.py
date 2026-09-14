"""Render the Phase 66 baseline calibration aggregate as an auditable report.

The input is produced by ``aggregate_phase58_calibration.py``.  This renderer
keeps every scene block, outcome interval, paired comparison, and the complete
predictor/planner/QDR-or-tube/safety/total p50/p95/p99 table visible.  It does
not recompute metrics or open confirmation/locked-test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METHOD_LABELS = {
    "M0_current_state_delayed_mpc": "M0 current-state delayed",
    "M1_known_delay_delayed_mpc": "M1 known-delay delayed",
    "M2_fixed_tube_mpc": "M2 fixed tube",
    "M3_queue_aware_tube_mpc": "M3 queue-aware tube",
    "M5_asynchronous_distributed_mpc": "M5 asynchronous distributed",
    "M6_qdr_synchronous_mpc": "M6 synchronous QDR",
    "M7_qdr_asynchronous_mpc": "M7 asynchronous QDR",
}
OUTCOME_COLUMNS = (
    ("safe_capture_success", "Safe capture", True),
    ("collision", "Collision", True),
    ("boundary_violation", "Boundary", True),
    ("timeout", "Timeout", True),
    ("capture_time_seconds", "Capture time (s)", False),
    ("min_clearance_m", "Min clearance (m)", False),
)
LATENCY_COLUMNS = (
    ("predictor_latency_ms", "Predictor"),
    ("planner_latency_ms", "Planner"),
    ("qdr_or_tube_latency_ms", "QDR/tube"),
    ("safety_latency_ms", "Safety"),
    ("total_control_latency_ms", "Total"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _interval(metric: dict[str, Any], *, percent: bool) -> str:
    mean = metric.get("mean")
    interval = metric.get("bootstrap_95_ci", [None, None])
    if mean is None or interval[0] is None or interval[1] is None:
        return "n/a"
    if percent:
        return f"{float(mean):.2%} [{float(interval[0]):.2%}, {float(interval[1]):.2%}]"
    return f"{float(mean):.3f} [{float(interval[0]):.3f}, {float(interval[1]):.3f}]"


def _latency_triplet(metric: dict[str, Any]) -> str:
    return "/".join(_number(metric.get(key), 2) for key in ("p50", "p95", "p99"))


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 66：强 delayed-MPC、tube-MPC 与异步分布式基线校准",
        "",
        "本报告只使用 fresh development-calibration manifest，不访问 confirmation 或 locked-test。",
        "local CBF 是经验过滤器；robust CBF-QP/R-CLBF-QP 仍是 diagnostic No-Go，不宣称安全证明。",
        "所有延迟表的格式均为 `p50/p95/p99`，单位为 ms。",
        "",
        "## 实验契约",
        "",
        f"- scene manifest SHA-256：`{payload['scene_manifest_sha256']}`；",
        f"- bootstrap：{payload['config']['bootstrap_samples']} 次，seed `{payload['config']['bootstrap_seed']}`，"
        "按 predictor seed 与 mirror group 分层；",
        f"- calibration 场景：{sum(block['episodes'] for block in payload['blocks'].values())} episodes，"
        f"{sum(block['mirror_groups'] for block in payload['blocks'].values())} mirror groups；",
        "- predictor：三个 Diagonal-SSM checkpoint seed，same scene order；",
        "- safety：local CBF empirical filter；",
        "",
        "## 主要观察",
        "",
        "- M1 known-delay 在 communication-tail 上明显优于 M0，说明“知道延迟”本身可以改善该轴，"
        "但不能替代 queued-state rollout；",
        "- M3 queue-aware tube 在 communication-tail 和 joint-tail-transfer 上比普通 delayed/tube 基线"
        "更有恢复能力，但 planner 与 QDR/tube 尾延迟显著增加；",
        "- M5 asynchronous distributed 在该强通信压力下出现明显安全/捕获退化，不能仅凭较低 planner"
        "计算量宣称异步优越；",
        "- M6/M7 QDR 的安全过滤轴仍受强压力下的可恢复性和 timeout 约束，本阶段不做 promotion。",
        "",
        "## Outcome（按 scene block）",
        "",
    ]
    for block_name, block in payload["blocks"].items():
        lines.extend(
            [
                f"### {block_name}（{block['episodes']} episodes / {block['mirror_groups']} mirror groups）",
                "",
                "| Method | Safe capture | Collision | Boundary | Timeout | Capture time (s) | Min clearance (m) |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method, method_payload in block["methods"].items():
            metrics = method_payload["episode_metrics"]
            cells = [METHOD_LABELS.get(method, method)]
            for key, _label, percent in OUTCOME_COLUMNS:
                cells.append(_interval(metrics[key], percent=percent))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(["## 完整组件延迟（按 scene block）", ""])
    for block_name, block in payload["blocks"].items():
        lines.extend(
            [
                f"### {block_name}",
                "",
                "| Method | Predictor | Planner | QDR/tube | Safety | Total |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for method, method_payload in block["methods"].items():
            latency = method_payload["pooled_step_latency_ms"]
            lines.append(
                "| "
                + " | ".join(
                    [
                        METHOD_LABELS.get(method, method),
                        *(_latency_triplet(latency[field]) for field, _label in LATENCY_COLUMNS),
                    ]
                )
                + " |"
            )
        lines.append("")

    lines.extend(["## Paired comparison", ""])
    for block_name, block in payload["blocks"].items():
        comparisons = block.get("paired_comparisons", {})
        if not comparisons:
            continue
        lines.extend(
            [
                f"### {block_name}",
                "",
                "正值 safe-capture delta 表示 candidate 优于 reference；区间为 mirror-group paired bootstrap 95% CI。",
                "",
                "| Comparison | Safe-capture delta | Collision delta | Timeout delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, comparison in comparisons.items():
            metrics = comparison
            cells = [name]
            for metric_name in ("safe_capture_success", "collision", "timeout"):
                item = metrics[metric_name]
                low, high = item["paired_bootstrap_95_ci"]
                cells.append(
                    f"{float(item['mean_delta_candidate_minus_reference']):.2%} "
                    f"[{float(low):.2%}, {float(high):.2%}]"
                )
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(
        [
            "## 阶段判定与边界",
            "",
            "本阶段只完成 baseline calibration evidence，不是模型晋级结论。后续必须在新的、"
            "预注册的 calibration 上处理 joint-tail-transfer 的可恢复性，再决定是否进入 confirmation。",
            "不得在本阶段结果上调参后访问 locked-test。",
            "",
            "## 复现入口",
            "",
            "```powershell",
            "python scripts\\aggregate_phase58_calibration.py `",
            "  --group \"phase66=results\\phase66_baseline_stress_seed*\" `",
            "  --scenes results\\phase65_qdr_segment_liveness_audit\\development_calibration\\scenes.jsonl `",
            "  --output results\\phase66_baseline_stress_aggregate.json `",
            "  --tensorboard-dir results\\phase66_baseline_stress_aggregate_tensorboard",
            "```",
            "",
            "详细 episode/step JSONL、effective config 和 TensorBoard event 保留在各 seed 运行目录；"
            "本报告由 `scripts/render_phase66_baseline_report.py` 从 aggregate JSON 渲染。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    payload = json.loads(args.aggregate.resolve().read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
