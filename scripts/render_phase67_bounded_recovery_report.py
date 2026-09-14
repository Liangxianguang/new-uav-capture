"""Render the Phase 67 bounded public-prefix recovery diagnostic report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.render_phase66_baseline_report import render as render_phase66
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from render_phase66_baseline_report import render as render_phase66


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def render(payload: dict[str, Any]) -> str:
    text = render_phase66(payload)
    text = text.replace(
        "# Phase 66：强 delayed-MPC、tube-MPC 与异步分布式基线校准",
        "# Phase 67：QDR bounded public-prefix recovery 诊断",
        1,
    )
    start = text.index("## 主要观察")
    end = text.index("## Outcome（按 scene block）")
    observations = """## 主要观察

- M9 bounded public-prefix recovery 在 execution-tail 上相对 M6 的 safe-capture
  delta 为 `+7.78 pp [-3.33,+18.89]`，区间跨零；collision delta 为
  `-4.44 pp [-15.56,+6.67]`，timeout delta 为 `-3.33 pp [-11.11,+3.33]`，
  因此不能判定为稳定收益。
- 在 communication-tail 上 M6 已达到 `100.00%` safe capture，M9 反而为
  `96.67%`，paired safe-capture delta 为 `-3.33 pp [-8.89,0.00]`；恢复规则
  不能替代已知延迟或通信建模。
- 在 joint-tail-transfer 上 M6/M9 safe capture 为 `31.11%/30.00%`，M9 的
  timeout 由 `33.33%` 变为 `35.56%`；paired intervals 很宽，说明公开候选集
  在联合尾部压力下仍缺乏可恢复性。
- M9 的 recovery budget 为 6 步、短前缀 horizon 为 3 步；恢复触发率和最大
  持续步数均已写入 JSONL、aggregate 与 TensorBoard。该策略没有通过本阶段的
  safe-capture、timeout、exhaustion-streak 联合门槛，冻结为 diagnostic No-Go，
  停止继续堆叠同类恢复规则。
"""
    text = text[:start] + observations + "\n" + text[end:]
    text = text.replace(
        "python scripts\\aggregate_phase58_calibration.py `\n"
        "  --group \"phase66=results\\phase66_baseline_stress_seed*\" `\n"
        "  --scenes results\\phase65_qdr_segment_liveness_audit\\development_calibration\\scenes.jsonl `\n"
        "  --output results\\phase66_baseline_stress_aggregate.json `\n"
        "  --tensorboard-dir results\\phase66_baseline_stress_aggregate_tensorboard",
        "python scripts\\aggregate_phase58_calibration.py `\n"
        "  --group \"phase67=results\\phase67_bounded_recovery_seed*\" `\n"
        "  --scenes results\\phase67_bounded_recovery_matrix\\development_calibration\\scenes.jsonl `\n"
        "  --output results\\phase67_bounded_recovery_aggregate.json `\n"
        "  --tensorboard-dir results\\phase67_bounded_recovery_aggregate_tensorboard `\n"
        "  --bootstrap-seed 20260914",
        1,
    )
    text = text.replace(
        "scripts/render_phase66_baseline_report.py",
        "scripts/render_phase67_bounded_recovery_report.py",
    )
    text = text.replace(
        "本阶段只完成 baseline calibration evidence，不是模型晋级结论。后续必须在新的、"
        "预注册的 calibration 上处理 joint-tail-transfer 的可恢复性，再决定是否进入 confirmation。\n"
        "不得在本阶段结果上调参后访问 locked-test。",
        "本阶段只完成 development calibration evidence，不是模型晋级结论。由于三个压力轴"
        "均未形成统计稳定的综合收益，本实验线按 diagnostic No-Go 冻结并停止继续堆叠同类"
        "恢复规则；不访问 confirmation/locked-test，也不据此宣称安全证明。",
        1,
    )
    return text


def main() -> None:
    args = parse_args()
    payload = json.loads(args.aggregate.resolve().read_text(encoding="utf-8"))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"status": "complete", "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
