"""Backfill TensorBoard scalars from retained closed-loop summaries.

This utility is for completed runs whose JSON artifacts exist but whose
TensorBoard writer was interrupted. It does not rerun an experiment and it
never derives metrics from locked-test data for tuning. Each method summary is
written below ``<run-dir>/<method>/tensorboard``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def _writer():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:  # pragma: no cover - depends on local optional package
        raise RuntimeError("TensorBoard logging requires the optional 'tensorboard' package.") from exc
    return SummaryWriter


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and value == value and abs(float(value)) != float("inf")


def _iter_method_summaries(run_dir: Path) -> Iterable[Path]:
    for summary_path in sorted(run_dir.glob("*/summary.json")):
        if summary_path.parent.name != "tensorboard":
            yield summary_path


def backfill_run(run_dir: Path) -> int:
    """Write one event file per method and return the number of methods."""

    config_path = run_dir / "config.yaml"
    config_text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    count = 0
    SummaryWriter = _writer()
    for summary_path in _iter_method_summaries(run_dir):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        overall = payload.get("overall", payload)
        method_dir = summary_path.parent
        with SummaryWriter(log_dir=str(method_dir / "tensorboard"), flush_secs=1) as writer:
            if config_text:
                writer.add_text("Evaluation/config", config_text, 0)
            writer.add_text("Evaluation/backfill_source", str(summary_path), 0)
            for key in (
                "safe_capture_rate",
                "capture_rate",
                "collision_rate",
                "boundary_violation_rate",
                "timeout_rate",
                "mean_capture_time_seconds",
                "mean_min_clearance_m",
                "future_action_condition_available_rate",
                "mean_prediction_age_steps",
                "prediction_refresh_rate",
                "solver_success_rate",
                "valid_plan_rate",
                "effective_plan_rate",
            ):
                value = overall.get(key)
                if _finite(value):
                    writer.add_scalar(f"Summary/Outcome/{key}", float(value), 0)
            for stage, metrics in (
                ("predictor", overall.get("predictor_latency_ms", {})),
                ("planner", overall.get("planner_latency_ms", {})),
                ("safety", overall.get("safety_latency_ms", {})),
                ("total", overall.get("total_control_latency_ms", {})),
            ):
                for quantile in ("p50", "p95", "p99"):
                    value = metrics.get(quantile)
                    if _finite(value):
                        writer.add_scalar(f"Summary/Latency/{stage}_{quantile}_ms", float(value), 0)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="completed run directories")
    args = parser.parse_args()
    total = 0
    for run_dir in args.run_dirs:
        if not run_dir.is_dir():
            raise FileNotFoundError(run_dir)
        total += backfill_run(run_dir)
    print(json.dumps({"runs": len(args.run_dirs), "method_summaries": total}, sort_keys=True))


if __name__ == "__main__":
    main()
