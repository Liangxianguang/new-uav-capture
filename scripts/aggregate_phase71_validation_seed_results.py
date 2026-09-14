"""Aggregate independent Phase 71 validation seed blocks.

This utility is deliberately validation-only.  It refuses locked-test inputs so
that a development model-selection report cannot silently become a test report.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-title", default="Phase 71 validation multi-seed aggregation")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=711901)
    return parser.parse_args()


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


def _load_validation_rows(directory: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    metadata_path = directory / "evaluation_metadata.json"
    episodes_path = directory / "episodes.csv"
    if not metadata_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"Validation output is incomplete: {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("split") != "validation" or not bool(metadata.get("not_a_locked_test")):
        raise ValueError(f"Refusing non-validation input: {directory}")
    if bool(metadata.get("locked_test")):
        raise ValueError(f"Refusing locked-test input: {directory}")
    with episodes_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Validation output has no episodes: {directory}")
    return metadata, rows


def _metrics(rows: list[dict[str, str]]) -> dict[str, Any]:
    capture_times = [
        value
        for value in (_float(row.get("capture_time_seconds")) for row in rows)
        if value is not None
    ]
    return {
        "episodes": len(rows),
        "safe_capture_rate": float(np.mean([_bool(row.get("safe_capture_success")) for row in rows])),
        "capture_rate": float(np.mean([_bool(row.get("capture_event")) for row in rows])),
        "collision_rate": float(np.mean([_bool(row.get("collision")) for row in rows])),
        "boundary_violation_rate": float(
            np.mean([(_float(row.get("world_violation_steps")) or 0.0) > 0.0 for row in rows])
        ),
        "timeout_rate": float(np.mean([row.get("termination_reason") == "timeout" for row in rows])),
        "mean_min_clearance_m": float(np.mean([_float(row.get("min_clearance_m")) or np.nan for row in rows])),
        "mean_capture_time_seconds": float(np.mean(capture_times)) if capture_times else None,
        "mean_target_maneuver_switch_count": float(
            np.mean([_float(row.get("target_maneuver_switch_count")) or 0.0 for row in rows])
        ),
    }


def _bootstrap_rate_ci(
    rows: list[dict[str, str]],
    extractor: Any,
    samples: int,
    seed: int,
) -> list[float]:
    if samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    values = np.asarray([float(extractor(row)) for row in rows], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    draws = values[indices].mean(axis=1)
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def aggregate(input_directories: list[Path], *, title: str, bootstrap_samples: int, bootstrap_seed: int) -> dict[str, Any]:
    if len(input_directories) < 2:
        raise ValueError("At least two independent validation seed directories are required.")
    loaded = [_load_validation_rows(path.resolve()) for path in input_directories]
    seed_blocks = [int(metadata["seed_block"]) for metadata, _rows in loaded]
    if len(set(seed_blocks)) != len(seed_blocks):
        raise ValueError("Validation seed blocks must be distinct.")
    all_rows = [row for _metadata, rows in loaded for row in rows]
    extractors = {
        "safe_capture_rate": lambda row: _bool(row.get("safe_capture_success")),
        "capture_rate": lambda row: _bool(row.get("capture_event")),
        "collision_rate": lambda row: _bool(row.get("collision")),
        "boundary_violation_rate": lambda row: (_float(row.get("world_violation_steps")) or 0.0) > 0.0,
    }
    per_seed = []
    for index, ((metadata, rows), seed_block) in enumerate(zip(loaded, seed_blocks)):
        per_seed.append(
            {
                "seed_block": seed_block,
                "directory": str(input_directories[index].resolve()),
                "metrics": _metrics(rows),
            }
        )
    pooled = _metrics(all_rows)
    pooled["bootstrap_ci95"] = {
        name: _bootstrap_rate_ci(all_rows, extractor, bootstrap_samples, bootstrap_seed + index)
        for index, (name, extractor) in enumerate(extractors.items())
    }
    seed_metric_values = {
        name: [float(item["metrics"][name]) for item in per_seed]
        for name in (
            "safe_capture_rate",
            "capture_rate",
            "collision_rate",
            "boundary_violation_rate",
            "timeout_rate",
        )
    }
    seed_variation = {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for name, values in seed_metric_values.items()
    }
    return {
        "report_title": title,
        "evaluation_type": "phase71_maneuvering_defender_validation_multiseed",
        "not_a_locked_test": True,
        "locked_test_included": False,
        "seed_blocks": seed_blocks,
        "total_episodes": len(all_rows),
        "per_seed": per_seed,
        "pooled": pooled,
        "seed_variation": seed_variation,
        "bootstrap": {"samples": bootstrap_samples, "seed": bootstrap_seed, "unit": "episode"},
    }


def main() -> None:
    args = parse_args()
    report = aggregate(
        args.input,
        title=str(args.report_title),
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
