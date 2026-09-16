"""Aggregate learned-policy validation results across seeds.

This report is development-only.  It uses the independent validation pool and
requires every seed to satisfy the predeclared Nominal-plus gate before Hard
or Stress evaluation is opened.  The trainer stores latency percentiles, but
not raw per-step samples, so latency is intentionally reported per seed rather
than as a fabricated pooled percentile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _wilson(rate: float, count: int, z: float = 1.96) -> list[float]:
    if count <= 0:
        return [0.0, 0.0]
    denominator = 1.0 + z * z / count
    centre = (rate + z * z / (2.0 * count)) / denominator
    radius = z * np.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count)) / denominator
    return [float(max(0.0, centre - radius)), float(min(1.0, centre + radius))]


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(row.get(key, False)) for row in rows])) if rows else 0.0


def _seed_from_path(path: Path, manifest: dict[str, Any]) -> int | None:
    match = re.search(r"seed(\d+)", path.name)
    if match:
        return int(match.group(1))
    value = manifest.get("seed")
    return int(value) if value is not None else None


def _summarize(path: Path, safe_gate: float, failure_gate: float) -> dict[str, Any]:
    evaluation = _read_json(path / "evaluation.json")
    manifest = _read_json(path / "manifest.json")
    if bool(evaluation.get("locked_test_used", False)) or bool(manifest.get("locked_test_used", False)):
        raise ValueError(f"locked-test output is not accepted: {path}")
    rows = evaluation.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"evaluation rows are required for pooled validation: {path}")
    if int(evaluation.get("episodes", len(rows))) != len(rows):
        raise ValueError(f"episode count does not match rows: {path}")
    safe_capture = _rate(rows, "safe_capture_success")
    collision = _rate(rows, "collision")
    boundary = _rate(rows, "boundary_violation")
    seed_gate = safe_capture >= safe_gate and collision <= failure_gate and boundary <= failure_gate
    return {
        "seed": _seed_from_path(path, manifest),
        "output_dir": str(path.resolve()),
        "checkpoint_sha256": evaluation.get("checkpoint_sha256"),
        "episodes": len(rows),
        "safe_capture_rate": safe_capture,
        "safe_capture_rate_percent": 100.0 * safe_capture,
        "safe_capture_95_ci": _wilson(safe_capture, len(rows)),
        "capture_event_rate": _rate(rows, "capture_event"),
        "collision_rate": collision,
        "boundary_violation_rate": boundary,
        "timeout_rate": _rate(rows, "timeout"),
        "mean_capture_time_seconds": evaluation.get("mean_capture_time_seconds"),
        "mean_min_clearance_m": evaluation.get("mean_min_clearance_m"),
        "termination_reason_counts": evaluation.get("termination_reason_counts", {}),
        "latency_ms": evaluation.get("latency_ms", {}),
        "gate": {
            "safe_capture_minimum": float(safe_gate),
            "collision_maximum": float(failure_gate),
            "boundary_maximum": float(failure_gate),
            "passed": bool(seed_gate),
        },
    }


def _pooled(paths: list[Path], safe_gate: float, failure_gate: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        evaluation = _read_json(path / "evaluation.json")
        current = evaluation.get("rows")
        if not isinstance(current, list):
            raise ValueError(f"evaluation rows are required: {path}")
        rows.extend(current)
    safe_capture = _rate(rows, "safe_capture_success")
    collision = _rate(rows, "collision")
    boundary = _rate(rows, "boundary_violation")
    return {
        "episodes": len(rows),
        "safe_capture_rate": safe_capture,
        "safe_capture_rate_percent": 100.0 * safe_capture,
        "safe_capture_95_ci": _wilson(safe_capture, len(rows)),
        "capture_event_rate": _rate(rows, "capture_event"),
        "collision_rate": collision,
        "boundary_violation_rate": boundary,
        "timeout_rate": _rate(rows, "timeout"),
        "gate": {
            "safe_capture_minimum": float(safe_gate),
            "collision_maximum": float(failure_gate),
            "boundary_maximum": float(failure_gate),
            "pooled_rate_gate_passed": bool(
                safe_capture >= safe_gate and collision <= failure_gate and boundary <= failure_gate
            ),
        },
    }


def aggregate(
    paths: list[Path],
    output: Path,
    *,
    safe_gate: float = 0.70,
    failure_gate: float = 0.05,
    experiment_name: str = "phase82_dnmcp_dagger_residual_eligible_three_seed_validation",
    evaluation_split: str = "development_validation_only",
) -> dict[str, Any]:
    if len(paths) < 3:
        raise ValueError("Phase 82 requires at least three seed outputs")
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("seed output directories must be unique")
    per_seed = [_summarize(path, safe_gate, failure_gate) for path in resolved]
    seed_values = [item["seed"] for item in per_seed]
    if None in seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("each output directory must identify a unique seed")
    source_hashes = {_read_json(path / "manifest.json").get("source_scene_sha256") for path in resolved}
    if len(source_hashes) != 1:
        raise ValueError("seed outputs must use the same training source scene pool")
    payload = {
        "experiment_name": str(experiment_name),
        "evaluation_split": str(evaluation_split),
        "locked_test_used": False,
        "source_scene_sha256": next(iter(source_hashes)),
        "seeds": per_seed,
        "pooled_episode_summary": _pooled(resolved, safe_gate, failure_gate),
        "three_seed_gate_passed": bool(all(item["gate"]["passed"] for item in per_seed)),
        "hard_stress_opened": False,
        "latency_aggregation_note": (
            "Only per-seed p50/p95/p99 are reported because evaluation.json stores "
            "summary percentiles rather than raw per-step latency samples."
        ),
        "local_cbf_is_empirical_filter_only": True,
        "formal_robust_cbf_qp_claim": False,
    }
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-output", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-capture-gate", type=float, default=0.70)
    parser.add_argument("--failure-rate-gate", type=float, default=0.05)
    parser.add_argument(
        "--experiment-name",
        default="phase82_dnmcp_dagger_residual_eligible_three_seed_validation",
    )
    parser.add_argument("--evaluation-split", default="development_validation_only")
    args = parser.parse_args()
    print(
        json.dumps(
            aggregate(
                args.seed_output,
                args.output,
                safe_gate=args.safe_capture_gate,
                failure_gate=args.failure_rate_gate,
                experiment_name=args.experiment_name,
                evaluation_split=args.evaluation_split,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
