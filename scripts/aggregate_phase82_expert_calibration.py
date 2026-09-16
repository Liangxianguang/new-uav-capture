"""Aggregate Phase 82 expert-screening results by single-factor variant.

This is a development-only audit.  It joins the immutable scene metadata with
the oracle and public-belief rollout logs, reports the predeclared 80 percent
training gate, and keeps route latency percentiles separate from any learned
policy runtime claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(row.get(key, False)) for row in rows])) if rows else 0.0


def _percentiles(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    values = [
        float(value)
        for row in rows
        for value in row.get("latency_samples_ms", [])
        if np.isfinite(float(value))
    ]
    if not values:
        return {"p50": None, "p95": None, "p99": None}
    array = np.asarray(values, dtype=np.float64)
    return {f"p{p}": float(np.percentile(array, p)) for p in (50, 95, 99)}


def _wilson(rate: float, count: int, z: float = 1.96) -> list[float]:
    if count <= 0:
        return [0.0, 0.0]
    denominator = 1.0 + z * z / count
    centre = (rate + z * z / (2.0 * count)) / denominator
    radius = z * np.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count)) / denominator
    return [float(max(0.0, centre - radius)), float(min(1.0, centre + radius))]


def _summarize(rows: list[dict[str, Any]], gate: float) -> dict[str, Any]:
    accepted = _rate(rows, "expert_calibration_accepted")
    return {
        "episodes": len(rows),
        "mirror_groups": len({str(row["mirror_group_id"]) for row in rows}),
        "expert_acceptance_rate": accepted,
        "expert_acceptance_rate_percent": 100.0 * accepted,
        "expert_acceptance_95_ci": _wilson(accepted, len(rows)),
        "training_gate_threshold": float(gate),
        "training_gate_passed": bool(accepted >= gate),
        "safe_capture_in_pursuit_rate": _rate(rows, "safe_capture_in_pursuit"),
        "physical_collision_rate": _rate(rows, "physical_collision"),
        "target_obstacle_collision_rate": _rate(rows, "target_obstacle_collision"),
        "target_boundary_violation_rate": _rate(rows, "target_boundary_violation"),
        "defender_boundary_violation_rate": _rate(rows, "defender_boundary_violation"),
        "timeout_rate": _rate(rows, "timeout"),
        "mean_min_clearance_m": float(np.mean([float(row["min_clearance_m"]) for row in rows])) if rows else None,
        "mean_capture_time_seconds": float(np.mean([float(row["capture_time_seconds"]) for row in rows])) if rows else None,
        "latency_ms_route_and_safety": _percentiles(rows),
    }


def aggregate(
    scenes_path: Path,
    oracle_dir: Path,
    public_dir: Path,
    output_path: Path,
    *,
    gate: float = 0.80,
) -> dict[str, Any]:
    scenes = _read_jsonl(scenes_path.resolve())
    manifest_path = scenes_path.resolve().parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("locked_test", False)) or not bool(manifest.get("not_a_locked_test", False)):
        raise ValueError("Phase 82 aggregation refuses locked-test or unlabelled scenes")
    oracle = _read_jsonl(oracle_dir.resolve() / "episodes.jsonl")
    public = _read_jsonl(public_dir.resolve() / "episodes.jsonl")
    scene_by_index = {int(row["episode_index"]): row for row in scenes}
    if len(scene_by_index) != len(scenes):
        raise ValueError("scene episode_index values must be unique")
    joined: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, rows in (("oracle_route", oracle), ("public_belief_route", public)):
        if {int(row["episode_index"]) for row in rows} != set(scene_by_index):
            raise ValueError(f"{name} episodes do not exactly cover the scene pool")
        for row in rows:
            scene = scene_by_index[int(row["episode_index"])]
            variant = str(scene["variant"])
            joined.setdefault(variant, {}).setdefault(name, []).append(row)

    variants: dict[str, Any] = {}
    for variant in sorted(joined):
        scene_rows = [row for row in scenes if str(row["variant"]) == variant]
        oracle_rows = joined[variant]["oracle_route"]
        public_rows = joined[variant]["public_belief_route"]
        oracle_summary = _summarize(oracle_rows, gate)
        public_summary = _summarize(public_rows, gate)
        variants[variant] = {
            "single_factor": scene_rows[0]["single_factor"],
            "single_factor_value": scene_rows[0]["single_factor_value"],
            "scene_count": len(scene_rows),
            "mirror_groups": len({str(row["mirror_group_id"]) for row in scene_rows}),
            "target_initial_obstacle_clearance_m": {
                "min": float(min(row["target_initial_obstacle_clearance_m"] for row in scene_rows)),
                "max": float(max(row["target_initial_obstacle_clearance_m"] for row in scene_rows)),
            },
            "formation_spacing_min_m": {
                "min": float(min(row["formation_spacing_min_m"] for row in scene_rows)),
                "max": float(max(row["formation_spacing_min_m"] for row in scene_rows)),
            },
            "oracle_route": oracle_summary,
            "public_belief_route": public_summary,
            "training_eligible": bool(
                oracle_summary["training_gate_passed"] and public_summary["training_gate_passed"]
            ),
        }

    payload = {
        "experiment_name": "phase82_nominal_plus1_expert_calibration_aggregate",
        "evaluation_split": "development_calibration_only",
        "locked_test_used": False,
        "source_scene_file": str(scenes_path.resolve()),
        "source_scene_file_sha256": _sha256(scenes_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "oracle_summary_file": str((oracle_dir.resolve() / "summary.json")),
        "public_belief_summary_file": str((public_dir.resolve() / "summary.json")),
        "gate": {"minimum_acceptance_rate_for_training": float(gate)},
        "variants": variants,
        "eligible_variants": [name for name, value in variants.items() if value["training_eligible"]],
        "local_cbf_is_empirical_filter_only": True,
        "formal_robust_cbf_qp_claim": False,
    }
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--public-belief-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate", type=float, default=0.80)
    args = parser.parse_args()
    print(
        json.dumps(
            aggregate(args.scenes, args.oracle_dir, args.public_belief_dir, args.output, gate=args.gate),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
