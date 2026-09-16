"""Merge reproducible Phase 82 expert-calibration shards.

Sharding changes only execution scheduling: the merged report is checked to
cover the immutable scene pool exactly once before any acceptance rate is
computed.  This utility is calibration-only and refuses locked-test scenes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_phase77_expert_calibration import _summarize


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def merge(
    scenes_path: Path,
    part_dirs: list[Path],
    output_dir: Path,
    *,
    gate: float = 0.80,
) -> dict[str, Any]:
    if len(part_dirs) < 2:
        raise ValueError("at least two calibration parts are required")
    scenes_path = scenes_path.resolve()
    manifest_path = scenes_path.parent / "manifest.json"
    manifest = _read_json(manifest_path)
    if bool(manifest.get("locked_test", False)) or not bool(manifest.get("not_a_locked_test", False)):
        raise ValueError("Phase 82 merge refuses locked-test or unlabelled scenes")
    scenes = _read_jsonl(scenes_path)
    scene_by_index = {int(row["episode_index"]): row for row in scenes}
    if len(scene_by_index) != len(scenes):
        raise ValueError("scene episode_index values must be unique")

    rows: list[dict[str, Any]] = []
    part_summaries: list[dict[str, Any]] = []
    for part in part_dirs:
        part = part.resolve()
        summary = _read_json(part / "summary.json")
        if bool(summary.get("locked_test_used", False)):
            raise ValueError(f"locked-test part is not accepted: {part}")
        if str(summary.get("source_scene_file_sha256")) != _sha256(scenes_path):
            raise ValueError(f"part uses a different scene file: {part}")
        current = _read_jsonl(part / "episodes.jsonl")
        if not current:
            raise ValueError(f"empty calibration part: {part}")
        rows.extend(current)
        part_summaries.append(summary)

    indices = [int(row["episode_index"]) for row in rows]
    if len(set(indices)) != len(indices):
        raise ValueError("calibration parts contain duplicate episode indices")
    if set(indices) != set(scene_by_index):
        missing = sorted(set(scene_by_index).difference(indices))
        extra = sorted(set(indices).difference(scene_by_index))
        raise ValueError(f"parts do not cover scenes exactly; missing={missing}, extra={extra}")
    rows.sort(key=lambda row: int(row["episode_index"]))

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    accepted = [
        scene_by_index[int(row["episode_index"])]
        for row in rows
        if bool(row.get("expert_calibration_accepted", False))
    ]
    route_exercise = [
        scene_by_index[int(row["episode_index"])]
        for row in rows
        if bool(row.get("expert_route_exercise_accepted", False))
    ]
    (output_dir / "accepted_calibration_scenes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted),
        encoding="utf-8",
    )
    (output_dir / "route_exercise_calibration_scenes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in route_exercise),
        encoding="utf-8",
    )

    rows_by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_profile[str(scene_by_index[int(row["episode_index"])] ["variant"])].append(row)
    summary = {
        "experiment_name": "phase82_target_boundary_guard_calibration_sharded_merge",
        "evaluation_split": "development_calibration_only",
        "locked_test_used": False,
        "source_scene_manifest_sha256": _sha256(manifest_path),
        "source_scene_file_sha256": _sha256(scenes_path),
        "controller": part_summaries[0].get("controller"),
        "controller_mode": part_summaries[0].get("controller_mode"),
        "target_contract_overrides": {
            key: part_summaries[0].get(key)
            for key in (
                "safety_margin_override_m",
                "target_boundary_margin_override_m",
                "target_maneuver_boundary_weight_override",
                "target_maneuver_route_boundary_buffer_override_m",
                "target_maneuver_predictive_boundary_recovery_override",
                "target_maneuver_predictive_boundary_lookahead_steps_override",
                "target_maneuver_boundary_recovery_speed_scale_override",
                "target_maneuver_safety_first_fallback_override",
            )
        },
        "expert_acceptance_contract": part_summaries[0].get("expert_acceptance_contract", {}),
        "expert_acceptance_gate": float(gate),
        "raw_pool_episodes": len(rows),
        "source_scene_start_index": "sharded",
        "source_scene_requested_episodes": len(rows),
        "accepted_calibration_episodes": len(accepted),
        "accepted_route_exercise_episodes": len(route_exercise),
        "raw_pool_summary": _summarize(rows),
        "by_variant": {
            name: _summarize(profile_rows)
            for name, profile_rows in sorted(rows_by_profile.items())
        },
        "part_output_dirs": [str(path.resolve()) for path in part_dirs],
        "part_count": len(part_dirs),
        "accepted_scene_file": str((output_dir / "accepted_calibration_scenes.jsonl").resolve()),
        "route_exercise_scene_file": str((output_dir / "route_exercise_calibration_scenes.jsonl").resolve()),
        "local_cbf_is_empirical_filter_only": True,
        "formal_robust_cbf_qp_claim": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "config.yaml").write_text(
        json.dumps(
            {
                "source_scene_file": str(scenes_path),
                "part_output_dirs": [str(path.resolve()) for path in part_dirs],
                "sharded_merge": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--part-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate", type=float, default=0.80)
    args = parser.parse_args()
    print(
        json.dumps(
            merge(args.scenes, args.part_dir, args.output_dir, gate=args.gate),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
