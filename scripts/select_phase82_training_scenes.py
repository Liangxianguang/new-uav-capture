"""Select Phase 82 training scenes from an audited public-belief gate.

Only variants that pass both the oracle and public-belief 80 percent gate are
selected.  The source calibration pool and its rejected variants remain
untouched; this command creates a separate development-only scene manifest
for training and records all source hashes for reproducibility.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def select(
    scenes_path: Path,
    accepted_path: Path,
    aggregate_path: Path,
    output_dir: Path,
    *,
    purpose: str = "training",
) -> dict[str, Any]:
    if purpose not in {"training", "validation"}:
        raise ValueError("purpose must be training or validation")
    scenes_path = scenes_path.resolve()
    accepted_path = accepted_path.resolve()
    aggregate_path = aggregate_path.resolve()
    scenes = _read_jsonl(scenes_path)
    accepted = _read_jsonl(accepted_path)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if bool(aggregate.get("locked_test_used", False)):
        raise ValueError("Phase 82 training selection refuses locked-test aggregate")
    eligible = {str(value) for value in aggregate.get("eligible_variants", [])}
    if not eligible:
        raise ValueError("no Phase 82 variant passed both expert gates")
    scene_by_index = {int(row["episode_index"]): row for row in scenes}
    if len(scene_by_index) != len(scenes):
        raise ValueError("scene episode_index values must be unique")
    accepted_indices = {int(row["episode_index"]) for row in accepted}
    if not accepted_indices.issubset(scene_by_index):
        raise ValueError("accepted calibration scenes are not a subset of the source pool")
    selected = [
        row
        for row in accepted
        if str(row.get("variant")) in eligible
    ]
    selected.sort(key=lambda row: int(row["episode_index"]))
    if len(selected) < 192:
        raise ValueError(f"eligible training pool has only {len(selected)} scenes; need at least 192")
    if any("locked" in str(row.get("scene_block", "")).lower() for row in selected):
        raise ValueError("selected training pool contains locked-test records")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_file = output_dir / "scenes.jsonl"
    scene_file.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    mirror_groups = sorted({str(row["mirror_group_id"]) for row in selected})
    manifest = {
        "dataset_name": f"phase82_nominal_plus1_eligible_{purpose}_pool",
        "dataset_version": "phase82.nominal_plus1.eligible.v1",
        "phase": f"development_{purpose}_only",
        "not_a_locked_test": True,
        "locked_test": False,
        "locked_test_modified": False,
        "source_scene_file": str(scenes_path),
        "source_scene_file_sha256": _sha256(scenes_path),
        "source_accepted_file": str(accepted_path),
        "source_accepted_file_sha256": _sha256(accepted_path),
        "source_aggregate_file": str(aggregate_path),
        "source_aggregate_file_sha256": _sha256(aggregate_path),
        "selected_variants": sorted(eligible),
        "total_scenes": len(selected),
        "mirror_groups": len(mirror_groups),
        "variant_counts": {
            name: sum(str(row.get("variant")) == name for row in selected)
            for name in sorted(eligible)
        },
        "scene_file": str(scene_file),
        "scene_file_sha256": _sha256(scene_file),
        "locked_test_used": False,
        "contract": {
            "teacher_gate_precondition": True,
            "one_factor_per_variant": True,
            "target_crossing_required": False,
            "intentional_reverse_lane_change": False,
            "local_cbf_is_empirical_filter_only": True,
            "formal_robust_cbf_qp_claim": False,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--purpose", choices=("training", "validation"), default="training")
    args = parser.parse_args()
    print(
        json.dumps(
            select(args.scenes, args.accepted, args.aggregate, args.output_dir, purpose=args.purpose),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
