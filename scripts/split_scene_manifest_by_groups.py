"""Create disjoint scene manifests from a canonical calibration artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.resolve().read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"scene row must be an object: {path}")
            rows.append(value)
    if not rows:
        raise ValueError(f"scene manifest is empty: {path}")
    return rows


def split_scene_manifest(
    scenes: Path,
    calibration_artifact: Path,
    calibration_output: Path,
    confirmation_output: Path,
) -> dict[str, Any]:
    records = _read_jsonl(scenes)
    artifact = json.loads(calibration_artifact.resolve().read_text(encoding="utf-8"))
    metadata = artifact.get("split_metadata", {})
    calibration_groups = {str(value) for value in metadata.get("calibration_groups", [])}
    confirmation_groups = {str(value) for value in metadata.get("confirmation_groups", [])}
    if not calibration_groups or not confirmation_groups:
        raise ValueError("calibration artifact must contain canonical group lists")
    if calibration_groups & confirmation_groups:
        raise ValueError("calibration and confirmation groups overlap")

    calibration_rows: list[dict[str, Any]] = []
    confirmation_rows: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    for record in records:
        group = str(record.get("mirror_group_id", f"episode:{record['episode_index']}"))
        seen_groups.add(group)
        if group in calibration_groups:
            calibration_rows.append(record)
        elif group in confirmation_groups:
            confirmation_rows.append(record)
        else:
            raise ValueError(f"scene group {group!r} is missing from calibration artifact")
    if seen_groups != calibration_groups | confirmation_groups:
        raise ValueError("artifact group lists do not cover the scene manifest")
    if not calibration_rows or not confirmation_rows:
        raise ValueError("both scene splits must be non-empty")

    for output, rows in ((calibration_output, calibration_rows), (confirmation_output, confirmation_rows)):
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8")
    return {
        "source_scenes": str(scenes.resolve()),
        "calibration_artifact": str(calibration_artifact.resolve()),
        "calibration_episode_count": len(calibration_rows),
        "confirmation_episode_count": len(confirmation_rows),
        "calibration_group_count": len(calibration_groups),
        "confirmation_group_count": len(confirmation_groups),
        "calibration_output": str(calibration_output.resolve()),
        "confirmation_output": str(confirmation_output.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--calibration-artifact", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--confirmation-output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(split_scene_manifest(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
