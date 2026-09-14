"""Extract one mirror-group-disjoint split from a canonical scene manifest.

The command is intentionally conservative: it preserves scene records, rejects
incomplete mirror pairs, and refuses to extract the locked diagnostic split
unless the caller explicitly opts in.  It is used to create the Phase 63
development-calibration input without editing the canonical manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("scene manifest must contain JSON objects")
    return rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def extract_split(
    scenes: Path,
    output: Path,
    *,
    evaluation_split: str,
    allow_locked: bool = False,
) -> dict[str, Any]:
    split = str(evaluation_split)
    if split == "locked_diagnostic" and not allow_locked:
        raise ValueError("locked_diagnostic extraction requires --allow-locked")
    records = _read_jsonl(scenes)
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for record in records:
        group = str(record.get("mirror_group_id", record.get("mirror_group", "")))
        if not group:
            raise ValueError("every scene record must contain mirror_group_id")
        groups.setdefault(group, []).append(record)
    selected_groups = [
        group
        for group, members in groups.items()
        if all(str(member.get("evaluation_split", "")) == split for member in members)
    ]
    selected: list[dict[str, Any]] = []
    for group in selected_groups:
        members = groups[group]
        if len(members) != 2 or {str(item.get("defender_bias")) for item in members} != {"upper", "lower"}:
            raise ValueError(f"mirror group {group} is incomplete")
        selected.extend(members)
    if not selected:
        raise ValueError(f"no records found for evaluation_split={split!r}")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    manifest = {
        "source_scene_file": str(scenes.resolve()),
        "source_scene_file_sha256": _sha256(scenes),
        "selected_scene_file": str(output),
        "selected_scene_file_sha256": _sha256(output),
        "evaluation_split": split,
        "episodes": len(selected),
        "mirror_groups": len(selected_groups),
        "locked_test_included": split == "locked_diagnostic",
        "selection_rule": "evaluation_split_with_complete_upper_lower_mirror_pairs",
    }
    manifest_path = output.with_name("manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluation-split", required=True)
    parser.add_argument("--allow-locked", action="store_true")
    args = parser.parse_args()
    print(json.dumps(extract_split(**vars(args)), indent=2), flush=True)


if __name__ == "__main__":
    main()
