"""Select a deterministic, mirror-group-disjoint scene block.

This utility is intended for validation-only confirmation blocks.  It keeps
complete mirror groups together, preserves the original scene records and
episode seeds, and writes a manifest containing source/selected hashes and the
selection rule.  It never reads or modifies locked-test data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


def read_scene_records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("input scene file is empty")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("every scene record must be a JSON object")
    return records


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def records_sha256(records: list[dict[str, Any]]) -> str:
    payload = "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_mirror_group_block(
    records: list[dict[str, Any]],
    *,
    drop_first_groups: int,
    expected_split: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if int(drop_first_groups) < 0:
        raise ValueError("drop_first_groups must be non-negative")

    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for record in records:
        group_value = record.get("mirror_group_id", record.get("mirror_group"))
        if group_value is None:
            raise ValueError("every record must contain mirror_group_id or mirror_group")
        group = str(group_value)
        groups.setdefault(group, []).append(record)

    if len(groups) <= int(drop_first_groups):
        raise ValueError("drop_first_groups leaves no mirror groups")
    incomplete = {group: len(members) for group, members in groups.items() if len(members) != 2}
    if incomplete:
        raise ValueError(f"all selected source mirror groups must have two members: {incomplete}")

    source_split_values = {str(record.get("scene_block", "")) for record in records}
    if expected_split is not None and source_split_values != {str(expected_split)}:
        raise ValueError(
            f"input scene block mismatch: expected {expected_split!r}, got {sorted(source_split_values)}"
        )

    group_items = list(groups.items())
    dropped_groups = [group for group, _members in group_items[: int(drop_first_groups)]]
    selected_groups = [group for group, _members in group_items[int(drop_first_groups) :]]
    selected = [record for group in selected_groups for record in groups[group]]
    if len(selected) % 2 != 0:
        raise AssertionError("mirror-group selection must contain an even number of scenes")

    selected_split_values = {str(record.get("scene_block", "")) for record in selected}
    if expected_split is not None and selected_split_values != {str(expected_split)}:
        raise ValueError("selected records changed scene block")

    metadata = {
        "selection_strategy": "ordered_mirror_group_holdout",
        "drop_first_groups": int(drop_first_groups),
        "source_scene_count": len(records),
        "source_mirror_group_count": len(groups),
        "dropped_mirror_group_count": len(dropped_groups),
        "selected_scene_count": len(selected),
        "selected_mirror_group_count": len(selected_groups),
        "dropped_mirror_groups": dropped_groups,
        "selected_mirror_groups": selected_groups,
        "source_scene_hash": records_sha256(records),
        "selected_scene_hash": records_sha256(selected),
        "locked_test_included": False,
    }
    return selected, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Source scenes.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for scenes.jsonl and manifest.json")
    parser.add_argument("--drop-first-groups", type=int, required=True)
    parser.add_argument("--expected-scene-block", default=None)
    parser.add_argument("--name", required=True)
    parser.add_argument("--evaluation-split", default="validation_confirmation_holdout")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = args.input.resolve()
    records = read_scene_records(source)
    selected, metadata = select_mirror_group_block(
        records,
        drop_first_groups=int(args.drop_first_groups),
        expected_split=args.expected_scene_block,
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "scenes.jsonl"
    selected_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in selected),
        encoding="utf-8",
    )
    manifest = {
        "name": str(args.name),
        "evaluation_split": str(args.evaluation_split),
        "source_scene_file": str(source),
        "source_scene_file_sha256": file_sha256(source),
        "selected_scene_file": str(selected_path),
        "selected_scene_file_sha256": file_sha256(selected_path),
        **metadata,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
