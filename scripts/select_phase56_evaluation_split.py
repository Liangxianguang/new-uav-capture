"""Select a Phase 56 evaluation split without breaking mirror groups."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any


SPLITS = ("development_calibration", "development_confirmation", "locked_diagnostic")


def _sha256_records(records: list[dict[str, Any]]) -> str:
    payload = "".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        for record in records
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_split(records: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for record in records:
        group = str(record.get("mirror_group_id", ""))
        if not group:
            raise ValueError("every scene must contain mirror_group_id")
        groups.setdefault(group, []).append(record)
    selected: list[dict[str, Any]] = []
    for group, members in groups.items():
        if len(members) != 2 or {str(item.get("defender_bias")) for item in members} != {"upper", "lower"}:
            raise ValueError(f"mirror group {group} is not a complete upper/lower pair")
        split_values = {str(item.get("evaluation_split", "")) for item in members}
        if len(split_values) != 1:
            raise ValueError(f"mirror group {group} crosses evaluation splits")
        if split_values == {split}:
            selected.extend(members)
    if not selected:
        raise ValueError(f"no records selected for {split}")
    metadata = {
        "selection": "exact_evaluation_split_with_complete_mirror_groups",
        "evaluation_split": split,
        "source_episodes": len(records),
        "source_mirror_groups": len(groups),
        "selected_episodes": len(selected),
        "selected_mirror_groups": len(selected) // 2,
        "selected_scene_sha256": _sha256_records(selected),
        "locked_tuning": split == "locked_diagnostic",
    }
    return selected, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.input.resolve()
    records = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected, metadata = select_split(records, args.split)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    scenes = output / "scenes.jsonl"
    scenes.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in selected),
        encoding="utf-8",
    )
    manifest = {
        **metadata,
        "source_scene": str(source),
        "source_scene_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "selected_scene": str(scenes),
        "selected_file_sha256": hashlib.sha256(scenes.read_bytes()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

