"""Freeze an S4 dataset into scene-disjoint train/validation/locked-test splits.

The split unit is a mirror group, never an overlapping trajectory window.  The
script copies every sample field, recomputes per-split sampling weights, and
writes a manifest that can be audited independently of the training code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_SIZES = {"train": 420, "validation": 90, "locked_test": 90}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Complete S4-v3 archive directory.")
    parser.add_argument("--output", type=Path, required=True, help="Directory receiving the three splits.")
    parser.add_argument("--seed", type=int, default=726901)
    parser.add_argument("--train-episodes", type=int, default=DEFAULT_SIZES["train"])
    parser.add_argument("--validation-episodes", type=int, default=DEFAULT_SIZES["validation"])
    parser.add_argument("--locked-test-episodes", type=int, default=DEFAULT_SIZES["locked_test"])
    return parser.parse_args()


def _read_scenes(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Scene record {line_number} is not a mapping.")
        records.append(value)
    if not records:
        raise ValueError(f"No scene records found in {path}.")
    return records


def _scene_key(record: dict[str, Any]) -> tuple[str, int]:
    if "episode_index" not in record:
        raise ValueError("Every scene record must contain episode_index.")
    scenario = record.get("scenario", {})
    layout_seed = record.get("layout_seed", scenario.get("layout_seed"))
    if layout_seed is None:
        raise ValueError("Every scene record must contain layout_seed.")
    group = record.get("mirror_group_id")
    if group is None:
        # Legacy archives are treated as singleton groups. They can still be
        # split, but no mirror-pair guarantee is claimed in their manifest.
        group = f"singleton:{int(record['episode_index'])}"
    return str(group), int(layout_seed)


def assign_splits(
    records: list[dict[str, Any]],
    sizes: dict[str, int],
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    if any(value <= 0 for value in sizes.values()):
        raise ValueError("All split episode counts must be positive.")
    if sum(sizes.values()) != len(records):
        raise ValueError(
            "Requested split sizes must sum to the number of scene records: "
            f"{sizes} versus {len(records)}."
        )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        group, _layout_seed = _scene_key(record)
        groups[group].append(record)
    for group, members in groups.items():
        member_indices = [int(item["episode_index"]) for item in members]
        if len(set(member_indices)) != len(member_indices):
            raise ValueError(f"Duplicate episode_index values in mirror group {group}.")
        if any("mirror_group_id" in item for item in members):
            if len(members) != 2:
                raise ValueError(f"Mirror group {group} must contain exactly two episodes.")
            biases = {str(item.get("defender_bias")) for item in members}
            if biases != {"upper", "lower"}:
                raise ValueError(f"Mirror group {group} must contain upper and lower members.")

    group_names = sorted(groups)
    rng = np.random.default_rng(int(seed))
    group_order = [group_names[index] for index in rng.permutation(len(group_names))]
    assigned_groups: dict[str, str] = {}
    cursor = 0
    for split in ("train", "validation", "locked_test"):
        target_count = sizes[split]
        consumed = 0
        while consumed < target_count:
            if cursor >= len(group_order):
                raise RuntimeError(f"Unable to fill {split} without splitting a mirror group.")
            group = group_order[cursor]
            group_size = len(groups[group])
            if consumed + group_size > target_count:
                raise RuntimeError(
                    f"Split size {target_count} for {split} is incompatible with mirror group size {group_size}."
                )
            assigned_groups[group] = split
            consumed += group_size
            cursor += 1

    split_records = {split: [] for split in ("train", "validation", "locked_test")}
    for record in records:
        group, _layout_seed = _scene_key(record)
        split_records[assigned_groups[group]].append(record)
    if any(len(split_records[key]) != sizes[key] for key in split_records):
        raise RuntimeError("Scene split counts do not match the requested contract.")
    return split_records, assigned_groups


def _balanced_weights(policy_ids: np.ndarray, branch_sign: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    policy_ids = np.asarray(policy_ids, dtype=np.int16)
    branch_sign = np.asarray(branch_sign, dtype=np.int16)
    if policy_ids.shape != branch_sign.shape or policy_ids.ndim != 1:
        raise ValueError("policy_ids and branch_sign must be aligned one-dimensional arrays.")
    if policy_ids.size == 0:
        return np.empty((0,), dtype=np.float32), {}
    strata = policy_ids * 10 + (branch_sign > 0).astype(np.int16)
    unique, inverse, counts = np.unique(strata, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights /= float(np.mean(weights))
    return weights.astype(np.float32), {
        str(int(key)): int(count) for key, count in zip(unique, counts, strict=True)
    }


def _write_split(
    source_archive: dict[str, np.ndarray],
    source_metadata: dict[str, Any],
    records: list[dict[str, Any]],
    split: str,
    output: Path,
) -> dict[str, Any]:
    episode_indices = np.asarray([int(item["episode_index"]) for item in records], dtype=np.int64)
    sample_episode_indices = np.asarray(source_archive["episode_indices"], dtype=np.int64)
    sample_mask = np.isin(sample_episode_indices, episode_indices)
    split_arrays: dict[str, np.ndarray] = {}
    sample_count = int(sample_mask.sum())
    for name, values in source_archive.items():
        array = np.asarray(values)
        if array.ndim == 0:
            split_arrays[name] = array.copy()
        elif array.shape[0] == sample_mask.shape[0]:
            split_arrays[name] = array[sample_mask]
        else:
            raise ValueError(f"Archive field {name} does not have a sample-aligned first dimension.")
    if {"rollout_policy_ids", "branch_sign"}.issubset(split_arrays):
        weights, strata = _balanced_weights(split_arrays["rollout_policy_ids"], split_arrays["branch_sign"])
        split_arrays["sampling_weights"] = weights
    else:
        strata = {}
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "dataset.npz", **split_arrays)
    (output / "scenes.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    metadata = dict(source_metadata)
    metadata.update(
        {
            "split": split,
            "episodes_requested": len(records),
            "episodes_with_samples": sum(int(item.get("sample_count", 0)) > 0 for item in records),
            "sample_count": sample_count,
            "split_manifest": {
                "episode_indices": episode_indices.tolist(),
                "layout_seeds": [int(item.get("layout_seed", item["scenario"]["layout_seed"])) for item in records],
                "mirror_group_ids": [item.get("mirror_group_id") for item in records],
                "sampling_strata": strata,
            },
            "source_archive_sha256": hashlib.sha256(
                np.asarray(source_archive["episode_indices"]).tobytes()
            ).hexdigest(),
        }
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    archive_path = input_dir / "dataset.npz"
    scenes_path = input_dir / "scenes.jsonl"
    metadata_path = input_dir / "metadata.json"
    for path in (archive_path, scenes_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _read_scenes(scenes_path)
    with np.load(archive_path, allow_pickle=False) as loaded:
        source_archive = {name: np.asarray(loaded[name]) for name in loaded.files}
    if "episode_indices" not in source_archive:
        raise ValueError("S4 archive must contain episode_indices.")
    source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    sizes = {
        "train": int(args.train_episodes),
        "validation": int(args.validation_episodes),
        "locked_test": int(args.locked_test_episodes),
    }
    split_records, assigned_groups = assign_splits(records, sizes, int(args.seed))
    manifest = {
        "source": str(input_dir),
        "seed": int(args.seed),
        "sizes": sizes,
        "group_count": len(set(assigned_groups.values())) and len({group for group in assigned_groups}),
        "split_group_counts": {
            split: len({group for group, value in assigned_groups.items() if value == split})
            for split in sizes
        },
        "split_episode_indices": {
            split: [int(item["episode_index"]) for item in items]
            for split, items in split_records.items()
        },
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    for split, items in split_records.items():
        _write_split(source_archive, source_metadata, items, split, output_dir / split)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
