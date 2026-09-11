"""Audit S4-v3 scene isolation, balance, completeness, and action alignment."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FIELDS = {
    "history_observations",
    "history_target_visible",
    "history_message_age_steps",
    "history_observation_timestamps",
    "history_message_timestamps",
    "history_communication_timestamps",
    "history_action_timestamps",
    "history_planned_actions",
    "history_commanded_actions",
    "history_delayed_actions",
    "history_executed_actions",
    "future_target_positions",
    "reference_target_positions",
    "reference_target_velocities",
    "future_defender_actions",
    "future_planned_actions",
    "future_commanded_actions",
    "future_delayed_actions",
    "future_executed_actions",
    "future_action_timestamps",
    "future_defender_positions",
    "future_defender_velocities",
    "episode_indices",
    "layout_seeds",
    "mirror_group_ids",
}
SPLITS = ("train", "validation", "locked_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-train-scenes", type=int, default=420)
    parser.add_argument("--expected-validation-scenes", type=int, default=90)
    parser.add_argument("--expected-locked-test-scenes", type=int, default=90)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object.")
        records.append(value)
    return records


def _counter_report(values: list[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(values).items(), key=lambda item: str(item[0]))}


def _scene_fields(record: dict[str, Any]) -> dict[str, Any]:
    scenario = record.get("scenario", {})
    return {
        "episode_index": int(record["episode_index"]),
        "layout_seed": int(record.get("layout_seed", scenario["layout_seed"])),
        "mirror_group_id": record.get("mirror_group_id"),
        "defender_bias": str(record.get("defender_bias", "unknown")),
        "target_speed_scale": float(record.get("target_speed_scale", np.nan)),
        "observation_condition": str(record.get("observation_condition", "unknown")),
        "rollout_policy": str(record.get("rollout_policy", "unknown")),
        "branch_sign": record.get("target_branch_sign"),
        "termination_reason": record.get("termination_reason"),
        "sample_count": int(record.get("sample_count", 0)),
        "frame_count": int(record.get("frame_count", 0)),
        "geometry_signature": scenario.get("geometry_signature"),
    }


def _audit_archive(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        fields = set(archive.files)
        missing = sorted(REQUIRED_FIELDS.difference(fields))
        if missing:
            raise ValueError(f"{path} is missing required fields: {', '.join(missing)}")
        values = {name: np.asarray(archive[name]) for name in archive.files}
    episode_indices = np.asarray(values["episode_indices"], dtype=np.int64)
    sample_count = int(episode_indices.shape[0])
    scene_indices = {int(record["episode_index"]) for record in records}
    sample_indices = set(episode_indices.tolist())
    if not sample_indices.issubset(scene_indices):
        raise ValueError(f"{path} contains samples from scenes outside its scenes.jsonl.")
    shape_report = {name: list(value.shape) for name, value in values.items()}
    nonfinite = [
        name
        for name, value in values.items()
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all()
    ]
    if nonfinite:
        raise ValueError(f"{path} contains non-finite fields: {', '.join(nonfinite)}")
    if not np.array_equal(values["future_defender_actions"], values["future_executed_actions"]):
        raise ValueError(f"{path} future_defender_actions is not the executed-action alias.")
    if sample_count:
        timesteps = np.asarray(values.get("sample_timesteps"), dtype=np.int64)
        history_action_ts = np.asarray(values["history_action_timestamps"], dtype=np.int64)
        future_action_ts = np.asarray(values["future_action_timestamps"], dtype=np.int64)
        communication_ts = np.asarray(values["history_communication_timestamps"], dtype=np.int64)
        if history_action_ts.shape[-1] != 4 or future_action_ts.shape[-1] != 4:
            raise ValueError(f"{path} action timestamp streams must retain one timestamp per defender.")
        expected_last_history = np.maximum(timesteps - 1, -1)
        if not np.all(history_action_ts[:, -1, :] == expected_last_history[:, None]):
            raise ValueError(f"{path} history action timestamps are not aligned to sample timesteps.")
        expected_future_first = timesteps
        if not np.all(future_action_ts[:, 0, :] == expected_future_first[:, None]):
            raise ValueError(f"{path} future action timestamps are not aligned to sample timesteps.")
        if not np.all(communication_ts[:, -1, :] == timesteps[:, None]):
            raise ValueError(f"{path} communication timestamps are not aligned to observation steps.")
        if not np.all(np.diff(future_action_ts, axis=1) >= 0):
            raise ValueError(f"{path} future action timestamps are not monotone.")
    return {
        "path": str(path),
        "sample_count": sample_count,
        "episode_indices": sorted(sample_indices),
        "fields": sorted(fields),
        "shapes": shape_report,
        "nonfinite_fields": nonfinite,
    }


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    output = args.output.resolve()
    expected_scene_counts = {
        "train": int(args.expected_train_scenes),
        "validation": int(args.expected_validation_scenes),
        "locked_test": int(args.expected_locked_test_scenes),
    }
    scene_records: dict[str, list[dict[str, Any]]] = {}
    archive_reports: dict[str, dict[str, Any]] = {}
    all_records: list[dict[str, Any]] = []
    split_of_episode: dict[int, str] = {}
    errors: list[str] = []
    for split in SPLITS:
        scenes_path = root / split / "scenes.jsonl"
        archive_path = root / split / "dataset.npz"
        if not scenes_path.is_file() or not archive_path.is_file():
            errors.append(f"Missing archive files for split {split}.")
            continue
        records = read_jsonl(scenes_path)
        scene_records[split] = records
        all_records.extend(records)
        if len(records) != expected_scene_counts[split]:
            errors.append(
                f"{split} has {len(records)} scenes; expected {expected_scene_counts[split]}."
            )
        for record in records:
            episode_index = int(record["episode_index"])
            previous = split_of_episode.setdefault(episode_index, split)
            if previous != split:
                errors.append(f"Episode {episode_index} appears in both {previous} and {split}.")
        try:
            archive_reports[split] = _audit_archive(archive_path, records)
        except (ValueError, KeyError) as exc:
            errors.append(str(exc))

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_layout: dict[int, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    by_geometry: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for split, records in scene_records.items():
        for record in records:
            fields = _scene_fields(record)
            group = str(fields["mirror_group_id"] if fields["mirror_group_id"] is not None else f"singleton:{fields['episode_index']}")
            by_group[group].append({"split": split, **fields})
            by_layout[fields["layout_seed"]].append((split, fields))
            if fields["geometry_signature"]:
                by_geometry[str(fields["geometry_signature"])].append((split, fields))

    group_split_report: dict[str, str] = {}
    group_sizes: Counter[str] = Counter()
    for group, members in sorted(by_group.items()):
        splits = {str(member["split"]) for member in members}
        group_sizes[str(len(members))] += 1
        if len(splits) != 1:
            errors.append(f"Mirror group {group} crosses split boundaries: {sorted(splits)}")
        else:
            group_split_report[group] = next(iter(splits))
        if len(members) == 2:
            biases = {str(member["defender_bias"]) for member in members}
            if biases != {"upper", "lower"}:
                errors.append(f"Mirror group {group} is not upper/lower paired.")
        elif members[0]["mirror_group_id"] is not None:
            errors.append(f"Mirror group {group} has {len(members)} members instead of two.")

    duplicate_layouts = {
        str(seed): [(split, int(fields["episode_index"])) for split, fields in members]
        for seed, members in by_layout.items()
        if len(members) > 1
    }
    cross_split_layouts = {
        str(seed): [(split, int(fields["episode_index"])) for split, fields in members]
        for seed, members in by_layout.items()
        if len({split for split, _fields in members}) > 1
    }
    if cross_split_layouts:
        errors.append("At least one layout_seed crosses split boundaries.")
    duplicate_geometry = {
        signature: [(split, int(fields["episode_index"])) for split, fields in members]
        for signature, members in by_geometry.items()
        if len(members) > 1
    }

    distribution_report: dict[str, Any] = {}
    for split, records in scene_records.items():
        fields = [_scene_fields(record) for record in records]
        distribution_report[split] = {
            "branch_sign": _counter_report([item["branch_sign"] for item in fields]),
            "target_speed_scale": _counter_report([item["target_speed_scale"] for item in fields]),
            "observation_condition": _counter_report([item["observation_condition"] for item in fields]),
            "rollout_policy": _counter_report([item["rollout_policy"] for item in fields]),
            "defender_bias": _counter_report([item["defender_bias"] for item in fields]),
            "termination_reason": _counter_report([item["termination_reason"] for item in fields]),
            "zero_window_scenes": [item["episode_index"] for item in fields if item["sample_count"] == 0],
            "missing_branch_label_scenes": [
                item["episode_index"] for item in fields if item["branch_sign"] not in (-1, 1)
            ],
            "min_frame_count": min((item["frame_count"] for item in fields), default=0),
            "max_frame_count": max((item["frame_count"] for item in fields), default=0),
        }
        if distribution_report[split]["missing_branch_label_scenes"]:
            errors.append(f"{split} contains scenes without a committed branch label.")

    report = {
        "dataset_root": str(root),
        "expected_scene_counts": expected_scene_counts,
        "actual_scene_counts": {split: len(records) for split, records in scene_records.items()},
        "archive_reports": archive_reports,
        "group_size_histogram": dict(group_sizes),
        "group_split_count": len(group_split_report),
        "cross_split_groups": [group for group, members in by_group.items() if len({m["split"] for m in members}) > 1],
        "duplicate_layout_seeds": duplicate_layouts,
        "cross_split_layout_seeds": cross_split_layouts,
        "duplicate_geometry_signatures": duplicate_geometry,
        "distribution": distribution_report,
        "errors": errors,
        "status": "failed" if errors else "passed",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
