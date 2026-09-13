"""Audit a frozen Phase 58 scene matrix and write an auditable summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment,misc]

from generate_phase58_delayed_planning_scenes import (
    BASE_EXECUTION,
    BASE_PURSUIT,
    BLOCKS,
    DELAY_LEVELS,
DRAG_LEVELS,
    DROPOUT_LEVELS,
    MESSAGE_DELAY_LEVELS,
    NOISE_LEVELS,
    SPLITS,
    TARGET_SPEEDS,
    TRACKING_LEVELS,
    TRAINING_GEOMETRY,
TRANSFER_GEOMETRY,
)

ALLOWED_MESSAGE_DROPOUT_LEVELS = tuple(
    sorted(set(DROPOUT_LEVELS) | {float(BASE_PURSUIT["message_dropout_probability"])})
)


def _read_records(path: Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("scene manifest is empty")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("every scene row must be a JSON object")
    return records


def _in_range(value: float, bounds: tuple[float, float]) -> bool:
    return float(bounds[0]) <= float(value) <= float(bounds[1])


def _assert_range_mapping(values: dict[str, Any], bounds: dict[str, tuple[float, float]], label: str) -> None:
    for key, interval in bounds.items():
        raw = values.get(key)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise ValueError(f"{label}.{key} must be a two-value range")
        if not _in_range(float(raw[0]), interval) or not _in_range(float(raw[1]), interval):
            raise ValueError(f"{label}.{key} leaves its declared range")


def audit_matrix(scenes: Path, manifest: Path) -> dict[str, Any]:
    scene_path = scenes.resolve()
    manifest_path = manifest.resolve()
    records = _read_records(scene_path)
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_data, dict):
        raise ValueError("manifest.json must contain an object")
    scene_hash = hashlib.sha256(scene_path.read_bytes()).hexdigest()
    if scene_hash != str(manifest_data.get("scene_manifest_sha256")):
        raise ValueError("scene SHA-256 does not match manifest")
    expected_episodes = int(manifest_data.get("episodes", -1))
    expected_groups = int(manifest_data.get("mirror_groups", -1))
    if len(records) != expected_episodes or len(records) != 2 * expected_groups:
        raise ValueError("episode/group counts do not match manifest")
    groups: dict[str, list[dict[str, Any]]] = {}
    episode_ids: set[int] = set()
    layout_seeds: set[int] = set()
    block_counts = {block: 0 for block in BLOCKS}
    split_counts = {split: 0 for split in SPLITS}
    for record in records:
        episode_index = int(record["episode_index"])
        if episode_index in episode_ids:
            raise ValueError(f"duplicate episode index {episode_index}")
        episode_ids.add(episode_index)
        group = str(record["mirror_group_id"])
        block = str(record["scene_block"])
        split = str(record["evaluation_split"])
        if block not in BLOCKS or split not in SPLITS:
            raise ValueError("unknown block or split")
        block_counts[block] += 1
        split_counts[split] += 1
        layout_seed = int(record["layout_seed"])
        layout_seeds.add(layout_seed)
        groups.setdefault(group, []).append(record)
        pursuit = record["pursuit_overrides"]
        execution = record["execution_overrides"]
        if pursuit["detection_dropout_probability"] != BASE_PURSUIT["detection_dropout_probability"]:
            raise ValueError("detection dropout is not frozen")
        if pursuit["observation_noise_std"] != BASE_PURSUIT["observation_noise_std"]:
            raise ValueError("observation noise is not frozen")
        if int(execution["action_delay_steps"]) not in DELAY_LEVELS:
            raise ValueError("unknown action-delay level")
        if float(execution["command_noise_std"]) not in NOISE_LEVELS:
            raise ValueError("unknown command-noise level")
        if int(pursuit["message_delay_steps"]) not in MESSAGE_DELAY_LEVELS:
            raise ValueError("unknown message-delay level")
        if float(pursuit["message_dropout_probability"]) not in ALLOWED_MESSAGE_DROPOUT_LEVELS:
            raise ValueError("unknown message-dropout level")
        if float(execution["velocity_time_constant_seconds"]) not in TRACKING_LEVELS:
            raise ValueError("unknown tracking level")
        if float(execution["drag_coefficient"]) not in DRAG_LEVELS:
            raise ValueError("unknown drag level")
        if float(record["target_speed_scale"]) not in TARGET_SPEEDS:
            raise ValueError("target speed leaves the training range")
        ranges = record.get("training_geometry_ranges")
        transfer_ranges = record.get("transfer_geometry_ranges")
        _assert_range_mapping(ranges, TRAINING_GEOMETRY, "training_geometry_ranges")
        _assert_range_mapping(transfer_ranges, TRANSFER_GEOMETRY, "transfer_geometry_ranges")
    if len(groups) != expected_groups:
        raise ValueError("mirror group count is inconsistent")
    if len(layout_seeds) != len(groups):
        raise ValueError("layout seeds are reused across mirror groups")
    for group, members in groups.items():
        if len(members) != 2 or {str(item["defender_bias"]) for item in members} != {"upper", "lower"}:
            raise ValueError(f"mirror group {group} is not a complete upper/lower pair")
        if len({str(item["evaluation_split"]) for item in members}) != 1:
            raise ValueError(f"mirror group {group} crosses split")
        if len({str(item["scene_block"]) for item in members}) != 1:
            raise ValueError(f"mirror group {group} crosses block")
        if len({int(item["layout_seed"]) for item in members}) != 1:
            raise ValueError(f"mirror group {group} does not share layout seed")
        if members[0]["scenario"]["obstacles"] != members[1]["scenario"]["obstacles"]:
            raise ValueError(f"mirror group {group} does not share obstacle geometry")
        block = str(members[0]["scene_block"])
        geometry_id = {"id_reference": "training_range", "delay_noise_factorial": "training_range", "communication_execution_factorial": "training_range", "joint_stress_transfer": "training_ood_transfer"}[block]
        if any(str(item["geometry_id"]) != geometry_id for item in members):
            raise ValueError(f"mirror group {group} geometry contract mismatch")
    expected_per_block = expected_episodes // len(BLOCKS)
    expected_per_split = expected_episodes // len(SPLITS)
    if any(value != expected_per_block for value in block_counts.values()):
        raise ValueError(f"unexpected block counts: {block_counts}")
    if any(value != expected_per_split for value in split_counts.values()):
        raise ValueError(f"unexpected split counts: {split_counts}")
    return {
        "experiment_name": "phase58_scene_matrix_audit",
        "scenes": str(scene_path),
        "manifest": str(manifest_path),
        "scene_manifest_sha256": scene_hash,
        "episodes": len(records),
        "mirror_groups": len(groups),
        "block_counts_episodes": block_counts,
        "split_counts_episodes": split_counts,
        "split_counts_mirror_groups": {key: value // 2 for key, value in split_counts.items()},
        "unique_layout_seeds": len(layout_seeds),
        "audit_pass": True,
        "claim_boundary": "scene and information-contract audit only; no performance or safety certificate",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_matrix(args.scenes, args.manifest)
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if SummaryWriter is not None:
        with SummaryWriter(log_dir=str(output / "tensorboard")) as writer:
            writer.add_text("Protocol/claim_boundary", result["claim_boundary"], 0)
            writer.add_scalar("Gate/scene_matrix_audit", 1.0, 0)
            writer.add_scalar("Dataset/episodes", result["episodes"], 0)
            writer.add_scalar("Dataset/mirror_groups", result["mirror_groups"], 0)
            for split, value in result["split_counts_episodes"].items():
                writer.add_scalar(f"Dataset/split_episodes/{split}", value, 0)
            for block, value in result["block_counts_episodes"].items():
                writer.add_scalar(f"Dataset/block_episodes/{block}", value, 0)
            writer.flush()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
