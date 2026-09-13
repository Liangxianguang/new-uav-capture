"""Generate the fresh, mirror-paired Phase 56 baseline matrix.

The generator creates three balanced blocks: an in-distribution reference, a
delay/noise grid, and a communication/execution grid.  It changes only the
declared factor for each block and never reuses the old v4 mirror groups.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import scenario_metadata, s4_adaptive_branching_scenario  # noqa: E402
from evaluate_minimax_mpc import DEFAULT_ENVIRONMENT_CONFIG  # noqa: E402
from evaluate_s4_branching import config_for_spec, load_protocol  # noqa: E402


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "phase15_s4_branching_pilot.yaml"
TRAINING_SPEED_SCALES = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
TRAINING_GEOMETRY = {
    "wall_half_extent_x_m": (0.45, 0.65),
    "wall_half_extent_y_m": (3.85, 4.35),
    "wall_height_m": (9.25, 9.55),
    "target_initial_x_m": (-4.25, -3.45),
    "target_initial_y_m": (-0.40, 0.40),
    "target_altitude_m": (4.30, 5.70),
    "defender_x_offset_m": (0.0, 0.35),
    "defender_y_scale": (0.92, 1.08),
    "defender_z_offset_m": (-0.30, 0.30),
}
BASE_PURSUIT = {
    "detection_range": 14.0,
    "detection_dropout_probability": 0.15,
    "observation_noise_std": 0.03,
    "message_delay_steps": 2,
    "message_dropout_probability": 0.05,
}
BASE_EXECUTION = {
    "enabled": True,
    "action_delay_steps": 4,
    "pending_command_authority": "immutable",
    "command_noise_std": 0.08,
    "command_noise_bound_sigma": 3.0,
    "clip_command_noise": True,
    "velocity_time_constant_seconds": 0.0,
    "drag_coefficient": 0.0,
}
SPLITS = ("development_calibration", "development_confirmation", "locked_diagnostic")
DELAY_LEVELS = (0, 2, 4, 6, 8)
NOISE_LEVELS = (0.00, 0.04, 0.08, 0.12)
DROPOUT_LEVELS = (0.00, 0.10, 0.20, 0.30)
MESSAGE_DELAY_LEVELS = (0, 2, 4)
TRACKING_LEVELS = (0.0, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-per-block", type=int, default=60)
    parser.add_argument("--seed-start", type=int, default=871101)
    parser.add_argument("--manifest-name", default="phase56_strong_baseline")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_ENVIRONMENT_CONFIG)
    return parser.parse_args()


def _split_for(block_index: int, group_in_block: int) -> str:
    return SPLITS[(block_index + group_in_block) % len(SPLITS)]


def _condition(block_index: int, group_in_block: int) -> tuple[str, dict[str, Any], dict[str, Any]]:
    pursuit = copy.deepcopy(BASE_PURSUIT)
    execution = copy.deepcopy(BASE_EXECUTION)
    if block_index == 0:
        condition_id = "id_reference"
    elif block_index == 1:
        cell = group_in_block % (len(DELAY_LEVELS) * len(NOISE_LEVELS))
        delay = DELAY_LEVELS[cell // len(NOISE_LEVELS)]
        noise = NOISE_LEVELS[cell % len(NOISE_LEVELS)]
        condition_id = f"delay{delay}_noise{noise:.2f}"
        execution["action_delay_steps"] = int(delay)
        execution["command_noise_std"] = float(noise)
    elif block_index == 2:
        cell = group_in_block % (len(DROPOUT_LEVELS) * len(MESSAGE_DELAY_LEVELS))
        dropout = DROPOUT_LEVELS[cell // len(MESSAGE_DELAY_LEVELS)]
        message_delay = MESSAGE_DELAY_LEVELS[cell % len(MESSAGE_DELAY_LEVELS)]
        tracking = TRACKING_LEVELS[(group_in_block // (len(DROPOUT_LEVELS) * len(MESSAGE_DELAY_LEVELS))) % len(TRACKING_LEVELS)]
        condition_id = f"dropout{dropout:.2f}_msgdelay{message_delay}_track{tracking:.2f}"
        pursuit["message_dropout_probability"] = float(dropout)
        pursuit["message_delay_steps"] = int(message_delay)
        execution["velocity_time_constant_seconds"] = float(tracking)
    else:  # pragma: no cover
        raise ValueError(f"unknown block index {block_index}")
    return condition_id, pursuit, execution


def build_records(
    protocol: dict[str, Any],
    environment_config: Path,
    *,
    groups_per_block: int,
    seed_start: int,
) -> list[dict[str, Any]]:
    if groups_per_block <= 0:
        raise ValueError("groups_per_block must be positive")
    records: list[dict[str, Any]] = []
    candidate_offset = 0
    for block_index in range(3):
        block_name = ("id_reference", "delay_noise_grid", "communication_execution_grid")[block_index]
        for group_in_block in range(groups_per_block):
            split = _split_for(block_index, group_in_block)
            condition_id, pursuit, execution = _condition(block_index, group_in_block)
            speed = float((0.65, 0.75)[group_in_block % 2])
            accepted: list[dict[str, Any]] | None = None
            for _attempt in range(10_000):
                layout_seed = int(seed_start + 1_000_000 + candidate_offset)
                candidate_offset += 1
                group_records: list[dict[str, Any]] = []
                try:
                    for member, defender_bias in enumerate(("upper", "lower")):
                        episode_index = 2 * (block_index * groups_per_block + group_in_block) + member
                        spec = {
                            "episode_index": episode_index,
                            "episode_seed": int(seed_start + episode_index),
                            "layout_seed": layout_seed,
                            "mirror_group_id": block_index * groups_per_block + group_in_block,
                            "mirror_pair_member": member,
                            "target_speed_scale": speed,
                            "defender_bias": defender_bias,
                            "observation_condition": condition_id,
                            "pursuit_overrides": copy.deepcopy(pursuit),
                            "execution_overrides": copy.deepcopy(execution),
                            "rollout_policy": "phase56_strong_baseline_matrix",
                            "scene_block": block_name,
                            "evaluation_split": split,
                            "condition_id": condition_id,
                            "training_target_speed_scales": list(TRAINING_SPEED_SCALES),
                            "training_geometry_ranges": copy.deepcopy(TRAINING_GEOMETRY),
                        }
                        config = config_for_spec(environment_config, protocol, spec, max_steps=None)
                        env = CaptureRadiusPursuit3DEnv(config, obstacle_count=1, target_speed_scale=speed)
                        scenario = s4_adaptive_branching_scenario(
                            env,
                            layout_seed=layout_seed,
                            defender_bias=defender_bias,
                            variation=TRAINING_GEOMETRY,
                        )
                        spec["scenario"] = scenario_metadata(scenario)
                        group_records.append(spec)
                except ValueError:
                    continue
                accepted = group_records
                break
            if accepted is None:
                raise RuntimeError(f"Unable to generate valid mirror group {block_name}:{group_in_block}.")
            records.extend(accepted)
        print(json.dumps({"status": "block_complete", "block": block_name, "mirror_groups": groups_per_block}), flush=True)
    return records


def validate_records(records: list[dict[str, Any]], groups_per_block: int) -> None:
    expected_episodes = 6 * groups_per_block
    if len(records) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} records, got {len(records)}")
    groups: dict[int, list[dict[str, Any]]] = {}
    episode_indices: set[int] = set()
    for record in records:
        episode_index = int(record["episode_index"])
        if episode_index in episode_indices:
            raise ValueError(f"duplicate episode_index {episode_index}")
        episode_indices.add(episode_index)
        if record.get("pursuit_overrides", {}).get("detection_dropout_probability") != BASE_PURSUIT["detection_dropout_probability"]:
            raise ValueError("detection dropout changed outside the declared matrix")
        groups.setdefault(int(record["mirror_group_id"]), []).append(record)
    if len(groups) != 3 * groups_per_block:
        raise ValueError("mirror-group count does not match the block contract")
    for group, members in groups.items():
        if len(members) != 2 or {item["defender_bias"] for item in members} != {"upper", "lower"}:
            raise ValueError(f"mirror group {group} is incomplete")
        if len({item["layout_seed"] for item in members}) != 1:
            raise ValueError(f"mirror group {group} does not share layout_seed")
        if len({item["evaluation_split"] for item in members}) != 1:
            raise ValueError(f"mirror group {group} crosses split")
        if members[0]["scenario"]["obstacles"] != members[1]["scenario"]["obstacles"]:
            raise ValueError(f"mirror group {group} does not share obstacles")
    split_counts = {split: sum(1 for group_members in groups.values() if group_members[0]["evaluation_split"] == split) for split in SPLITS}
    if sum(split_counts.values()) != len(groups):
        raise ValueError("split accounting is incomplete")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    protocol = load_protocol(args.protocol)
    records = build_records(
        protocol,
        args.environment_config.resolve(),
        groups_per_block=args.groups_per_block,
        seed_start=args.seed_start,
    )
    validate_records(records, args.groups_per_block)
    output.mkdir(parents=True, exist_ok=True)
    scenes_path = output / "scenes.jsonl"
    scenes_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    split_counts = {
        split: sum(1 for record in records if record["evaluation_split"] == split)
        for split in SPLITS
    }
    manifest = {
        "name": args.manifest_name,
        "episodes": len(records),
        "mirror_groups": len(records) // 2,
        "groups_per_block": int(args.groups_per_block),
        "seed_start": int(args.seed_start),
        "blocks": ["id_reference", "delay_noise_grid", "communication_execution_grid"],
        "split_counts_episodes": split_counts,
        "split_counts_mirror_groups": {split: count // 2 for split, count in split_counts.items()},
        "delay_levels": list(DELAY_LEVELS),
        "noise_levels": list(NOISE_LEVELS),
        "dropout_levels": list(DROPOUT_LEVELS),
        "message_delay_levels": list(MESSAGE_DELAY_LEVELS),
        "tracking_levels": list(TRACKING_LEVELS),
        "training_target_speed_scales": list(TRAINING_SPEED_SCALES),
        "training_geometry_ranges": TRAINING_GEOMETRY,
        "base_pursuit": BASE_PURSUIT,
        "base_execution": BASE_EXECUTION,
        "scene_manifest_sha256": hashlib.sha256(scenes_path.read_bytes()).hexdigest(),
        "protocol": str(args.protocol.resolve()),
        "environment_config": str(args.environment_config.resolve()),
        "claim_boundary": "fresh baseline matrix; no locked-test tuning and no safety proof",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()

