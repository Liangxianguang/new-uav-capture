"""Generate a fresh Phase 65 tail-stress matrix for QDR segment auditing.

The matrix is intentionally separate from the Phase 58--64 manifests.  It
extends the declared delay/noise ranges (action delay 12, command noise 0.20,
message delay 8, dropout 0.40, and stronger tracking/drag combinations) while
preserving complete upper/lower mirror groups and split disjointness.  Scene
generation uses only public geometry and the same route validator as the
earlier reproducible matrices.
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


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "phase58_delayed_planning_protocol.yaml"
SPLITS = ("development_calibration", "development_confirmation", "locked_diagnostic")
BLOCKS = ("id_reference", "execution_tail", "communication_tail", "joint_tail_transfer")
TARGET_SPEEDS = (0.50, 0.65, 0.80)
TRAINING_TARGET_SPEEDS = TARGET_SPEEDS

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
TRANSFER_GEOMETRY = {
    "wall_half_extent_x_m": (0.70, 0.85),
    "wall_half_extent_y_m": (4.38, 4.42),
    "wall_height_m": (9.60, 9.80),
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

EXECUTION_TAIL = tuple(
    (delay, noise, tracking, drag)
    for delay in (6, 8, 10, 12)
    for noise in (0.12, 0.16, 0.20)
    for tracking, drag in ((0.0, 0.0), (0.10, 0.05), (0.20, 0.10), (0.30, 0.10))
)
COMMUNICATION_TAIL = tuple(
    (message_delay, dropout, tracking, drag)
    for message_delay in (4, 6, 8)
    for dropout in (0.20, 0.30, 0.40)
    for tracking, drag in ((0.10, 0.05), (0.20, 0.10), (0.30, 0.10))
)
JOINT_TAIL = tuple(
    (delay, noise, message_delay, dropout, tracking, drag)
    for delay in (10, 12)
    for noise in (0.16, 0.20)
    for message_delay in (6, 8)
    for dropout in (0.30, 0.40)
    for tracking in (0.20, 0.30)
    for drag in (0.05, 0.10)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-per-block", type=int, default=45)
    parser.add_argument("--seed-start", type=int, default=992101)
    parser.add_argument("--manifest-name", default="phase65_qdr_tail_stress_matrix")
    parser.add_argument(
        "--rollout-policy",
        default="phase65_qdr_segment_liveness_audit",
        help="Immutable provenance label stored in each scene record.",
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_ENVIRONMENT_CONFIG)
    return parser.parse_args()


def split_for(block_index: int, group_in_block: int) -> str:
    return SPLITS[(block_index + group_in_block) % len(SPLITS)]


def _format_float(value: float) -> str:
    return f"{float(value):.2f}"


def condition_for(
    block_index: int,
    group_in_block: int,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    pursuit = copy.deepcopy(BASE_PURSUIT)
    execution = copy.deepcopy(BASE_EXECUTION)
    factors: dict[str, Any] = {
        "action_delay_steps": int(execution["action_delay_steps"]),
        "command_noise_std": float(execution["command_noise_std"]),
        "message_delay_steps": int(pursuit["message_delay_steps"]),
        "message_dropout_probability": float(pursuit["message_dropout_probability"]),
        "velocity_time_constant_seconds": float(execution["velocity_time_constant_seconds"]),
        "drag_coefficient": float(execution["drag_coefficient"]),
    }
    if block_index == 0:
        condition_id = "id_reference"
    elif block_index == 1:
        delay, noise, tracking, drag = EXECUTION_TAIL[group_in_block % len(EXECUTION_TAIL)]
        execution.update(
            action_delay_steps=int(delay),
            command_noise_std=float(noise),
            velocity_time_constant_seconds=float(tracking),
            drag_coefficient=float(drag),
        )
        factors.update(
            action_delay_steps=int(delay),
            command_noise_std=float(noise),
            velocity_time_constant_seconds=float(tracking),
            drag_coefficient=float(drag),
        )
        condition_id = (
            f"tail_delay{delay}_noise{_format_float(noise)}"
            f"_track{_format_float(tracking)}_drag{_format_float(drag)}"
        )
    elif block_index == 2:
        message_delay, dropout, tracking, drag = COMMUNICATION_TAIL[
            group_in_block % len(COMMUNICATION_TAIL)
        ]
        pursuit.update(
            message_delay_steps=int(message_delay),
            message_dropout_probability=float(dropout),
        )
        execution.update(
            velocity_time_constant_seconds=float(tracking),
            drag_coefficient=float(drag),
        )
        factors.update(
            message_delay_steps=int(message_delay),
            message_dropout_probability=float(dropout),
            velocity_time_constant_seconds=float(tracking),
            drag_coefficient=float(drag),
        )
        condition_id = (
            f"tail_msgdelay{message_delay}_dropout{_format_float(dropout)}"
            f"_track{_format_float(tracking)}_drag{_format_float(drag)}"
        )
    elif block_index == 3:
        delay, noise, message_delay, dropout, tracking, drag = JOINT_TAIL[
            group_in_block % len(JOINT_TAIL)
        ]
        pursuit.update(
            message_delay_steps=int(message_delay),
            message_dropout_probability=float(dropout),
        )
        execution.update(
            action_delay_steps=int(delay),
            command_noise_std=float(noise),
            velocity_time_constant_seconds=float(tracking),
            drag_coefficient=float(drag),
        )
        factors.update(
            action_delay_steps=int(delay),
            command_noise_std=float(noise),
            message_delay_steps=int(message_delay),
            message_dropout_probability=float(dropout),
            velocity_time_constant_seconds=float(tracking),
            drag_coefficient=float(drag),
        )
        condition_id = (
            f"joint_tail_delay{delay}_noise{_format_float(noise)}"
            f"_msgdelay{message_delay}_dropout{_format_float(dropout)}"
            f"_track{_format_float(tracking)}_drag{_format_float(drag)}"
        )
    else:  # pragma: no cover
        raise ValueError(f"unknown block index: {block_index}")
    return condition_id, pursuit, execution, factors


def geometry_for(block_index: int) -> tuple[str, dict[str, tuple[float, float]]]:
    if block_index == 3:
        return "training_ood_transfer", copy.deepcopy(TRANSFER_GEOMETRY)
    return "training_range", copy.deepcopy(TRAINING_GEOMETRY)


def build_records(
    protocol: dict[str, Any],
    environment_config: Path,
    *,
    groups_per_block: int,
    seed_start: int,
    rollout_policy: str,
) -> list[dict[str, Any]]:
    if groups_per_block <= 0 or groups_per_block % len(SPLITS):
        raise ValueError("groups_per_block must be positive and divisible by three")
    records: list[dict[str, Any]] = []
    candidate_offset = 0
    for block_index, block_name in enumerate(BLOCKS):
        for group_in_block in range(groups_per_block):
            split = split_for(block_index, group_in_block)
            condition_id, pursuit, execution, factors = condition_for(block_index, group_in_block)
            geometry_id, geometry_ranges = geometry_for(block_index)
            speed = float(TARGET_SPEEDS[group_in_block % len(TARGET_SPEEDS)])
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
                            "rollout_policy": str(rollout_policy),
                            "scene_block": block_name,
                            "evaluation_split": split,
                            "condition_id": condition_id,
                            "geometry_id": geometry_id,
                            "phase65_factors": copy.deepcopy(factors),
                            "training_target_speed_scales": list(TRAINING_TARGET_SPEEDS),
                            "training_geometry_ranges": copy.deepcopy(TRAINING_GEOMETRY),
                            "transfer_geometry_ranges": copy.deepcopy(TRANSFER_GEOMETRY),
                        }
                        config = config_for_spec(environment_config, protocol, spec, max_steps=None)
                        env = CaptureRadiusPursuit3DEnv(config, obstacle_count=1, target_speed_scale=speed)
                        scenario = s4_adaptive_branching_scenario(
                            env,
                            layout_seed=layout_seed,
                            defender_bias=defender_bias,
                            variation=geometry_ranges,
                        )
                        spec["scenario"] = scenario_metadata(scenario)
                        group_records.append(spec)
                except (RuntimeError, ValueError):
                    continue
                accepted = group_records
                break
            if accepted is None:
                raise RuntimeError(f"unable to generate valid mirror group {block_name}:{group_in_block}")
            records.extend(accepted)
        print(json.dumps({"status": "block_complete", "block": block_name, "mirror_groups": groups_per_block}), flush=True)
    return records


def validate_records(records: list[dict[str, Any]], groups_per_block: int) -> None:
    expected_episodes = 2 * len(BLOCKS) * groups_per_block
    if len(records) != expected_episodes:
        raise ValueError(f"expected {expected_episodes} records, got {len(records)}")
    groups: dict[int, list[dict[str, Any]]] = {}
    episode_indices: set[int] = set()
    for record in records:
        episode_index = int(record["episode_index"])
        if episode_index in episode_indices:
            raise ValueError(f"duplicate episode_index {episode_index}")
        episode_indices.add(episode_index)
        if record["pursuit_overrides"]["detection_dropout_probability"] != BASE_PURSUIT["detection_dropout_probability"]:
            raise ValueError("detection dropout changed outside the declared matrix")
        if record["pursuit_overrides"]["observation_noise_std"] != BASE_PURSUIT["observation_noise_std"]:
            raise ValueError("observation noise changed outside the declared matrix")
        groups.setdefault(int(record["mirror_group_id"]), []).append(record)
    if len(groups) != len(BLOCKS) * groups_per_block:
        raise ValueError("mirror-group count does not match block contract")
    for group, members in groups.items():
        if len(members) != 2 or {item["defender_bias"] for item in members} != {"upper", "lower"}:
            raise ValueError(f"mirror group {group} is incomplete")
        if len({item["layout_seed"] for item in members}) != 1:
            raise ValueError(f"mirror group {group} does not share layout_seed")
        if len({item["evaluation_split"] for item in members}) != 1:
            raise ValueError(f"mirror group {group} crosses split")
        if members[0]["scenario"]["obstacles"] != members[1]["scenario"]["obstacles"]:
            raise ValueError(f"mirror group {group} does not share obstacles")
    split_counts = {
        split: sum(1 for members in groups.values() if members[0]["evaluation_split"] == split)
        for split in SPLITS
    }
    expected_per_split = len(BLOCKS) * groups_per_block // len(SPLITS)
    if any(count != expected_per_split for count in split_counts.values()):
        raise ValueError(f"unexpected split counts: {split_counts}")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    protocol = load_protocol(args.protocol)
    records = build_records(
        protocol,
        args.environment_config.resolve(),
        groups_per_block=args.groups_per_block,
        seed_start=args.seed_start,
        rollout_policy=str(args.rollout_policy),
    )
    validate_records(records, args.groups_per_block)
    output.mkdir(parents=True, exist_ok=True)
    scenes_path = output / "scenes.jsonl"
    scenes_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    split_counts = {
        split: sum(1 for record in records if record["evaluation_split"] == split)
        for split in SPLITS
    }
    block_counts = {
        block: sum(1 for record in records if record["scene_block"] == block)
        for block in BLOCKS
    }
    manifest = {
        "name": args.manifest_name,
        "episodes": len(records),
        "mirror_groups": len(records) // 2,
        "groups_per_block": int(args.groups_per_block),
        "seed_start": int(args.seed_start),
        "rollout_policy": str(args.rollout_policy),
        "blocks": list(BLOCKS),
        "block_counts_episodes": block_counts,
        "split_counts_episodes": split_counts,
        "split_counts_mirror_groups": {split: count // 2 for split, count in split_counts.items()},
        "action_delay_levels": [0, 4, 6, 8, 10, 12],
        "command_noise_levels": [0.00, 0.08, 0.12, 0.16, 0.20],
        "message_delay_levels": [2, 4, 6, 8],
        "message_dropout_levels": [0.05, 0.20, 0.30, 0.40],
        "tracking_levels": [0.0, 0.10, 0.20, 0.30],
        "drag_levels": [0.0, 0.05, 0.10],
        "training_target_speed_scales": list(TRAINING_TARGET_SPEEDS),
        "training_geometry_ranges": TRAINING_GEOMETRY,
        "transfer_geometry_ranges": TRANSFER_GEOMETRY,
        "base_pursuit": BASE_PURSUIT,
        "base_execution": BASE_EXECUTION,
        "scene_manifest_sha256": hashlib.sha256(scenes_path.read_bytes()).hexdigest(),
        "protocol": str(args.protocol.resolve()),
        "environment_config": str(args.environment_config.resolve()),
        "claim_boundary": "fresh tail-stress calibration matrix; no locked-test tuning; local CBF empirical only; no safety proof",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
