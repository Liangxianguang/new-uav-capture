"""Freeze a fresh ID validation-confirmation block for QDR liveness.

Only the delayed execution contract is stressed: sensing and communication
remain at the nominal Phase 15 support, while the plant uses the frozen
delay-4, bounded-command-noise, immutable-authority contract.  Complete mirror
groups are kept together and the output is never allowed to overwrite data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import s4_adaptive_branching_scenario, scenario_metadata  # noqa: E402
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
NOMINAL_PURSUIT = {
    "detection_range": 14.0,
    "detection_dropout_probability": 0.15,
    "observation_noise_std": 0.03,
    "message_delay_steps": 2,
    "message_dropout_probability": 0.05,
}
LIVENESS_EXECUTION = {
    "enabled": True,
    "action_delay_steps": 4,
    "pending_command_authority": "immutable",
    "command_noise_std": 0.08,
    "command_noise_bound_sigma": 3.0,
    "clip_command_noise": True,
    "velocity_time_constant_seconds": 0.0,
    "drag_coefficient": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=831101)
    parser.add_argument("--manifest-name", default="phase48_qdr_liveness_confirmation")
    parser.add_argument("--evaluation-split", default="validation_confirmation")
    parser.add_argument("--rollout-policy", default="qdr_liveness_confirmation")
    parser.add_argument("--scene-block", default="phase48_qdr_liveness_confirmation")
    parser.add_argument("--observation-condition", default="nominal_liveness_confirmation")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_ENVIRONMENT_CONFIG)
    return parser.parse_args()


def build_records(
    protocol: dict[str, Any],
    environment_config: Path,
    episodes: int,
    seed_start: int,
    *,
    evaluation_split: str = "validation_confirmation",
    rollout_policy: str = "qdr_liveness_confirmation",
    scene_block: str = "phase48_qdr_liveness_confirmation",
    observation_condition: str = "nominal_liveness_confirmation",
) -> list[dict[str, Any]]:
    if episodes <= 0 or episodes % 2:
        raise ValueError("episodes must be a positive even number for mirror pairing.")
    records: list[dict[str, Any]] = []
    candidate_offset = 0
    for group_index in range(episodes // 2):
        speed = float((0.65, 0.75)[group_index % 2])
        accepted: list[dict[str, Any]] | None = None
        for _attempt in range(10_000):
            layout_seed = int(seed_start + 1_000_000 + candidate_offset)
            candidate_offset += 1
            group_records: list[dict[str, Any]] = []
            try:
                for member, defender_bias in enumerate(("upper", "lower")):
                    episode_index = 2 * group_index + member
                    spec = {
                        "episode_index": episode_index,
                        "episode_seed": int(seed_start + episode_index),
                        "layout_seed": layout_seed,
                        "mirror_group_id": group_index,
                        "mirror_pair_member": member,
                        "target_speed_scale": speed,
                        "defender_bias": defender_bias,
                        "observation_condition": observation_condition,
                        "pursuit_overrides": copy.deepcopy(NOMINAL_PURSUIT),
                        "execution_overrides": copy.deepcopy(LIVENESS_EXECUTION),
                        "rollout_policy": rollout_policy,
                        "scene_block": scene_block,
                        "evaluation_split": evaluation_split,
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
            raise RuntimeError(f"Unable to generate valid liveness mirror group {group_index}.")
        records.extend(accepted)
        if (group_index + 1) % 10 == 0:
            print(json.dumps({"status": "generating", "mirror_groups_completed": group_index + 1}), flush=True)
    return records


def validate_records(records: list[dict[str, Any]]) -> None:
    if len(records) <= 0 or len(records) % 2:
        raise ValueError("records must have positive paired cardinality.")
    groups: dict[int, list[dict[str, Any]]] = {}
    episode_indices: set[int] = set()
    for record in records:
        episode_index = int(record["episode_index"])
        if episode_index in episode_indices:
            raise ValueError(f"duplicate episode_index {episode_index}.")
        episode_indices.add(episode_index)
        if record.get("pursuit_overrides") != NOMINAL_PURSUIT:
            raise ValueError("Nominal pursuit contract is missing or inconsistent.")
        if record.get("execution_overrides") != LIVENESS_EXECUTION:
            raise ValueError("Liveness execution contract is missing or inconsistent.")
        if float(record["target_speed_scale"]) not in {0.65, 0.75}:
            raise ValueError("Liveness confirmation speed must remain in-distribution.")
        groups.setdefault(int(record["mirror_group_id"]), []).append(record)
    for group, members in groups.items():
        if len(members) != 2 or {item["defender_bias"] for item in members} != {"upper", "lower"}:
            raise ValueError(f"Mirror group {group} is incomplete.")
        first, second = members
        if first["layout_seed"] != second["layout_seed"]:
            raise ValueError(f"Mirror group {group} does not share layout_seed.")
        if first["scenario"]["obstacles"] != second["scenario"]["obstacles"]:
            raise ValueError(f"Mirror group {group} does not share obstacles.")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    protocol = load_protocol(args.protocol)
    records = build_records(
        protocol,
        args.environment_config.resolve(),
        args.episodes,
        args.seed_start,
        evaluation_split=args.evaluation_split,
        rollout_policy=args.rollout_policy,
        scene_block=args.scene_block,
        observation_condition=args.observation_condition,
    )
    validate_records(records)
    output.mkdir(parents=True, exist_ok=True)
    scenes_path = output / "scenes.jsonl"
    scenes_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    manifest = {
        "name": args.manifest_name,
        "evaluation_split": args.evaluation_split,
        "episodes": len(records),
        "mirror_groups": len(records) // 2,
        "seed_start": int(args.seed_start),
        "training_target_speed_scales": list(TRAINING_SPEED_SCALES),
        "training_geometry_ranges": TRAINING_GEOMETRY,
        "nominal_pursuit": NOMINAL_PURSUIT,
        "liveness_execution": LIVENESS_EXECUTION,
        "pre_registered_liveness_gate": {
            "timeout_rate_max": 0.05,
            "timeout_delta_vs_qdr_off_max": 0.05,
            "max_exhaustion_streak_steps_max": 24,
        },
        "scene_manifest_sha256": hashlib.sha256(scenes_path.read_bytes()).hexdigest(),
        "protocol": str(args.protocol.resolve()),
        "environment_config": str(args.environment_config.resolve()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
