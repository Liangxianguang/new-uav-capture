"""Freeze an unseen adaptive-branch behavior S4 OOD block for Phase 16."""

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
OOD_BEHAVIORS = (
    {"name": "short_lookahead", "target_branch_defender_lookahead_seconds": 0.20, "target_branch_commit_margin_seconds": 0.0},
    {"name": "long_lookahead_margin", "target_branch_defender_lookahead_seconds": 1.20, "target_branch_commit_margin_seconds": 0.20},
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=739101)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_ENVIRONMENT_CONFIG)
    return parser.parse_args()


def build_records(protocol: dict[str, Any], environment_config: Path, episodes: int, seed_start: int) -> list[dict[str, Any]]:
    if episodes <= 0 or episodes % 2:
        raise ValueError("episodes must be a positive even number for mirror pairing.")
    observations = list(protocol["s4"]["observation_conditions"])
    records: list[dict[str, Any]] = []
    for group_index in range(episodes // 2):
        behavior = OOD_BEHAVIORS[group_index % len(OOD_BEHAVIORS)]
        speed = float((0.65, 0.75)[group_index % 2])
        observation = observations[(group_index // len(OOD_BEHAVIORS)) % len(observations)]
        layout_seed = int(seed_start + 1_000_000 + group_index)
        group_records: list[dict[str, Any]] = []
        for member, defender_bias in enumerate(("upper", "lower")):
            episode_index = 2 * group_index + member
            pursuit_overrides = copy.deepcopy(observation["pursuit_overrides"])
            pursuit_overrides.update({
                "target_branch_defender_lookahead_seconds": behavior["target_branch_defender_lookahead_seconds"],
                "target_branch_commit_margin_seconds": behavior["target_branch_commit_margin_seconds"],
            })
            spec = {
                "episode_index": episode_index,
                "episode_seed": int(seed_start + episode_index),
                "layout_seed": layout_seed,
                "mirror_group_id": group_index,
                "mirror_pair_member": member,
                "target_speed_scale": speed,
                "defender_bias": defender_bias,
                "observation_condition": str(observation["name"]),
                "pursuit_overrides": pursuit_overrides,
                "rollout_policy": "ood_target_behavior_evaluation_only",
                "ood_block": "target_behavior_shift",
                "ood_behavior": copy.deepcopy(behavior),
                "training_target_speed_scales": list(TRAINING_SPEED_SCALES),
                "training_geometry_ranges": copy.deepcopy(TRAINING_GEOMETRY),
            }
            config = config_for_spec(environment_config, protocol, spec, max_steps=None)
            env = CaptureRadiusPursuit3DEnv(config, obstacle_count=1, target_speed_scale=speed)
            scenario = s4_adaptive_branching_scenario(env, layout_seed=layout_seed, defender_bias=defender_bias, variation=TRAINING_GEOMETRY)
            spec["scenario"] = scenario_metadata(scenario)
            group_records.append(spec)
        records.extend(group_records)
        if (group_index + 1) % 10 == 0:
            print(json.dumps({"status": "generating", "mirror_groups_completed": group_index + 1}), flush=True)
    return records


def validate_records(records: list[dict[str, Any]]) -> None:
    if len(records) % 2:
        raise ValueError("records must have paired cardinality.")
    groups: dict[int, list[dict[str, Any]]] = {}
    behavior_names = {item["name"] for item in OOD_BEHAVIORS}
    for record in records:
        if record.get("ood_behavior", {}).get("name") not in behavior_names:
            raise ValueError("Unknown target behavior OOD condition.")
        if float(record["target_speed_scale"]) not in {0.65, 0.75}:
            raise ValueError("Target behavior OOD speed must remain in-distribution.")
        groups.setdefault(int(record["mirror_group_id"]), []).append(record)
    for group, members in groups.items():
        if len(members) != 2 or {item["defender_bias"] for item in members} != {"upper", "lower"}:
            raise ValueError(f"Mirror group {group} is incomplete.")
        first, second = members
        if first["layout_seed"] != second["layout_seed"] or first["ood_behavior"] != second["ood_behavior"] or first["scenario"]["obstacles"] != second["scenario"]["obstacles"]:
            raise ValueError(f"Mirror group {group} is not frozen consistently.")
    if {item["ood_behavior"]["name"] for item in records} != behavior_names:
        raise ValueError("Both target behavior OOD conditions are required.")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    protocol = load_protocol(args.protocol)
    records = build_records(protocol, args.environment_config.resolve(), args.episodes, args.seed_start)
    validate_records(records)
    output.mkdir(parents=True, exist_ok=True)
    scenes_path = output / "scenes.jsonl"
    scenes_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    manifest = {
        "name": "phase16_ood_target_behavior_shift",
        "evaluation_split": "ood_diagnostic",
        "episodes": len(records),
        "mirror_groups": len(records) // 2,
        "seed_start": int(args.seed_start),
        "training_target_speed_scales": list(TRAINING_SPEED_SCALES),
        "training_geometry_ranges": TRAINING_GEOMETRY,
        "ood_behaviors": list(OOD_BEHAVIORS),
        "scene_manifest_sha256": hashlib.sha256(scenes_path.read_bytes()).hexdigest(),
        "protocol": str(args.protocol.resolve()),
        "environment_config": str(args.environment_config.resolve()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
