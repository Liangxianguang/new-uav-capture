"""Freeze fresh, mirror-paired in-distribution scenes for Phase 27 RNIC.

The generator keeps geometry, speed and observation ranges inside the released
S4 training support while using a disjoint seed block from the historical
Phase 16/17 validation scenes.  It writes evaluator-compatible records only;
no predictor labels or locked-test records are created.
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
from encirclement3d.showcase import scenario_metadata, s4_adaptive_branching_scenario  # noqa: E402
from evaluate_minimax_mpc import DEFAULT_ENVIRONMENT_CONFIG  # noqa: E402
from evaluate_s4_branching import config_for_spec, load_protocol  # noqa: E402


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "phase15_s4_branching_pilot.yaml"
TRAINING_VARIATION = {
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed-start", type=int, default=781101)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_ENVIRONMENT_CONFIG)
    parser.add_argument("--scene-block", default="phase27_rnic_fresh_validation")
    parser.add_argument("--rollout-policy", default="rnic_confirmation_evaluation_only")
    parser.add_argument("--name", default=None)
    parser.add_argument("--evaluation-split", default="validation_confirmation")
    return parser.parse_args()


def build_records(
    protocol: dict[str, Any],
    environment_config: Path,
    episodes: int,
    seed_start: int,
    *,
    scene_block: str = "phase27_rnic_fresh_validation",
    rollout_policy: str = "rnic_confirmation_evaluation_only",
) -> list[dict[str, Any]]:
    if episodes <= 0 or episodes % 2:
        raise ValueError("episodes must be a positive even number for mirror pairing")
    settings = dict(protocol["s4"])
    observations = list(settings["observation_conditions"])
    speeds = [float(value) for value in settings["target_speed_scales"]]
    if not observations or not speeds:
        raise ValueError("protocol must define S4 target speeds and observation conditions")

    records: list[dict[str, Any]] = []
    condition_count = len(speeds) * len(observations)
    for group_index in range(episodes // 2):
        condition_index = group_index % condition_count
        speed = speeds[condition_index % len(speeds)]
        observation = observations[(condition_index // len(speeds)) % len(observations)]
        layout_seed = int(seed_start + 1_000_000 + group_index)
        for member, defender_bias in enumerate(("upper", "lower")):
            episode_index = 2 * group_index + member
            spec = {
                "episode_index": episode_index,
                "episode_seed": int(seed_start + episode_index),
                "layout_seed": layout_seed,
                "mirror_group_id": group_index,
                "mirror_pair_member": member,
                "target_speed_scale": speed,
                "observation_condition": str(observation["name"]),
                "pursuit_overrides": copy.deepcopy(observation["pursuit_overrides"]),
                "defender_bias": defender_bias,
                "rollout_policy": str(rollout_policy),
                "condition_index": condition_index,
                "condition_table_size": condition_count,
                "scene_block": str(scene_block),
                "training_variation_ranges": copy.deepcopy(TRAINING_VARIATION),
            }
            config = config_for_spec(environment_config, protocol, spec, max_steps=None)
            env = CaptureRadiusPursuit3DEnv(
                config,
                obstacle_count=1,
                target_speed_scale=speed,
            )
            scenario = s4_adaptive_branching_scenario(
                env,
                layout_seed=layout_seed,
                defender_bias=defender_bias,
                variation=TRAINING_VARIATION,
            )
            spec["scenario"] = scenario_metadata(scenario)
            records.append(spec)
    return records


def validate_records(records: list[dict[str, Any]], previous_scene_hashes: set[str] | None = None) -> None:
    if len(records) == 0 or len(records) % 2:
        raise ValueError("records must have positive paired cardinality")
    groups: dict[int, list[dict[str, Any]]] = {}
    layout_seeds: set[int] = set()
    episode_seeds: set[int] = set()
    for record in records:
        group = int(record["mirror_group_id"])
        groups.setdefault(group, []).append(record)
        layout_seeds.add(int(record["layout_seed"]))
        episode_seeds.add(int(record["episode_seed"]))
        obstacle = record["scenario"]["obstacles"][0]
        half = obstacle["half_extents_xy"]
        values = {
            "wall_half_extent_x_m": float(half[0]),
            "wall_half_extent_y_m": float(half[1]),
            "wall_height_m": float(obstacle["height"]),
        }
        for name, value in values.items():
            low, high = TRAINING_VARIATION[name]
            if not low <= value <= high:
                raise ValueError(f"training-support geometry contract failed for {name}: {value}")
    if len(layout_seeds) != len(records) // 2 or len(episode_seeds) != len(records):
        raise ValueError("layout and episode seeds must be unique at their expected granularity")
    for group, members in groups.items():
        if len(members) != 2 or {item["defender_bias"] for item in members} != {"upper", "lower"}:
            raise ValueError(f"mirror group {group} is incomplete")
        first, second = members
        if first["layout_seed"] != second["layout_seed"]:
            raise ValueError(f"mirror group {group} does not share layout seed")
        if first["scenario"]["obstacles"] != second["scenario"]["obstacles"]:
            raise ValueError(f"mirror group {group} does not share geometry")
    if previous_scene_hashes:
        current_hashes = {
            hashlib.sha256(
                json.dumps(record["scenario"], sort_keys=True).encode("utf-8")
            ).hexdigest()
            for record in records
        }
        if current_hashes.intersection(previous_scene_hashes):
            raise ValueError("fresh scene block overlaps a previous scene geometry hash")


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    protocol = load_protocol(args.protocol.resolve())
    records = build_records(
        protocol,
        args.environment_config.resolve(),
        args.episodes,
        args.seed_start,
        scene_block=args.scene_block,
        rollout_policy=args.rollout_policy,
    )
    validate_records(records)
    output.mkdir(parents=True, exist_ok=True)
    scenes_path = output / "scenes.jsonl"
    scenes_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "name": str(args.name or args.scene_block),
        "evaluation_split": str(args.evaluation_split),
        "episodes": len(records),
        "mirror_groups": len(records) // 2,
        "seed_start": int(args.seed_start),
        "training_variation_ranges": TRAINING_VARIATION,
        "scene_manifest_sha256": hashlib.sha256(scenes_path.read_bytes()).hexdigest(),
        "protocol": str(args.protocol.resolve()),
        "environment_config": str(args.environment_config.resolve()),
        "locked_test_included": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
