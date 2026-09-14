"""Generate reusable, validation-only Phase 71 randomized scenes.

Scene generation is separated from policy rollout so multiple defenders can be
compared on exactly the same maps without repeating route-feasibility search.
The command accepts validation seed blocks only; it has no locked-test mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import random_central_mixed_obstacle_scenario, scenario_metadata  # noqa: E402
from evaluate_random_central_mixed_obstacles import (  # noqa: E402
    config_for_spec,
    effective_seed_block,
    episode_spec,
    load_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--validation-seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def generate(
    protocol_path: Path,
    environment_path: Path,
    validation_seed: int,
    episodes: int,
    output_dir: Path,
) -> None:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    protocol = load_protocol(protocol_path.resolve())
    seed_block = effective_seed_block(protocol, "validation", validation_seed)
    environment = environment_path.resolve()
    if not environment.is_file():
        raise FileNotFoundError(environment)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    config_base = yaml.safe_load(environment.read_text(encoding="utf-8"))
    if not isinstance(config_base, dict):
        raise ValueError("environment config must be a YAML mapping")
    scene_path = output_dir / "scenes.jsonl"
    records: list[dict[str, object]] = []
    with scene_path.open("w", encoding="utf-8") as stream:
        for episode_index in range(episodes):
            spec = episode_spec(protocol, "validation", episode_index, seed_block=seed_block)
            config = config_for_spec("capture", spec, environment)
            env = CaptureRadiusPursuit3DEnv(
                config,
                obstacle_count=0,
                target_speed_scale=float(spec["target_speed_scale"]),
            )
            scenario = random_central_mixed_obstacle_scenario(
                env,
                layout_seed=int(spec["layout_seed"]),
                initial_side_distance=float(spec["initial_side_distance"]),
                defender_side=str(spec["defender_side"]),
                target_crossing_required=bool(spec["target_crossing_required"]),
                obstacle_count_range=(int(spec["obstacle_count"]), int(spec["obstacle_count"])),
                max_attempts=int(protocol["s3"].get("max_sampling_attempts", 500)),
                required_defender_zone_entries=int(protocol["s3"].get("required_defender_zone_entries", 1)),
            )
            record = {
                "episode_index": episode_index,
                "spec": spec,
                "scenario": scenario_metadata(scenario),
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            records.append(record)
            print(f"generated validation seed={seed_block} episode={episode_index + 1}/{episodes}", flush=True)
    metadata = {
        "evaluation_type": "phase71_validation_scene_cache",
        "not_a_locked_test": True,
        "locked_test": False,
        "split": "validation",
        "seed_block": seed_block,
        "episodes": episodes,
        "protocol": str(protocol_path.resolve()),
        "environment_config": str(environment),
        "scene_file": str(scene_path),
        "scene_count": len(records),
    }
    (output_dir / "scene_cache_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    generate(args.protocol, args.environment_config, args.validation_seed, args.episodes, args.output_dir)


if __name__ == "__main__":
    main()
