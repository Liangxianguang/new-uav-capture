"""Freeze a geometry-shifted S4 OOD scene block for Phase 16 evaluation.

The training archive varies wall half-x in [0.45, 0.65], half-y in
[3.85, 4.35], and height in [9.25, 9.55].  This generator samples only from
disjoint, still-valid ranges and pairs each layout with upper/lower defender
mirrors.  It writes evaluator-compatible scene records without touching a
training, validation, or locked-test manifest.
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
TRAINING_GEOMETRY = {
    "wall_half_extent_x_m": (0.45, 0.65),
    "wall_half_extent_y_m": (3.85, 4.35),
    "wall_height_m": (9.25, 9.55),
}
OOD_GEOMETRY = {
    "wall_half_extent_x_m": (0.70, 0.85),
    # Disjoint from training [3.85, 4.35], while retaining clearance for
    # the fixed 5.30 m branch exits under the route validator.
    "wall_half_extent_y_m": (4.38, 4.42),
    "wall_height_m": (9.60, 9.80),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=736101)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_ENVIRONMENT_CONFIG)
    return parser.parse_args()


def build_records(
    protocol: dict[str, Any],
    environment_config: Path,
    episodes: int,
    seed_start: int,
) -> list[dict[str, Any]]:
    if episodes <= 0 or episodes % 2:
        raise ValueError("episodes must be a positive even number for mirror pairing.")
    settings = dict(protocol["s4"])
    observations = list(settings["observation_conditions"])
    speeds = [float(value) for value in settings["target_speed_scales"]]
    if not observations or not speeds:
        raise ValueError("protocol must define S4 speeds and observation conditions.")
    records: list[dict[str, Any]] = []
    candidate_offset = 0
    for group_index in range(episodes // 2):
        speed = speeds[group_index % len(speeds)]
        observation = observations[(group_index // len(speeds)) % len(observations)]
        accepted: list[dict[str, Any]] | None = None
        # Long OOD walls occasionally close one conservative route. Reject the
        # whole mirror group and advance deterministically until both routes pass.
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
                        "observation_condition": str(observation["name"]),
                        "pursuit_overrides": copy.deepcopy(observation["pursuit_overrides"]),
                        "rollout_policy": "ood_geometry_evaluation_only",
                        "ood_block": "geometry_shift",
                        "ood_geometry_ranges": copy.deepcopy(OOD_GEOMETRY),
                        "training_geometry_ranges": copy.deepcopy(TRAINING_GEOMETRY),
                    }
                    config = config_for_spec(environment_config, protocol, spec, max_steps=None)
                    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=1, target_speed_scale=speed)
                    scenario = s4_adaptive_branching_scenario(
                        env,
                        layout_seed=layout_seed,
                        defender_bias=defender_bias,
                        variation=OOD_GEOMETRY,
                    )
                    spec["scenario"] = scenario_metadata(scenario)
                    group_records.append(spec)
            except ValueError:
                continue
            accepted = group_records
            break
        if accepted is None:
            raise RuntimeError(f"Unable to generate valid OOD mirror group {group_index}.")
        records.extend(accepted)
        if (group_index + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "status": "generating",
                        "mirror_groups_completed": group_index + 1,
                        "mirror_groups_requested": episodes // 2,
                        "candidate_layouts_examined": candidate_offset,
                    }
                ),
                flush=True,
            )
    return records


def validate_records(records: list[dict[str, Any]]) -> None:
    if len(records) % 2:
        raise ValueError("records must have paired cardinality.")
    groups: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(int(record["mirror_group_id"]), []).append(record)
        obstacle = record["scenario"]["obstacles"][0]
        half = obstacle["half_extents_xy"]
        values = {
            "wall_half_extent_x_m": float(half[0]),
            "wall_half_extent_y_m": float(half[1]),
            "wall_height_m": float(obstacle["height"]),
        }
        for name, value in values.items():
            low, high = OOD_GEOMETRY[name]
            train_low, train_high = TRAINING_GEOMETRY[name]
            if not low <= value <= high or train_low <= value <= train_high:
                raise ValueError(f"OOD geometry contract failed for {name}: {value}")
    for group, members in groups.items():
        if len(members) != 2 or {item["defender_bias"] for item in members} != {"upper", "lower"}:
            raise ValueError(f"mirror group {group} is incomplete")
        first, second = members
        if first["layout_seed"] != second["layout_seed"] or first["scenario"]["obstacles"] != second["scenario"]["obstacles"]:
            raise ValueError(f"mirror group {group} does not share frozen geometry")


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
        "name": "phase16_ood_geometry_shift",
        "evaluation_split": "ood_diagnostic",
        "episodes": len(records),
        "mirror_groups": len(records) // 2,
        "seed_start": int(args.seed_start),
        "training_geometry_ranges": TRAINING_GEOMETRY,
        "ood_geometry_ranges": OOD_GEOMETRY,
        "scene_manifest_sha256": hashlib.sha256(scenes_path.read_bytes()).hexdigest(),
        "protocol": str(args.protocol.resolve()),
        "environment_config": str(args.environment_config.resolve()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
