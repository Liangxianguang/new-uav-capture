"""Generate single-factor Nominal-plus-1 calibration scene blocks.

This generator keeps the Phase 78 obstacle-avoidance-first target contract and
creates independent, calibration-only blocks.  Each block changes exactly one
factor relative to the Phase 81 transition profile: command noise, execution
delay, maneuver replanning frequency, target/obstacle proximity, or initial
defender formation spacing.  The raw pool is retained even when a later expert
screen rejects a scene; no locked-test data is read or modified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    ShowcaseScenario,
    _obstacle_x_extent,
    scenario_metadata,
)
from generate_phase78_obstacle_avoidance_scenes import (  # noqa: E402
    _initial_escape_certificate,
    _sample_obstacle_avoidance_pair,
)


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _minimum_formation_spacing(scenario: ShowcaseScenario) -> float:
    positions = np.asarray(scenario.defender_positions, dtype=np.float64)
    pairwise = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    pairwise += np.eye(len(positions), dtype=np.float64) * 1.0e6
    return float(np.min(pairwise))


def _target_obstacle_clearance(
    env: CaptureRadiusPursuit3DEnv,
    scenario: ShowcaseScenario,
) -> float:
    if not scenario.obstacles:
        return float("inf")
    return float(
        min(env._obstacle_clearance(scenario.target_position, obstacle) for obstacle in scenario.obstacles)
    )


def _record(
    index: int,
    variant_name: str,
    variant: dict[str, Any],
    env: CaptureRadiusPursuit3DEnv,
    scenario: ShowcaseScenario,
    certificate: dict[str, Any],
    episode_seed: int,
    layout_seed: int,
    pair_index: int,
    target_speed_scale: float,
) -> dict[str, Any]:
    member = str(scenario.defender_side)
    member_name = "left" if member == "left" else "right"
    spec = {
        "episode_index": int(index),
        "episode_seed": int(episode_seed),
        "layout_seed": int(layout_seed),
        "defender_side": member,
        "initial_side_distance": float(abs(scenario.target_position[0])),
        "target_speed_scale": float(target_speed_scale),
        "target_motion_mode": str(variant["target_motion_mode"]),
        "target_crossing_required": False,
        "observation_condition": str(variant["observation_condition"]),
        "pursuit_overrides": copy.deepcopy(variant["pursuit_overrides"]),
        "obstacle_count": int(len(scenario.obstacles)),
        "difficulty": str(variant.get("difficulty_label", "nominal_plus1")),
        "variant": str(variant_name),
        "single_factor": str(variant["single_factor"]),
        "single_factor_value": copy.deepcopy(variant.get("single_factor_value")),
        "scene_block": str(variant.get("scene_block", "phase82_calibration")),
        "execution": copy.deepcopy(variant["execution"]),
        "mirror_group_id": (
            f"{variant.get('mirror_group_prefix', 'phase82')}-"
            f"{variant_name}-{pair_index:04d}"
        ),
        "mirror_pair_member": member_name,
        "formation_spacing_min_m": _minimum_formation_spacing(scenario),
        "target_initial_obstacle_clearance_m": _target_obstacle_clearance(env, scenario),
        "defender_y_scale": float(variant.get("defender_y_scale", 1.0)),
    }
    certificate = copy.deepcopy(certificate)
    certificate["mirror_pair_member"] = member_name
    certificate["layout_seed"] = int(layout_seed)
    certificate["single_factor"] = str(variant["single_factor"])
    return {
        "dataset_index": int(index),
        **spec,
        "spec": copy.deepcopy(spec),
        "scenario": scenario_metadata(scenario),
        "avoidance_certificate": certificate,
    }


def _variant_profile(protocol: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    base = protocol.get("base_profile")
    if not isinstance(base, dict):
        raise ValueError("protocol must define base_profile")
    merged = _deep_merge(base, variant)
    required = {
        "target_motion_mode",
        "target_speed_scales",
        "initial_side_distances",
        "obstacle_count_range",
        "observation_condition",
        "pursuit_overrides",
        "execution",
        "single_factor",
    }
    missing = sorted(required.difference(merged))
    if missing:
        raise ValueError(f"variant is missing fields: {', '.join(missing)}")
    return merged


def generate(
    protocol_path: Path,
    environment_path: Path,
    output_dir: Path,
    episodes_per_variant: int | None = None,
    *,
    allow_small: bool = False,
    seed_offset: int = 0,
) -> dict[str, Any]:
    protocol = _load(protocol_path)
    environment_payload = _load(environment_path)
    variants = protocol.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ValueError("protocol must define a non-empty variants mapping")
    requested = int(
        protocol.get("episodes_per_variant", 100)
        if episodes_per_variant is None
        else episodes_per_variant
    )
    minimum = 2 if allow_small else 100
    if requested < minimum or requested % 2 != 0:
        raise ValueError(f"episodes-per-variant must be even and >= {minimum}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir.resolve()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks_dir = output_dir / "blocks"
    blocks_dir.mkdir()

    seed_blocks = protocol.get("seed_blocks", {})
    if not isinstance(seed_blocks, dict):
        raise ValueError("seed_blocks must be a mapping")
    records: list[dict[str, Any]] = []
    variant_counts: dict[str, int] = {}
    attempts_by_variant: dict[str, int] = {}
    variant_manifests: dict[str, dict[str, Any]] = {}
    max_attempts = int(protocol.get("common", {}).get("max_sampling_attempts_per_group", 500))
    horizon = int(protocol.get("common", {}).get("initial_escape_lookahead_steps", 4))
    clearance_limit = float(protocol.get("common", {}).get("initial_escape_clearance_limit_m", 0.60))

    for variant_index, (variant_name, raw_variant) in enumerate(variants.items()):
        if not isinstance(raw_variant, dict):
            raise ValueError(f"variant {variant_name} must be a mapping")
        variant = _variant_profile(protocol, raw_variant)
        block_seed = int(seed_blocks.get(variant_name, 82_1001 + variant_index * 1000)) + int(seed_offset)
        assignment_rng = np.random.default_rng(block_seed + 3_000_000)
        distances = np.asarray(variant["initial_side_distances"], dtype=np.float64)
        speeds = np.asarray(variant["target_speed_scales"], dtype=np.float64)
        count_range = tuple(int(value) for value in variant["obstacle_count_range"])
        y_scale = float(variant.get("defender_y_scale", 1.0))
        if y_scale <= 0.0:
            raise ValueError(f"variant {variant_name} defender_y_scale must be positive")
        profile_records: list[dict[str, Any]] = []
        attempts_total = 0
        for pair_index in range(requested // 2):
            initial_distance = float(assignment_rng.choice(distances))
            target_speed = float(assignment_rng.choice(speeds))
            layout_seed = block_seed + 1_000_000 + pair_index
            base_env = CaptureRadiusPursuit3DEnv(
                copy.deepcopy(environment_payload),
                obstacle_count=0,
                target_speed_scale=target_speed,
            )
            left, right, attempts = _sample_obstacle_avoidance_pair(
                base_env,
                layout_seed,
                initial_distance,
                count_range,
                max_attempts,
                clearance_limit,
                # Keep one full route audit per single-factor block while
                # using the same fast transit certificate for every pair.
                # This matches the Phase78 calibration contract and keeps a
                # 500-scene pool reproducible without an avoidable quadratic
                # validation cost.
                strict_audit=pair_index == 0,
                defender_y_scale=y_scale,
                target_obstacle_clearance_max=(
                    float(variant["target_obstacle_clearance_max_m"])
                    if variant.get("target_obstacle_clearance_max_m") is not None
                    else None
                ),
            )
            attempts_total += int(attempts)
            left_certificate = _initial_escape_certificate(
                base_env, left, target_speed, horizon, clearance_limit
            )
            right_certificate = _initial_escape_certificate(
                base_env, right, target_speed, horizon, clearance_limit
            )
            for scenario, certificate, side_offset in (
                (left, left_certificate, 0),
                (right, right_certificate, 1),
            ):
                if not certificate["initial_escape_clearance_certified"]:
                    raise RuntimeError(f"sampled pair is not outward-feasible: {layout_seed}")
                episode_seed = block_seed + pair_index * 2 + side_offset
                profile_records.append(
                    _record(
                        len(records) + len(profile_records),
                        str(variant_name),
                        variant,
                        base_env,
                        scenario,
                        certificate,
                        episode_seed,
                        layout_seed,
                        pair_index,
                        target_speed,
                    )
                )
        records.extend(profile_records)
        variant_counts[str(variant_name)] = len(profile_records)
        attempts_by_variant[str(variant_name)] = attempts_total
        block_path = blocks_dir / f"{variant_name}.jsonl"
        block_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in profile_records),
            encoding="utf-8",
        )
        variant_manifests[str(variant_name)] = {
            "single_factor": str(variant["single_factor"]),
            "single_factor_value": copy.deepcopy(variant.get("single_factor_value")),
            "scene_count": len(profile_records),
            "mirror_groups": len(profile_records) // 2,
            "seed": block_seed,
            "block_file": str(block_path.resolve()),
        }
        print(
            f"generated {variant_name}: {len(profile_records)} scenes / "
            f"{len(profile_records) // 2} mirror groups",
            flush=True,
        )

    scene_path = output_dir / "scenes.jsonl"
    scene_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    mirror_groups = Counter(str(item["mirror_group_id"]) for item in records)
    manifest = {
        "dataset_name": str(protocol["dataset_name"]),
        "dataset_version": str(protocol.get("dataset_version", "phase82.nominal_plus1.v1")),
        "phase": "development_calibration_only",
        "not_a_locked_test": True,
        "locked_test": False,
        "locked_test_modified": False,
        "total_scenes": len(records),
        "episodes_per_variant": requested,
        "variant_count": len(variants),
        "variant_counts": variant_counts,
        "variant_manifests": variant_manifests,
        "mirror_groups": len(mirror_groups),
        "all_mirror_groups_have_two_members": bool(all(count == 2 for count in mirror_groups.values())),
        "initial_escape_clearance_certified_rate": float(
            np.mean(
                [
                    item["avoidance_certificate"]["initial_escape_clearance_certified"]
                    for item in records
                ]
            )
        ),
        "sampling_attempts_by_variant": attempts_by_variant,
        "seed_offset": int(seed_offset),
        "scene_file": str(scene_path.resolve()),
        "scene_file_sha256": _sha256(scene_path),
        "protocol": str(protocol_path.resolve()),
        "environment_config": str(environment_path.resolve()),
        "contract": {
            "obstacle_avoidance_first": True,
            "target_crossing_required": False,
            "intentional_reverse_lane_change": False,
            "one_factor_per_variant": True,
            "expert_filter_is_calibration_only": True,
            "route_certificate_is_not_a_closed_loop_safety_proof": True,
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme_title = str(
        protocol.get("readme_title", "Phase 82 Nominal-plus-1 calibration pool")
    )
    (output_dir / "README.md").write_text(
        f"# {readme_title}\n\n"
        "Each variant changes one factor relative to its declared base profile. "
        "The target initially moves away from the nearest obstacle, does not require an "
        "intentional lane change or central crossing, and all generated scenes are "
        "calibration-only.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "configs" / "phase82_nominal_plus1.yaml",
    )
    parser.add_argument(
        "--environment-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-variant", type=int, default=None)
    parser.add_argument("--allow-small", action="store_true")
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    print(
        json.dumps(
            generate(
                args.protocol,
                args.environment_config,
                args.output_dir,
                args.episodes_per_variant,
                allow_small=args.allow_small,
                seed_offset=args.seed_offset,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
