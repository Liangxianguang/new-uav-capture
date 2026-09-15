"""Generate reproducible Easy/Nominal/Hard calibration scene pools.

The three pools use the Phase 73 opposite-side crossing geometry contract,
but stratify target behavior, sensing, execution, initial distance, and
obstacle count.  They are development-calibration data only; the existing
locked-test manifest is never read or modified.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import scenario_metadata  # noqa: E402
from generate_phase73_crossing_scene_dataset import (  # noqa: E402
    sample_crossing_pair,
    validate_showcase_scenario,
)


PROFILES = ("easy", "nominal", "hard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs" / "phase77_calibrated_difficulty.yaml")
    parser.add_argument("--environment-config", type=Path, default=PROJECT_ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml")
    parser.add_argument("--source-protocol", type=Path, default=PROJECT_ROOT / "configs" / "phase73_crossing_scene_dataset.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-difficulty", type=int, default=None)
    parser.add_argument("--allow-small", action="store_true", help="Allow a small smoke dataset for tests.")
    return parser.parse_args()


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile_settings(source: dict[str, Any], profile: dict[str, Any]) -> dict[str, Any]:
    settings = copy.deepcopy(source)
    settings["obstacle_count_range"] = list(profile["obstacle_count_range"])
    settings["initial_side_distances"] = list(profile["initial_side_distances"])
    settings["target_speed_scales"] = list(profile["target_speed_scales"])
    settings["observation_conditions"] = [
        {
            "name": str(profile["observation_condition"]),
            "pursuit_overrides": copy.deepcopy(profile["pursuit_overrides"]),
        }
    ]
    settings["block_order"] = ["calibration"]
    settings["total_scenes"] = 2
    settings["seed_blocks"] = {"calibration": 1}
    return settings


def _episode_record(
    index: int,
    profile_name: str,
    profile: dict[str, Any],
    scenario: Any,
    certificate: dict[str, Any],
    episode_seed: int,
    layout_seed: int,
    pair_index: int,
    target_speed_scale: float,
    member: str,
) -> dict[str, Any]:
    pursuit_overrides = copy.deepcopy(profile["pursuit_overrides"])
    spec = {
        "episode_index": int(index),
        "episode_seed": int(episode_seed),
        "layout_seed": int(layout_seed),
        "defender_side": str(scenario.defender_side),
        "initial_side_distance": float(abs(scenario.target_position[0])),
        "target_speed_scale": float(target_speed_scale),
        "target_motion_mode": str(profile["target_motion_mode"]),
        "target_crossing_required": True,
        "observation_condition": str(profile["observation_condition"]),
        "pursuit_overrides": pursuit_overrides,
        "obstacle_count": int(len(scenario.obstacles)),
        "difficulty": str(profile_name),
        "scene_block": "development_calibration",
        "execution": copy.deepcopy(profile["execution"]),
        "mirror_group_id": f"phase77-{profile_name}-{pair_index:04d}",
        "mirror_pair_member": str(member),
    }
    record_certificate = copy.deepcopy(certificate)
    record_certificate["mirror_pair_member"] = str(member)
    record_certificate["layout_seed"] = int(layout_seed)
    return {
        "dataset_index": int(index),
        **spec,
        "defender_bias": "upper" if member == "left" else "lower",
        "spec": copy.deepcopy(spec),
        "scenario": scenario_metadata(scenario),
        "route_certificate": record_certificate,
    }


def generate(
    protocol_path: Path,
    environment_path: Path,
    source_protocol_path: Path,
    output_dir: Path,
    episodes_per_difficulty: int | None = None,
    *,
    allow_small: bool = False,
) -> dict[str, Any]:
    protocol = _load_mapping(protocol_path)
    source = _load_mapping(source_protocol_path)
    environment_payload = _load_mapping(environment_path)
    profiles = protocol.get("difficulty_profiles")
    if not isinstance(profiles, dict) or any(name not in profiles for name in PROFILES):
        raise ValueError("phase77 protocol must define easy, nominal, and hard difficulty_profiles")
    requested = int(protocol.get("scenes_per_difficulty", 100) if episodes_per_difficulty is None else episodes_per_difficulty)
    minimum = 2 if allow_small else 100
    if requested < minimum or requested % 2 != 0:
        raise ValueError(f"episodes-per-difficulty must be even and >= {minimum}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir.resolve()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    block_dir = output_dir / "blocks"
    block_dir.mkdir()

    records: list[dict[str, Any]] = []
    attempts_by_profile: dict[str, int] = {}
    profile_counts: dict[str, int] = {}
    for profile_name in PROFILES:
        profile = profiles[profile_name]
        if not isinstance(profile, dict):
            raise ValueError(f"difficulty profile {profile_name} must be a mapping")
        settings = _profile_settings(source, profile)
        block_seed = int(protocol["seed_blocks"][profile_name])
        assignment_rng = np.random.default_rng(block_seed + 3_000_000)
        profile_records: list[dict[str, Any]] = []
        attempts_total = 0
        for pair_index in range(requested // 2):
            distances = np.asarray(profile["initial_side_distances"], dtype=np.float64)
            speeds = np.asarray(profile["target_speed_scales"], dtype=np.float64)
            initial_side_distance = float(assignment_rng.choice(distances))
            target_speed_scale = float(assignment_rng.choice(speeds))
            layout_seed = block_seed + 1_000_000 + pair_index
            mirror_group_id = f"phase77-{profile_name}-{pair_index:04d}"
            base_env = CaptureRadiusPursuit3DEnv(
                copy.deepcopy(environment_payload),
                obstacle_count=0,
                target_speed_scale=target_speed_scale,
            )
            left, right, certificate, attempts = sample_crossing_pair(
                base_env,
                layout_seed,
                initial_side_distance,
                target_speed_scale,
                settings,
                mirror_group_id,
            )
            if pair_index == 0:
                validate_showcase_scenario(base_env, left)
                validate_showcase_scenario(base_env, right)
            attempts_total += int(attempts)
            for member, scenario in (("left", left), ("right", right)):
                episode_seed = block_seed + pair_index * 2 + (0 if member == "left" else 1)
                profile_records.append(
                    _episode_record(
                        len(records) + len(profile_records),
                        profile_name,
                        profile,
                        scenario,
                        certificate,
                        episode_seed,
                        layout_seed,
                        pair_index,
                        target_speed_scale,
                        member,
                    )
                )
        records.extend(profile_records)
        profile_counts[profile_name] = len(profile_records)
        attempts_by_profile[profile_name] = attempts_total
        (block_dir / f"{profile_name}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in profile_records),
            encoding="utf-8",
        )
        print(f"generated {profile_name}: {len(profile_records)} scenes / {len(profile_records) // 2} mirror groups", flush=True)

    scene_path = output_dir / "scenes.jsonl"
    scene_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    mirror_groups = Counter(str(record["mirror_group_id"]) for record in records)
    manifest = {
        "dataset_name": str(protocol["dataset_name"]),
        "dataset_version": "phase77.v1",
        "phase": "development_calibration_only",
        "not_a_locked_test": True,
        "locked_test": False,
        "locked_test_modified": False,
        "total_scenes": len(records),
        "scenes_per_difficulty": requested,
        "mirror_groups": len(mirror_groups),
        "all_mirror_groups_have_two_members": bool(all(count == 2 for count in mirror_groups.values())),
        "difficulty_counts": profile_counts,
        "difficulty_mirror_group_counts": {
            name: requested // 2 for name in PROFILES
        },
        "target_crossing_required_rate": float(np.mean([bool(item["target_crossing_required"]) for item in records])),
        "direct_path_blocked_rate": float(np.mean([bool(item["route_certificate"]["direct_path_blocked"]) for item in records])),
        "minimum_bypass_route_count": int(min(int(item["route_certificate"]["bypass_route_count"]) for item in records)),
        "difficulty_profiles": copy.deepcopy(profiles),
        "sampling_attempts_by_profile": attempts_by_profile,
        "scene_file": str(scene_path.resolve()),
        "scene_file_sha256": _sha256(scene_path),
        "protocol": str(protocol_path.resolve()),
        "source_protocol": str(source_protocol_path.resolve()),
        "environment_config": str(environment_path.resolve()),
        "contract": {
            "same_scene_manifest_required_for_all_models": True,
            "locked_test_tuning_forbidden": True,
            "raw_pool_retained_after_expert_filter": True,
            "expert_filter_is_calibration_only": True,
            "route_certificate_is_not_a_closed_loop_safety_proof": True,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Phase 77 calibrated difficulty scene pool\n\n"
        "This development-calibration pool contains equal Easy/Nominal/Hard blocks, "
        "each with horizontal mirror pairs and the Phase 73 opposite-side crossing contract. "
        "The raw pool is retained even after expert filtering; filtered records are only for "
        "calibration/training and must not replace a fixed validation pool. The existing "
        "locked-test manifest is not read or modified.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    manifest = generate(
        args.protocol,
        args.environment_config,
        args.source_protocol,
        args.output_dir,
        args.episodes_per_difficulty,
        allow_small=bool(args.allow_small),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
