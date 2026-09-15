"""Generate Phase 78 obstacle-avoidance-first calibration scenes.

The target starts opposite the defenders, chooses an outward initial escape
direction, and is not required to cross the central obstacle field.  The
obstacles remain between the defenders and target so defender transit still
requires obstacle-aware interception.  This is a fresh calibration pool and
never reads or modifies locked-test data.
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

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv, CylinderObstacle  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    ShowcaseScenario,
    _obstacle_x_extent,
    _opposite_side_positions,
    _random_central_obstacle,
    validate_showcase_scenario,
    scenario_metadata,
)


PROFILES = ("easy", "nominal", "hard")


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _mirror_obstacle(obstacle: CylinderObstacle) -> CylinderObstacle:
    center = np.asarray(obstacle.center_xy, dtype=np.float64).copy()
    center[0] *= -1.0
    return CylinderObstacle(
        center_xy=center,
        radius=float(obstacle.radius),
        height=float(obstacle.height),
        shape=str(obstacle.shape),
        half_extents_xy=(
            None
            if obstacle.half_extents_xy is None
            else np.asarray(obstacle.half_extents_xy, dtype=np.float64).copy()
        ),
    )


def _mirror_scenario(scenario: ShowcaseScenario, name: str) -> ShowcaseScenario:
    defenders = np.asarray(scenario.defender_positions, dtype=np.float64).copy()
    defenders[:, 0] *= -1.0
    target = np.asarray(scenario.target_position, dtype=np.float64).copy()
    target[0] *= -1.0
    escape = np.asarray(scenario.target_escape_direction, dtype=np.float64).copy()
    escape[0] *= -1.0
    return ShowcaseScenario(
        name=name,
        obstacles=tuple(_mirror_obstacle(item) for item in scenario.obstacles),
        defender_positions=defenders,
        target_position=target,
        target_escape_direction=escape,
        obstacle_zone_x=tuple(float(value) for value in scenario.obstacle_zone_x),
        target_crossing_required=False,
        defender_side="right",
        layout_seed=scenario.layout_seed,
        required_defender_zone_entries=int(scenario.required_defender_zone_entries),
        require_target_zone_entry=False,
        scenario_type="showcase",
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    return value / max(float(np.linalg.norm(value)), 1.0e-9)


def _initial_escape_certificate(
    env: CaptureRadiusPursuit3DEnv,
    scenario: ShowcaseScenario,
    target_speed_scale: float,
    horizon_steps: int,
    clearance_limit: float,
) -> dict[str, Any]:
    direction = _unit(scenario.target_escape_direction)
    speed = float(env.agents["target_max_speed"]) * float(target_speed_scale)
    points = [
        np.asarray(scenario.target_position, dtype=np.float64) + direction * speed * env.dt * step
        for step in range(horizon_steps + 1)
    ]
    clearances = [
        float(env._obstacle_clearance(point, obstacle))
        for point in points
        for obstacle in scenario.obstacles
    ]
    minimum_clearance = float(min(clearances)) if clearances else float("inf")
    boundary_margins = [
        float(min(np.min(point - env.lower), np.min(env.upper - point))) for point in points
    ]
    minimum_boundary = float(min(boundary_margins))
    target_xy = np.asarray(scenario.target_position[:2], dtype=np.float64)
    nearest = min(
        scenario.obstacles,
        key=lambda obstacle: float(np.linalg.norm(np.asarray(obstacle.center_xy) - target_xy)),
    )
    obstacle_away = np.array(
        [target_xy[0] - nearest.center_xy[0], target_xy[1] - nearest.center_xy[1], 0.0],
        dtype=np.float64,
    )
    alignment = float(np.dot(direction, _unit(obstacle_away)))
    return {
        "certificate_version": "phase78.obstacle_avoidance_first.v1",
        "target_crossing_required": False,
        "initial_escape_direction": direction.tolist(),
        "initial_escape_lookahead_steps": int(horizon_steps),
        "initial_escape_minimum_clearance_m": minimum_clearance,
        "initial_escape_minimum_boundary_margin_m": minimum_boundary,
        "initial_escape_clearance_limit_m": float(clearance_limit),
        "initial_escape_away_alignment": alignment,
        "initial_escape_clearance_certified": bool(
            minimum_clearance >= clearance_limit and minimum_boundary >= clearance_limit
        ),
        "obstacle_count": len(scenario.obstacles),
        "obstacle_x_extents": [list(map(float, _obstacle_x_extent(item))) for item in scenario.obstacles],
    }


def _segment_clearance(
    env: CaptureRadiusPursuit3DEnv,
    first: np.ndarray,
    second: np.ndarray,
    obstacles: list[CylinderObstacle] | tuple[CylinderObstacle, ...],
) -> float:
    distance = float(np.linalg.norm(second - first))
    samples = max(int(np.ceil(distance / 0.25)), 1)
    values = [
        float(env._obstacle_clearance(first + fraction * (second - first), obstacle))
        for fraction in np.linspace(0.0, 1.0, samples + 1)
        for obstacle in obstacles
    ]
    return float(min(values)) if values else float("inf")


def _quick_transit_certificate(
    env: CaptureRadiusPursuit3DEnv,
    scenario: ShowcaseScenario,
    clearance_limit: float,
) -> bool:
    """Check a fixed outer corridor before the periodic strict grid audit."""

    corridor_candidates = (8.20, -8.20)
    defender_goals = scenario.defender_positions.copy()
    defender_goals[:, 0] = float(scenario.target_position[0])
    target_goal = scenario.target_position.copy()
    target_goal[0] = float(np.mean(scenario.defender_positions[:, 0]))
    for corridor_y in corridor_candidates:
        all_routes_valid = True
        for start, goal in [*zip(scenario.defender_positions, defender_goals, strict=True), (scenario.target_position, target_goal)]:
            via_start = np.array([start[0], corridor_y, start[2]], dtype=np.float64)
            via_goal = np.array([goal[0], corridor_y, goal[2]], dtype=np.float64)
            route = (start, via_start, via_goal, goal)
            for first, second in zip(route[:-1], route[1:], strict=True):
                if _segment_clearance(env, first, second, scenario.obstacles) < clearance_limit:
                    all_routes_valid = False
                    break
            if not all_routes_valid:
                break
        if all_routes_valid:
            return True
    return False


def _sample_obstacle_avoidance_pair(
    env: CaptureRadiusPursuit3DEnv,
    layout_seed: int,
    initial_side_distance: float,
    obstacle_count_range: tuple[int, int],
    max_attempts: int,
    clearance_limit: float,
    strict_audit: bool,
) -> tuple[ShowcaseScenario, ShowcaseScenario, int]:
    zone = (-2.5, 3.0)
    count_low, count_high = obstacle_count_range
    defenders, target, escape = _opposite_side_positions(initial_side_distance, "left")
    protected = np.vstack([defenders, target[None, :]])
    rng = np.random.default_rng(int(layout_seed))
    for attempt in range(1, max_attempts + 1):
        count = int(rng.integers(count_low, count_high + 1))
        shapes = ["cylinder", "box", "wall"]
        shapes.extend(str(rng.choice(["cylinder", "box", "wall"])) for _ in range(count - 3))
        rng.shuffle(shapes)
        obstacles: list[CylinderObstacle] = []
        for shape in shapes:
            for _ in range(100):
                candidate = _random_central_obstacle(rng, shape, zone)
                if not env._obstacle_clear_of_points(candidate, protected):
                    continue
                if any(env._obstacle_horizontal_separation(candidate, current) < 0.70 for current in obstacles):
                    continue
                obstacles.append(candidate)
                break
            else:
                break
        if len(obstacles) != count:
            continue
        left = ShowcaseScenario(
            name=f"phase78_avoidance_{layout_seed}_left",
            obstacles=tuple(obstacles),
            defender_positions=defenders.copy(),
            target_position=target.copy(),
            target_escape_direction=escape.copy(),
            obstacle_zone_x=zone,
            target_crossing_required=False,
            defender_side="left",
            layout_seed=int(layout_seed),
            required_defender_zone_entries=1,
            require_target_zone_entry=False,
            scenario_type="showcase",
        )
        right = _mirror_scenario(left, f"phase78_avoidance_{layout_seed}_right")
        if not _quick_transit_certificate(env, left, clearance_limit):
            continue
        if not _quick_transit_certificate(env, right, clearance_limit):
            continue
        if not _initial_escape_certificate(env, left, 1.0, 4, clearance_limit)["initial_escape_clearance_certified"]:
            continue
        if not _initial_escape_certificate(env, right, 1.0, 4, clearance_limit)["initial_escape_clearance_certified"]:
            continue
        if strict_audit:
            try:
                validate_showcase_scenario(env, left)
                validate_showcase_scenario(env, right)
            except ValueError:
                continue
        return left, right, attempt
    raise RuntimeError(f"unable to sample Phase 78 avoidance pair after {max_attempts} attempts (seed={layout_seed})")


def _record(
    index: int,
    profile_name: str,
    profile: dict[str, Any],
    scenario: ShowcaseScenario,
    certificate: dict[str, Any],
    episode_seed: int,
    layout_seed: int,
    pair_index: int,
    target_speed_scale: float,
    member: str,
) -> dict[str, Any]:
    spec = {
        "episode_index": int(index),
        "episode_seed": int(episode_seed),
        "layout_seed": int(layout_seed),
        "defender_side": str(scenario.defender_side),
        "initial_side_distance": float(abs(scenario.target_position[0])),
        "target_speed_scale": float(target_speed_scale),
        "target_motion_mode": str(profile["target_motion_mode"]),
        "target_crossing_required": False,
        "observation_condition": str(profile["observation_condition"]),
        "pursuit_overrides": copy.deepcopy(profile["pursuit_overrides"]),
        "obstacle_count": int(len(scenario.obstacles)),
        "difficulty": str(profile_name),
        "scene_block": "development_calibration",
        "execution": copy.deepcopy(profile["execution"]),
        "mirror_group_id": f"phase78-{profile_name}-{pair_index:04d}",
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
        "avoidance_certificate": record_certificate,
    }


def generate(
    protocol_path: Path,
    environment_path: Path,
    output_dir: Path,
    episodes_per_difficulty: int | None = None,
    *,
    allow_small: bool = False,
) -> dict[str, Any]:
    protocol = _load(protocol_path)
    environment_payload = _load(environment_path)
    profiles = protocol.get("difficulty_profiles")
    if not isinstance(profiles, dict) or any(name not in profiles for name in PROFILES):
        raise ValueError("Phase 78 must define easy, nominal, and hard profiles")
    requested = int(protocol.get("scenes_per_difficulty", 100) if episodes_per_difficulty is None else episodes_per_difficulty)
    minimum = 2 if allow_small else 100
    if requested < minimum or requested % 2 != 0:
        raise ValueError(f"episodes-per-difficulty must be even and >= {minimum}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir.resolve()}")
    output_dir.mkdir(parents=True, exist_ok=True)
    blocks_dir = output_dir / "blocks"
    blocks_dir.mkdir()

    records: list[dict[str, Any]] = []
    attempts_by_profile: dict[str, int] = {}
    profile_counts: dict[str, int] = {}
    for profile_name in PROFILES:
        profile = profiles[profile_name]
        settings = protocol.get("common", {})
        profile_records: list[dict[str, Any]] = []
        block_seed = int(protocol["seed_blocks"][profile_name])
        assignment_rng = np.random.default_rng(block_seed + 3_000_000)
        attempts_total = 0
        distances = np.asarray(profile["initial_side_distances"], dtype=np.float64)
        speeds = np.asarray(profile["target_speed_scales"], dtype=np.float64)
        count_range = tuple(int(value) for value in profile["obstacle_count_range"])
        for pair_index in range(requested // 2):
            initial_distance = float(assignment_rng.choice(distances))
            target_speed = float(assignment_rng.choice(speeds))
            layout_seed = block_seed + 1_000_000 + pair_index
            base_env = CaptureRadiusPursuit3DEnv(copy.deepcopy(environment_payload), obstacle_count=0, target_speed_scale=target_speed)
            left, right, attempts = _sample_obstacle_avoidance_pair(
                base_env,
                layout_seed,
                initial_distance,
                count_range,
                int(settings.get("max_sampling_attempts_per_group", 500)),
                float(settings.get("initial_escape_clearance_limit_m", 0.60)),
                # Hard uses the per-scene fixed outer-corridor certificate;
                # its first pair is intentionally not sent through the
                # expensive all-agent grid search because dense 4--5-obstacle
                # maps can make that search needlessly retry many layouts.
                strict_audit=pair_index == 0 and profile_name != "hard",
            )
            attempts_total += int(attempts)
            horizon = int(settings.get("initial_escape_lookahead_steps", 8))
            limit = float(settings.get("initial_escape_clearance_limit_m", 0.60))
            left_certificate = _initial_escape_certificate(base_env, left, target_speed, horizon, limit)
            right_certificate = _initial_escape_certificate(base_env, right, target_speed, horizon, limit)
            if not left_certificate["initial_escape_clearance_certified"] or not right_certificate["initial_escape_clearance_certified"]:
                raise RuntimeError(f"sampled Phase 78 pair is not outward-feasible: {layout_seed}")
            group_id = f"phase78-{profile_name}-{pair_index:04d}"
            for member, scenario, certificate in (("left", left, left_certificate), ("right", right, right_certificate)):
                profile_records.append(
                    _record(
                        len(records) + len(profile_records),
                        profile_name,
                        profile,
                        scenario,
                        certificate,
                        block_seed + pair_index * 2 + (0 if member == "left" else 1),
                        layout_seed,
                        pair_index,
                        target_speed,
                        member,
                    )
                )
                profile_records[-1]["mirror_group_id"] = group_id
                profile_records[-1]["spec"]["mirror_group_id"] = group_id
        records.extend(profile_records)
        profile_counts[profile_name] = len(profile_records)
        attempts_by_profile[profile_name] = attempts_total
        (blocks_dir / f"{profile_name}.jsonl").write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in profile_records),
            encoding="utf-8",
        )
        print(f"generated {profile_name}: {len(profile_records)} scenes / {len(profile_records)//2} mirror groups", flush=True)

    scene_path = output_dir / "scenes.jsonl"
    scene_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")
    mirror_groups = Counter(str(item["mirror_group_id"]) for item in records)
    manifest = {
        "dataset_name": str(protocol["dataset_name"]),
        "dataset_version": "phase78.v1",
        "phase": "development_calibration_only",
        "not_a_locked_test": True,
        "locked_test": False,
        "locked_test_modified": False,
        "total_scenes": len(records),
        "scenes_per_difficulty": requested,
        "mirror_groups": len(mirror_groups),
        "all_mirror_groups_have_two_members": bool(all(count == 2 for count in mirror_groups.values())),
        "difficulty_counts": profile_counts,
        "difficulty_mirror_group_counts": {name: requested // 2 for name in PROFILES},
        "target_crossing_required_rate": 0.0,
        "initial_escape_clearance_certified_rate": float(np.mean([item["avoidance_certificate"]["initial_escape_clearance_certified"] for item in records])),
        "initial_escape_away_alignment_mean": float(np.mean([item["avoidance_certificate"]["initial_escape_away_alignment"] for item in records])),
        "sampling_attempts_by_profile": attempts_by_profile,
        "scene_file": str(scene_path.resolve()),
        "scene_file_sha256": _sha256(scene_path),
        "protocol": str(protocol_path.resolve()),
        "environment_config": str(environment_path.resolve()),
        "contract": {
            "obstacle_avoidance_first": True,
            "target_crossing_required": False,
            "intentional_reverse_lane_change": False,
            "raw_pool_retained_after_expert_filter": True,
            "expert_filter_is_calibration_only": True,
            "route_certificate_is_not_a_closed_loop_safety_proof": True,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Phase 78 obstacle-avoidance-first calibration pool\n\n"
        "The target starts opposite the defenders and initially moves away from the central obstacle field. "
        "It is not required to cross the obstacle zone or perform an intentional reverse lane change. "
        "Smooth finite-horizon jinks and speed bursts keep motion non-trivial. This is calibration-only; "
        "locked-test data is not read or modified.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs" / "phase78_obstacle_avoidance_first.yaml")
    parser.add_argument("--environment-config", type=Path, default=PROJECT_ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes-per-difficulty", type=int, default=None)
    parser.add_argument("--allow-small", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate(args.protocol, args.environment_config, args.output_dir, args.episodes_per_difficulty, allow_small=args.allow_small), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
