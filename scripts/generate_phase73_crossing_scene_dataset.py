"""Generate a large fixed scene pool for genuine obstacle-bypass evaluation.

The pool is intentionally separate from the benchmark's existing locked-test
manifest.  Every accepted scene has defenders and target on opposite sides of
the central obstacle zone, a blocked straight target route, two certified
lateral bypass routes, and ``target_crossing_required=True``.  Horizontal
mirror pairs share exactly the same obstacle layout so that paired comparisons
can separate controller effects from map effects.
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

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv, CylinderObstacle  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    ShowcaseScenario,
    _obstacle_x_extent,
    _opposite_side_positions,
    _random_central_obstacle,
    scenario_metadata,
    validate_showcase_scenario,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True, help="Phase 73 dataset contract YAML.")
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=None, help="Optional even override; default is protocol total_scenes.")
    return parser.parse_args()


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def validate_dataset_contract(
    settings: dict[str, Any],
    episodes_override: int | None = None,
    *,
    allow_small: bool = False,
) -> tuple[int, int, list[str]]:
    if not bool(settings.get("not_a_locked_test", False)):
        raise ValueError("Phase 73 generation requires not_a_locked_test=true")
    if bool(settings.get("locked_test", False)):
        raise ValueError("Phase 73 must not be marked as a locked test")
    total = int(settings["total_scenes"] if episodes_override is None else episodes_override)
    minimum_total = 2 if allow_small else 500
    if total < minimum_total or total % 2 != 0:
        raise ValueError(f"Phase 73 requires at least {minimum_total} even-numbered scenes")
    blocks = [str(value) for value in settings["block_order"]]
    if not blocks or total % len(blocks) != 0:
        raise ValueError("total_scenes must divide evenly across block_order")
    scenes_per_block = total // len(blocks)
    if scenes_per_block % 2 != 0:
        raise ValueError("each block must contain complete mirror pairs")
    seeds = settings["seed_blocks"]
    if not isinstance(seeds, dict) or any(block not in seeds for block in blocks):
        raise ValueError("seed_blocks must define every block")
    if len({int(seeds[block]) for block in blocks}) != len(blocks):
        raise ValueError("seed blocks must be distinct")
    zone = tuple(float(value) for value in settings["obstacle_zone_x"])
    if len(zone) != 2 or not zone[0] < zone[1]:
        raise ValueError("obstacle_zone_x must contain increasing bounds")
    count_range = tuple(int(value) for value in settings["obstacle_count_range"])
    if count_range[0] < 3 or count_range[0] > count_range[1]:
        raise ValueError("obstacle_count_range must satisfy 3 <= lower <= upper")
    if not bool(settings["route_certificate"]["target_crossing_required"]):
        raise ValueError("Phase 73 requires target_crossing_required=true")
    if not bool(settings["route_certificate"]["direct_path_must_be_blocked"]):
        raise ValueError("Phase 73 requires direct_path_must_be_blocked=true")
    if int(settings["route_certificate"]["minimum_bypass_routes"]) < 2:
        raise ValueError("Phase 73 requires two bypass routes")
    return total, scenes_per_block, blocks


def _new_obstacle_center_y(obstacle: CylinderObstacle, center_y: float) -> CylinderObstacle:
    center = np.asarray(obstacle.center_xy, dtype=np.float64).copy()
    center[1] = float(center_y)
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


def _mirror_scenario(scenario: ShowcaseScenario, name: str, mirror_group_id: str) -> ShowcaseScenario:
    defenders = np.asarray(scenario.defender_positions, dtype=np.float64).copy()
    defenders[:, 0] *= -1.0
    target = np.asarray(scenario.target_position, dtype=np.float64).copy()
    target[0] *= -1.0
    escape = np.asarray(scenario.target_escape_direction, dtype=np.float64).copy()
    escape[0] *= -1.0
    return ShowcaseScenario(
        name=name,
        obstacles=tuple(_mirror_obstacle(obstacle) for obstacle in scenario.obstacles),
        defender_positions=defenders,
        target_position=target,
        target_escape_direction=escape,
        obstacle_zone_x=tuple(float(value) for value in scenario.obstacle_zone_x),
        target_crossing_required=True,
        defender_side="right",
        layout_seed=scenario.layout_seed,
        required_defender_zone_entries=int(scenario.required_defender_zone_entries),
        require_target_zone_entry=True,
        scenario_type="showcase",
    )


def _sample_points_on_segment(first: np.ndarray, second: np.ndarray, spacing: float = 0.05) -> np.ndarray:
    distance = float(np.linalg.norm(second - first))
    count = max(int(np.ceil(distance / max(spacing, 1.0e-9))), 1)
    fractions = np.linspace(0.0, 1.0, count + 1)
    return first[None, :] + fractions[:, None] * (second - first)[None, :]


def _polyline_metrics(
    env: CaptureRadiusPursuit3DEnv,
    points: list[np.ndarray],
    obstacles: tuple[CylinderObstacle, ...],
) -> dict[str, float]:
    samples = [
        segment_point
        for first, second in zip(points, points[1:])
        for segment_point in _sample_points_on_segment(first, second)
    ]
    trace = np.asarray(samples, dtype=np.float64)
    clearance_values = [
        float(env._obstacle_clearance(point, obstacle))
        for point in trace
        for obstacle in obstacles
    ]
    clearance = float(min(clearance_values)) if clearance_values else float("inf")
    safe_lower = np.asarray(env.lower, dtype=np.float64) + float(env.agents["drone_radius"]) + float(env.pursuit["safety_margin"])
    safe_upper = np.asarray(env.upper, dtype=np.float64) - float(env.agents["drone_radius"]) - float(env.pursuit["safety_margin"])
    boundary_margin = float(np.min(np.minimum(trace - safe_lower[None, :], safe_upper[None, :] - trace)))
    length = float(sum(np.linalg.norm(second - first) for first, second in zip(points, points[1:])))
    return {
        "length_m": length,
        "minimum_clearance_m": clearance,
        "minimum_boundary_margin_m": boundary_margin,
    }


def build_route_certificate(
    env: CaptureRadiusPursuit3DEnv,
    scenario: ShowcaseScenario,
    target_speed_scale: float,
    settings: dict[str, Any],
) -> dict[str, Any] | None:
    route_settings = settings["route_certificate"]
    clearance_limit = float(env.agents["drone_radius"]) + float(env.pursuit["safety_margin"])
    configured_limit = float(route_settings["clearance_limit_m"])
    if not np.isclose(clearance_limit, configured_limit):
        raise ValueError(
            f"route certificate clearance mismatch: environment={clearance_limit}, config={configured_limit}"
        )
    target_goal = np.asarray(scenario.target_position, dtype=np.float64).copy()
    target_goal[0] = float(np.mean(scenario.defender_positions[:, 0]))
    direct = _polyline_metrics(env, [scenario.target_position, target_goal], scenario.obstacles)
    direct_blocked = bool(direct["minimum_clearance_m"] < clearance_limit)
    if not direct_blocked:
        return None
    target_x_sign = 1.0 if float(scenario.target_position[0]) > 0.0 else -1.0
    waypoint_x = float(route_settings["target_waypoint_x_abs_m"])
    waypoint_y = float(route_settings["target_waypoint_y_abs_m"])
    routes: dict[str, dict[str, Any]] = {}
    for label, lateral_sign in (("positive_y", 1.0), ("negative_y", -1.0)):
        waypoints = [
            np.asarray(scenario.target_position, dtype=np.float64),
            np.array([target_x_sign * waypoint_x, lateral_sign * waypoint_y, scenario.target_position[2]], dtype=np.float64),
            np.array([-target_x_sign * waypoint_x, lateral_sign * waypoint_y, scenario.target_position[2]], dtype=np.float64),
            target_goal,
        ]
        metrics = _polyline_metrics(env, waypoints, scenario.obstacles)
        speed = float(env.agents["target_max_speed"]) * float(target_speed_scale)
        estimated_time = metrics["length_m"] / max(speed, 1.0e-9) + 2.0
        routes[label] = {
            "waypoints": [point.tolist() for point in waypoints],
            **metrics,
            "estimated_transit_time_seconds": float(estimated_time),
            "clearance_certified": bool(
                metrics["minimum_clearance_m"] >= clearance_limit
                and metrics["minimum_boundary_margin_m"] >= 0.0
                and estimated_time <= float(route_settings["estimated_transit_time_limit_seconds"])
            ),
        }
    certified_routes = [label for label, route in routes.items() if bool(route["clearance_certified"])]
    if len(certified_routes) < int(route_settings["minimum_bypass_routes"]):
        return None
    # The target routes above are also used as a fast, deterministic
    # reachability certificate for the defenders.  Each defender takes one of
    # the same two lateral channels and then approaches the opposite-side
    # target endpoint outside the obstacle zone.  This avoids repeating the
    # repository's expensive 0.5 m grid search 600 times during data export;
    # strict grid validation is still performed for one pair per block below.
    defender_route_certificates: dict[str, list[dict[str, Any]]] = {}
    defender_side_sign = -1.0 if scenario.defender_side == "left" else 1.0
    for label, lateral_sign in (("positive_y", 1.0), ("negative_y", -1.0)):
        route_results: list[dict[str, Any]] = []
        all_certified = True
        for defender_position in scenario.defender_positions:
            defender_goal = np.asarray(scenario.target_position, dtype=np.float64).copy()
            waypoints = [
                np.asarray(defender_position, dtype=np.float64),
                np.array([defender_side_sign * waypoint_x, lateral_sign * waypoint_y, defender_position[2]], dtype=np.float64),
                np.array([-defender_side_sign * waypoint_x, lateral_sign * waypoint_y, defender_position[2]], dtype=np.float64),
                defender_goal,
            ]
            metrics = _polyline_metrics(env, waypoints, scenario.obstacles)
            certified = bool(
                metrics["minimum_clearance_m"] >= clearance_limit
                and metrics["minimum_boundary_margin_m"] >= 0.0
            )
            all_certified = all_certified and certified
            route_results.append({"certified": certified, **metrics})
        defender_route_certificates[label] = route_results
        if all_certified:
            break
    certified_defender_routes = [
        label
        for label, route_results in defender_route_certificates.items()
        if route_results and all(bool(result["certified"]) for result in route_results)
    ]
    if not certified_defender_routes:
        return None
    return {
        "certificate_version": "phase73.v1",
        "obstacle_zone_x": [float(value) for value in scenario.obstacle_zone_x],
        "target_crossing_required": True,
        "require_target_zone_entry": True,
        "direct_path_blocked": True,
        "direct_path_length_m": float(direct["length_m"]),
        "direct_path_minimum_clearance_m": float(direct["minimum_clearance_m"]),
        "minimum_clearance_limit_m": float(clearance_limit),
        "bypass_route_count": int(len(certified_routes)),
        "certified_bypass_routes": certified_routes,
        "bypass_routes": routes,
        "target_goal": target_goal.tolist(),
        "independent_transit_route_feasible": True,
        "independent_transit_route_metrics": {
            "certificate_type": "lateral_polyline",
            "certified_defender_routes": certified_defender_routes,
            "defender_route_metrics": defender_route_certificates,
        },
    }


def sample_crossing_pair(
    env: CaptureRadiusPursuit3DEnv,
    layout_seed: int,
    initial_side_distance: float,
    target_speed_scale: float,
    settings: dict[str, Any],
    mirror_group_id: str,
) -> tuple[ShowcaseScenario, ShowcaseScenario, dict[str, Any], int]:
    zone = tuple(float(value) for value in settings["obstacle_zone_x"])
    count_low, count_high = (int(value) for value in settings["obstacle_count_range"])
    max_attempts = int(settings["generation"]["max_sampling_attempts_per_group"])
    y_limit = float(settings["generation"]["obstacle_center_y_abs_limit_m"])
    min_separation = float(settings["generation"]["obstacle_min_horizontal_separation_m"])
    defenders, target, escape = _opposite_side_positions(initial_side_distance, "left")
    escape = -escape
    protected_points = np.vstack([defenders, target[None, :]])
    rng = np.random.default_rng(int(layout_seed))
    for attempt in range(1, max_attempts + 1):
        count = int(rng.integers(count_low, count_high + 1))
        # The first obstacle is a central gate.  It is always tall enough to
        # block the target's z=4.2 direct route; the remaining shapes provide
        # geometry diversity without making the two lateral channels vanish.
        gate_sample = _random_central_obstacle(rng, "cylinder", zone)
        gate = CylinderObstacle(
            center_xy=np.array([gate_sample.center_xy[0], 0.0], dtype=np.float64),
            radius=float(gate_sample.radius),
            height=max(float(gate_sample.height), 4.8),
            shape="cylinder",
            half_extents_xy=None,
        )
        obstacles: list[CylinderObstacle] = [gate]
        shapes = ["box", "wall"]
        shapes.extend(str(rng.choice(["cylinder", "box", "wall"])) for _ in range(count - 3))
        rng.shuffle(shapes)
        for shape in shapes:
            for _candidate_attempt in range(100):
                candidate = _random_central_obstacle(rng, shape, zone)
                candidate_y = float(np.clip(candidate.center_xy[1], -y_limit, y_limit))
                candidate = _new_obstacle_center_y(candidate, candidate_y)
                if not env._obstacle_clear_of_points(candidate, protected_points):
                    continue
                if any(env._obstacle_horizontal_separation(candidate, current) < min_separation for current in obstacles):
                    continue
                obstacles.append(candidate)
                break
            else:
                break
        if len(obstacles) != count:
            continue
        scenario = ShowcaseScenario(
            name=f"phase73_crossing_{mirror_group_id}_left",
            obstacles=tuple(obstacles),
            defender_positions=defenders.copy(),
            target_position=target.copy(),
            target_escape_direction=escape.copy(),
            obstacle_zone_x=zone,
            target_crossing_required=True,
            defender_side="left",
            layout_seed=int(layout_seed),
            required_defender_zone_entries=1,
            require_target_zone_entry=True,
            scenario_type="showcase",
        )
        # Full grid validation is retained as a periodic audit rather than
        # repeated for every record.  Cheap geometry checks plus the explicit
        # polyline certificates are applied to every candidate below.
        if not _cheap_scenario_sanity(env, scenario):
            continue
        certificate = build_route_certificate(env, scenario, target_speed_scale, settings)
        if certificate is None:
            continue
        mirrored = _mirror_scenario(scenario, f"phase73_crossing_{mirror_group_id}_right", mirror_group_id)
        if not _cheap_scenario_sanity(env, mirrored):
            continue
        mirrored_certificate = build_route_certificate(env, mirrored, target_speed_scale, settings)
        if mirrored_certificate is None:
            continue
        if certificate["certified_bypass_routes"] != mirrored_certificate["certified_bypass_routes"]:
            continue
        certificate = copy.deepcopy(certificate)
        certificate["mirror_group_id"] = mirror_group_id
        certificate["mirror_invariant"] = True
        return scenario, mirrored, certificate, attempt
    raise RuntimeError(f"unable to sample Phase 73 crossing pair after {max_attempts} attempts (seed={layout_seed})")


def _cheap_scenario_sanity(env: CaptureRadiusPursuit3DEnv, scenario: ShowcaseScenario) -> bool:
    """Validate invariants that do not require a grid search."""
    if scenario.defender_positions.shape != (env.n_defenders, 3) or scenario.target_position.shape != (3,):
        return False
    low, high = map(float, scenario.obstacle_zone_x)
    if scenario.defender_side == "left":
        if np.any(scenario.defender_positions[:, 0] >= low) or scenario.target_position[0] <= high:
            return False
    elif scenario.defender_side == "right":
        if np.any(scenario.defender_positions[:, 0] <= high) or scenario.target_position[0] >= low:
            return False
    else:
        return False
    all_positions = np.vstack([scenario.defender_positions, scenario.target_position[None, :]])
    boundary_buffer = float(env.agents["drone_radius"]) + float(env.pursuit["safety_margin"])
    if np.any(all_positions < env.lower[None, :] + boundary_buffer) or np.any(
        all_positions > env.upper[None, :] - boundary_buffer
    ):
        return False
    pairwise = np.linalg.norm(
        scenario.defender_positions[:, None, :] - scenario.defender_positions[None, :, :], axis=2
    ) + np.eye(env.n_defenders) * 1.0e6
    if float(np.min(pairwise)) < 2.0 * float(env.agents["drone_radius"]):
        return False
    for obstacle in scenario.obstacles:
        obstacle_low, obstacle_high = _obstacle_x_extent(obstacle)
        if obstacle_low < low or obstacle_high > high:
            return False
        if not env._obstacle_clear_of_points(obstacle, all_positions):
            return False
    return bool(scenario.target_crossing_required and scenario.require_target_zone_entry)


def _record_spec(
    episode_seed: int,
    layout_seed: int,
    scenario: ShowcaseScenario,
    target_speed_scale: float,
    condition: dict[str, Any],
    block: str,
    block_episode_index: int,
    mirror_group_id: str,
    mirror_pair_member: str,
) -> dict[str, Any]:
    return {
        "episode_seed": int(episode_seed),
        "layout_seed": int(layout_seed),
        "defender_side": str(scenario.defender_side),
        "initial_side_distance": float(abs(scenario.target_position[0])),
        "target_speed_scale": float(target_speed_scale),
        "target_motion_mode": "adaptive_maneuvering",
        "target_crossing_required": True,
        "observation_condition": str(condition["name"]),
        "pursuit_overrides": copy.deepcopy(condition["pursuit_overrides"]),
        "obstacle_count": int(len(scenario.obstacles)),
        "condition_index": int(block_episode_index % 2),
        "condition_table_size": 2,
        "scene_block": block,
        "block_episode_index": int(block_episode_index),
        "mirror_group_id": mirror_group_id,
        "mirror_pair_member": mirror_pair_member,
        "dataset_split_role": block,
    }


def generate_records(settings: dict[str, Any], environment_config: Path, total: int, *, allow_small: bool = False) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _total, scenes_per_block, blocks = validate_dataset_contract(settings, total, allow_small=allow_small)
    environment_payload = _load_mapping(environment_config)
    records: list[dict[str, Any]] = []
    attempts_by_block: dict[str, int] = {}
    conditions = list(settings["observation_conditions"])
    distances = [float(value) for value in settings["initial_side_distances"]]
    speeds = [float(value) for value in settings["target_speed_scales"]]
    for block_index, block in enumerate(blocks):
        block_seed = int(settings["seed_blocks"][block])
        assignment_rng = np.random.default_rng(block_seed + 3_000_000)
        block_attempts = 0
        for pair_index in range(scenes_per_block // 2):
            layout_seed = block_seed + 1_000_000 + pair_index
            initial_side_distance = float(assignment_rng.choice(np.asarray(distances, dtype=np.float64)))
            target_speed_scale = float(assignment_rng.choice(np.asarray(speeds, dtype=np.float64)))
            condition = copy.deepcopy(conditions[int(assignment_rng.integers(0, len(conditions)))])
            mirror_group_id = f"phase73-{block}-{pair_index:04d}"
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
                # One strict audit per block catches regressions in the
                # repository's canonical grid validator while keeping the
                # large export practical.  The per-record certificate remains
                # the authoritative generation filter for all other pairs.
                validate_showcase_scenario(base_env, left)
                validate_showcase_scenario(base_env, right)
            block_attempts += int(attempts)
            for member_index, (member_name, scenario) in enumerate((("left", left), ("right", right))):
                global_index = len(records)
                episode_seed = block_seed + pair_index * 2 + member_index
                spec = _record_spec(
                    episode_seed,
                    layout_seed,
                    scenario,
                    target_speed_scale,
                    condition,
                    block,
                    pair_index * 2 + member_index,
                    mirror_group_id,
                    member_name,
                )
                record_certificate = copy.deepcopy(certificate)
                record_certificate["mirror_pair_member"] = member_name
                record_certificate["layout_seed"] = int(layout_seed)
                records.append(
                    {
                        "dataset_index": global_index,
                        "episode_index": global_index,
                        "episode_seed": int(episode_seed),
                        "layout_seed": int(layout_seed),
                        "target_speed_scale": float(target_speed_scale),
                        "defender_side": str(scenario.defender_side),
                        "initial_side_distance": float(abs(scenario.target_position[0])),
                        "target_motion_mode": "adaptive_maneuvering",
                        "target_crossing_required": True,
                        "observation_condition": str(condition["name"]),
                        "pursuit_overrides": copy.deepcopy(condition["pursuit_overrides"]),
                        # The closed-loop S4 evaluator requires a branch-label
                        # field for legacy protocol adaptation.  Phase 73 does
                        # not use branch geometry; this is only a schema
                        # compatibility label and is never passed to control.
                        "defender_bias": "upper" if member_name == "left" else "lower",
                        "spec": spec,
                        "scenario": scenario_metadata(scenario),
                        "route_certificate": record_certificate,
                    }
                )
        attempts_by_block[block] = block_attempts
        print(f"generated {block}: {scenes_per_block} scenes / {scenes_per_block // 2} mirror groups", flush=True)
    return records, {"sampling_attempts_by_block": attempts_by_block}


def _write_jsonl(path: Path, records: list[dict[str, Any]], *, local_episode_index: bool = False) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for local_index, record in enumerate(records):
            item = copy.deepcopy(record)
            if local_episode_index:
                item["episode_index"] = int(local_index)
                item["spec"]["block_episode_index"] = int(local_index)
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset(
    output_dir: Path,
    records: list[dict[str, Any]],
    settings: dict[str, Any],
    protocol_path: Path,
    environment_path: Path,
    generation_metadata: dict[str, Any],
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_path = output_dir / "scenes.jsonl"
    _write_jsonl(scene_path, records)
    block_dir = output_dir / "blocks"
    block_dir.mkdir()
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_block[str(record["spec"]["scene_block"])].append(record)
    for block, block_records in by_block.items():
        _write_jsonl(block_dir / f"{block}.jsonl", block_records, local_episode_index=True)
    mirror_groups = Counter(str(record["spec"]["mirror_group_id"]) for record in records)
    shapes = Counter(
        str(obstacle["shape"])
        for record in records
        for obstacle in record["scenario"]["obstacles"]
    )
    manifest = {
        "dataset_name": str(settings["dataset_name"]),
        "dataset_version": "phase73.v1",
        "phase": str(settings["phase"]),
        "not_a_locked_test": True,
        "locked_test": False,
        "locked_test_modified": False,
        "purpose": str(settings["purpose"]),
        "total_scenes": len(records),
        "mirror_groups": len(mirror_groups),
        "all_mirror_groups_have_two_members": bool(all(count == 2 for count in mirror_groups.values())),
        "block_counts": {block: len(items) for block, items in sorted(by_block.items())},
        "block_mirror_group_counts": {block: len({str(item["spec"]["mirror_group_id"]) for item in items}) for block, items in sorted(by_block.items())},
        "target_crossing_required_rate": float(np.mean([bool(item["spec"]["target_crossing_required"]) for item in records])),
        "direct_path_blocked_rate": float(np.mean([bool(item["route_certificate"]["direct_path_blocked"]) for item in records])),
        "minimum_bypass_route_count": int(min(int(item["route_certificate"]["bypass_route_count"]) for item in records)),
        "obstacle_shape_counts": dict(sorted(shapes.items())),
        "observation_condition_counts": dict(sorted(Counter(str(item["spec"]["observation_condition"]) for item in records).items())),
        "target_speed_scale_counts": dict(sorted(Counter(str(item["spec"]["target_speed_scale"]) for item in records).items())),
        "scene_file": str(scene_path),
        "scene_file_sha256": _sha256(scene_path),
        "protocol": str(protocol_path.resolve()),
        "environment_config": str(environment_path.resolve()),
        "generation": generation_metadata,
        "contract": {
            "same_scene_manifest_required_for_all_models": True,
            "locked_test_tuning_forbidden": True,
            "route_certificate_is_generation_time_only": True,
            "route_certificate_is_not_a_closed_loop_safety_proof": True,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.md").write_text(
        "# Phase 73 target-crossing scene dataset\n\n"
        "This is a 600-scene development/validation pool, not the existing locked test. "
        "Each record contains frozen geometry, a mirror-pair identifier, observation-condition metadata, "
        "and a generation-time route certificate. Use the same `scenes.jsonl` for every model comparison; "
        "do not tune on `external_holdout`. The route certificate is not a closed-loop safety proof.\n",
        encoding="utf-8",
    )
    return manifest


def generate(
    protocol_path: Path,
    environment_path: Path,
    output_dir: Path,
    episodes_override: int | None = None,
    *,
    allow_small: bool = False,
) -> dict[str, Any]:
    settings = _load_mapping(protocol_path.resolve())
    total, _scenes_per_block, _blocks = validate_dataset_contract(
        settings,
        episodes_override,
        allow_small=allow_small,
    )
    environment_path = environment_path.resolve()
    if not environment_path.is_file():
        raise FileNotFoundError(environment_path)
    records, generation_metadata = generate_records(settings, environment_path, total, allow_small=allow_small)
    manifest = write_dataset(
        output_dir,
        records,
        settings,
        protocol_path,
        environment_path,
        generation_metadata,
    )
    print(json.dumps({"output_dir": str(output_dir.resolve()), **manifest}, ensure_ascii=False, indent=2), flush=True)
    return manifest


def main() -> None:
    args = parse_args()
    generate(args.protocol, args.environment_config, args.output_dir, args.episodes)


if __name__ == "__main__":
    main()
