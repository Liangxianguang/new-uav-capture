"""Collect prediction windows from policy-safe observations and hidden labels."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.pursuit_controllers import (  # noqa: E402
    DynamicEncirclementController,
    SafetyFilteredPursuitController,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.trajectory_dataset import (  # noqa: E402
    build_episode_samples,
    concatenate_prediction_datasets,
    save_prediction_dataset,
    team_belief_reference,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "capture_radius_pursuit_central_v4_flee.yaml"
TARGET_MOTION_MODES = (
    "flee_persistence",
    "random_turn",
    "s_curve",
    "burst",
    "boundary_escape",
    "adaptive_adversarial",
    "adaptive_branching",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True, help="Empty directory for dataset.npz and metadata.json.")
    parser.add_argument("--split", choices=("train", "validation", "locked_test"), default="train")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=645101)
    parser.add_argument("--obstacle-count", type=int, default=3)
    parser.add_argument("--target-speed-scale", type=float, default=0.55)
    parser.add_argument(
        "--rollout-policy",
        choices=("stationary", "expert"),
        default="stationary",
        help="Policy used to generate observable histories; stationary avoids early capture truncation.",
    )
    parser.add_argument(
        "--target-motion-mode",
        choices=TARGET_MOTION_MODES,
        default="flee_persistence",
    )
    parser.add_argument(
        "--target-motion-modes",
        nargs="+",
        choices=TARGET_MOTION_MODES,
        help=(
            "Optional deterministic round-robin target modes. When supplied, this overrides "
            "--target-motion-mode and records the episode-to-mode schedule in metadata."
        ),
    )
    parser.add_argument("--history-length", type=int, default=16)
    parser.add_argument("--horizon-steps", type=int, default=12)
    return parser.parse_args()


def load_config(path: Path, target_motion_mode: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Environment config does not exist: {resolved}")
    config = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Environment config must be a mapping.")
    config = copy.deepcopy(config)
    config.setdefault("task", {}).setdefault("pursuit", {})["target_motion_mode"] = target_motion_mode
    return config


def collect_episode(
    config: dict[str, Any],
    *,
    episode_index: int,
    episode_seed: int,
    obstacle_count: int,
    target_speed_scale: float,
    history_length: int,
    horizon_steps: int,
    target_motion_mode: str,
    rollout_policy: str,
):
    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=obstacle_count,
        target_speed_scale=target_speed_scale,
    )
    observation = env.reset(seed=episode_seed)
    controller = (
        SafetyFilteredPursuitController(DynamicEncirclementController(env))
        if rollout_policy == "expert"
        else None
    )
    local_frames: list[np.ndarray] = []
    references: list[np.ndarray] = []
    reference_velocities: list[np.ndarray] = []
    target_positions: list[np.ndarray] = []
    while True:
        # This is the only actor input saved in the dataset. Target truth is
        # used below only as a future supervised label.
        local_frames.append(policy_observations(env, observation).copy())
        references.append(team_belief_reference(observation))
        belief_velocities = np.asarray(observation["target_belief_velocities"], dtype=np.float64)
        confidences = np.asarray(observation["target_observation_confidence"], dtype=np.float64)
        ages = np.asarray(observation["message_age_steps"], dtype=np.float64)
        weights = np.maximum(confidences, 1e-3) / (1.0 + np.maximum(ages, 0.0))
        weights /= np.sum(weights)
        reference_velocities.append(np.sum(belief_velocities * weights[:, None], axis=0).astype(np.float32))
        target_positions.append(env.target_position.copy().astype(np.float32))
        action = controller.act(observation) if controller is not None else np.zeros((env.n_defenders, 3))
        observation, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break
    shape_codes = {"cylinder": 0, "box": 1, "wall": 2}
    obstacle_count = len(env.obstacles)
    centers = np.empty((obstacle_count, 2), dtype=np.float32)
    radii = np.empty(obstacle_count, dtype=np.float32)
    heights = np.empty(obstacle_count, dtype=np.float32)
    half_extents = np.empty((obstacle_count, 2), dtype=np.float32)
    obstacle_shapes = np.empty(obstacle_count, dtype=np.int8)
    for index, obstacle in enumerate(env.obstacles):
        if obstacle.shape not in shape_codes:
            raise ValueError(f"Unsupported obstacle shape: {obstacle.shape}")
        centers[index] = np.asarray(obstacle.center_xy, dtype=np.float32)
        radii[index] = float(obstacle.radius)
        heights[index] = float(obstacle.height)
        half_extents[index] = (
            np.array([obstacle.radius, obstacle.radius], dtype=np.float32)
            if obstacle.half_extents_xy is None
            else np.asarray(obstacle.half_extents_xy, dtype=np.float32)
        )
        obstacle_shapes[index] = shape_codes[obstacle.shape]
    return build_episode_samples(
        local_frames,
        references,
        target_positions,
        reference_velocities,
        dt_seconds=float(env.dt),
        history_length=history_length,
        horizon_steps=horizon_steps,
        episode_index=episode_index,
        episode_seed=episode_seed,
        target_motion_mode=target_motion_mode,
        world_lower_bounds=env.lower.astype(np.float32),
        world_upper_bounds=env.upper.astype(np.float32),
        obstacle_centers_xy=centers,
        obstacle_radii=radii,
        obstacle_heights=heights,
        obstacle_half_extents_xy=half_extents,
        obstacle_shape_codes=obstacle_shapes,
    ), len(local_frames)


def source_hashes() -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "scripts" / "collect_prediction_dataset.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "trajectory_dataset.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "observation_encoding.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_env.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_controllers.py",
    )
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def main() -> None:
    args = parse_args()
    if args.episodes <= 0 or args.obstacle_count < 0:
        raise ValueError("episodes must be positive and obstacle-count must be non-negative.")
    if args.history_length <= 0 or args.horizon_steps <= 0:
        raise ValueError("history-length and horizon-steps must be positive.")
    if args.target_speed_scale <= 0.0:
        raise ValueError("target-speed-scale must be positive.")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    target_motion_modes = tuple(args.target_motion_modes or (args.target_motion_mode,))
    config = load_config(args.environment_config, target_motion_modes[0])
    datasets = []
    episode_lengths: list[int] = []
    mode_schedule: list[str] = []
    for episode_index in range(args.episodes):
        target_motion_mode = target_motion_modes[episode_index % len(target_motion_modes)]
        dataset, frame_count = collect_episode(
            load_config(args.environment_config, target_motion_mode),
            episode_index=episode_index,
            episode_seed=args.seed + episode_index,
            obstacle_count=args.obstacle_count,
            target_speed_scale=args.target_speed_scale,
            history_length=args.history_length,
            horizon_steps=args.horizon_steps,
            target_motion_mode=target_motion_mode,
            rollout_policy=args.rollout_policy,
        )
        datasets.append(dataset)
        episode_lengths.append(frame_count)
        mode_schedule.append(target_motion_mode)
    merged = concatenate_prediction_datasets(datasets)
    dataset_path = output / "dataset.npz"
    save_prediction_dataset(merged, str(dataset_path))
    metadata = {
        "dataset_type": "policy_safe_target_trajectory_windows",
        "split": args.split,
        "environment_config": str(args.environment_config.resolve()),
        "config": config,
        "episodes": int(args.episodes),
        "episode_seed_start": int(args.seed),
        "episode_lengths": episode_lengths,
        "obstacle_count": int(args.obstacle_count),
        "target_speed_scale": float(args.target_speed_scale),
        "target_motion_mode": target_motion_modes[0] if len(target_motion_modes) == 1 else "stratified_cycle",
        "target_motion_modes": list(target_motion_modes),
        "episode_target_motion_modes": mode_schedule,
        "rollout_policy": args.rollout_policy,
        "history_length": int(merged.history_length),
        "horizon_steps": int(merged.horizon_steps),
        "defender_count": int(merged.defender_count),
        "feature_dim": int(merged.feature_dim),
        "sample_count": int(merged.sample_count),
        "dt_seconds": float(merged.dt_seconds),
        "input_contract": {
            "source": "encirclement3d.observation_encoding.policy_observations",
            "uses_target_truth": False,
            "uses_delayed_beliefs": True,
        },
        "label_contract": {
            "source": "CaptureRadiusPursuit3DEnv.target_position",
            "uses_target_truth": True,
            "representation": "future_target_position_minus_published_team_belief_reference",
            "future_velocity_representation": "finite_difference_of_target_position_over_dt",
        },
        "geometry_contract": {
            "available": bool(merged.has_geometry_context),
            "obstacle_shape_codes": {"cylinder": 0, "box": 1, "wall": 2},
            "world_bounds": "per-sample static environment bounds",
        },
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "source_hashes": source_hashes(),
    }
    output.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": str(dataset_path), "metadata": str(output / "metadata.json"), **metadata}, indent=2))


if __name__ == "__main__":
    main()
