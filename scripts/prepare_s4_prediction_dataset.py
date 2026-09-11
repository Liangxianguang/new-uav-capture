"""Convert an audited raw S4 archive into the common predictor dataset format.

The converted archive keeps observation history and action conditioning as
separate tensors.  This prevents action streams from being silently dropped by
the generic predictor trainer and makes the action source explicit in metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.trajectory_dataset import PredictionDataset, save_prediction_dataset


ACTION_STREAMS = ("planned", "commanded", "delayed", "executed")
SHAPE_CODES = {"cylinder": 0, "box": 1, "wall": 2}

# The first 15 legacy policy-observation values are stable across the v3
# collection contract. Shape-aware geometry is appended after that prefix.
_BELIEF_RELATIVE_POSITION = slice(3, 6)
_BELIEF_VELOCITY = slice(6, 9)
_BELIEF_CONFIDENCE_INDEX = 11


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--history-action-stream", choices=ACTION_STREAMS, default="executed")
    parser.add_argument("--future-action-stream", choices=ACTION_STREAMS, default="planned")
    return parser.parse_args()


def load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return value


def load_scene_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            records[int(value["episode_index"])] = value
    return records


def policy_safe_team_references(
    raw: dict[str, np.ndarray],
    environment: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the public team belief used as the forecast reference.

    S4-v3 records the policy-safe local observations and the defender state at
    every forecast origin.  The former v3 conversion used the simulator target
    position as the trajectory origin, which is unsuitable for an online
    prediction baseline.  This reconstruction uses only the public belief
    fields stored in the raw archive; target truth is not read here.
    """

    required = {
        "history_observations",
        "history_message_age_steps",
        "future_defender_positions",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(
            "Policy-safe S4 reference reconstruction requires: " + ", ".join(missing)
        )
    pursuit = dict(environment.get("task", {}).get("pursuit", {}))
    if not bool(pursuit.get("include_uncertainty_features", False)):
        raise ValueError("S4-v3 belief reconstruction requires uncertainty features in policy observations.")
    world = dict(environment.get("world", {}))
    agents = dict(environment.get("agents", {}))
    half_extent = float(world.get("half_extent_xy", 0.0))
    target_max_speed = float(agents.get("target_max_speed", 0.0))
    if not np.isfinite([half_extent, target_max_speed]).all() or half_extent <= 0.0 or target_max_speed <= 0.0:
        raise ValueError("Environment must define positive world.half_extent_xy and agents.target_max_speed.")

    observations = np.asarray(raw["history_observations"], dtype=np.float32)
    message_ages = np.asarray(raw["history_message_age_steps"][:, -1], dtype=np.float32)
    defender_positions = np.asarray(raw["future_defender_positions"][:, 0], dtype=np.float32)
    if (
        observations.ndim != 4
        or observations.shape[-1] <= _BELIEF_CONFIDENCE_INDEX
        or message_ages.shape != observations.shape[:1] + observations.shape[2:3]
        or defender_positions.shape != observations.shape[:1] + observations.shape[2:3] + (3,)
    ):
        raise ValueError("S4-v3 public belief fields have incompatible shapes.")

    last_observations = observations[:, -1]
    belief_positions = (
        defender_positions + last_observations[..., _BELIEF_RELATIVE_POSITION] * half_extent
    )
    belief_velocities = last_observations[..., _BELIEF_VELOCITY] * target_max_speed
    confidences = last_observations[..., _BELIEF_CONFIDENCE_INDEX]
    weights = np.maximum(confidences, 1.0e-3) / (1.0 + np.maximum(message_ages, 0.0))
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1.0e-12)
    reference_positions = np.sum(belief_positions * weights[..., None], axis=1)
    reference_velocities = np.sum(belief_velocities * weights[..., None], axis=1)
    if not np.isfinite(reference_positions).all() or not np.isfinite(reference_velocities).all():
        raise RuntimeError("Policy-safe team belief reconstruction produced non-finite values.")
    return reference_positions.astype(np.float32), reference_velocities.astype(np.float32)


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dataset.resolve()
    output = args.output.resolve()
    raw_path = raw_dir / "dataset.npz"
    raw_metadata_path = raw_dir / "metadata.json"
    for path in (raw_path, raw_metadata_path, args.environment_config.resolve()):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")

    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {name: np.asarray(archive[name]) for name in archive.files}
    scenes = load_scene_records(raw_dir / "scenes.jsonl")
    required = {
        "history_observations",
        "history_message_age_steps",
        "future_target_positions",
        "reference_target_positions",
        "future_defender_positions",
        "episode_indices",
        "episode_seeds",
        "sample_timesteps",
        "branch_sign",
        "branch_decision_steps",
    }
    missing = sorted(required.difference(raw))
    if missing:
        raise ValueError(f"Raw S4 archive is missing fields: {', '.join(missing)}")
    history_key = f"history_{args.history_action_stream}_actions"
    future_key = f"future_{args.future_action_stream}_actions"
    for key in (history_key, future_key):
        if key not in raw:
            raise ValueError(f"Raw S4 archive is missing action stream {key}.")

    observations = np.asarray(raw["history_observations"], dtype=np.float32)
    absolute_targets = np.asarray(raw["future_target_positions"], dtype=np.float32)
    target_truth_at_origin = np.asarray(raw["reference_target_positions"], dtype=np.float32)
    history_actions = np.asarray(raw[history_key], dtype=np.float32)
    future_actions = np.asarray(raw[future_key], dtype=np.float32)
    if observations.ndim != 4 or absolute_targets.ndim != 3:
        raise ValueError("Raw S4 observations or target positions have incompatible shapes.")
    sample_count, history_length, defender_count, _feature_dim = observations.shape
    if (
        absolute_targets.shape[0] != sample_count
        or target_truth_at_origin.shape != (sample_count, 3)
        or history_actions.shape[:3] != (sample_count, history_length, defender_count)
    ):
        raise ValueError("Raw S4 history and target sample dimensions do not match.")
    if future_actions.shape[:3] != (sample_count, absolute_targets.shape[1], defender_count):
        raise ValueError("Raw S4 future action stream does not align with the target horizon.")

    environment = load_mapping(args.environment_config)
    dt_seconds = float(environment.get("world", {}).get("dt", 0.0))
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError("Environment config must define a positive world.dt.")
    reference_positions, reference_velocities = policy_safe_team_references(raw, environment)
    targets = absolute_targets - reference_positions[:, None, :]
    future_velocities = np.diff(
        np.concatenate([target_truth_at_origin[:, None, :], absolute_targets], axis=1), axis=1
    ) / dt_seconds
    episode_indices = np.asarray(raw["episode_indices"], dtype=np.int64)
    half_extent = float(environment["world"]["half_extent_xy"])
    lower_bounds = np.repeat(
        np.asarray([-half_extent, -half_extent, float(environment["world"]["minimum_altitude"])], dtype=np.float32)[None, :],
        sample_count,
        axis=0,
    )
    upper_bounds = np.repeat(
        np.asarray(
            [half_extent, half_extent, float(environment["world"]["height"])], dtype=np.float32
        )[None, :],
        sample_count,
        axis=0,
    )
    obstacle_centers: list[list[list[float]]] = []
    obstacle_radii: list[list[float]] = []
    obstacle_heights: list[list[float]] = []
    obstacle_extents: list[list[list[float]]] = []
    obstacle_codes: list[list[int]] = []
    for episode_index in episode_indices.tolist():
        scenario = scenes[int(episode_index)].get("scenario", {})
        obstacles = scenario.get("obstacles", [])
        if len(obstacles) != 1:
            raise ValueError("S4 prediction conversion expects exactly one obstacle per scene.")
        obstacle = obstacles[0]
        shape = str(obstacle.get("shape", "cylinder"))
        if shape not in SHAPE_CODES:
            raise ValueError(f"Unsupported obstacle shape in S4 archive: {shape}")
        extents = obstacle.get("half_extents_xy") or [obstacle["radius"], obstacle["radius"]]
        obstacle_centers.append([list(obstacle["center_xy"])])
        obstacle_radii.append([float(obstacle["radius"])])
        obstacle_heights.append([float(obstacle["height"])])
        obstacle_extents.append([list(extents)])
        obstacle_codes.append([SHAPE_CODES[shape]])
    dataset = PredictionDataset(
        history_observations=observations,
        future_target_displacements=targets,
        reference_positions=reference_positions,
        reference_velocities=reference_velocities,
        episode_indices=np.asarray(raw["episode_indices"], dtype=np.int64),
        timesteps=np.asarray(raw["sample_timesteps"], dtype=np.int64),
        episode_seeds=np.asarray(raw["episode_seeds"], dtype=np.int64),
        target_motion_modes=np.full(sample_count, "adaptive_branching"),
        dt_seconds=dt_seconds,
        future_target_velocities=future_velocities.astype(np.float32),
        history_action_features=history_actions.reshape(sample_count, history_length, defender_count * 3),
        future_action_conditions=future_actions.reshape(sample_count, targets.shape[1], defender_count * 3),
        target_branch_signs=np.asarray(raw["branch_sign"], dtype=np.int8),
        target_branch_decision_steps=np.asarray(raw["branch_decision_steps"], dtype=np.int64),
        world_lower_bounds=lower_bounds,
        world_upper_bounds=upper_bounds,
        obstacle_centers_xy=np.asarray(obstacle_centers, dtype=np.float32),
        obstacle_radii=np.asarray(obstacle_radii, dtype=np.float32),
        obstacle_heights=np.asarray(obstacle_heights, dtype=np.float32),
        obstacle_half_extents_xy=np.asarray(obstacle_extents, dtype=np.float32),
        obstacle_shape_codes=np.asarray(obstacle_codes, dtype=np.int8),
    )
    output.mkdir(parents=True, exist_ok=True)
    save_prediction_dataset(dataset, str(output / "dataset.npz"))
    raw_metadata = json.loads(raw_metadata_path.read_text(encoding="utf-8"))
    metadata = {
        "dataset_name": f"{raw_metadata.get('dataset_name', 's4')}_predictor",
        "schema_version": 2,
        "source_raw_dataset": str(raw_dir),
        "source_raw_schema_version": raw_metadata.get("schema_version"),
        "sample_count": sample_count,
        "history_length": history_length,
        "horizon_steps": int(targets.shape[1]),
        "feature_dim": int(observations.shape[-1]),
        "history_action_stream": args.history_action_stream,
        "future_action_stream": args.future_action_stream,
        "action_condition_contract": {
            "history_action_features": "flattened per-defender action stream aligned with observation history",
            "future_action_conditions": "flattened per-defender planned/execution stream aligned with forecast horizon",
            "future_action_online_availability": "must be supplied by the online planner; logged rollout sequence is an offline condition",
        },
        "reference_contract": {
            "prediction_origin": "confidence-and-age weighted public team target belief",
            "prediction_velocity": "confidence-and-age weighted public team belief velocity",
            "target_truth_usage": "future supervised labels and physical future-velocity diagnostics only",
            "legacy_v3_truth_origin": "not used by this predictor-v4 conversion",
        },
        "branch_label_contract": {
            "target_branch_signs": "supervised/evaluation-only lower=-1 upper=+1 labels",
            "target_branch_decision_steps": "simulator-private decision timestep; not a predictor input",
        },
        "dt_seconds": dt_seconds,
        "config": environment,
        "raw_metadata": raw_metadata,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
