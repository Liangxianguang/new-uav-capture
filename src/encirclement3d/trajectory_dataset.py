"""Reproducible target-trajectory windows built from policy-safe observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PredictionDataset:
    """Validated in-memory representation of a prediction dataset archive."""

    history_observations: np.ndarray
    future_target_displacements: np.ndarray
    reference_positions: np.ndarray
    reference_velocities: np.ndarray
    episode_indices: np.ndarray
    timesteps: np.ndarray
    episode_seeds: np.ndarray
    target_motion_modes: np.ndarray
    dt_seconds: float
    world_lower_bounds: np.ndarray | None = None
    world_upper_bounds: np.ndarray | None = None
    obstacle_centers_xy: np.ndarray | None = None
    obstacle_radii: np.ndarray | None = None
    obstacle_heights: np.ndarray | None = None
    obstacle_half_extents_xy: np.ndarray | None = None
    obstacle_shape_codes: np.ndarray | None = None

    @property
    def sample_count(self) -> int:
        return int(self.history_observations.shape[0])

    @property
    def history_length(self) -> int:
        return int(self.history_observations.shape[1])

    @property
    def horizon_steps(self) -> int:
        return int(self.future_target_displacements.shape[1])

    @property
    def defender_count(self) -> int:
        return int(self.history_observations.shape[2])

    @property
    def feature_dim(self) -> int:
        return int(self.history_observations.shape[3])

    @property
    def has_geometry_context(self) -> bool:
        return all(
            value is not None
            for value in (
                self.world_lower_bounds,
                self.world_upper_bounds,
                self.obstacle_centers_xy,
                self.obstacle_radii,
                self.obstacle_heights,
                self.obstacle_half_extents_xy,
                self.obstacle_shape_codes,
            )
        )

    def as_dict(self) -> dict[str, np.ndarray]:
        values = {
            "history_observations": self.history_observations,
            "future_target_displacements": self.future_target_displacements,
            "reference_positions": self.reference_positions,
            "reference_velocities": self.reference_velocities,
            "episode_indices": self.episode_indices,
            "timesteps": self.timesteps,
            "episode_seeds": self.episode_seeds,
            "target_motion_modes": self.target_motion_modes,
            "dt_seconds": np.asarray(self.dt_seconds, dtype=np.float32),
        }
        if self.has_geometry_context:
            values.update(
                {
                    "world_lower_bounds": self.world_lower_bounds,
                    "world_upper_bounds": self.world_upper_bounds,
                    "obstacle_centers_xy": self.obstacle_centers_xy,
                    "obstacle_radii": self.obstacle_radii,
                    "obstacle_heights": self.obstacle_heights,
                    "obstacle_half_extents_xy": self.obstacle_half_extents_xy,
                    "obstacle_shape_codes": self.obstacle_shape_codes,
                }
            )
        return values


def team_belief_reference(observation: dict[str, Any]) -> np.ndarray:
    """Fuse only published beliefs into a shared target reference point."""

    beliefs = np.asarray(observation["target_belief_positions"], dtype=np.float64)
    confidences = np.asarray(observation["target_observation_confidence"], dtype=np.float64)
    ages = np.asarray(observation["message_age_steps"], dtype=np.float64)
    if beliefs.ndim != 2 or beliefs.shape[1] != 3:
        raise ValueError("target_belief_positions must have shape [defenders, 3].")
    if confidences.shape != (beliefs.shape[0],) or ages.shape != (beliefs.shape[0],):
        raise ValueError("Belief confidence and age arrays must match the defender count.")
    weights = np.maximum(confidences, 1e-3) / (1.0 + np.maximum(ages, 0.0))
    weights /= np.sum(weights)
    reference = np.sum(beliefs * weights[:, None], axis=0)
    if not np.isfinite(reference).all():
        raise RuntimeError("Belief reference is non-finite.")
    return reference.astype(np.float32)


def padded_history(frames: list[np.ndarray], end_index: int, history_length: int) -> np.ndarray:
    """Return [history, defenders, features] ending at ``end_index``."""

    if history_length <= 0:
        raise ValueError("history_length must be positive.")
    if not frames or end_index < 0 or end_index >= len(frames):
        raise ValueError("end_index must address an available observation frame.")
    selected = frames[max(0, end_index - history_length + 1) : end_index + 1]
    selected = [np.asarray(frame, dtype=np.float32) for frame in selected]
    first = selected[0]
    if first.ndim != 2 or first.shape[1] <= 0:
        raise ValueError("Observation frames must have shape [defenders, features].")
    if any(frame.shape != first.shape for frame in selected):
        raise ValueError("All observation frames must have identical shapes.")
    selected = [first] * (history_length - len(selected)) + selected
    result = np.stack(selected, axis=0).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise RuntimeError("Padded observation history is non-finite.")
    return result


def build_episode_samples(
    local_frames: list[np.ndarray],
    reference_positions: list[np.ndarray],
    target_positions: list[np.ndarray],
    reference_velocities: list[np.ndarray] | None = None,
    *,
    dt_seconds: float,
    history_length: int,
    horizon_steps: int,
    episode_index: int,
    episode_seed: int,
    target_motion_mode: str,
    world_lower_bounds: np.ndarray | None = None,
    world_upper_bounds: np.ndarray | None = None,
    obstacle_centers_xy: np.ndarray | None = None,
    obstacle_radii: np.ndarray | None = None,
    obstacle_heights: np.ndarray | None = None,
    obstacle_half_extents_xy: np.ndarray | None = None,
    obstacle_shape_codes: np.ndarray | None = None,
) -> PredictionDataset:
    """Build non-overlapping-episode-safe windows without exposing target truth."""

    if len(local_frames) != len(reference_positions) or len(local_frames) != len(target_positions):
        raise ValueError("Observation, reference, and target frame counts must match.")
    if reference_velocities is not None and len(reference_velocities) != len(local_frames):
        raise ValueError("Reference velocity frame counts must match observations.")
    if horizon_steps <= 0:
        raise ValueError("horizon_steps must be positive.")
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be finite and positive.")
    if len(local_frames) <= horizon_steps:
        raise ValueError("An episode must contain more frames than horizon_steps.")

    histories: list[np.ndarray] = []
    future_displacements: list[np.ndarray] = []
    references: list[np.ndarray] = []
    reference_velocity_values: list[np.ndarray] = []
    timesteps: list[int] = []
    for timestep in range(len(local_frames) - horizon_steps):
        reference = np.asarray(reference_positions[timestep], dtype=np.float32)
        velocity = (
            np.zeros(3, dtype=np.float32)
            if reference_velocities is None
            else np.asarray(reference_velocities[timestep], dtype=np.float32)
        )
        target = np.asarray(target_positions, dtype=np.float32)
        if reference.shape != (3,) or velocity.shape != (3,) or target.ndim != 2 or target.shape[1] != 3:
            raise ValueError("Reference and target positions must use three-dimensional coordinates.")
        future = target[timestep + 1 : timestep + 1 + horizon_steps] - reference[None, :]
        histories.append(padded_history(local_frames, timestep, history_length))
        future_displacements.append(future.astype(np.float32, copy=False))
        references.append(reference)
        reference_velocity_values.append(velocity)
        timesteps.append(timestep)

    sample_count = len(histories)
    defender_count, feature_dim = histories[0].shape[1:]
    geometry_fields = (
        world_lower_bounds,
        world_upper_bounds,
        obstacle_centers_xy,
        obstacle_radii,
        obstacle_heights,
        obstacle_half_extents_xy,
        obstacle_shape_codes,
    )
    if any(value is None for value in geometry_fields) and any(value is not None for value in geometry_fields):
        raise ValueError("Geometry context must include every world and obstacle field or none of them.")
    if all(value is not None for value in geometry_fields):
        lower = np.asarray(world_lower_bounds, dtype=np.float32)
        upper = np.asarray(world_upper_bounds, dtype=np.float32)
        centers = np.asarray(obstacle_centers_xy, dtype=np.float32)
        radii = np.asarray(obstacle_radii, dtype=np.float32)
        heights = np.asarray(obstacle_heights, dtype=np.float32)
        half_extents = np.asarray(obstacle_half_extents_xy, dtype=np.float32)
        shape_codes = np.asarray(obstacle_shape_codes, dtype=np.int8)
        obstacle_count = centers.shape[0]
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or centers.shape != (obstacle_count, 2)
            or radii.shape != (obstacle_count,)
            or heights.shape != (obstacle_count,)
            or half_extents.shape != (obstacle_count, 2)
            or shape_codes.shape != (obstacle_count,)
        ):
            raise ValueError("Geometry context has incompatible world or obstacle shapes.")
        geometry = {
            "world_lower_bounds": np.repeat(lower[None, :], sample_count, axis=0),
            "world_upper_bounds": np.repeat(upper[None, :], sample_count, axis=0),
            "obstacle_centers_xy": np.repeat(centers[None, :, :], sample_count, axis=0),
            "obstacle_radii": np.repeat(radii[None, :], sample_count, axis=0),
            "obstacle_heights": np.repeat(heights[None, :], sample_count, axis=0),
            "obstacle_half_extents_xy": np.repeat(half_extents[None, :, :], sample_count, axis=0),
            "obstacle_shape_codes": np.repeat(shape_codes[None, :], sample_count, axis=0),
        }
    else:
        geometry = {}
    dataset = PredictionDataset(
        history_observations=np.stack(histories).astype(np.float32),
        future_target_displacements=np.stack(future_displacements).astype(np.float32),
        reference_positions=np.stack(references).astype(np.float32),
        reference_velocities=np.stack(reference_velocity_values).astype(np.float32),
        episode_indices=np.full(sample_count, int(episode_index), dtype=np.int64),
        timesteps=np.asarray(timesteps, dtype=np.int64),
        episode_seeds=np.full(sample_count, int(episode_seed), dtype=np.int64),
        target_motion_modes=np.full(sample_count, str(target_motion_mode)),
        dt_seconds=float(dt_seconds),
        **geometry,
    )
    validate_prediction_dataset(dataset)
    if dataset.defender_count != defender_count or dataset.feature_dim != feature_dim:
        raise RuntimeError("Prediction dataset dimensions changed while building samples.")
    return dataset


def concatenate_prediction_datasets(datasets: list[PredictionDataset]) -> PredictionDataset:
    if not datasets:
        raise ValueError("At least one prediction dataset is required.")
    first = datasets[0]
    for dataset in datasets[1:]:
        if (
            dataset.history_length != first.history_length
            or dataset.horizon_steps != first.horizon_steps
            or dataset.defender_count != first.defender_count
            or dataset.feature_dim != first.feature_dim
            or not np.isclose(dataset.dt_seconds, first.dt_seconds)
            or dataset.has_geometry_context != first.has_geometry_context
            or (
                dataset.has_geometry_context
                and dataset.obstacle_centers_xy.shape[1] != first.obstacle_centers_xy.shape[1]
            )
        ):
            raise ValueError("Prediction datasets have incompatible dimensions.")
    merged = PredictionDataset(
        history_observations=np.concatenate([item.history_observations for item in datasets], axis=0),
        future_target_displacements=np.concatenate(
            [item.future_target_displacements for item in datasets], axis=0
        ),
        reference_positions=np.concatenate([item.reference_positions for item in datasets], axis=0),
        reference_velocities=np.concatenate([item.reference_velocities for item in datasets], axis=0),
        episode_indices=np.concatenate([item.episode_indices for item in datasets], axis=0),
        timesteps=np.concatenate([item.timesteps for item in datasets], axis=0),
        episode_seeds=np.concatenate([item.episode_seeds for item in datasets], axis=0),
        target_motion_modes=np.concatenate([item.target_motion_modes for item in datasets], axis=0),
        dt_seconds=first.dt_seconds,
        world_lower_bounds=(
            np.concatenate([item.world_lower_bounds for item in datasets], axis=0)
            if first.has_geometry_context
            else None
        ),
        world_upper_bounds=(
            np.concatenate([item.world_upper_bounds for item in datasets], axis=0)
            if first.has_geometry_context
            else None
        ),
        obstacle_centers_xy=(
            np.concatenate([item.obstacle_centers_xy for item in datasets], axis=0)
            if first.has_geometry_context
            else None
        ),
        obstacle_radii=(
            np.concatenate([item.obstacle_radii for item in datasets], axis=0)
            if first.has_geometry_context
            else None
        ),
        obstacle_heights=(
            np.concatenate([item.obstacle_heights for item in datasets], axis=0)
            if first.has_geometry_context
            else None
        ),
        obstacle_half_extents_xy=(
            np.concatenate([item.obstacle_half_extents_xy for item in datasets], axis=0)
            if first.has_geometry_context
            else None
        ),
        obstacle_shape_codes=(
            np.concatenate([item.obstacle_shape_codes for item in datasets], axis=0)
            if first.has_geometry_context
            else None
        ),
    )
    validate_prediction_dataset(merged)
    return merged


def validate_prediction_dataset(dataset: PredictionDataset) -> None:
    sample_count = dataset.sample_count
    expected_history = dataset.history_observations
    expected_future = dataset.future_target_displacements
    if expected_history.ndim != 4 or expected_history.shape[1] <= 0:
        raise ValueError("history_observations must have shape [samples, history, defenders, features].")
    if expected_future.ndim != 3 or expected_future.shape[0] != sample_count or expected_future.shape[2] != 3:
        raise ValueError("future_target_displacements must have shape [samples, horizon, 3].")
    if dataset.reference_positions.shape != (sample_count, 3):
        raise ValueError("reference_positions must have shape [samples, 3].")
    if dataset.reference_velocities.shape != (sample_count, 3):
        raise ValueError("reference_velocities must have shape [samples, 3].")
    if not np.isfinite(dataset.dt_seconds) or dataset.dt_seconds <= 0.0:
        raise ValueError("dt_seconds must be finite and positive.")
    for values, name in (
        (dataset.episode_indices, "episode_indices"),
        (dataset.timesteps, "timesteps"),
        (dataset.episode_seeds, "episode_seeds"),
        (dataset.target_motion_modes, "target_motion_modes"),
    ):
        if values.shape != (sample_count,):
            raise ValueError(f"{name} must have one value per sample.")
    for values, name in (
        (expected_history, "history_observations"),
        (expected_future, "future_target_displacements"),
        (dataset.reference_positions, "reference_positions"),
        (dataset.reference_velocities, "reference_velocities"),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains non-finite values.")
    geometry_fields = (
        dataset.world_lower_bounds,
        dataset.world_upper_bounds,
        dataset.obstacle_centers_xy,
        dataset.obstacle_radii,
        dataset.obstacle_heights,
        dataset.obstacle_half_extents_xy,
        dataset.obstacle_shape_codes,
    )
    if any(value is None for value in geometry_fields) and any(value is not None for value in geometry_fields):
        raise ValueError("Geometry context must include every world and obstacle field or none of them.")
    if dataset.has_geometry_context:
        lower = np.asarray(dataset.world_lower_bounds)
        upper = np.asarray(dataset.world_upper_bounds)
        centers = np.asarray(dataset.obstacle_centers_xy)
        radii = np.asarray(dataset.obstacle_radii)
        heights = np.asarray(dataset.obstacle_heights)
        half_extents = np.asarray(dataset.obstacle_half_extents_xy)
        shape_codes = np.asarray(dataset.obstacle_shape_codes)
        obstacle_count = centers.shape[1] if centers.ndim == 3 else -1
        if (
            lower.shape != (sample_count, 3)
            or upper.shape != (sample_count, 3)
            or centers.shape != (sample_count, obstacle_count, 2)
            or radii.shape != (sample_count, obstacle_count)
            or heights.shape != (sample_count, obstacle_count)
            or half_extents.shape != (sample_count, obstacle_count, 2)
            or shape_codes.shape != (sample_count, obstacle_count)
        ):
            raise ValueError("Geometry context has incompatible sample shapes.")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("World bounds contain non-finite values.")
        if not np.all(lower < upper):
            raise ValueError("World lower bounds must be strictly smaller than upper bounds.")
        for values, name in (
            (centers, "obstacle_centers_xy"),
            (radii, "obstacle_radii"),
            (heights, "obstacle_heights"),
            (half_extents, "obstacle_half_extents_xy"),
        ):
            if not np.isfinite(values).all():
                raise ValueError(f"{name} contains non-finite values.")
        if not np.isin(shape_codes, (0, 1, 2)).all():
            raise ValueError("obstacle_shape_codes must use cylinder=0, box=1, wall=2.")
        if np.any(radii <= 0.0) or np.any(heights <= 0.0) or np.any(half_extents <= 0.0):
            raise ValueError("Obstacle radii, heights, and half extents must be positive.")


def save_prediction_dataset(dataset: PredictionDataset, path: str) -> None:
    validate_prediction_dataset(dataset)
    np.savez_compressed(path, **dataset.as_dict())


def load_prediction_dataset(path: str) -> PredictionDataset:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "history_observations",
            "future_target_displacements",
            "reference_positions",
            "reference_velocities",
            "episode_indices",
            "timesteps",
            "episode_seeds",
            "target_motion_modes",
            "dt_seconds",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Prediction dataset is missing fields: {', '.join(sorted(missing))}")
        geometry_keys = {
            "world_lower_bounds",
            "world_upper_bounds",
            "obstacle_centers_xy",
            "obstacle_radii",
            "obstacle_heights",
            "obstacle_half_extents_xy",
            "obstacle_shape_codes",
        }
        present_geometry_keys = geometry_keys.intersection(archive.files)
        if present_geometry_keys and present_geometry_keys != geometry_keys:
            missing_geometry = geometry_keys.difference(archive.files)
            raise ValueError(f"Prediction dataset has incomplete geometry context: {', '.join(sorted(missing_geometry))}")
        geometry: dict[str, np.ndarray | None]
        if present_geometry_keys:
            geometry = {
                "world_lower_bounds": np.asarray(archive["world_lower_bounds"], dtype=np.float32),
                "world_upper_bounds": np.asarray(archive["world_upper_bounds"], dtype=np.float32),
                "obstacle_centers_xy": np.asarray(archive["obstacle_centers_xy"], dtype=np.float32),
                "obstacle_radii": np.asarray(archive["obstacle_radii"], dtype=np.float32),
                "obstacle_heights": np.asarray(archive["obstacle_heights"], dtype=np.float32),
                "obstacle_half_extents_xy": np.asarray(archive["obstacle_half_extents_xy"], dtype=np.float32),
                "obstacle_shape_codes": np.asarray(archive["obstacle_shape_codes"], dtype=np.int8),
            }
        else:
            geometry = {key: None for key in geometry_keys}
        dataset = PredictionDataset(
            history_observations=np.asarray(archive["history_observations"], dtype=np.float32),
            future_target_displacements=np.asarray(archive["future_target_displacements"], dtype=np.float32),
            reference_positions=np.asarray(archive["reference_positions"], dtype=np.float32),
            reference_velocities=np.asarray(archive["reference_velocities"], dtype=np.float32),
            episode_indices=np.asarray(archive["episode_indices"], dtype=np.int64),
            timesteps=np.asarray(archive["timesteps"], dtype=np.int64),
            episode_seeds=np.asarray(archive["episode_seeds"], dtype=np.int64),
            target_motion_modes=np.asarray(archive["target_motion_modes"], dtype=str),
            dt_seconds=float(np.asarray(archive["dt_seconds"]).item()),
            **geometry,
        )
    validate_prediction_dataset(dataset)
    return dataset
