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

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
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
        )
    validate_prediction_dataset(dataset)
    return dataset
