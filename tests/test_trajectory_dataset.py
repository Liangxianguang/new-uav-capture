from __future__ import annotations

import numpy as np
import pytest

from encirclement3d.trajectory_dataset import (
    build_episode_samples,
    concatenate_prediction_datasets,
    load_prediction_dataset,
    padded_history,
    save_prediction_dataset,
)


def _geometry() -> dict[str, np.ndarray]:
    return {
        "world_lower_bounds": np.array([-3.0, -3.0, 0.0], dtype=np.float32),
        "world_upper_bounds": np.array([3.0, 3.0, 3.0], dtype=np.float32),
        "obstacle_centers_xy": np.array([[0.0, 0.0]], dtype=np.float32),
        "obstacle_radii": np.array([0.5], dtype=np.float32),
        "obstacle_heights": np.array([2.0], dtype=np.float32),
        "obstacle_half_extents_xy": np.array([[0.5, 0.75]], dtype=np.float32),
        "obstacle_shape_codes": np.array([0], dtype=np.int8),
    }


def test_padded_history_has_fixed_shape_and_repeats_first_frame() -> None:
    frames = [
        np.zeros((4, 3), dtype=np.float32),
        np.ones((4, 3), dtype=np.float32),
    ]

    history = padded_history(frames, end_index=1, history_length=4)

    assert history.shape == (4, 4, 3)
    np.testing.assert_allclose(history[0], frames[0])
    np.testing.assert_allclose(history[1], frames[0])
    np.testing.assert_allclose(history[2], frames[0])
    np.testing.assert_allclose(history[3], frames[1])


def test_episode_samples_use_future_truth_only_as_labels() -> None:
    frames = [np.full((4, 2), float(index), dtype=np.float32) for index in range(6)]
    references = [np.array([float(index), 0.0, 0.0], dtype=np.float32) for index in range(6)]
    reference_velocities = [np.array([0.5, 0.0, 0.0], dtype=np.float32) for _ in range(6)]
    targets = [np.array([10.0 + index, 1.0, 2.0], dtype=np.float32) for index in range(6)]

    dataset = build_episode_samples(
        frames,
        references,
        targets,
        reference_velocities,
        dt_seconds=0.1,
        history_length=3,
        horizon_steps=2,
        episode_index=7,
        episode_seed=123,
        target_motion_mode="s_curve",
    )

    assert dataset.history_observations.shape == (4, 3, 4, 2)
    assert dataset.future_target_displacements.shape == (4, 2, 3)
    np.testing.assert_allclose(dataset.future_target_displacements[0], [[11.0, 1.0, 2.0], [12.0, 1.0, 2.0]])
    np.testing.assert_array_equal(dataset.episode_indices, 7)
    np.testing.assert_array_equal(dataset.episode_seeds, 123)


def test_prediction_dataset_round_trip_rejects_non_finite_values(tmp_path) -> None:
    frames = [np.zeros((4, 2), dtype=np.float32) for _ in range(4)]
    references = [np.zeros(3, dtype=np.float32) for _ in range(4)]
    targets = [np.ones(3, dtype=np.float32) for _ in range(4)]
    reference_velocities = [np.zeros(3, dtype=np.float32) for _ in range(4)]
    dataset = build_episode_samples(
        frames,
        references,
        targets,
        reference_velocities,
        dt_seconds=0.1,
        history_length=2,
        horizon_steps=1,
        episode_index=0,
        episode_seed=10,
        target_motion_mode="flee_persistence",
    )
    path = tmp_path / "dataset.npz"
    save_prediction_dataset(dataset, str(path))
    loaded = load_prediction_dataset(str(path))
    np.testing.assert_allclose(loaded.history_observations, dataset.history_observations)
    np.testing.assert_allclose(loaded.future_target_displacements, dataset.future_target_displacements)

    invalid = dataset.history_observations.copy()
    invalid[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        from encirclement3d.trajectory_dataset import PredictionDataset, validate_prediction_dataset

        validate_prediction_dataset(
            PredictionDataset(
                invalid,
                dataset.future_target_displacements,
                dataset.reference_positions,
                dataset.reference_velocities,
                dataset.episode_indices,
                dataset.timesteps,
                dataset.episode_seeds,
                dataset.target_motion_modes,
                dataset.dt_seconds,
            )
        )


def test_geometry_context_round_trips_and_validates_bounds_and_obstacle_sizes(tmp_path) -> None:
    frames = [np.zeros((4, 2), dtype=np.float32) for _ in range(4)]
    references = [np.zeros(3, dtype=np.float32) for _ in range(4)]
    targets = [np.ones(3, dtype=np.float32) for _ in range(4)]
    velocities = [np.zeros(3, dtype=np.float32) for _ in range(4)]
    dataset = build_episode_samples(
        frames,
        references,
        targets,
        velocities,
        dt_seconds=0.1,
        history_length=2,
        horizon_steps=1,
        episode_index=0,
        episode_seed=10,
        target_motion_mode="flee_persistence",
        **_geometry(),
    )
    path = tmp_path / "geometry_dataset.npz"
    save_prediction_dataset(dataset, str(path))
    loaded = load_prediction_dataset(str(path))
    assert loaded.has_geometry_context
    np.testing.assert_allclose(loaded.world_lower_bounds[0], _geometry()["world_lower_bounds"])
    np.testing.assert_array_equal(loaded.obstacle_shape_codes[:, 0], 0)

    invalid_bounds = _geometry()
    invalid_bounds["world_upper_bounds"] = invalid_bounds["world_lower_bounds"].copy()
    with pytest.raises(ValueError, match="strictly smaller"):
        build_episode_samples(
            frames,
            references,
            targets,
            velocities,
            dt_seconds=0.1,
            history_length=2,
            horizon_steps=1,
            episode_index=0,
            episode_seed=10,
            target_motion_mode="flee_persistence",
            **invalid_bounds,
        )

    invalid_obstacle = _geometry()
    invalid_obstacle["obstacle_radii"] = np.array([0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="must be positive"):
        build_episode_samples(
            frames,
            references,
            targets,
            velocities,
            dt_seconds=0.1,
            history_length=2,
            horizon_steps=1,
            episode_index=0,
            episode_seed=10,
            target_motion_mode="flee_persistence",
            **invalid_obstacle,
        )


def test_geometry_datasets_concatenate_and_retain_target_modes() -> None:
    frames = [np.zeros((4, 2), dtype=np.float32) for _ in range(4)]
    references = [np.zeros(3, dtype=np.float32) for _ in range(4)]
    velocities = [np.zeros(3, dtype=np.float32) for _ in range(4)]
    first = build_episode_samples(
        frames,
        references,
        [np.array([1.0, 0.0, 0.0], dtype=np.float32) for _ in frames],
        velocities,
        dt_seconds=0.1,
        history_length=2,
        horizon_steps=1,
        episode_index=0,
        episode_seed=10,
        target_motion_mode="flee_persistence",
        **_geometry(),
    )
    second = build_episode_samples(
        frames,
        references,
        [np.array([2.0, 0.0, 0.0], dtype=np.float32) for _ in frames],
        velocities,
        dt_seconds=0.1,
        history_length=2,
        horizon_steps=1,
        episode_index=1,
        episode_seed=11,
        target_motion_mode="s_curve",
        **_geometry(),
    )
    merged = concatenate_prediction_datasets([first, second])
    assert merged.has_geometry_context
    assert merged.sample_count == first.sample_count + second.sample_count
    assert set(merged.target_motion_modes.tolist()) == {"flee_persistence", "s_curve"}
    np.testing.assert_allclose(merged.future_target_displacements[-1, 0], [2.0, 0.0, 0.0])
