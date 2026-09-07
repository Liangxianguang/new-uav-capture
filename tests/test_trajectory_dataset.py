from __future__ import annotations

import numpy as np
import pytest

from encirclement3d.trajectory_dataset import (
    build_episode_samples,
    load_prediction_dataset,
    padded_history,
    save_prediction_dataset,
)


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
    targets = [np.array([10.0 + index, 1.0, 2.0], dtype=np.float32) for index in range(6)]

    dataset = build_episode_samples(
        frames,
        references,
        targets,
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
    dataset = build_episode_samples(
        frames,
        references,
        targets,
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
                dataset.episode_indices,
                dataset.timesteps,
                dataset.episode_seeds,
                dataset.target_motion_modes,
            )
        )
