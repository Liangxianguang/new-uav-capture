from __future__ import annotations

import torch

from encirclement3d.prediction import (
    CandidateTrajectorySet,
    ConditionalDiffusionTrajectoryPredictor,
    DiagonalSSMEncoder,
    TrajectoryNormalizer,
    assess_candidate_feasibility,
    prediction_metrics,
)


def test_diagonal_ssm_is_linear_time_shape_preserving() -> None:
    model = DiagonalSSMEncoder(input_dim=12, hidden_dim=16, num_layers=2)
    output = model(torch.randn(3, 7, 12))
    assert output.shape == (3, 7, 16)
    assert torch.isfinite(output).all()


def test_diffusion_predictor_trains_and_samples_candidate_trajectories() -> None:
    model = ConditionalDiffusionTrajectoryPredictor(
        input_dim=12,
        horizon_count=5,
        hidden_dim=16,
        num_layers=2,
        diffusion_steps=12,
    )
    inputs = torch.randn(3, 7, 12)
    targets = torch.randn(3, 5, 3)
    loss = model.diffusion_loss(inputs, targets)
    assert torch.isfinite(loss)
    loss.backward()
    candidates = model.sample(inputs, num_samples=4, sampling_steps=4)
    assert candidates.shape == (3, 4, 5, 3)
    assert torch.isfinite(candidates).all()


def test_prediction_metrics_report_best_of_k() -> None:
    targets = torch.zeros(2, 3, 3)
    candidates = torch.ones(2, 2, 3, 3)
    candidates[:, 1] = 0.0
    metrics = prediction_metrics(candidates, targets)
    assert metrics["min_ade"] == 0.0
    assert metrics["min_fde"] == 0.0
    assert metrics["candidate_spread"] > 0.0


def test_trajectory_normalizer_round_trips_train_targets_and_labels_legacy_scale() -> None:
    targets = torch.tensor(
        [
            [[-2.0, 1.0, 3.0], [0.0, 2.0, 4.0]],
            [[2.0, 3.0, 5.0], [4.0, 4.0, 6.0]],
        ]
    )
    normalizer = TrajectoryNormalizer.fit(targets.numpy())
    torch.testing.assert_close(normalizer.denormalize(normalizer.normalize(targets)), targets)

    legacy = TrajectoryNormalizer.fixed(horizon_count=2, scale=10.0)
    assert legacy.kind == "fixed_scalar_scale_legacy_compatibility"
    torch.testing.assert_close(legacy.denormalize(legacy.normalize(targets)), targets)


def test_diffusion_sampling_and_candidate_scores_are_deterministic_on_cpu() -> None:
    model = ConditionalDiffusionTrajectoryPredictor(
        input_dim=6,
        horizon_count=3,
        hidden_dim=8,
        num_layers=1,
        diffusion_steps=8,
    )
    inputs = torch.randn(2, 4, 6)
    first = model.sample(inputs, num_samples=3, sampling_steps=3, generator=torch.Generator().manual_seed(4))
    second = model.sample(inputs, num_samples=3, sampling_steps=3, generator=torch.Generator().manual_seed(4))
    torch.testing.assert_close(first, second)

    candidate_set = model.sample_set(
        inputs,
        num_samples=3,
        sampling_steps=3,
        generator=torch.Generator().manual_seed(4),
    )
    assert candidate_set.score_kind == "uniform_uncalibrated"
    torch.testing.assert_close(candidate_set.relative_weights, torch.full((2, 3), 1.0 / 3.0))


def test_candidate_trajectory_set_rejects_incompatible_scores() -> None:
    with torch.no_grad():
        trajectories = torch.zeros(1, 2, 3, 3)
        try:
            CandidateTrajectorySet(trajectories, torch.zeros(1, 3), "uniform_uncalibrated")
        except ValueError as error:
            assert "logits" in str(error)
        else:
            raise AssertionError("CandidateTrajectorySet accepted incompatible logits.")


def _feasibility_inputs(candidates: torch.Tensor, *, max_speed: float = 10.0, max_acceleration: float | None = 10.0):
    return assess_candidate_feasibility(
        candidates,
        reference_positions=torch.zeros(1, 3),
        reference_velocities=torch.zeros(1, 3),
        dt_seconds=1.0,
        lower_bounds=torch.full((1, 3), -2.0),
        upper_bounds=torch.full((1, 3), 2.0),
        max_speed=max_speed,
        max_acceleration=max_acceleration,
    )


def test_candidate_feasibility_detects_bounds_speed_and_acceleration() -> None:
    in_bounds = _feasibility_inputs(torch.tensor([[[[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]]]))
    assert bool(in_bounds.feasible.item())

    out_of_bounds = _feasibility_inputs(torch.tensor([[[[3.0, 0.0, 0.0]]]]))
    assert not bool(out_of_bounds.within_bounds.item())

    too_fast = _feasibility_inputs(torch.tensor([[[[1.5, 0.0, 0.0]]]]), max_speed=1.0)
    assert not bool(too_fast.speed_feasible.item())

    accelerating = assess_candidate_feasibility(
        torch.tensor([[[[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]]]]),
        reference_positions=torch.zeros(1, 3),
        reference_velocities=torch.tensor([[0.5, 0.0, 0.0]]),
        dt_seconds=1.0,
        lower_bounds=torch.full((1, 3), -2.0),
        upper_bounds=torch.full((1, 3), 2.0),
        max_speed=2.0,
        max_acceleration=0.2,
    )
    assert bool(accelerating.speed_feasible.item())
    assert not bool(accelerating.acceleration_feasible.item())


def test_candidate_feasibility_detects_cylinder_box_and_wall_collisions() -> None:
    for shape_code in (0, 1, 2):
        feasibility = assess_candidate_feasibility(
            torch.tensor([[[[0.0, 0.0, 0.5]]]]),
            reference_positions=torch.zeros(1, 3),
            reference_velocities=torch.zeros(1, 3),
            dt_seconds=1.0,
            lower_bounds=torch.full((1, 3), -2.0),
            upper_bounds=torch.full((1, 3), 2.0),
            max_speed=2.0,
            max_acceleration=2.0,
            obstacle_centers_xy=torch.zeros(1, 1, 2),
            obstacle_radii=torch.ones(1, 1),
            obstacle_heights=torch.ones(1, 1),
            obstacle_half_extents_xy=torch.ones(1, 1, 2),
            obstacle_shape_codes=torch.tensor([[shape_code]], dtype=torch.int8),
            minimum_obstacle_clearance=0.0,
        )
        assert feasibility.obstacle_checked
        assert not bool(feasibility.obstacle_clear.item())
