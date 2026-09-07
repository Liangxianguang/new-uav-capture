from __future__ import annotations

import torch

from encirclement3d.prediction import (
    ConditionalDiffusionTrajectoryPredictor,
    DiagonalSSMEncoder,
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
