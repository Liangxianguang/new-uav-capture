"""Supervised local-history target trajectory predictors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class TrajectoryNormalizer:
    """Train-split-only affine normalization for trajectory coordinates."""

    center: np.ndarray
    scale: np.ndarray
    kind: str = "per_horizon_coordinate_train_split_standardization"

    @classmethod
    def fit(cls, targets: np.ndarray, minimum_scale: float = 1e-3) -> "TrajectoryNormalizer":
        values = np.asarray(targets, dtype=np.float32)
        if values.ndim != 3 or values.shape[-1] != 3:
            raise ValueError("targets must have shape [samples, horizon, 3].")
        if not np.isfinite(values).all() or minimum_scale <= 0.0:
            raise ValueError("targets must be finite and minimum_scale must be positive.")
        center = values.mean(axis=0, keepdims=True)
        scale = np.maximum(values.std(axis=0, keepdims=True), minimum_scale)
        return cls(center.astype(np.float32), scale.astype(np.float32))

    @classmethod
    def fixed(cls, horizon_count: int, scale: float) -> "TrajectoryNormalizer":
        if horizon_count <= 0 or not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("horizon_count and scale must be positive.")
        return cls(
            np.zeros((1, horizon_count, 3), dtype=np.float32),
            np.full((1, horizon_count, 3), scale, dtype=np.float32),
            kind="fixed_scalar_scale_legacy_compatibility",
        )

    def _tensors(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if values.ndim < 2 or values.shape[-2:] != self.center.shape[-2:]:
            raise ValueError("Trajectory values have an incompatible horizon or coordinate shape.")
        center = torch.as_tensor(self.center, device=values.device, dtype=values.dtype)
        scale = torch.as_tensor(self.scale, device=values.device, dtype=values.dtype)
        return center, scale

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        center, scale = self._tensors(values)
        return (values - center) / scale

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        center, scale = self._tensors(values)
        return values * scale + center

    def variance_to_physical(self, log_variance: torch.Tensor) -> torch.Tensor:
        _center, scale = self._tensors(log_variance)
        return log_variance + 2.0 * torch.log(scale)

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
        }


@dataclass(frozen=True)
class CandidateTrajectorySet:
    """Stable planner-facing candidate contract.

    ``logits`` are deliberately uniform until a calibrated scorer is trained.
    They provide a fixed interface without misrepresenting uncalibrated
    diffusion samples as probabilistic mode estimates.
    """

    trajectories: torch.Tensor
    logits: torch.Tensor
    score_kind: str

    def __post_init__(self) -> None:
        if self.trajectories.ndim != 4 or self.trajectories.shape[-1] != 3:
            raise ValueError("trajectories must have shape [batch, candidates, horizon, 3].")
        if self.logits.shape != self.trajectories.shape[:2]:
            raise ValueError("logits must have shape [batch, candidates].")
        if not torch.isfinite(self.trajectories).all() or not torch.isfinite(self.logits).all():
            raise ValueError("Candidate trajectories and logits must be finite.")

    @classmethod
    def uniform(cls, trajectories: torch.Tensor) -> "CandidateTrajectorySet":
        return cls(
            trajectories=trajectories,
            logits=torch.zeros(trajectories.shape[:2], dtype=trajectories.dtype, device=trajectories.device),
            score_kind="uniform_uncalibrated",
        )

    @property
    def relative_weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=1)


@dataclass(frozen=True)
class CandidateFeasibility:
    """Per-candidate geometric and kinematic feasibility diagnostics."""

    finite: torch.Tensor
    within_bounds: torch.Tensor
    speed_feasible: torch.Tensor
    acceleration_feasible: torch.Tensor
    obstacle_clear: torch.Tensor
    obstacle_checked: bool
    acceleration_checked: bool

    @property
    def feasible(self) -> torch.Tensor:
        return (
            self.finite
            & self.within_bounds
            & self.speed_feasible
            & self.acceleration_feasible
            & self.obstacle_clear
        )

    def metrics(self) -> dict[str, float]:
        return {
            "candidate_finite_fraction": float(self.finite.float().mean().cpu()),
            "candidate_within_bounds_fraction": float(self.within_bounds.float().mean().cpu()),
            "candidate_speed_feasible_fraction": float(self.speed_feasible.float().mean().cpu()),
            "candidate_acceleration_feasible_fraction": float(
                self.acceleration_feasible.float().mean().cpu()
            ),
            "candidate_obstacle_clear_fraction": float(self.obstacle_clear.float().mean().cpu()),
            "candidate_feasible_fraction": float(self.feasible.float().mean().cpu()),
            "candidate_obstacle_checked": float(self.obstacle_checked),
            "candidate_acceleration_checked": float(self.acceleration_checked),
        }


def assess_candidate_feasibility(
    candidates: torch.Tensor,
    reference_positions: torch.Tensor,
    reference_velocities: torch.Tensor,
    dt_seconds: float,
    lower_bounds: torch.Tensor,
    upper_bounds: torch.Tensor,
    max_speed: float,
    max_acceleration: float | None = None,
    obstacle_centers_xy: torch.Tensor | None = None,
    obstacle_radii: torch.Tensor | None = None,
    obstacle_heights: torch.Tensor | None = None,
    obstacle_half_extents_xy: torch.Tensor | None = None,
    obstacle_shape_codes: torch.Tensor | None = None,
    minimum_obstacle_clearance: float = 0.0,
) -> CandidateFeasibility:
    """Check candidate position, speed, acceleration, and obstacle constraints.

    Candidates are target displacements from each sample's published team-belief
    reference. Obstacle codes use cylinder=0, box=1, and wall=2, matching the
    archived prediction-dataset contract.
    """

    if candidates.ndim != 4 or candidates.shape[-1] != 3:
        raise ValueError("candidates must have shape [batch, candidates, horizon, 3].")
    batch_size, candidate_count, _horizon, _coordinates = candidates.shape
    if reference_positions.shape != (batch_size, 3) or reference_velocities.shape != (batch_size, 3):
        raise ValueError("Reference positions and velocities must have shape [batch, 3].")
    if lower_bounds.shape != (batch_size, 3) or upper_bounds.shape != (batch_size, 3):
        raise ValueError("World bounds must have shape [batch, 3].")
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0 or max_speed <= 0.0:
        raise ValueError("dt_seconds and max_speed must be positive and finite.")
    if max_acceleration is not None and (not np.isfinite(max_acceleration) or max_acceleration <= 0.0):
        raise ValueError("max_acceleration must be positive and finite when supplied.")
    if minimum_obstacle_clearance < 0.0:
        raise ValueError("minimum_obstacle_clearance must be non-negative.")

    finite = torch.isfinite(candidates).all(dim=(2, 3))
    positions = candidates + reference_positions[:, None, None, :]
    within_bounds = (
        (positions >= lower_bounds[:, None, None, :])
        & (positions <= upper_bounds[:, None, None, :])
    ).all(dim=(2, 3))
    previous_positions = torch.cat(
        [
            reference_positions[:, None, None, :].expand(-1, candidate_count, -1, -1),
            positions[:, :, :-1, :],
        ],
        dim=2,
    )
    velocities = (positions - previous_positions) / float(dt_seconds)
    speed_feasible = (torch.linalg.vector_norm(velocities, dim=-1) <= max_speed).all(dim=2)

    if max_acceleration is None:
        acceleration_feasible = torch.ones_like(speed_feasible)
        acceleration_checked = False
    else:
        previous_velocities = torch.cat(
            [
                reference_velocities[:, None, None, :].expand(-1, candidate_count, -1, -1),
                velocities[:, :, :-1, :],
            ],
            dim=2,
        )
        accelerations = (velocities - previous_velocities) / float(dt_seconds)
        acceleration_feasible = (
            torch.linalg.vector_norm(accelerations, dim=-1) <= max_acceleration
        ).all(dim=2)
        acceleration_checked = True

    obstacle_fields = (
        obstacle_centers_xy,
        obstacle_radii,
        obstacle_heights,
        obstacle_half_extents_xy,
        obstacle_shape_codes,
    )
    if any(value is None for value in obstacle_fields) and any(value is not None for value in obstacle_fields):
        raise ValueError("Obstacle context must include every obstacle field or none of them.")
    if all(value is None for value in obstacle_fields):
        obstacle_clear = torch.ones_like(speed_feasible)
        obstacle_checked = False
    else:
        assert obstacle_centers_xy is not None
        assert obstacle_radii is not None
        assert obstacle_heights is not None
        assert obstacle_half_extents_xy is not None
        assert obstacle_shape_codes is not None
        obstacle_count = obstacle_centers_xy.shape[1]
        if (
            obstacle_centers_xy.shape != (batch_size, obstacle_count, 2)
            or obstacle_radii.shape != (batch_size, obstacle_count)
            or obstacle_heights.shape != (batch_size, obstacle_count)
            or obstacle_half_extents_xy.shape != (batch_size, obstacle_count, 2)
            or obstacle_shape_codes.shape != (batch_size, obstacle_count)
        ):
            raise ValueError("Obstacle context has incompatible shapes.")
        obstacle_clear = torch.ones_like(speed_feasible)
        for obstacle_index in range(obstacle_count):
            center_xy = obstacle_centers_xy[:, obstacle_index, None, None, :]
            radius = obstacle_radii[:, obstacle_index, None, None]
            height = obstacle_heights[:, obstacle_index, None, None]
            half_xy = obstacle_half_extents_xy[:, obstacle_index, None, None, :]
            radial_gap = torch.linalg.vector_norm(positions[..., :2] - center_xy, dim=-1) - radius
            below = (-positions[..., 2]).clamp_min(0.0)
            above = (positions[..., 2] - height).clamp_min(0.0)
            vertical_gap = below + above
            cylinder_clearance = torch.where(
                vertical_gap == 0.0,
                radial_gap,
                torch.where(
                    radial_gap <= 0.0,
                    vertical_gap,
                    torch.hypot(radial_gap, vertical_gap),
                ),
            )
            box_center = torch.cat(
                [center_xy, height[..., None] * 0.5],
                dim=-1,
            )
            box_half_extent = torch.cat(
                [half_xy, height[..., None] * 0.5],
                dim=-1,
            )
            signed_distance = torch.abs(positions - box_center) - box_half_extent
            outside = signed_distance.clamp_min(0.0)
            outside_norm = torch.linalg.vector_norm(outside, dim=-1)
            box_clearance = torch.where(
                outside_norm > 0.0,
                outside_norm,
                -(-signed_distance).amin(dim=-1),
            )
            is_cylinder = obstacle_shape_codes[:, obstacle_index, None, None] == 0
            clearance = torch.where(is_cylinder, cylinder_clearance, box_clearance)
            obstacle_clear &= (clearance >= minimum_obstacle_clearance).all(dim=2)
        obstacle_checked = True
    return CandidateFeasibility(
        finite=finite,
        within_bounds=within_bounds,
        speed_feasible=speed_feasible,
        acceleration_feasible=acceleration_feasible,
        obstacle_clear=obstacle_clear,
        obstacle_checked=obstacle_checked,
        acceleration_checked=acceleration_checked,
    )


def project_candidate_trajectories(
    candidates: torch.Tensor,
    reference_positions: torch.Tensor,
    reference_velocities: torch.Tensor,
    dt_seconds: float,
    max_speed: float,
    max_acceleration: float | None = None,
    lower_bounds: torch.Tensor | None = None,
    upper_bounds: torch.Tensor | None = None,
    iterations: int = 4,
) -> torch.Tensor:
    """Project position candidates onto a bounded velocity/acceleration rollout.

    The projection is a lightweight runtime contract for a downstream planner:
    each point is reconstructed from a velocity state, then alternately clipped
    to the speed ball, acceleration ball, and world-bound velocity box. It does
    not claim obstacle avoidance or replace the safety filter.
    """

    if candidates.ndim != 4 or candidates.shape[-1] != 3:
        raise ValueError("candidates must have shape [batch, candidates, horizon, 3].")
    batch_size, candidate_count, _horizon, _coordinates = candidates.shape
    if reference_positions.shape != (batch_size, 3) or reference_velocities.shape != (batch_size, 3):
        raise ValueError("Reference positions and velocities must have shape [batch, 3].")
    if not np.isfinite(dt_seconds) or dt_seconds <= 0.0 or not np.isfinite(max_speed) or max_speed <= 0.0:
        raise ValueError("dt_seconds and max_speed must be finite and positive.")
    if max_acceleration is not None and (not np.isfinite(max_acceleration) or max_acceleration <= 0.0):
        raise ValueError("max_acceleration must be finite and positive when supplied.")
    if (lower_bounds is None) != (upper_bounds is None):
        raise ValueError("lower_bounds and upper_bounds must be supplied together.")
    if lower_bounds is not None and upper_bounds is not None:
        if lower_bounds.shape != (batch_size, 3) or upper_bounds.shape != (batch_size, 3):
            raise ValueError("World bounds must have shape [batch, 3].")
        if not torch.all(lower_bounds < upper_bounds):
            raise ValueError("lower_bounds must be strictly smaller than upper_bounds.")
    if iterations <= 0:
        raise ValueError("iterations must be positive.")
    if not torch.isfinite(candidates).all():
        raise ValueError("candidates must be finite.")

    previous_position = reference_positions[:, None, :].expand(-1, candidate_count, -1).clone()
    previous_velocity = reference_velocities[:, None, :].expand(-1, candidate_count, -1).clone()
    projected: list[torch.Tensor] = []
    # Leave a small margin for the strict <= checks used by the independent
    # feasibility auditor and for floating-point roundoff at the boundary.
    acceleration_delta = (
        None if max_acceleration is None else 0.999 * float(max_acceleration) * float(dt_seconds)
    )

    for timestep in range(candidates.shape[2]):
        desired_position = candidates[:, :, timestep, :] + reference_positions[:, None, :]
        velocity = (desired_position - previous_position) / float(dt_seconds)
        for _ in range(iterations):
            speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
            velocity = velocity * torch.clamp(float(max_speed) / speed.clamp_min(1e-9), max=1.0)
            if acceleration_delta is not None:
                delta = velocity - previous_velocity
                delta_norm = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
                velocity = previous_velocity + delta * torch.clamp(
                    float(acceleration_delta) / delta_norm.clamp_min(1e-9), max=1.0
                )
            if lower_bounds is not None and upper_bounds is not None:
                lower_velocity = (lower_bounds[:, None, :] - previous_position) / float(dt_seconds)
                upper_velocity = (upper_bounds[:, None, :] - previous_position) / float(dt_seconds)
                velocity = torch.minimum(torch.maximum(velocity, lower_velocity), upper_velocity)
        position = previous_position + velocity * float(dt_seconds)
        if lower_bounds is not None and upper_bounds is not None:
            position = torch.minimum(torch.maximum(position, lower_bounds[:, None, :]), upper_bounds[:, None, :])
            velocity = (position - previous_position) / float(dt_seconds)
        projected.append(position - reference_positions[:, None, :])
        previous_position = position
        previous_velocity = velocity
    result = torch.stack(projected, dim=2)
    if not torch.isfinite(result).all():
        raise RuntimeError("Candidate trajectory projection emitted non-finite values.")
    return result


class HistoryTargetPredictor(nn.Module):
    """GRU predictor that maps local observation history to target means/uncertainty."""

    def __init__(
        self,
        input_dim: int,
        horizon_count: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or horizon_count <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("Predictor dimensions must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.input_dim = int(input_dim)
        self.horizon_count = int(horizon_count)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.encoder = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.horizon_count * 6),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected [batch, history, {self.input_dim}] inputs, got {tuple(inputs.shape)}."
            )
        encoded, _hidden = self.encoder(inputs)
        output = self.head(encoded[:, -1])
        output = output.view(inputs.shape[0], self.horizon_count, 6)
        mean = output[..., :3]
        log_variance = torch.clamp(output[..., 3:], min=-8.0, max=5.0)
        return mean, log_variance


def gaussian_nll(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if mean.shape != log_variance.shape or mean.shape != target.shape:
        raise ValueError("mean, log_variance, and target must have identical shapes.")
    variance = torch.exp(log_variance)
    return 0.5 * (log_variance + (target - mean).square() / variance).mean()


def deterministic_mse(mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mean.shape != target.shape:
        raise ValueError("mean and target must have identical shapes.")
    return (mean - target).square().mean()


class LearnedPredictionObserver:
    """Replace an environment's constant-velocity feature block with GRU output.

    The adapter keeps the actor input width unchanged. It maintains a short
    history separately for each episode, runs the frozen predictor without
    gradients, and replaces only the four prediction columns (relative xyz
    plus one uncertainty scalar). All other local observation fields and the
    centralized critic state remain unchanged.
    """

    def __init__(
        self,
        env: Any,
        predictor: HistoryTargetPredictor,
        device: torch.device,
        history_length: int,
        horizon_index: int,
    ) -> None:
        if history_length <= 0:
            raise ValueError("history_length must be positive.")
        if horizon_index < 0 or horizon_index >= predictor.horizon_count:
            raise ValueError("horizon_index is outside the predictor output range.")
        self.env = env
        self.predictor = predictor.to(device).eval()
        self.device = device
        self.history_length = int(history_length)
        self.horizon_index = int(horizon_index)
        self._history: list[np.ndarray] = []
        self.last_prediction_mean: np.ndarray | None = None
        self.last_prediction_std: np.ndarray | None = None

    @classmethod
    def from_checkpoint(
        cls,
        env: Any,
        checkpoint_path: Path,
        device: torch.device,
        history_length: int,
        horizon_index: int,
    ) -> "LearnedPredictionObserver":
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model_config = checkpoint.get("model")
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(model_config, dict) or not isinstance(state_dict, dict):
            raise ValueError("Prediction checkpoint must contain model and model_state_dict.")
        predictor = HistoryTargetPredictor(**model_config)
        predictor.load_state_dict(state_dict, strict=True)
        return cls(env, predictor, device, history_length, horizon_index)

    def reset(self, observation: dict[str, Any]) -> np.ndarray:
        self._history = []
        self.last_prediction_mean = None
        self.last_prediction_std = None
        return self.observe(observation)

    def observe(self, observation: dict[str, Any]) -> np.ndarray:
        base = np.asarray(self.env.policy_observations(observation), dtype=np.float32)
        feature_slice = self.env.prediction_feature_slice()
        if base.shape[-1] < feature_slice.stop:
            raise ValueError("Environment observation is shorter than its prediction feature block.")
        if base.shape[-1] != self.predictor.input_dim:
            raise ValueError(
                "Prediction checkpoint input dimension "
                f"{self.predictor.input_dim} does not match environment base dimension {base.shape[-1]}."
            )
        self._history.append(base.copy())
        if len(self._history) > self.history_length:
            self._history.pop(0)
        padded = [self._history[0]] * (self.history_length - len(self._history)) + self._history
        # [history, defenders, features] -> [defenders, history, features].
        window = np.stack(padded, axis=0)
        window = np.transpose(window, (1, 0, 2)).copy()
        with torch.no_grad():
            mean, log_variance = self.predictor(torch.as_tensor(window, device=self.device))
        selected_mean = mean[:, self.horizon_index].detach().cpu().numpy().astype(np.float32)
        selected_std = torch.exp(0.5 * log_variance[:, self.horizon_index]).detach().cpu().numpy().astype(np.float32)
        uncertainty = np.mean(selected_std, axis=1, keepdims=True)
        augmented = base.copy()
        augmented[:, feature_slice] = np.concatenate([selected_mean, uncertainty], axis=1)
        self.last_prediction_mean = selected_mean
        self.last_prediction_std = selected_std
        if not np.isfinite(augmented).all():
            raise RuntimeError("Learned prediction adapter emitted a non-finite observation.")
        return augmented


class DiagonalSSMEncoder(nn.Module):
    """Small dependency-free diagonal state-space encoder.

    This is a portable SSM backend for environments where the optional
    ``mamba_ssm`` CUDA extension is unavailable. It preserves the intended
    linear-time sequence recurrence while keeping the backend explicit in
    checkpoint metadata.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0 or num_layers <= 0:
            raise ValueError("SSM dimensions must be positive.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.input_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layers)]
        )
        self.state_layers = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) for _ in range(self.num_layers)]
        )
        self.log_decay = nn.Parameter(torch.zeros(self.num_layers, self.hidden_dim))
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected [batch, time, {self.input_dim}] SSM inputs, got {tuple(inputs.shape)}."
            )
        sequence = torch.tanh(self.input_projection(inputs))
        state = [
            torch.zeros(
                inputs.shape[0], self.hidden_dim,
                device=inputs.device,
                dtype=inputs.dtype,
            )
            for _ in range(self.num_layers)
        ]
        outputs: list[torch.Tensor] = []
        for timestep in range(sequence.shape[1]):
            layer_input = sequence[:, timestep]
            layer_states: list[torch.Tensor] = []
            for layer in range(self.num_layers):
                decay = torch.sigmoid(self.log_decay[layer]).view(1, -1)
                candidate = torch.tanh(
                    self.input_layers[layer](layer_input)
                    + self.state_layers[layer](state[layer])
                )
                next_state = decay * state[layer] + (1.0 - decay) * candidate
                next_state = self.norms[layer](next_state)
                layer_input = next_state
                layer_states.append(next_state)
            state = layer_states
            outputs.append(layer_states[-1])
        return torch.stack(outputs, dim=1)


class ConditionalDiffusionTrajectoryPredictor(nn.Module):
    """SSM-conditioned trajectory diffusion model with few-step DDIM sampling."""

    backend = "portable_diagonal_ssm"

    def __init__(
        self,
        input_dim: int,
        horizon_count: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        diffusion_steps: int = 100,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or horizon_count <= 0 or hidden_dim <= 0:
            raise ValueError("Diffusion predictor dimensions must be positive.")
        if diffusion_steps < 2:
            raise ValueError("diffusion_steps must be at least 2.")
        self.input_dim = int(input_dim)
        self.horizon_count = int(horizon_count)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.diffusion_steps = int(diffusion_steps)
        self.target_dim = self.horizon_count * 3
        self.encoder = DiagonalSSMEncoder(self.input_dim, self.hidden_dim, self.num_layers)
        self.condition_norm = nn.LayerNorm(self.hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.denoiser = nn.Sequential(
            nn.Linear(self.target_dim + 2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.target_dim),
        )
        betas = torch.linspace(1e-4, 0.02, self.diffusion_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

    @property
    def model_config(self) -> dict[str, int]:
        return {
            "input_dim": self.input_dim,
            "horizon_count": self.horizon_count,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "diffusion_steps": self.diffusion_steps,
        }

    def encode_condition(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence = self.encoder(inputs)
        return self.condition_norm(sequence[:, -1])

    def _time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.hidden_dim // 2
        frequencies = torch.exp(
            -np.log(10000.0)
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if embedding.shape[1] < self.hidden_dim:
            embedding = F.pad(embedding, (0, self.hidden_dim - embedding.shape[1]))
        return self.time_projection(embedding)

    def predict_noise(
        self,
        noisy_target: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_target.ndim != 2 or noisy_target.shape[-1] != self.target_dim:
            raise ValueError("noisy_target has incompatible shape.")
        time_embedding = self._time_embedding(timesteps)
        return self.denoiser(torch.cat([noisy_target, condition, time_embedding], dim=-1))

    def diffusion_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if targets.ndim != 3 or targets.shape[1:] != (self.horizon_count, 3):
            raise ValueError("targets must have shape [batch, horizon, 3].")
        condition = self.encode_condition(inputs)
        clean = targets.reshape(targets.shape[0], -1)
        timesteps = torch.randint(
            0,
            self.diffusion_steps,
            (targets.shape[0],),
            device=targets.device,
            generator=generator,
        )
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        alpha = self.alphas_cumprod[timesteps].unsqueeze(1)
        noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise
        predicted_noise = self.predict_noise(noisy, timesteps, condition)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(
        self,
        inputs: torch.Tensor,
        num_samples: int = 8,
        sampling_steps: int = 8,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if num_samples <= 0 or sampling_steps <= 0:
            raise ValueError("num_samples and sampling_steps must be positive.")
        condition = self.encode_condition(inputs)
        batch_size = inputs.shape[0]
        expanded_condition = condition[:, None, :].expand(batch_size, num_samples, -1).reshape(
            batch_size * num_samples, -1
        )
        if generator is None:
            noise = torch.randn(
                batch_size * num_samples, self.target_dim,
                device=inputs.device,
                dtype=inputs.dtype,
            )
        else:
            noise = torch.randn(
                batch_size * num_samples, self.target_dim,
                device=inputs.device,
                dtype=inputs.dtype,
                generator=generator,
            )
        current = noise
        schedule = torch.linspace(
            self.diffusion_steps - 1,
            0,
            min(sampling_steps, self.diffusion_steps),
            device=inputs.device,
        ).long()
        for index, timestep in enumerate(schedule):
            timesteps = torch.full(
                (batch_size * num_samples,), int(timestep), device=inputs.device, dtype=torch.long
            )
            predicted_noise = self.predict_noise(current, timesteps, expanded_condition)
            alpha = self.alphas_cumprod[timestep]
            clean = (current - (1.0 - alpha).sqrt() * predicted_noise) / alpha.sqrt()
            if index + 1 == len(schedule):
                current = clean
                continue
            previous_timestep = schedule[index + 1]
            previous_alpha = self.alphas_cumprod[previous_timestep]
            current = previous_alpha.sqrt() * clean + (1.0 - previous_alpha).sqrt() * predicted_noise
        return current.reshape(batch_size, num_samples, self.horizon_count, 3)

    @torch.no_grad()
    def sample_set(
        self,
        inputs: torch.Tensor,
        num_samples: int = 8,
        sampling_steps: int = 8,
        generator: torch.Generator | None = None,
    ) -> CandidateTrajectorySet:
        """Return planner-facing candidates with explicitly uncalibrated scores."""

        return CandidateTrajectorySet.uniform(
            self.sample(
                inputs,
                num_samples=num_samples,
                sampling_steps=sampling_steps,
                generator=generator,
            )
        )


def prediction_metrics(
    candidates: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Return ADE/FDE and best-of-K metrics for candidate trajectories."""

    if candidates.ndim != 4 or targets.ndim != 3:
        raise ValueError("Expected candidates [batch, modes, horizon, 3] and targets [batch, horizon, 3].")
    distances = torch.linalg.norm(candidates - targets[:, None], dim=-1)
    ade = distances.mean(dim=-1)
    fde = distances[:, :, -1]
    return {
        "ade_top1": float(ade[:, 0].mean().detach().cpu()),
        "fde_top1": float(fde[:, 0].mean().detach().cpu()),
        "min_ade": float(ade.min(dim=1).values.mean().detach().cpu()),
        "min_fde": float(fde.min(dim=1).values.mean().detach().cpu()),
        "candidate_spread": float(candidates.std(dim=1, unbiased=False).mean().detach().cpu()),
    }


def candidate_energy_score(candidates: torch.Tensor, targets: torch.Tensor) -> float:
    """Return the multivariate energy score for an equally weighted candidate set."""

    if candidates.ndim != 4 or targets.ndim != 3 or candidates.shape[0] != targets.shape[0]:
        raise ValueError("Expected candidates [batch, modes, horizon, 3] and targets [batch, horizon, 3].")
    batch_size, candidate_count, horizon_count, coordinates = candidates.shape
    if targets.shape[1:] != (horizon_count, coordinates) or candidate_count <= 0:
        raise ValueError("Candidate and target horizons must match and candidate count must be positive.")
    candidate_vectors = candidates.reshape(batch_size, candidate_count, -1)
    target_vectors = targets.reshape(batch_size, -1)[:, None, :]
    fit = torch.linalg.vector_norm(candidate_vectors - target_vectors, dim=-1).mean(dim=1)
    pairwise = torch.linalg.vector_norm(
        candidate_vectors[:, :, None, :] - candidate_vectors[:, None, :, :], dim=-1
    ).mean(dim=(1, 2))
    return float((fit - 0.5 * pairwise).mean().detach().cpu())


def conformal_nonconformity(candidates: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return one split-conformal trajectory error per sample in physical units."""

    if candidates.ndim != 4 or targets.ndim != 3 or candidates.shape[0] != targets.shape[0]:
        raise ValueError("Expected candidates [batch, modes, horizon, 3] and targets [batch, horizon, 3].")
    distances = torch.linalg.vector_norm(candidates - targets[:, None], dim=-1)
    return distances.amax(dim=-1).amin(dim=1)


def conformal_radius(calibration_scores: torch.Tensor, coverage: float = 0.9) -> float:
    """Compute a finite-sample split-conformal radius for desired marginal coverage."""

    if calibration_scores.ndim != 1 or calibration_scores.numel() == 0:
        raise ValueError("calibration_scores must be a non-empty vector.")
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be strictly between zero and one.")
    if not torch.isfinite(calibration_scores).all():
        raise ValueError("calibration_scores must be finite.")
    quantile_level = min(
        1.0,
        float(np.ceil((calibration_scores.numel() + 1) * coverage) / calibration_scores.numel()),
    )
    return float(torch.quantile(calibration_scores, quantile_level, interpolation="higher").cpu())


def conformal_coverage(
    candidates: torch.Tensor,
    targets: torch.Tensor,
    radius: float,
) -> dict[str, float]:
    """Measure full-trajectory and per-horizon coverage for a calibrated radius."""

    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be finite and non-negative.")
    distances = torch.linalg.vector_norm(candidates - targets[:, None], dim=-1)
    within = distances <= float(radius)
    full_coverage = within.all(dim=-1).any(dim=1).float().mean()
    horizon_coverage = within.any(dim=1).float().mean(dim=0)
    return {
        "coverage_full_trajectory": float(full_coverage.cpu()),
        "coverage_horizon_mean": float(horizon_coverage.mean().cpu()),
        "coverage_horizon_min": float(horizon_coverage.min().cpu()),
        "coverage_horizon_max": float(horizon_coverage.max().cpu()),
    }
