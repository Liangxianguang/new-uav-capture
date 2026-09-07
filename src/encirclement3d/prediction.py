"""Supervised local-history target trajectory predictors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


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

    def diffusion_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.ndim != 3 or targets.shape[1:] != (self.horizon_count, 3):
            raise ValueError("targets must have shape [batch, horizon, 3].")
        condition = self.encode_condition(inputs)
        clean = targets.reshape(targets.shape[0], -1)
        timesteps = torch.randint(
            0, self.diffusion_steps, (targets.shape[0],), device=targets.device
        )
        noise = torch.randn_like(clean)
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
