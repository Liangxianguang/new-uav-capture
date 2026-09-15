"""Recurrent residual policy around a public-belief pursuit baseline.

The residual actor is deliberately small and auditable.  A deterministic
public-belief route controller provides the base action; the recurrent network
learns a bounded correction from route-aware local observations.  This keeps
the DN-MPC teacher in the data-generation loop without requiring the planner
to run as the deployed policy.
"""

from __future__ import annotations

import torch
from torch import nn


class RecurrentResidualActor(nn.Module):
    """Parameter-sharing GRU actor that predicts a bounded action residual."""

    def __init__(
        self,
        local_observation_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 128,
        residual_scale: float = 1.5,
    ) -> None:
        super().__init__()
        if local_observation_dim <= 0 or action_dim <= 0 or hidden_dim <= 0:
            raise ValueError("Residual actor dimensions must be positive.")
        if residual_scale <= 0.0:
            raise ValueError("residual_scale must be positive.")
        self.local_observation_dim = int(local_observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)
        self.encoder = nn.Sequential(
            nn.Linear(self.local_observation_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.gru = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.residual_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    def initial_hidden(
        self,
        defender_count: int,
        *,
        batch_size: int | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if defender_count <= 0:
            raise ValueError("defender_count must be positive.")
        shape = (
            (defender_count, self.hidden_dim)
            if batch_size is None
            else (batch_size, defender_count, self.hidden_dim)
        )
        return torch.zeros(shape, device=device, dtype=next(self.parameters()).dtype)

    def step(
        self,
        local_observations: torch.Tensor,
        base_actions: torch.Tensor,
        hidden_state: torch.Tensor,
        action_scale: float,
        reset_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(action, next_hidden, residual)`` for one frame."""

        if local_observations.ndim != 2:
            raise ValueError("local_observations must have shape [defenders, features].")
        if base_actions.shape != (local_observations.shape[0], self.action_dim):
            raise ValueError("base_actions has an incompatible shape.")
        if hidden_state.shape != (local_observations.shape[0], self.hidden_dim):
            raise ValueError("hidden_state has an incompatible shape.")
        if action_scale <= 0.0:
            raise ValueError("action_scale must be positive.")
        if reset_mask is not None:
            mask = torch.as_tensor(reset_mask, dtype=hidden_state.dtype, device=hidden_state.device)
            if mask.ndim == 0:
                hidden_state = hidden_state * (1.0 - mask)
            elif mask.shape == (local_observations.shape[0],):
                hidden_state = hidden_state * (1.0 - mask[:, None])
            else:
                raise ValueError("reset_mask must be scalar or one value per defender.")
        encoded = self.encoder(local_observations)
        next_hidden = self.gru(encoded, hidden_state)
        residual = torch.tanh(self.residual_head(next_hidden)) * self.residual_scale
        action = torch.clamp(base_actions + residual, -float(action_scale), float(action_scale))
        return action, next_hidden, residual

    def sequence(
        self,
        local_observations: torch.Tensor,
        base_actions: torch.Tensor,
        initial_hidden: torch.Tensor,
        reset_masks: torch.Tensor,
        action_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a batch of padded sequences.

        Inputs use ``[batch, time, defenders, ...]``.  The returned tensors are
        ``actions`` and ``residuals`` with matching time dimensions.
        """

        if local_observations.ndim != 4:
            raise ValueError("local_observations must have shape [batch, time, defenders, features].")
        if base_actions.shape[:3] != local_observations.shape[:3]:
            raise ValueError("base_actions has an incompatible sequence shape.")
        if reset_masks.shape != local_observations.shape[:2]:
            raise ValueError("reset_masks has an incompatible sequence shape.")
        batch_size, sequence_length, defender_count, _ = local_observations.shape
        expected_hidden = (batch_size, defender_count, self.hidden_dim)
        if initial_hidden.shape != expected_hidden:
            raise ValueError(f"Expected initial_hidden shape {expected_hidden}, got {tuple(initial_hidden.shape)}.")
        hidden = initial_hidden
        actions: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        for index in range(sequence_length):
            hidden = hidden * (1.0 - reset_masks[:, index, None, None].to(hidden.dtype))
            flattened_local = local_observations[:, index].reshape(-1, self.local_observation_dim)
            flattened_base = base_actions[:, index].reshape(-1, self.action_dim)
            flattened_hidden = hidden.reshape(-1, self.hidden_dim)
            action, next_hidden, residual = self.step(
                flattened_local,
                flattened_base,
                flattened_hidden,
                action_scale,
            )
            hidden = next_hidden.reshape(expected_hidden)
            actions.append(action.reshape(batch_size, defender_count, self.action_dim))
            residuals.append(residual.reshape(batch_size, defender_count, self.action_dim))
        return torch.stack(actions, dim=1), torch.stack(residuals, dim=1)

    def actor_parameters(self):
        return list(self.parameters())


__all__ = ["RecurrentResidualActor"]
