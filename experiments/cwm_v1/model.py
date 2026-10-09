"""Matched plain and structured intervention-trained target-response surrogates.

The structural prior is additive autonomous motion plus pooled per-defender
command pressure. It is a testable inductive bias, NOT a proven identified SCM.
"""
from __future__ import annotations

import torch
from torch import nn


class ResponseModel(nn.Module):
    def __init__(self, kind: str, history_dim: int = 252, hidden: int = 32, horizon: int = 8):
        super().__init__()
        if kind not in ("plain", "structured"):
            raise ValueError("Unsupported response model")
        self.kind, self.horizon = kind, horizon
        self.encoder = nn.GRU(history_dim, hidden, batch_first=True)
        if kind == "plain":
            self.decoder = nn.Sequential(nn.Linear(hidden + horizon * 12, 64), nn.Tanh(), nn.Linear(64, horizon * 3))
        else:
            self.autonomous = nn.Sequential(nn.Linear(hidden, 48), nn.Tanh(), nn.Linear(48, horizon * 3))
            self.agent_pressure = nn.Sequential(nn.Linear(6 + horizon * 3, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
            self.response = nn.Sequential(nn.Linear(hidden + 32, 48), nn.Tanh(), nn.Linear(48, horizon * 3))

    def pressure(self, relative: torch.Tensor, proposed: torch.Tensor) -> torch.Tensor:
        # Per-agent intervention nodes with shared parameters, invariant to
        # permutation of the defender nodes when the context is held fixed.
        per_agent_actions = proposed.transpose(1, 2).flatten(2) / 5.0
        relative_scale = relative.new_tensor([10, 10, 10, 5, 5, 5])
        states = relative / relative_scale
        acted = self.agent_pressure(torch.cat([states, per_agent_actions], dim=-1))
        reference = self.agent_pressure(torch.cat([states, torch.zeros_like(per_agent_actions)], dim=-1))
        return (acted - reference).mean(dim=1)

    def forward(self, history, relative, proposed, reference_velocity):
        _, encoded = self.encoder(history)
        context = encoded[-1]
        if self.kind == "plain":
            residual = self.decoder(torch.cat([context, proposed.flatten(1) / 5.0], dim=-1))
        else:
            residual = self.autonomous(context) + self.response(torch.cat([context, self.pressure(relative, proposed)], dim=-1))
        times = torch.arange(1, self.horizon + 1, device=history.device, dtype=history.dtype) * 0.1
        return reference_velocity[:, None, :] * times[None, :, None] + residual.reshape(-1, self.horizon, 3)


def group_loss(prediction, labels, mask, effect_weight: float = 1.0):
    valid = mask[..., None].expand_as(labels)
    if not valid.any():
        raise ValueError("Batch has no valid labels")
    reconstruction = torch.nn.functional.smooth_l1_loss(prediction[valid], labels[valid])
    common = (mask[:, 1:] & mask[:, :1])[..., None].expand_as(labels[:, 1:])
    if common.any():
        predicted_effect = prediction[:, 1:] - prediction[:, :1]
        observed_effect = labels[:, 1:] - labels[:, :1]
        effect = torch.nn.functional.smooth_l1_loss(predicted_effect[common], observed_effect[common])
    else:
        effect = reconstruction * 0
    return reconstruction + effect_weight * effect, reconstruction, effect
