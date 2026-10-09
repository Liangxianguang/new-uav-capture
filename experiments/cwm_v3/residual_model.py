"""Bounded intervention response around an externally supplied frozen GRU path."""
import torch
from torch import nn


class AnchoredResponse(nn.Module):
    def __init__(self, kind, response_scale_m=.05):
        super().__init__()
        if kind not in ("plain", "structured") or response_scale_m <= 0:
            raise ValueError("Invalid anchored response configuration")
        self.kind = kind
        self.response_scale_m = float(response_scale_m)
        self.history_encoder = nn.GRU(252, 24, batch_first=True)
        self.context = nn.Sequential(nn.Linear(24 + 24, 32), nn.Tanh())
        if kind == "plain":
            self.nodes = nn.Sequential(nn.Linear(24 + 96, 32), nn.Tanh())
        else:
            self.nodes = nn.Sequential(nn.Linear(6 + 24, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.head = nn.Sequential(nn.Linear(64, 48), nn.Tanh(), nn.Linear(48, 24))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def action_features(self, relative, actions):
        scale = relative.new_tensor([10., 10., 10., 5., 5., 5.])
        state = relative / scale
        if self.kind == "plain":
            return self.nodes(torch.cat([state.flatten(1), actions.flatten(1) / 5.], dim=-1))
        per_agent = actions.transpose(1, 2).flatten(2) / 5.
        return self.nodes(torch.cat([state, per_agent], dim=-1)).mean(dim=1)

    def forward(self, history, relative, proposed, anchor, backbone):
        if history.shape[1:] != (8, 252) or relative.shape[1:] != (4, 6) or proposed.shape[1:] != (8, 4, 3) or anchor.shape != proposed.shape or backbone.shape[1:] != (8, 3):
            raise ValueError("Public input shape contract mismatch")
        _, hidden = self.history_encoder(history)
        context = self.context(torch.cat([hidden[-1], backbone.flatten(1) / 5.], dim=-1))
        acted = self.head(torch.cat([context, self.action_features(relative, proposed)], dim=-1))
        reference = self.head(torch.cat([context, self.action_features(relative, anchor)], dim=-1))
        response = self.response_scale_m * torch.tanh(acted - reference).reshape(-1, 8, 3)
        equal_anchor = (proposed == anchor).flatten(1).all(dim=1)
        response = torch.where(equal_anchor[:, None, None], torch.zeros_like(response), response)
        # The backbone is data, never a trainable parameter or fine-tuning target.
        prediction = torch.where(equal_anchor[:, None, None], backbone, backbone + response)
        return prediction, response


def paired_loss(response, labels, masks, residual_penalty=.01):
    common = masks[:, 1:] & masks[:, :1]
    valid = common[..., None].expand(-1, -1, -1, 3)
    if not valid.any():
        raise ValueError("No valid paired labels")
    observed = labels[:, 1:] - labels[:, :1]
    predicted = response[:, 1:] - response[:, :1]
    effect = ((predicted[valid] - observed[valid]) ** 2).mean()
    magnitude = (predicted[valid] ** 2).mean()
    return effect + residual_penalty * magnitude, effect, magnitude
