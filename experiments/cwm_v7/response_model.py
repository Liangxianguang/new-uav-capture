"""Matched action-conditioned and graph-structured anchored response predictors."""
import torch
from torch import nn


class ResponseModel(nn.Module):
    def __init__(self, kind, response_scale_m=2.5):
        super().__init__()
        if kind not in ("plain", "structured") or response_scale_m <= 0:
            raise ValueError("Invalid response model contract")
        self.kind, self.response_scale_m = kind, float(response_scale_m)
        self.history_encoder = nn.GRU(252, 24, batch_first=True)
        self.context = nn.Sequential(nn.Linear(48, 32), nn.Tanh())
        if kind == "plain":
            self.nodes = nn.Sequential(nn.Linear(120, 32), nn.Tanh())
        else:
            self.nodes = nn.Sequential(nn.Linear(30, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
            self.edges = nn.Sequential(nn.Linear(30, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        self.shared = nn.Sequential(nn.Linear(64, 64), nn.Tanh())
        self.response_head = nn.Linear(64, 24)
        self.branch_head = nn.Linear(64, 8)
        nn.init.zeros_(self.response_head.weight)
        nn.init.zeros_(self.response_head.bias)
        nn.init.zeros_(self.branch_head.weight)
        nn.init.zeros_(self.branch_head.bias)

    def action_features(self, relative, actions):
        state = relative / relative.new_tensor([10., 10., 10., 5., 5., 5.])
        if self.kind == "plain":
            return self.nodes(torch.cat([state.flatten(1), actions.flatten(1) / 5.], -1))
        per_agent = actions.transpose(1, 2).flatten(2) / 5.
        nodes = torch.cat([state, per_agent], -1)
        pair_differences = nodes[:, :, None] - nodes[:, None, :]
        messages = self.edges(pair_differences)
        mask = (~torch.eye(4, dtype=torch.bool, device=nodes.device))[None, :, :, None]
        messages = (messages * mask).sum(2) / 3.
        return (self.nodes(nodes) + messages).mean(1)

    def forward(self, history, relative, proposed, anchor, backbone):
        if history.shape[1:] != (8, 252) or relative.shape[1:] != (4, 6) or proposed.shape[1:] != (8, 4, 3) or anchor.shape != proposed.shape or backbone.shape[1:] != (8, 3):
            raise ValueError("Public input shape contract mismatch")
        _, hidden = self.history_encoder(history.float())
        context = self.context(torch.cat([hidden[-1], backbone.flatten(1).float() / 5.], -1))
        acted = self.shared(torch.cat([context, self.action_features(relative.float(), proposed.float())], -1))
        reference = self.shared(torch.cat([context, self.action_features(relative.float(), anchor.float())], -1))
        difference = self.response_head(acted) - self.response_head(reference)
        response = self.response_scale_m * torch.tanh(difference).reshape(-1, 8, 3)
        equal = (proposed == anchor).flatten(1).all(1)
        response = torch.where(equal[:, None, None], torch.zeros_like(response), response)
        prediction = torch.where(equal[:, None, None], backbone, backbone + response.to(backbone.dtype))
        return prediction, response, self.branch_head(acted)


def loss_terms(response, branch_logits, target, valid, signs, weights, protocol):
    common = valid[:, 1:] & valid[:, :1]
    observed = target[:, 1:] - target[:, :1]
    predicted = response[:, 1:] - response[:, :1]
    support = common.sum((1, 2)).clamp_min(1)
    effect = (((predicted - observed) ** 2).sum(-1) * common).sum((1, 2)) / (3 * support)
    energy = ((predicted ** 2).sum(-1) * common).sum((1, 2)) / (3 * support)
    committed = valid & (signs != 0)
    branch = nn.functional.binary_cross_entropy_with_logits(branch_logits, (signs > 0).float(), reduction="none")
    branch = (branch * committed).sum((1, 2)) / committed.sum((1, 2)).clamp_min(1)
    eligible = common.any((1, 2))
    weights = weights * eligible
    if weights.sum() <= 0:
        raise ValueError("No common valid paired support")
    combine = lambda value: (value * weights).sum() / weights.sum()
    effect, energy, branch = combine(effect), combine(energy), combine(branch)
    loss = effect + protocol["residual_penalty"] * energy + protocol["branch_auxiliary_weight"] * branch
    return loss, effect, energy, branch
