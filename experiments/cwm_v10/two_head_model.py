"""Optional reference-motion calibration and anchored interventional response."""
import torch
from torch import nn


class LocalTwoHead(nn.Module):
    def __init__(self, kind, motion_scale_m=2.5, response_scale_m=2.5):
        super().__init__()
        if kind not in ("motion_only", "plain", "structured") or min(motion_scale_m, response_scale_m) <= 0:
            raise ValueError("Invalid optional model contract")
        self.kind = kind
        self.motion_scale_m, self.response_scale_m = float(motion_scale_m), float(response_scale_m)
        # This is a new optional history encoder, NOT the frozen original GRU.
        self.history_encoder = nn.GRU(252, 24, batch_first=True)
        self.context = nn.Sequential(nn.Linear(48, 32), nn.Tanh())
        if kind == "structured":
            self.nodes = nn.Sequential(nn.Linear(30, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
            self.edges = nn.Sequential(nn.Linear(30, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh())
        else:
            self.nodes = nn.Sequential(nn.Linear(120, 32), nn.Tanh())
        self.shared = nn.Sequential(nn.Linear(64, 64), nn.Tanh())
        self.motion_head, self.response_head = nn.Linear(64, 24), nn.Linear(64, 24)
        for head in (self.motion_head, self.response_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if kind == "motion_only":
            self.response_head.requires_grad_(False)

    def action_features(self, relative, actions):
        state = relative / relative.new_tensor([10., 10., 10., 5., 5., 5.])
        if self.kind != "structured":
            return self.nodes(torch.cat([state.flatten(1), actions.flatten(1) / 5.], -1))
        per_agent = actions.transpose(1, 2).flatten(2) / 5.
        nodes = torch.cat([state, per_agent], -1)
        messages = self.edges(nodes[:, :, None] - nodes[:, None, :])
        mask = (~torch.eye(4, dtype=torch.bool, device=nodes.device))[None, :, :, None]
        return (self.nodes(nodes) + (messages * mask).sum(2) / 3.).mean(1)

    def forward(self, history, relative, proposed, anchor, backbone):
        if history.shape[1:] != (8, 252) or relative.shape[1:] != (4, 6) or proposed.shape[1:] != (8, 4, 3) or proposed.shape != anchor.shape or backbone.shape[1:] != (8, 3):
            raise ValueError("Public input shape contract mismatch")
        _, hidden = self.history_encoder(history.float())
        context = self.context(torch.cat([hidden[-1], backbone.flatten(1).float() / 5.], -1))
        reference = self.shared(torch.cat([context, self.action_features(relative.float(), anchor.float())], -1))
        motion = self.motion_scale_m * torch.tanh(self.motion_head(reference)).reshape(-1, 8, 3)
        if self.kind == "motion_only":
            response = torch.zeros_like(motion)
        else:
            acted = self.shared(torch.cat([context, self.action_features(relative.float(), proposed.float())], -1))
            response = self.response_scale_m * torch.tanh(self.response_head(acted) - self.response_head(reference)).reshape(-1, 8, 3)
            equal = (proposed == anchor).flatten(1).all(1)
            response = torch.where(equal[:, None, None], torch.zeros_like(response), response)
        prediction = backbone + motion.to(backbone.dtype) + response.to(backbone.dtype)
        prediction = torch.where((motion == 0) & (response == 0), backbone, prediction)
        return prediction, motion, response
