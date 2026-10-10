"""UNTRAINED optional prefix-response hypothesis and capacity-matched control.

Same five public tensors as LocalTwoHead. No original GRU/controller import,
weight mutation, scenario access, training, deployment gate or active entry.
The query-control pair differs ONLY in masking future proposed/anchor commands.
Both share hard prefix-anchor equality, a24-output diagonal query head, parameter
count and neural tensor shapes. Fixed history/GRU forecast are known context.
"""
import math
import torch
from torch import nn


class PrefixQueryResponse(nn.Module):
    kind = 'plain'  # Existing FrozenOriginResponse wrapper's response interface.

    def __init__(self, causal_prefix, motion_scale_m=2.5, response_scale_m=2.5):
        super().__init__()
        if (type(causal_prefix) is not bool or
            any(not isinstance(s, (int, float)) or isinstance(s, bool) or not math.isfinite(s) or s <= 0
                for s in (motion_scale_m, response_scale_m))):
            raise ValueError('Explicit fixed prefix/control mode and positive original scales required')
        self.causal_prefix = causal_prefix
        self.motion_scale_m = float(motion_scale_m)
        self.response_scale_m = float(response_scale_m)
        self.history_encoder = nn.GRU(252, 24, batch_first=True)
        self.context = nn.Sequential(nn.Linear(48, 32), nn.Tanh())
        # Same action/context width as V10, plus a fixed eight-offset query code.
        self.nodes = nn.Sequential(nn.Linear(128, 32), nn.Tanh())
        self.shared = nn.Sequential(nn.Linear(64, 64), nn.Tanh())
        self.response_head = nn.Linear(64, 24)
        # Only for compatibility with the existing wrapper. Not used/trained.
        self.motion_head = nn.Linear(64, 24).requires_grad_(False)
        nn.init.zeros_(self.response_head.weight); nn.init.zeros_(self.response_head.bias)
        nn.init.zeros_(self.motion_head.weight); nn.init.zeros_(self.motion_head.bias)
        self.register_buffer('query_code', torch.eye(8))
        self.register_buffer('prefix_mask', torch.tril(torch.ones(8, 8)))

    def action_queries(self, relative, actions):
        batch = len(actions)
        state = relative/relative.new_tensor([10., 10., 10., 5., 5., 5.])
        masked = actions[:, None]*self.prefix_mask[None, :, :, None, None] if self.causal_prefix else actions[:, None].expand(-1, 8, -1, -1, -1)
        features = torch.cat([state.flatten(1)[:, None].expand(-1, 8, -1),
            masked.flatten(2)/5., self.query_code[None].expand(batch, -1, -1)], dim=-1)
        return self.nodes(features)

    def forward(self, history, relative, proposed, anchor, backbone):
        batch = len(history)
        if (history.shape != (batch, 8, 252) or relative.shape != (batch, 4, 6) or
            proposed.shape != (batch, 8, 4, 3) or anchor.shape != proposed.shape or
            backbone.shape != (batch, 8, 3) or batch < 1):
            raise ValueError('Original five public tensor shapes required')
        if any(not torch.isfinite(v).all() for v in (history, relative, proposed, anchor, backbone)):
            raise ValueError('Finite public features required')
        _, hidden = self.history_encoder(history.float())
        context = self.context(torch.cat([hidden[-1], backbone.flatten(1).float()/5.], dim=-1))
        repeated = context[:, None].expand(-1, 8, -1)
        # Acted/reference rows share ONE neural block and shapes for both modes.
        queries = torch.cat([self.action_queries(relative.float(), proposed.float()),
            self.action_queries(relative.float(), anchor.float())], dim=0)
        joined = torch.cat([torch.cat([repeated, repeated], dim=0), queries], dim=-1)
        all_offsets = self.response_head(self.shared(joined)).reshape(2*batch, 8, 8, 3)
        query = torch.arange(8, device=history.device)
        diagonal = all_offsets[:, query, query]
        response = self.response_scale_m*torch.tanh(diagonal[:batch]-diagonal[batch:])
        # A identical command prefix has physically zero paired response so far.
        # Apply the SAME public constraint to the full-sequence query control.
        equal_prefix = (proposed == anchor).flatten(2).all(2).long().cumprod(1).bool()
        response = torch.where(equal_prefix[:, :, None], torch.zeros_like(response), response)
        motion = torch.zeros_like(response)
        prediction = torch.where(response == 0, backbone, backbone+response.to(backbone.dtype))
        return prediction, motion, response
