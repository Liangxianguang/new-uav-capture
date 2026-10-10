"""Public GRU/CV reference origins with unchanged, matched V10 neural cores.

The sixth tensor is an analytic PUBLIC CV path, not a new neural feature.
Every core still receives only the same original five public tensors, including
the original GRU prediction. Truth/labels/costs are never forward inputs.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'cwm_v10'))
from two_head_model import LocalTwoHead
from train_two_head import pack_inputs, metrics, normalization


def public_cost_inputs(calls, indices, mean, scale):
    inputs, slices = pack_inputs(calls, indices, mean, scale)
    origins = []
    for i in indices:
        v = calls[i]['values']
        cv = v['reference'][None] + .1 * np.arange(1, 9)[:, None] * v['velocity'][None]
        origins.append(np.repeat(cv[None], len(v['proposed']), 0))
    cv = torch.as_tensor(np.concatenate(origins), dtype=torch.float64)
    if not torch.isfinite(cv).all():
        raise ValueError('Finite independently audited public CV required')
    return (*inputs, cv), slices


class CalibratedOrigin(nn.Module):
    """Only the learned calibration is penalized, never the CV-GRU offset."""
    kind = 'motion_only'

    def __init__(self, core, origin):
        super().__init__()
        if origin not in ('gru', 'cv') or core.kind != 'motion_only':
            raise ValueError('Matched motion-only GRU/CV origin required')
        self.core, self.origin = core, origin

    def calibration(self, history, relative, proposed, anchor, backbone, cv):
        if cv.shape != backbone.shape or cv.dtype != backbone.dtype or not torch.isfinite(cv).all():
            raise ValueError('Finite public CV path contract mismatch')
        _, calibration, _ = self.core(history, relative, proposed, anchor, backbone)
        return backbone if self.origin == 'gru' else cv, calibration

    def forward(self, history, relative, proposed, anchor, backbone, cv):
        origin, calibration = self.calibration(history, relative, proposed, anchor, backbone, cv)
        prediction = torch.where(calibration == 0, origin, origin + calibration.to(origin.dtype))
        # Total offset is for common diagnostics only, not motion regularization.
        return prediction, prediction - backbone, torch.zeros_like(calibration)


class FrozenOriginResponse(nn.Module):
    def __init__(self, common, response):
        super().__init__()
        if not isinstance(common, CalibratedOrigin) or response.kind != 'plain':
            raise ValueError('Calibrated public common and raw plain response required')
        self.common = common.eval().requires_grad_(False)
        for p in self.common.parameters():
            p.grad = None
        self.response_core = response
        self.response_core.motion_head.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.common.eval()
        return self

    def forward(self, history, relative, proposed, anchor, backbone, cv):
        with torch.no_grad():
            reference, motion, _ = self.common(history, relative, proposed, anchor, backbone, cv)
        _, _, response = self.response_core(history, relative, proposed, anchor, backbone)
        equal = (proposed == anchor).flatten(1).all(1)
        response = torch.where(equal[:, None, None], torch.zeros_like(response), response)
        prediction = torch.where(response == 0, reference, reference + response.to(reference.dtype))
        return prediction, motion, response


def calibrated_motion_loss(model, calls, indices, mean, scale, group_counts, penalty):
    """V10 group-call masked coordinate MSE/energy, relative to chosen origin.

    Uses the unchanged V10 per-batch group normalization for motion pretraining;
    response/cost stages separately use fixed full-population denominators.
    """
    inputs, slices = public_cost_inputs(calls, indices, mean, scale)
    origin, calibration = model.calibration(*inputs)
    terms, weights = [], []
    for i, (start, _) in zip(indices, slices):
        v = calls[i]['values']
        mask = torch.as_tensor(v['anchor_valid'])
        target = torch.as_tensor(v['anchor_target'], dtype=torch.float32)
        denom = 3 * mask.sum().clamp_min(1)
        # Original V10 label precision and coordinate loss remain unchanged.
        predicted = origin[start] + calibration[start].double()
        error = ((predicted - target).square().sum(-1) * mask).sum() / denom
        energy = (calibration[start].square().sum(-1) * mask).sum() / denom
        terms.append(torch.stack([error, energy]))
        weights.append(1. / group_counts[calls[i]['record']['group']] if mask.any() else 0.)
    weight = torch.as_tensor(weights, dtype=torch.float64)
    if not len(indices) or weight.sum() <= 0:
        raise ValueError('Nonempty masked motion training support required')
    measured = (torch.stack(terms) * weight[:, None]).sum(0) / weight.sum()
    return measured[0] + penalty * measured[1], measured


def evaluate_cost_model(model, calls, indices, mean, scale, batch_calls=32):
    model.eval()
    outputs = []
    for start in range(0, len(indices), batch_calls):
        inputs, slices = public_cost_inputs(calls, indices[start:start + batch_calls], mean, scale)
        with torch.no_grad():
            tensors = [v.numpy() for v in model(*inputs)]
            reference = model.common(*inputs)[0].numpy() if isinstance(model, FrozenOriginResponse) else tensors[0]
        outputs.extend({key: value[a:b].copy() for key, value in
                        zip(('prediction', 'motion', 'response', 'reference'), [*tensors, reference])} for a, b in slices)
    return metrics(calls, indices, outputs), outputs
