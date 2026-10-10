"""Response cannot alter or differentiate through the pretrained common motion."""
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'cwm_v10'))
from two_head_model import LocalTwoHead


class FrozenCommonResponse(nn.Module):
    def __init__(self, common, response, mediated=False):
        super().__init__()
        self.common = common.eval().requires_grad_(False)
        self.response_core = response
        self.response_core.motion_head.requires_grad_(False)
        self.mediated = mediated

    def train(self, mode=True):
        super().train(mode)
        self.common.eval()
        return self

    def forward(self, history, relative, proposed, anchor, backbone,
                estimated_proposed=None, estimated_anchor=None):
        with torch.no_grad():
            _, motion, _ = self.common(history,relative,proposed,anchor,backbone)
        if self.mediated:
            if estimated_proposed is None or estimated_anchor is None:
                raise ValueError('Frozen public estimated commands required')
            used_proposed,used_anchor = estimated_proposed,estimated_anchor
        else:
            used_proposed,used_anchor = proposed,anchor
        _, _, response = self.response_core(history,relative,used_proposed,used_anchor,backbone)
        equal = (proposed == anchor).flatten(1).all(1)
        response = torch.where(equal[:,None,None],torch.zeros_like(response),response)
        prediction = backbone + motion.to(backbone.dtype) + response.to(backbone.dtype)
        prediction = torch.where((motion == 0) & (response == 0),backbone,prediction)
        return prediction,motion,response


def response_loss(response, calls, indices, slices, group_counts, config):
    losses, weights = [], []
    for i,(start,end) in zip(indices,slices):
        values = calls[i]['values']
        nonanchor = ~(values['proposed'] == values['anchor'][None]).all((1,2,3))
        mask = torch.as_tensor(values['valid'] & values['anchor_valid'][None] & nonanchor[:,None])
        label = torch.as_tensor(values['target']-values['anchor_target'][None],dtype=torch.float32)
        denominator = (3*mask.sum()).clamp_min(1)
        error = (((response[start:end]-label)**2).sum(-1)*mask).sum()/denominator
        energy = ((response[start:end]**2).sum(-1)*mask).sum()/denominator
        losses.append(torch.stack([error,energy]))
        weights.append(1./group_counts[calls[i]['record']['group']] if mask.any() else 0.)
    weight = torch.as_tensor(weights,dtype=torch.float64)
    if weight.sum() <= 0:
        raise ValueError('No valid nonanchor response support')
    terms = (torch.stack(losses)*weight[:,None]).sum(0)/weight.sum()
    return terms[0]+config['residual_penalty']*terms[1],terms
