"""Small supervision-only mechanism estimators with strictly public inputs."""
import numpy as np
import torch
from torch import nn


class PublicMechanismModel(nn.Module):
    def __init__(self, kind):
        super().__init__()
        if kind not in ('branch_snapshot', 'branch_history', 'cbf_history'):
            raise ValueError('Unknown public mechanism configuration')
        self.kind = kind
        self.encoder = nn.Sequential(nn.Linear(252, 24), nn.Tanh()) if kind == 'branch_snapshot' else nn.GRU(252, 24, batch_first=True)
        self.shared = nn.Sequential(nn.Linear(144, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.head = nn.Linear(64, 96 if kind == 'cbf_history' else 24)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, history, relative, proposed):
        if history.shape[1:] != (8, 252) or relative.shape[1:] != (4, 6) or proposed.shape[1:] != (8, 4, 3):
            raise ValueError('Public-only input shape mismatch')
        if self.kind == 'branch_snapshot':
            encoded = self.encoder(history[:, -1])
        else:
            _, hidden = self.encoder(history)
            encoded = hidden[-1]
        state_scale = relative.new_tensor([10., 10., 10., 5., 5., 5.])
        features = torch.cat([encoded, (relative/state_scale).flatten(1), proposed.flatten(1)/5.], -1)
        output = self.head(self.shared(features))
        if self.kind == 'cbf_history':
            return proposed + 5. * torch.tanh(output).reshape(-1, 8, 4, 3)
        return output.reshape(-1, 8, 3)


def public_inputs(values, mean, scale):
    # This whitelist is intentional: labels, target, masks and execution cannot
    # affect inference. Anchor is another supplied candidate, not future truth.
    history = np.asarray(values['history'])
    relative = np.asarray(values['relative'])
    proposed = np.concatenate([values['anchor'][None], values['proposed']], axis=0)
    if history.shape != (8, 252) or relative.shape != (4, 6) or proposed.shape[1:] != (8, 4, 3):
        raise ValueError('Public-only array shape mismatch')
    if any(not np.isfinite(x).all() for x in (history, relative, proposed, mean, scale)) or (scale <= 0).any():
        raise ValueError('Nonfinite public input or normalizer')
    n = len(proposed)
    return (torch.from_numpy(((history - mean) / scale).astype(np.float32))[None].repeat(n, 1, 1),
            torch.from_numpy(relative.astype(np.float32))[None].repeat(n, 1, 1), torch.from_numpy(proposed.astype(np.float32)))


def branch_labels(values, private):
    signs = np.concatenate([private['anchor_branch_after_label_only'][None], private['branch_after_label_only']], axis=0)
    valid = np.concatenate([values['anchor_valid'][None], values['valid']], axis=0)
    if signs.shape != valid.shape or not np.isin(signs, [-1, 0, 1]).all():
        raise ValueError('True branch class/support mismatch')
    return torch.from_numpy((signs + 1).astype(np.int64)), torch.from_numpy(valid.copy())


def cbf_labels(values):
    commanded = np.concatenate([values['anchor_commanded'][None], values['commanded']], axis=0)
    observed = np.concatenate([(values['anchor_termination'] != 'after_terminal')[None], values['termination'] != 'after_terminal'], axis=0)
    return torch.from_numpy(commanded.astype(np.float32)), torch.from_numpy(observed.copy())


def single_call_loss(model, values, private, mean, scale, energy_weight):
    inputs = public_inputs(values, mean, scale)
    predicted = model(*inputs)
    if model.kind == 'cbf_history':
        truth, support = cbf_labels(values)
        squared = (predicted - truth).square().sum((-1, -2)) / 12.
        penalty = (predicted - inputs[-1]).square().sum((-1, -2)) / 12.
        loss = squared + energy_weight * penalty
    else:
        truth, support = branch_labels(values, private)
        loss = nn.functional.cross_entropy(predicted.transpose(1, 2), truth, reduction='none')
    if not support.any():
        raise ValueError('No observed mechanism support')
    return loss[support].mean()


def batch_call_loss(model, calls, indices, mean, scale, group_counts, energy_weight):
    packed = [public_inputs(calls[i]['values'], mean, scale) for i in indices]
    inputs = tuple(torch.cat([row[k] for row in packed], 0) for k in range(3))
    output = model(*inputs)
    labels = [cbf_labels(calls[i]['values']) if model.kind == 'cbf_history' else branch_labels(calls[i]['values'], calls[i]['private']) for i in indices]
    truth = torch.cat([row[0] for row in labels], 0)
    support = torch.cat([row[1] for row in labels], 0)
    if model.kind == 'cbf_history':
        loss = ((output-truth).square() + energy_weight*(output-inputs[-1]).square()).sum((-1,-2))/12.
    else:
        loss = nn.functional.cross_entropy(output.transpose(1,2),truth,reduction='none')
    selected = []
    offset = 0
    for row in packed:
        count = len(row[0])
        mask = support[offset:offset+count]
        if not mask.any():
            raise ValueError('No observed mechanism support')
        selected.append(loss[offset:offset+count][mask].mean())
        offset += count
    weights = loss.new_tensor([1./group_counts[calls[i]['record']['group']] for i in indices])
    return (torch.stack(selected)*weights).sum()/weights.sum()
