"""Frozen public CBF mediator plus unchanged optional plain two-head core."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent/'cwm_v14'), str(ROOT.parent/'cwm_v10')]
from public_mechanism_model import PublicMechanismModel, public_inputs
from two_head_model import LocalTwoHead


class FrozenMediator:
    def __init__(self, archive_path, protocol):
        if hashlib.sha256(archive_path.read_bytes()).hexdigest() != protocol['mediator_archive_sha256']:
            raise ValueError('Frozen V14 mediator archive mismatch')
        self.protocol = protocol
        with zipfile.ZipFile(archive_path) as archive:
            report = json.loads(archive.read('primary/summary.json'))
            gate = report['qualification']['cbf_history']
            if not gate['mechanism_research_eligible'] or gate['selected_median_seed'] != protocol['mediator_seed']:
                raise ValueError('Frozen mediator is not the qualified V14 median')
            row = next(r for r in report['models'] if r['kind'] == protocol['mediator_kind'] and r['seed'] == protocol['mediator_seed'])
            raw = archive.read(f"primary/cbf_history_seed{protocol['mediator_seed']}.pt")
            self.checkpoint_sha256 = hashlib.sha256(raw).hexdigest()
            if self.checkpoint_sha256 != row['checkpoint_sha256']:
                raise ValueError('Frozen mediator checkpoint mismatch')
            checkpoint = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        if checkpoint['kind'] != 'cbf_history' or checkpoint['seed'] != protocol['mediator_seed'] or checkpoint['online_promoted'] or checkpoint['baseline_weights_included']:
            raise ValueError('Frozen mediator checkpoint contract mismatch')
        with torch.random.fork_rng(devices=[]):
            self.model = PublicMechanismModel('cbf_history')
        self.model.load_state_dict(checkpoint['model_state'], strict=True)
        self.model.eval().requires_grad_(False)
        self.mean = checkpoint['normalizer_mean'].numpy().copy()
        self.scale = checkpoint['normalizer_scale'].numpy().copy()
        self.digest = state_digest(self.model)

    def estimate(self, values):
        inputs = public_inputs(values, self.mean, self.scale)
        with torch.no_grad():
            commands = self.model(*inputs).numpy().copy()
        if commands.shape != inputs[-1].shape or not np.isfinite(commands).all():
            raise ValueError('Nonfinite frozen public command estimate')
        # Enforce exact reference invariance independent of batch roundoff.
        equal = (values['proposed'] == values['anchor'][None]).all((1,2,3))
        commands[1:][equal] = commands[0]
        return commands

    def unchanged(self):
        return (state_digest(self.model) == self.digest and not self.model.training
                and all(not p.requires_grad and p.grad is None for p in self.model.parameters()))


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def attach_public_estimates(calls, mediator):
    diagnostics = []
    for call in calls:
        estimate = mediator.estimate(call['values'])
        call['estimated_commands'] = estimate
        values = call['values']
        truth = np.concatenate([values['anchor_commanded'][None], values['commanded']],0)
        proposed = np.concatenate([values['anchor'][None], values['proposed']],0)
        observed = np.concatenate([(values['anchor_termination'] != 'after_terminal')[None],values['termination'] != 'after_terminal'],0)
        diagnostics.append({**{k:call['record'][k] for k in ('episode_index','step','agent','group','split')},
                            'estimated_error_mps':float(np.linalg.norm(estimate-truth,axis=-1).mean(-1)[observed].mean()),
                            'proposed_error_mps':float(np.linalg.norm(proposed-truth,axis=-1).mean(-1)[observed].mean())})
    if not mediator.unchanged():
        raise ValueError('Frozen mediator mutated')
    return diagnostics


def pack(calls, indices, mean, scale, mediated):
    packed, slices, offset = [[] for _ in range(5)], [], 0
    for index in indices:
        call = calls[index]
        values = call['values']
        n = len(values['proposed'])
        if mediated:
            commands = call['estimated_commands']
            proposed, anchor = commands[1:],commands[0]
        else:
            proposed, anchor = values['proposed'],values['anchor']
        for dest, data in zip(packed,(np.repeat(values['history'][None],n,0),np.repeat(values['relative'][None],n,0),
                                      proposed,np.repeat(anchor[None],n,0),np.repeat(values['backbone'][None],n,0))):
            dest.append(data)
        slices.append((offset,offset+n))
        offset += n
    history,relative,proposed,anchor,backbone = [np.concatenate(v) for v in packed]
    return (torch.as_tensor((history-mean)/scale,dtype=torch.float32),torch.as_tensor(relative,dtype=torch.float32),
            torch.as_tensor(proposed,dtype=torch.float32),torch.as_tensor(anchor,dtype=torch.float32),torch.as_tensor(backbone,dtype=torch.float64)),slices


def physical_loss(model, name, calls, indices, mean, scale, counts, config):
    inputs,slices = pack(calls,indices,mean,scale,name == 'mediated_plain')
    _,motion,response = model(*inputs)
    terms,weights = [],[]
    for i,(start,end) in zip(indices,slices):
        values = calls[i]['values']
        valid = torch.as_tensor(values['anchor_valid'])
        nonanchor = ~(values['proposed'] == values['anchor'][None]).all((1,2,3))
        common = torch.as_tensor(values['valid'] & values['anchor_valid'][None] & nonanchor[:,None])
        target = torch.as_tensor(values['target'],dtype=torch.float32)
        anchor_target = torch.as_tensor(values['anchor_target'],dtype=torch.float32)
        motion_prediction = inputs[-1][start] + motion[start].double()
        motion_error = (((motion_prediction-anchor_target)**2).sum(-1)*valid).sum()/(3*valid.sum().clamp_min(1))
        response_error = (((response[start:end]-(target-anchor_target[None]))**2).sum(-1)*common).sum()/(3*common.sum().clamp_min(1))
        if model.kind == 'motion_only':
            response_error = response_error * 0.
        energy = ((motion[start]**2).sum(-1)*valid).sum()/(3*valid.sum().clamp_min(1))
        energy += ((response[start:end]**2).sum(-1)*common).sum()/(3*common.sum().clamp_min(1))
        terms.append(torch.stack([motion_error,response_error,energy]))
        weights.append(1./counts[calls[i]['record']['group']] if valid.any() else 0.)
    weights = torch.as_tensor(weights,dtype=torch.float64)
    if weights.sum() <= 0:
        raise ValueError('No valid target-response support')
    terms = (torch.stack(terms)*weights[:,None]).sum(0)/weights.sum()
    loss = config['motion_weight']*terms[0] + config['response_weight']*terms[1] + config['residual_penalty']*terms[2]
    return loss,terms
