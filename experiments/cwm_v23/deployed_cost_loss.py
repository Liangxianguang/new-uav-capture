"""Fixed-population L1 effect loss at the detached DEPLOYED motion reference."""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'cwm_v11'))
sys.path.insert(0, str(ROOT.parent / 'cwm_v21'))
from task_score import FrozenLocalScore, original_scores
from factorial_loss import population_weights, paired_mask, factorial_response_loss


def cost_population_weights(calls, indices):
    physical, report = population_weights(calls, indices)
    counts = collections.Counter(calls[i]['record']['group'] for i in indices
                                 if calls[i]['values']['valid'].all() and calls[i]['values']['anchor_valid'].all())
    groups = {calls[i]['record']['group'] for i in indices}
    if set(counts) != groups:
        raise ValueError('Complete original cost support required in every training group')
    factor = len(indices) / len(groups)
    full = {i: factor / counts[calls[i]['record']['group']]
            if calls[i]['values']['valid'].all() and calls[i]['values']['anchor_valid'].all() else 0.
            for i in indices}
    if not np.isclose(sum(full.values()), len(indices), rtol=0, atol=1e-10):
        raise ValueError('Full cost population weights do not normalize')
    return {**physical, 'cost': full}, {**report, 'group_complete_cost_calls': dict(counts)}


def contrast_cache(score, reference, values):
    """Labels are detached supervision; all calls require full union support."""
    if not (values['valid'].all() and values['anchor_valid'].all()):
        raise ValueError('Deployed full-cost loss rejects terminal-incomplete calls')
    nonanchor = ~(values['proposed'] == values['anchor'][None]).all((1, 2, 3))
    if not nonanchor.any():
        raise ValueError('Nonanchor original cost support required')
    ref = torch.as_tensor(reference, dtype=torch.float64).detach().clone().requires_grad_(True)
    if ref.shape != values['target'].shape or not torch.equal(ref, ref[:1].expand_as(ref)):
        raise ValueError('Action-independent deployed reference required')
    cost = score(ref)
    gradient = torch.autograd.grad(cost.sum(), ref, create_graph=False)[0].detach()
    scale = torch.linalg.vector_norm(gradient.flatten(1), dim=1).clamp_min(1.)
    target = torch.as_tensor(values['target'], dtype=torch.float64).detach()
    anchor = torch.as_tensor(values['anchor_target'], dtype=torch.float64)[None].expand_as(ref).detach()
    label = (score(target) - score(anchor)).detach()
    if not all(torch.isfinite(v).all() for v in (cost, scale, label, gradient)):
        raise ValueError('Nonfinite deployed-reference cost supervision')
    return {'score': score, 'reference': ref.detach(), 'reference_cost': cost.detach(),
            'gradient': gradient, 'scale': scale.detach(), 'label_effect': label,
            'nonanchor': torch.as_tensor(nonanchor)}


def deployed_contrast(cache, response):
    if response.shape != cache['reference'].shape or not torch.isfinite(response).all():
        raise ValueError('Finite paired response shape required')
    predicted = cache['score'](cache['reference'] + response.double()) - cache['reference_cost']
    error = (predicted - cache['label_effect']).abs() / cache['scale']
    return error[cache['nonanchor']].mean()


def cost_response_loss(response, calls, indices, slices, weights, caches, penalty, cost_weight):
    loss, physical = factorial_response_loss(response, calls, indices, slices, weights,
        {'geometry': 'vector_l2', 'weighting': 'point'}, penalty)
    if cost_weight not in (0., .1):
        raise ValueError('Only preregistered L2 / .1 deployed cost contrast supported')
    auxiliary = []
    for i, (start, end) in zip(indices, slices):
        if weights['cost'][i] > 0:
            if i not in caches:
                raise ValueError('Complete cost call missing deployed-reference cache')
            term = deployed_contrast(caches[i], response[start:end]) * weights['cost'][i]
        else:
            if i in caches:
                raise ValueError('Incomplete call entered deployed-reference cost cache')
            term = response[start:end].sum().double() * 0.
        auxiliary.append(term)
    measured = torch.stack(auxiliary).mean()  # fixed batch-size denominator
    return loss + cost_weight * measured, torch.cat([physical.double(), measured[None]])


def audit_deployed_score(context, values, reference, score, cache):
    """Original-engine full costs + V11 finite-difference criteria at actual M.

    Piecewise derivatives only; role switches/norm/abs kinks explicitly counted.
    This is an implementation audit, not a smoothness/safety proof.
    """
    protocol = json.loads((ROOT.parent / 'cwm_v11/protocol.json').read_text())
    actions = values['proposed'][:, :, context['agent']]
    reference = np.asarray(reference, dtype=np.float64)
    count = len(actions)
    paths = {'deployed_reference': reference, 'action_truth': values['target'],
             'anchor_truth': np.repeat(values['anchor_target'][None], count, 0),
             'stress': reference + .2 * np.sin(np.arange(reference.size).reshape(reference.shape))}
    errors, original = {}, {}
    for name, path in paths.items():
        original[name] = original_scores(context, actions, path)
        calculated = score(torch.from_numpy(path)).detach().numpy()
        if not np.allclose(original[name], calculated, rtol=protocol['score_rtol'], atol=protocol['score_atol']):
            raise ValueError('Deployed full original cost mismatch: ' + name)
        errors[name] = float(np.max(np.abs(original[name] - calculated)))
    if not np.allclose(cache['label_effect'].numpy(), original['action_truth'] - original['anchor_truth'],
                       rtol=protocol['score_rtol'], atol=protocol['score_atol']):
        raise ValueError('Original truth effect does not match detached label')
    step = protocol['finite_difference_step_m']
    gaps = score.role_gap(torch.from_numpy(reference)).detach().numpy()
    checked, skipped = [], 0
    for candidate in sorted({0, count // 2, count - 1}):
        for t, xyz in protocol['finite_difference_coordinates']:
            plus, minus = reference[candidate:candidate + 1].copy(), reference[candidate:candidate + 1].copy()
            plus[0, t, xyz] += step
            minus[0, t, xyz] -= step
            high = original_scores(context, actions[candidate:candidate + 1], plus)[0]
            low = original_scores(context, actions[candidate:candidate + 1], minus)[0]
            midpoint = original['deployed_reference'][candidate]
            second = abs(high + low - 2 * midpoint) / step
            if gaps[candidate] <= 4 * step or second > protocol['nondifferentiable_second_difference_per_step_threshold']:
                skipped += 1
                continue
            numerical = (high - low) / (2 * step)
            automatic = float(cache['gradient'][candidate, t, xyz])
            if not np.isclose(numerical, automatic, rtol=protocol['gradient_rtol'], atol=protocol['gradient_atol']):
                raise ValueError('Original deployed gradient mismatch')
            checked.append(abs(numerical - automatic))
    if not checked:
        raise ValueError('No smooth deployed derivative coordinate verified for call')
    return {'score_max_absolute_errors': errors, 'gradient_checked_coordinates': len(checked),
            'gradient_skipped_nonsmooth_coordinates': skipped,
            'gradient_max_absolute_error': max(checked)}


def contrast_metrics(calls, indices, outputs, caches):
    grouped = collections.defaultdict(list)
    for i, output in zip(indices, outputs):
        if i not in caches:
            continue
        with torch.no_grad():
            measured = float(deployed_contrast(caches[i], torch.from_numpy(output['response'])))
        grouped[calls[i]['record']['group']].append(measured)
    if set(grouped) != {calls[i]['record']['group'] for i in indices}:
        raise ValueError('Complete cost metric missing a group')
    by_group = {g: {'calls': len(v), 'normalized_cost_contrast_l1': float(np.mean(v))}
                for g, v in grouped.items()}
    return {'group_equal_normalized_cost_contrast_l1': float(np.mean([v['normalized_cost_contrast_l1']
                                                                                for v in by_group.values()])),
            'complete_cost_calls': sum(len(v) for v in grouped.values()), 'cost_contrast_by_group': by_group}
