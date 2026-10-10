"""Fixed population weighting for group-equal call/point paired objectives."""
import collections

import numpy as np
import torch


def paired_mask(values):
    nonanchor = ~(values['proposed'] == values['anchor'][None]).all((1, 2, 3))
    return values['valid'] & values['anchor_valid'][None] & nonanchor[:, None]


def population_weights(calls, indices):
    if not len(indices) or len(set(indices)) != len(indices):
        raise ValueError('Nonempty unique full training call population required')
    counts = collections.Counter(calls[i]['record']['group'] for i in indices)
    totals = collections.Counter()
    point_counts = {}
    for i in indices:
        point_counts[i] = int(paired_mask(calls[i]['values']).sum())
        if not point_counts[i]:
            raise ValueError('Empty full-train paired support')
        totals[calls[i]['record']['group']] += point_counts[i]
    factor = len(indices) / len(counts)
    weights = {'call': {}, 'point': {}}
    for i in indices:
        group = calls[i]['record']['group']
        weights['call'][i] = factor / counts[group]
        weights['point'][i] = factor * point_counts[i] / totals[group]
    if any(not np.isclose(sum(v.values()), len(indices), rtol=0, atol=1e-10) for v in weights.values()):
        raise ValueError('Full population weights do not normalize')
    return weights, {'calls': len(indices), 'groups': len(counts), 'group_calls': dict(counts),
                     'group_valid_nonanchor_points': dict(totals)}


def factorial_response_loss(response, calls, indices, slices, weights, objective, residual_penalty):
    if objective['geometry'] not in ('coordinate_mse', 'vector_l2') or objective['weighting'] not in ('call', 'point'):
        raise ValueError('Unknown fixed factorial objective')
    if len(indices) != len(slices) or not len(indices):
        raise ValueError('Incomplete paired batch')
    terms = []
    for i, (start, end) in zip(indices, slices):
        values = calls[i]['values']
        mask = torch.as_tensor(paired_mask(values), device=response.device)
        label = torch.as_tensor(values['target']-values['anchor_target'][None], dtype=response.dtype, device=response.device)
        predicted = response[start:end]
        if predicted.shape != label.shape or not mask.any():
            raise ValueError('Invalid response output/support')
        error = (predicted-label)[mask]
        distance = error.square().sum(-1)/3. if objective['geometry'] == 'coordinate_mse' else torch.linalg.vector_norm(error, dim=-1)
        energy = predicted[mask].square().sum(-1)/3.
        terms.append(torch.stack([distance.mean(), energy.mean()]) * weights[objective['weighting']][i])
    # Fixed batch denominator is an unbiased full-population objective estimate.
    measured = torch.stack(terms).mean(0)
    return measured[0]+residual_penalty*measured[1], measured
