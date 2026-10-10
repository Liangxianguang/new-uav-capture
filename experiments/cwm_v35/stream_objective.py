"""Fixed V34 augmented-physical / BASE-only ranking batch primitives.

No optimizer, selection, training CLI or qualification. Caller must complete
native data/normalizer/original-score gradient audits BEFORE optimization.
Reuse original packing/physical loss/KL; never edit those frozen sources.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from audited_stream import validate_call

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'cwm_v28'))
from train_ranking_response import public_cost_inputs
from ranking_loss import ranking_kl, factorial_response_loss


def public_batch(calls, mean, scale):
    if not 1 <= len(calls) <= 16:
        raise ValueError('Fixed bounded16-call public batch required')
    for call in calls: validate_call(call['record'], call['values'])
    if (mean.shape != (1, 1, 252) or scale.shape != mean.shape or
        mean.dtype != np.float64 or scale.dtype != np.float64 or
        not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale < .01)):
        raise ValueError('Audited fixed TRAIN-only normalizer shape/dtype/floor required')
    return public_cost_inputs(calls, list(range(len(calls))), mean, scale)


def batch_objective(response, calls, global_ids, slices, weights, caches):
    """All augmentation in vectorL2; ONLY original base rows in cost KL."""
    n = len(calls)
    if (not 1 <= n <= 16 or len(global_ids) != n or len(slices) != n or
        any(type(i) is not int or i < 0 for i in global_ids) or len(set(global_ids)) != n or
        response.ndim != 3 or response.shape[1:] != (8, 3) or not torch.isfinite(response).all()):
        raise ValueError('Finite full augmented response and unique bounded global batch required')
    cursor = 0
    for call, bounds in zip(calls, slices):
        validate_call(call['record'], call['values'])
        end = cursor+len(call['values']['proposed'])
        if tuple(bounds) != (cursor, end):
            raise ValueError('Complete contiguous augmented call slices required')
        cursor = end
    if cursor != len(response): raise ValueError('Full response batch length differs')
    local_weights = {'point': {j: weights['point'][i] for j, i in enumerate(global_ids)}}
    for i in global_ids:
        if not np.isfinite(weights['point'][i]) or weights['point'][i] <= 0 or not np.isfinite(weights['cost'][i]) or weights['cost'][i] < 0:
            raise ValueError('Fixed complete-TRAIN point/cost weights required')
    physical, terms = factorial_response_loss(response, calls, list(range(n)), slices, local_weights,
        {'geometry': 'vector_l2', 'weighting': 'point'}, .001)
    auxiliary = []
    for i, call, (start, end) in zip(global_ids, calls, slices):
        v = call['values']; count = int(v['base_candidate_count'])
        complete = bool(v['valid'][:count].all() and v['anchor_valid'].all())
        if complete != (weights['cost'][i] > 0):
            raise ValueError('Cost support MUST use unchanged BASE library, not augmented completeness')
        if complete:
            if i not in caches or caches[i]['reference'].shape != (count, 8, 3):
                raise ValueError('Complete fixed BASE-only frozen-reference ranking cache required')
            auxiliary.append(ranking_kl(caches[i], response[start:start+count])*weights['cost'][i])
        else:
            if i in caches: raise ValueError('Incomplete BASE call cannot enter cost supervision')
            auxiliary.append(response[start:end].sum().double()*0.)
    measured = torch.stack(auxiliary).mean()  # Fixed actual batch-call denominator.
    return physical+.1*measured, torch.cat([terms.double(), measured[None]])
