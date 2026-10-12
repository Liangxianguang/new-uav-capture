"""Prospective coordinate-local audit. Old V23/V36 outputs remain invalid.

No training entry is installed by this module. All original numerical thresholds,
candidate indices, coordinates and full-cost comparisons remain unchanged.
FrozenLocalScore.roles depends ONLY on paths[:,0]; a later-time perturbation
cannot change that discrete role, even at an exact first-time role tie.
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'cwm_v23'))
from deployed_cost_loss import original_scores, FrozenLocalScore


def role_coordinate_isolated(t, gap, step):
    if type(t) is not int or not 0 <= t < 8:
        raise ValueError('Frozen horizon coordinate required')
    # Keep the old conservative margin for first-time coordinates.
    return t > 0 or gap > 4 * step


def audit_deployed_score(context, values, reference, score, cache):
    if type(score) is not FrozenLocalScore:
        raise ValueError('Only frozen first-time-role score supports this audit')
    protocol = json.loads((HERE.parent/'cwm_v11/protocol.json').read_bytes())
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
    checked, skipped, evidence = [], 0, []
    for candidate in sorted({0, count // 2, count - 1}):
        for t, xyz in protocol['finite_difference_coordinates']:
            plus, minus = reference[candidate:candidate+1].copy(), reference[candidate:candidate+1].copy()
            plus[0,t,xyz] += step; minus[0,t,xyz] -= step
            roles = [int(score.roles(torch.from_numpy(p))[0]) for p in (minus, reference[candidate:candidate+1], plus)]
            if t > 0 and (not np.array_equal(plus[:,0], minus[:,0]) or len(set(roles)) != 1):
                raise ValueError('Later-time role isolation invariant failed')
            high = original_scores(context, actions[candidate:candidate+1], plus)[0]
            low = original_scores(context, actions[candidate:candidate+1], minus)[0]
            midpoint = original['deployed_reference'][candidate]
            second = abs(high + low - 2 * midpoint) / step
            role_safe = role_coordinate_isolated(t, float(gaps[candidate]), step)
            reason = 'role_margin' if not role_safe else ('cost_kink' if second > protocol['nondifferentiable_second_difference_per_step_threshold'] else 'checked')
            row = dict(candidate=candidate,t=t,xyz=xyz,roles=roles,role_gap=float(gaps[candidate]),
                       second_difference_per_step=float(second),status=reason)
            evidence.append(row)
            if reason != 'checked':
                skipped += 1
                continue
            numerical = (high-low)/(2*step)
            automatic = float(cache['gradient'][candidate,t,xyz])
            if not np.isclose(numerical, automatic, rtol=protocol['gradient_rtol'], atol=protocol['gradient_atol']):
                raise ValueError('Original deployed gradient mismatch')
            row.update(numerical=float(numerical),automatic=automatic)
            checked.append(abs(numerical-automatic))
    if not checked:
        raise ValueError('No smooth deployed derivative coordinate verified for call')
    return dict(score_max_absolute_errors=errors,gradient_checked_coordinates=len(checked),
                gradient_skipped_nonsmooth_coordinates=skipped,gradient_max_absolute_error=max(checked),
                coordinate_evidence=evidence,audit_version='v47_coordinate_local_role_isolation')
