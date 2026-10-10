"""Optional research local selection; original plan deadline and CBF untouched."""
import copy
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'cwm_v27'), str(HERE.parent / 'cwm_v11')]
from research_bundle import ResearchBundle
from bounded_candidates import expand_candidates
from qualification_gate import preserved_cpu_rng
from task_score import original_scores

SCORES = ('original_gru', 'constant_velocity', 'own_motion', 'cv_motion_only', 'cv_l2', 'cv_rank_l2')


def select_research(call, history=None, belief_reference=None, bundle=None, *, mode='off',
                    score='cv_rank_l2', expand=expand_candidates, scorer=original_scores):
    """Off/refusal ignores histories, models, candidates and public contexts.

    Injectable expansion/score functions are implementation-test fixtures only.
    Always start from the original solver's selected anchor and exact cost row.
    On failure retain these, discard ALL expansion, disable further research.
    Failure fallback is numeric, NOT a claim about real-time plan equivalence:
    original plan timeout still counts proposal/inference/scoring time.
    """
    selected = np.asarray(call['selected']).copy()
    costs = np.asarray(call['selected_costs']).copy()
    fallback = {'selected': selected, 'selected_costs': costs, 'evidence': None,
                'info': {'status': 'off', 'deployment_control_eligible': False,
                         'research_action_selected': False, 'latency_ms': 0.}}
    if mode == 'off':
        return fallback
    if (mode not in ('research_eval', 'shadow') or not isinstance(bundle, ResearchBundle) or
        bundle.authorized is not True or bundle.deployment_control_eligible is not False):
        fallback['info']['status'] = 'research_refused'
        return fallback
    started = time.perf_counter()
    try:
        with preserved_cpu_rng():
            if score not in SCORES:
                raise ValueError('Fixed public scoring control required')
            public = copy.deepcopy(call)
            config = public['planner'].config
            if (config.horizon_steps != 8 or config.dt_seconds != .1 or config.max_speed_mps != 5 or
                np.asarray(public['scenarios'].trajectories).shape != (1, 8, 3)):
                raise ValueError('Frozen K1/horizon/speed/time-step contract differs')
            result = expand(public, history, belief_reference, bundle.proposer, mode='shadow', source='shared')
            if result['inputs'] is None:
                if result['info']['status'] != 'missing_received_peer_context':
                    raise ValueError('Public proposal/forecast failed')
                fallback['info']['status'] = result['info']['status']
                return fallback
            if result['info']['status'] != 'shadow_candidates_ready' or result['info']['control_eligible']:
                raise ValueError('Only bounded noncontrolling public proposals are supported')
            actions, values = result['actions'].copy(), tuple(x.copy() for x in result['inputs'])
            if (actions.ndim != 3 or actions.shape[1:] != (8, 3) or not 1 <= len(actions) <= 24 or
                actions.dtype != np.float64 or not np.isfinite(actions).all() or
                np.linalg.norm(actions, axis=-1).max() > 5.+1e-8 or
                not any(np.array_equal(a, selected) for a in actions) or
                values[2].shape != (len(actions), 8, 4, 3) or
                not np.array_equal(values[2][:, :, public['agent']], actions) or
                not np.array_equal(values[3][:, public['agent']], selected)):
                raise ValueError('Bounded original-anchor joint action contract differs')
            if score in ('original_gru', 'constant_velocity'):
                path = values[4 if score == 'original_gru' else 5]
                paths = np.repeat(path[None], len(actions), axis=0)
                forecast = None
            else:
                name = 'cv_rank_l2' if score == 'own_motion' else score
                forecast, info = bundle.scorers[name].forecast(*values)
                if forecast is None or info['control_eligible']:
                    raise ValueError('Public selected scoring forecast unavailable')
                paths = forecast['reference' if score == 'own_motion' else 'prediction'].copy()
            score_values = np.asarray(scorer(copy.deepcopy(public), actions.copy(), paths.copy()))
            if score_values.shape != (len(actions),) or not np.isfinite(score_values).all():
                raise ValueError('Finite original full cost per candidate required')
            index = int(np.argmin(score_values))  # Same first-index tie rule as original.
            chosen = actions[index].copy()
            # Preserve parent API's ORIGINAL-scenario cost row. Team metadata is
            # still evaluated by the unchanged original planner, not fabricated
            # from learned scores or branch labels.
            original_cost = public['planner']._local_scenario_cost_matrix(public['observation'], public['scenarios'],
                public['agent'], chosen[None], public['known'], public['peer_sequences'])[0].copy()
            if original_cost.shape != costs.shape or not np.isfinite(original_cost).all():
                raise ValueError('Original selected cost-row contract differs')
            evidence = dict(zip(('history', 'relative', 'proposed', 'anchor', 'backbone', 'cv'), values))
            evidence.update(actions=actions, scoring_paths=paths, scoring_costs=score_values,
                            original_anchor=selected.copy(), original_anchor_costs=costs.copy(), proposed_selection=chosen.copy())
            if forecast is not None:
                evidence.update({k: v.copy() for k, v in forecast.items()})
            chosen_for_control = chosen if mode == 'research_eval' else selected
            return {'selected': chosen_for_control.copy(),
                'selected_costs': original_cost if mode == 'research_eval' else costs,
                'evidence': evidence,
                'info': {'status': 'research_selection_ready' if mode == 'research_eval' else 'shadow_selection_only',
                    'deployment_control_eligible': False, 'research_action_selected': mode == 'research_eval',
                    'choice_changed': not np.array_equal(chosen_for_control, selected), 'score': score,
                    'candidate_count': len(actions), 'selected_index': index,
                    'proposal_info': result['info'], 'latency_ms': (time.perf_counter()-started)*1000}}
    except Exception as error:
        bundle.authorized, bundle.status = False, 'selection_failed'
        fallback['info'].update(status='selection_failed', error_type=type(error).__name__)
        return fallback
    finally:
        fallback['info']['latency_ms'] = (time.perf_counter()-started)*1000
