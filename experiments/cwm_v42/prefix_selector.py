"""Fixed prefix research local selection under the unchanged planner deadline."""
import copy
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent/n) for n in ('cwm_v41', 'cwm_v27', 'cwm_v11')]
from benchmark_loader import BenchmarkScorers
from bounded_candidates import expand_candidates
from qualification_gate import preserved_cpu_rng
from task_score import original_scores

SCORES = ('original_gru', 'constant_velocity', 'own_motion', 'cv_motion_only',
          'cv_l2', 'cv_full_query_rank_l2', 'cv_prefix_rank_l2')
NEURAL = {'cv_motion_only', 'cv_l2', 'cv_full_query_rank_l2', 'cv_prefix_rank_l2'}


def ready(bundle):
    return (isinstance(bundle, BenchmarkScorers) and bundle.authorized is True and
        bundle.deployment_control_eligible is False and bundle.legacy_l2_research_eligible is False and
        bundle.prefix_bundle is not None and bundle.prefix_bundle.authorized is True and
        bundle.prefix_bundle.deployment_control_eligible is False and
        set(bundle.scorers) == NEURAL and bundle.proposer is not None)


def revoke(bundle, status):
    bundle.authorized, bundle.status = False, status
    if bundle.prefix_bundle is not None:
        bundle.prefix_bundle.authorized, bundle.prefix_bundle.status = False, status


def select_prefix(call, history=None, belief_reference=None, bundle=None, *, mode='off',
                  score='cv_prefix_rank_l2', expand=expand_candidates, scorer=original_scores):
    """Public forecasts score UAV candidates; they never become UAV actions.

    Off/refusal never accesses optional contexts. Runtime failure discards all
    expanded proposals and returns original local anchor/costs, revoking research.
    Numeric fallback cannot undo prior changed actions or elapsed deadline time.
    Injectable expansion/scoring functions are unit-test fixtures only.
    """
    selected = np.asarray(call['selected']).copy()
    costs = np.asarray(call['selected_costs']).copy()
    fallback = {'selected': selected, 'selected_costs': costs, 'evidence': None,
        'info': {'status': 'off', 'deployment_control_eligible': False,
                 'research_action_selected': False, 'latency_ms': 0.}}
    if mode == 'off':
        return fallback
    if mode not in ('shadow', 'research_eval') or not ready(bundle):
        fallback['info']['status'] = 'research_refused'
        return fallback
    started = time.perf_counter()
    try:
        with preserved_cpu_rng():
            if score not in SCORES:
                raise ValueError('Fixed seven public scoring controls required')
            public = copy.deepcopy(call)
            config = public['planner'].config
            if (config.horizon_steps != 8 or config.dt_seconds != .1 or config.max_speed_mps != 5 or
                np.asarray(public['scenarios'].trajectories).shape != (1, 8, 3)):
                raise ValueError('Unchanged K1/horizon/speed/time-step contract required')
            result = expand(public, history, belief_reference, bundle.proposer, mode='shadow', source='shared')
            if result['inputs'] is None:
                if result['info']['status'] != 'missing_received_peer_context':
                    raise ValueError('Public bounded proposal/forecast failed')
                fallback['info']['status'] = 'missing_received_peer_context'
                return fallback
            if result['info']['status'] != 'shadow_candidates_ready' or result['info']['control_eligible'] is not False:
                raise ValueError('Only bounded noncontrolling public proposals supported')
            actions = result['actions'].copy(); values = tuple(x.copy() for x in result['inputs'])
            if (len(values) != 6 or actions.ndim != 3 or actions.shape[1:] != (8, 3) or
                actions.dtype != np.float64 or not 1 <= len(actions) <= 24 or not np.isfinite(actions).all() or
                np.linalg.norm(actions, axis=-1).max() > 5.+1e-8 or
                not any(a.tobytes() == selected.tobytes() for a in actions) or
                values[2].shape != (len(actions), 8, 4, 3) or values[3].shape != (8, 4, 3) or
                not np.array_equal(values[2][:, :, public['agent']], actions) or
                not np.array_equal(values[3][:, public['agent']], selected)):
                raise ValueError('Bounded BASE actions and original-anchor joint contract required')
            for peer in range(4):
                if peer == public['agent']: continue
                received = np.asarray(public['peer_sequences'][peer])
                if (not np.array_equal(values[3][:, peer], received) or
                    any(not np.array_equal(joint[:, peer], received) for joint in values[2])):
                    raise ValueError('Fresh received peer plans must remain unchanged in every candidate')
            forecast = None
            if score in ('original_gru', 'constant_velocity'):
                paths = np.repeat(values[4 if score == 'original_gru' else 5][None], len(actions), 0)
            else:
                name = 'cv_prefix_rank_l2' if score == 'own_motion' else score
                forecast, info = bundle.scorers[name].forecast(*values)
                if forecast is None or info['control_eligible'] is not False:
                    raise ValueError('Selected public scoring forecast unavailable')
                paths = forecast['reference' if score == 'own_motion' else 'prediction'].copy()
            score_values = np.asarray(scorer(copy.deepcopy(public), actions.copy(), paths.copy()))
            if score_values.shape != (len(actions),) or not np.isfinite(score_values).all():
                raise ValueError('Finite original full cost for every BASE candidate required')
            index = int(np.argmin(score_values))
            chosen = actions[index].copy()
            original_costs = public['planner']._local_scenario_cost_matrix(public['observation'], public['scenarios'],
                public['agent'], chosen[None], public['known'], public['peer_sequences'])[0].copy()
            if original_costs.shape != costs.shape or not np.isfinite(original_costs).all():
                raise ValueError('Original-scenario selected cost-row contract required')
            evidence = dict(zip(('history', 'relative', 'proposed', 'anchor', 'backbone', 'cv'), values))
            evidence.update(actions=actions, scoring_paths=paths, scoring_costs=score_values,
                original_anchor=selected.copy(), original_anchor_costs=costs.copy(), proposed_selection=chosen.copy())
            if forecast is not None: evidence.update({k: v.copy() for k, v in forecast.items()})
            active = mode == 'research_eval'
            controlled = chosen if active else selected
            return {'selected': controlled.copy(), 'selected_costs': original_costs if active else costs,
                'evidence': evidence, 'info': {'status': 'research_selection_ready' if active else 'shadow_selection_only',
                    'deployment_control_eligible': False, 'research_action_selected': active,
                    'choice_changed': not np.array_equal(controlled, selected), 'score': score,
                    'candidate_count': len(actions), 'selected_index': index, 'proposal_info': result['info'],
                    'latency_ms': (time.perf_counter()-started)*1000}}
    except Exception as error:
        revoke(bundle, 'selection_failed')
        fallback['info'].update(status='selection_failed', error_type=type(error).__name__)
        return fallback
    finally:
        fallback['info']['latency_ms'] = (time.perf_counter()-started)*1000
