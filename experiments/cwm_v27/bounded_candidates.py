"""Default-off bounded PUBLIC candidate expansion; never a controller.

Target forecasts [N,8,3] are passed to the ORIGINAL candidate generator,
never interpreted as UAV actions. Every expanded pool is re-forecast with
the received delayed joint context and the unchanged original action anchor.
"""
import copy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / 'cwm_v18'), str(HERE.parent / 'cwm_v9'),
               str(HERE.parent / 'cwm_v25')]
from geometry_probes import public_geometry_pool, stable_unique
from public_provider import delayed_joint_context
from qualification_gate import preserved_cpu_rng

MAX_CANDIDATES = 24
MAX_ROUNDS = 2
SOURCES = ('cv', 'motion', 'response', 'shared')


def original_path_candidates(call, path):
    from encirclement3d.minimax_mpc import ScenarioTrajectorySet
    path = np.asarray(path, dtype=np.float64)
    if path.shape != (8, 3) or not np.isfinite(path).all():
        raise ValueError('Finite public TARGET trajectory required')
    planner = copy.deepcopy(call['planner'])
    return planner._local_candidate_sequences(call['observation'],
        ScenarioTrajectorySet(path[None], np.ones(1), dynamics_status='raw'),
        call['agent'], call['observation']['defender_positions'][call['agent']], call['known'])


def expand_candidates(call, history=None, belief_reference=None, adapter=None, *,
                      mode='off', source='shared', geometry_factory=public_geometry_pool,
                      path_factory=original_path_candidates):
    """Return research candidates plus forecasts, NOT a selected control action.

    off/refusal only copies the original local candidates; all other public
    context and checkpoint IO remain untouched. Any shadow exception discards
    ALL proposals and returns the complete original order/multiplicity.
    Factories are injectable only for implementation unit tests.
    """
    actual = np.asarray(call['local_candidates']).copy()
    fallback = {'actions': actual, 'inputs': None, 'forecast': None,
                'info': {'status': 'off', 'control_eligible': False,
                         'expanded': False, 'rounds': []}}
    if mode == 'off':
        return fallback
    if mode != 'shadow':
        fallback['info']['status'] = 'control_refused'
        return fallback
    try:
        with preserved_cpu_rng():
            if source not in SOURCES or adapter.gate.mode != 'shadow':
                raise ValueError('Noncontrolling shadow adapter/source required')
            # Neither factories nor predictors can mutate captured public context.
            public = copy.deepcopy(call)
            h = np.asarray(history).copy()
            if h.shape != (8, 252) or not np.issubdtype(h.dtype, np.floating) or not np.isfinite(h).all():
                raise ValueError('Original public eight-frame 252 encoding required')
            reference, velocity = belief_reference(public['observation'], public['planner'].config)
            reference, velocity = np.asarray(reference).copy(), np.asarray(velocity).copy()
            backbone = np.asarray(public['scenarios'].trajectories[0], dtype=np.float64).copy()
            cv = reference[None] + .1 * np.arange(1, 9)[:, None] * velocity[None]

            def inputs_for(pool):
                context = delayed_joint_context(public['observation'], public['known'],
                    public['peer_sequences'], public['agent'], pool, public['selected'], reference, velocity)
                if context is None:
                    return None
                proposed, anchor, relative = context
                return tuple(x.copy() for x in (h, relative, proposed, anchor, backbone, cv))

            # Skip before invoking ANY proposal factory/checkpoint on missing peers.
            if inputs_for(actual) is None:
                fallback['info']['status'] = 'missing_received_peer_context'
                return fallback
            pool = stable_unique(geometry_factory(copy.deepcopy(public), reference.copy(), velocity.copy()))
            if len(pool) > MAX_CANDIDATES:
                raise ValueError('Initial pool exceeds fixed candidate budget')
            actual_unique = stable_unique(actual)
            if not np.array_equal(pool[:len(actual_unique)], actual_unique):
                raise ValueError('Original candidates must be unchanged leading subset')
            if not any(np.array_equal(public['selected'], a) for a in pool):
                raise ValueError('Original action anchor missing')
            seed_count, rounds = len(pool), []
            for round_index in range(MAX_ROUNDS):
                values = inputs_for(pool)
                forecast, info = adapter.forecast(*values)
                if forecast is None or info['control_eligible']:
                    raise ValueError('Noncontrolling forecast unavailable')
                paths = []
                if source in ('cv', 'shared'):
                    paths.append(('cv', -1, cv))
                if source in ('motion', 'shared'):
                    paths.append(('motion', -1, forecast['reference'][0]))
                if source in ('response', 'shared'):
                    paths.extend(('response', i, p) for i, p in enumerate(forecast['prediction']))
                before, considered, added = len(pool), 0, []
                for origin, path_index, path in paths:
                    if len(pool) == MAX_CANDIDATES:
                        break
                    # Validate every returned action BEFORE appending; dedup is stable.
                    generated = stable_unique(path_factory(copy.deepcopy(public), path.copy()))
                    considered += 1
                    for action in generated:
                        if len(pool) == MAX_CANDIDATES:
                            break
                        if not any(np.array_equal(action, old) for old in pool):
                            added.append({'index': len(pool), 'origin': origin, 'path_index': path_index})
                            pool = np.concatenate([pool, action[None]])
                rounds.append({'round': round_index, 'before': before, 'after': len(pool),
                               'paths_considered': considered, 'added': added})
                if len(pool) == before or len(pool) == MAX_CANDIDATES:
                    break
            # Proposed actions changed. Do NOT reuse predictions from the old pool.
            values = inputs_for(pool)
            forecast, info = adapter.forecast(*values)
            if forecast is None or info['control_eligible']:
                raise ValueError('Final expanded-pool forecast unavailable')
            return {'actions': pool.copy(), 'inputs': values, 'forecast': forecast,
                    'info': {'status': 'shadow_candidates_ready', 'control_eligible': False,
                             'expanded': len(pool) > len(actual_unique), 'seed_count': seed_count,
                             'candidate_count': len(pool), 'source': source, 'rounds': rounds}}
    except Exception as error:
        fallback['info'].update(status='candidate_expansion_failed', error_type=type(error).__name__)
        return fallback
