"""Pure public finite local-action probes; off path does not inspect context."""
import copy

import numpy as np


def stable_unique(values):
    result = []
    for value in values:
        value = np.asarray(value,dtype=np.float64)
        if value.shape != (8,3) or not np.isfinite(value).all() or np.linalg.norm(value,axis=-1).max() > 5.+1e-8:
            raise ValueError('Original finite local command contract violated')
        if not any(np.array_equal(value,old) for old in result):
            result.append(value.copy())
    if not result:
        raise ValueError('Empty original local candidate population')
    return np.stack(result)


def public_geometry_pool(call,reference,velocity,magnitude=3.,mode='shadow'):
    actual = np.asarray(call['local_candidates']).copy()
    if mode == 'off':
        return actual  # Preserve original order/multiplicity without inspecting anything else.
    if mode != 'shadow':
        raise ValueError('Geometry probes are not an enabled controller')
    actual_unique = stable_unique(actual)
    reference,velocity = np.asarray(reference),np.asarray(velocity)
    if reference.shape != (3,) or velocity.shape != (3,) or not np.isfinite(reference).all() or not np.isfinite(velocity).all() or magnitude != 3.:
        raise ValueError('Frozen public geometry probe contract mismatch')
    from encirclement3d.minimax_mpc import ScenarioTrajectorySet
    own = np.asarray(call['observation']['defender_positions'][call['agent']])
    direction = (reference-own).copy()
    direction[2] = 0.
    radial = direction/np.linalg.norm(direction) if np.linalg.norm(direction) > 1e-9 else np.array([1.,0.,0.])
    tangent = np.array([-radial[1],radial[0],0.])
    anchor = np.asarray(call['selected'])
    if not any(np.array_equal(anchor,a) for a in actual_unique):
        raise ValueError('Original reference action absent from actual candidate set')
    cv = reference[None]+.1*np.arange(1,9)[:,None]*velocity[None]
    planner = copy.deepcopy(call['planner'])
    cv_candidates = planner._local_candidate_sequences(call['observation'],ScenarioTrajectorySet(cv[None],np.ones(1),dynamics_status='raw'),
                                                        call['agent'],own,call['known'])
    pool = list(actual_unique)+list(cv_candidates)
    for axis in (radial,tangent):
        for sign in (-1.,1.):
            candidate = anchor+sign*magnitude*axis[None]
            candidate *= np.minimum(1.,5./np.maximum(np.linalg.norm(candidate,axis=-1,keepdims=True),1e-9))
            pool.append(candidate)
    pool.append(np.zeros((8,3)))
    return stable_unique(pool)
