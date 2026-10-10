import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_cardinality import clip_rows, duplicate_reason, public_interceptor, tracking_commands, summarize_records


def config(speed=5., perimeter=1.):
    return SimpleNamespace(horizon_steps=8, control_horizon_steps=8, perimeter_scales=[.75, 1.],
        role_perimeter_m=perimeter, slot_gain=1., target_velocity_gain=0., dt_seconds=.1, max_speed_mps=speed)


def test_public_delayed_position_role_and_tie_order():
    reference = np.zeros((8, 3))
    assert public_interceptor([2., 0., 0.], 0, {1: [1., 0., 0.]}, reference) == 1
    assert public_interceptor([1., 0., 0.], 0, {1: [1., 0., 0.]}, reference) == 0


def test_interceptor_candidates_do_not_depend_on_perimeter():
    reference, directions = np.ones((8, 3)), np.eye(3)
    actual = tracking_commands(np.zeros(3), 0, 0, reference, np.zeros(3), config(), directions)
    raw = tracking_commands(np.zeros(3), 0, 0, reference, np.zeros(3), config(), directions, False)
    assert duplicate_reason(actual, raw, True) == 'interceptor_perimeter_invariant'


def test_clip_collapse_vs_distinct_and_other_degeneracy():
    reference, directions, velocity = np.tile([100., 0., 0.], (8, 1)), np.eye(3), np.zeros(3)
    cfg = config(speed=1.)
    actual = tracking_commands(np.zeros(3), 0, 1, reference, velocity, cfg, directions)
    raw = tracking_commands(np.zeros(3), 0, 1, reference, velocity, cfg, directions, False)
    assert duplicate_reason(actual, raw, False) == 'speed_clipping_induced_duplicate'
    actual = tracking_commands(np.zeros(3), 0, 1, np.zeros((8, 3)), velocity, config(), directions)
    assert duplicate_reason(actual, actual, False) == 'distinct'
    identical = np.zeros((2, 8, 3))
    assert duplicate_reason(identical, identical, False) == 'other_public_geometry_duplicate'


def test_speed_limit_and_control_suffix():
    clipped = clip_rows(np.array([[100., 0., 0.], [0., 0., 0.]]), 5.)
    np.testing.assert_array_equal(clipped, [[5., 0., 0.], [0., 0., 0.]])
    cfg = config()
    cfg.control_horizon_steps = 2
    actual = tracking_commands(np.zeros(3), 0, 0, np.ones((8, 3)), np.zeros(3), cfg, np.eye(3))
    np.testing.assert_array_equal(actual[:, 2:], np.repeat(actual[:, 1:2], 6, axis=1))


@pytest.mark.parametrize('bad', ['shape', 'nan', 'role', 'unclipped_role'])
def test_diagnostic_contract_rejects(bad):
    actual, raw = np.zeros((2, 8, 3)), np.zeros((2, 8, 3))
    role = False
    if bad == 'shape':
        actual = actual[:1]
    elif bad == 'nan':
        actual[0, 0, 0] = np.nan
    elif bad == 'role':
        actual[1, 0, 0] = 1.
        role = True
    else:
        raw[1, 0, 0] = 1.
        role = True
    with pytest.raises(ValueError):
        duplicate_reason(actual, raw, role)


def test_group_equal_summary_separates_full_support():
    rows = []
    for split in ('train', 'development_validation'):
        rows += [{'split': split, 'group': g, 'complete_union': full, 'unique_candidates': n,
                  'duplicate_reason': 'distinct' if n == 2 else 'interceptor_perimeter_invariant'}
                 for g, full, n in [('a', True, 1), ('a', False, 1), ('b', True, 2)]]
    result = summarize_records(rows)
    assert result['train']['all_collected']['group_equal_single_candidate_fraction'] == .5
    assert result['development_validation']['complete_union']['calls'] == 2


def test_protocol_remains_diagnostic_and_holdout_unread():
    import json
    p = json.loads((Path(__file__).resolve().parent / 'diagnostic_protocol.json').read_text())
    assert p['source_calls'] == 1536
    assert p['two_independent_runs_required']
    assert not any(p[k] for k in ('enhanced_control_enabled', 'new_model_trained', 'holdout_used', 'prior_gates_overridden'))
