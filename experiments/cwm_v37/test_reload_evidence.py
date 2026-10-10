"""Synthetic mutation and independent original-score reconstruction fixtures."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent/'cwm_v36'),
    str(Path(__file__).resolve().parent.parent/'cwm_v35')]
import reload_streamed as reload
import decision_gate
from test_audited_stream import toy_call
from evidence import require_same_summary


def test_existing_reload_output_rejected_before_native_reads(tmp_path, monkeypatch):
    tmp_path.joinpath('marker').write_bytes(b'user-owned')
    monkeypatch.setattr(reload, 'AuditedCorpus', lambda *a: pytest.fail('Must refuse existing output first'))
    with pytest.raises(ValueError): reload.replay(None, None, None, None, tmp_path)
    assert (tmp_path/'marker').read_bytes() == b'user-owned'


@pytest.mark.parametrize('kind', ['int_float', 'signed_zero', 'dtype', 'shape', 'keys'])
def test_all_typed_prediction_array_fields_strict(kind):
    left = {'a': np.zeros((2, 3), dtype=np.float64)}; right = copy.deepcopy(left)
    if kind == 'int_float': right['a'] = right['a'].astype(np.int64)
    elif kind == 'signed_zero': right['a'][0, 0] = -0.
    elif kind == 'dtype': right['a'] = right['a'].astype(np.float32)
    elif kind == 'shape': right['a'] = right['a'].reshape(3, 2)
    else: right['extra'] = np.zeros(1)
    with pytest.raises(ValueError): reload.require_arrays(left, right, 'toy')


def test_summary_serialization_hash_exception_does_not_ignore_scientific_failure():
    left = {'models': [{'checkpoint_sha256': 'a'*64, 'development': {'gain': .1}}], 'primary_research_eligible': False}
    right = copy.deepcopy(left); right['models'][0]['checkpoint_sha256'] = 'b'*64
    require_same_summary(left, right)
    right['primary_research_eligible'] = True
    with pytest.raises(ValueError): require_same_summary(left, right)


def test_independent_full_cost_rows_match_trainer_and_exclude_extra_probes(monkeypatch):
    call = toy_call(); row, v = call['record'], call['values']
    row.update(episode_index=1, step=2, ordinal=0, variant='toy', actual_choice_pool_index=0)
    v.update(actual_candidates=v['proposed'][:3, :, 0].copy(), original_local_costs=np.arange(3, dtype=np.float64),
        public_geometry_indices=np.arange(3, dtype=np.int64))
    output = {'prediction': np.zeros((7, 8, 3)), 'reference': np.zeros((7, 8, 3))}
    score = lambda context, actions, paths: np.arange(len(actions), dtype=np.float64)+paths[:, 0, 0]
    monkeypatch.setattr(reload, 'original_scores', score); monkeypatch.setattr(decision_gate, 'original_scores', score)
    a = reload.independent_decisions(call, output, {'scope': 'SYNTHETIC'})
    b = decision_gate.build_decisions(call, output, {'scope': 'SYNTHETIC'})
    assert reload.exact_tree(a, b)
    assert a['bounded_shared']['indices'] == [0, 1, 2]
    output['prediction'][3:] = 100.
    assert reload.exact_tree(a, reload.independent_decisions(call, output, {}))
