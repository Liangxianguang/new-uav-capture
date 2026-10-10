"""Mock score/cache fixtures; no native finite-difference certificate."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent/'cwm_v35')]
import cost_runtime as runtime
from test_audited_stream import toy_call
from audited_stream import base_values
from stream_objective import public_batch
from ranking_loss import ranking_cache


class ToyCommon(torch.nn.Module):
    def __init__(self): super().__init__(); self.blocks = []
    def forward(self, history, relative, proposed, anchor, backbone, cv):
        self.blocks.append(len(history)); return cv, torch.zeros_like(cv), torch.zeros_like(cv)


@pytest.fixture
def toy_cache(tmp_path, monkeypatch):
    calls = [toy_call('a', tail_invalid=True), toy_call('b')]
    class Corpus:
        def __len__(self): return len(calls)
        def batch(self, ids): return [calls[i] for i in ids]
    corpus = Corpus(); corpus.records = [c['record'] for c in calls]
    def audited(corpus, call, configs, reference):
        v = base_values(call['values'])
        return ranking_cache(lambda t: t.square().sum((1, 2)), reference, v), {'scope': 'SYNTHETIC_NOT_NATIVE'}
    monkeypatch.setattr(runtime, 'make_audited_cache', audited)
    monkeypatch.setattr(runtime, 'require_cost_population', lambda corpus, ids: {
        'groups': 64, 'complete_calls': len(ids), 'family_axes': {'SYNTHETIC': ['all4']}})
    monkeypatch.setattr(runtime, 'context_for', lambda *a: {'scope': 'SYNTHETIC'})
    monkeypatch.setattr(runtime, 'FrozenLocalScore', lambda *a: lambda t: t.square().sum((1, 2)))
    mean = np.zeros((1, 1, 252)); scale = np.ones_like(mean); common = ToyCommon()
    cache = runtime.CanonicalCache.build(common, corpus, None, mean, scale, tmp_path/'cache')
    return cache, calls, common


def test_all_global_block_canonical_arrays_and_base_only_cache_rebuild(toy_cache):
    cache, calls, common = toy_cache
    assert common.blocks == [14]
    assert cache.manifest['calls'] == 2 and cache.manifest['complete_calls'] == 2
    cache.require_response_permit()
    for i, call in enumerate(calls):
        assert cache.arrays(i)['reference_augmented'].shape == (7, 8, 3)
        assert cache.load(i, call)['reference'].shape == (3, 8, 3)
    assert not calls[0]['values']['valid'].all()  # Extra terminal probe does NOT drop BASE cost support.
    assert set(cache.batch([0, 1], calls)) == {0, 1}


@pytest.mark.parametrize('kind', ['array', 'manifest', 'score_audit'])
def test_changed_canonical_evidence_refuses_response_permit(toy_cache, kind):
    cache, _, _ = toy_cache
    path = cache.directory/('0.npz' if kind == 'array' else 'manifest.json' if kind == 'manifest' else 'score_audit.json')
    path.write_bytes(b'changed')
    with pytest.raises((ValueError, json.JSONDecodeError)): cache.require_response_permit()


def test_cache_tensor_semantics_verified_even_if_manifest_hash_updated(toy_cache):
    cache, calls, _ = toy_cache; values = cache.arrays(0)
    values['temperature'] += 1.
    path = cache.directory/'0.npz'; np.savez_compressed(path, **values)
    cache.manifest['files']['0']['sha256'] = runtime.sha(path)
    with pytest.raises(ValueError, match='ranking tensor differs'): cache.load(0, calls[0])


def test_bad_candidate_dependent_common_cannot_create_certificate(tmp_path, monkeypatch):
    call = toy_call()
    class Corpus:
        def __len__(self): return 1
        def batch(self, ids): return [call]
    class BadCommon(ToyCommon):
        def forward(self, *inputs):
            target, motion, response = super().forward(*inputs)
            return target+torch.arange(len(target), dtype=torch.float64)[:, None, None], motion, response
    monkeypatch.setattr(runtime, 'make_audited_cache', lambda *args: pytest.fail('Must reject common first'))
    directory = tmp_path/'bad'
    with pytest.raises(ValueError, match='candidate-dependent'):
        runtime.CanonicalCache.build(BadCommon(), Corpus(), None, np.zeros((1, 1, 252)), np.ones((1, 1, 252)), directory)
    assert not (directory/'manifest.json').exists()
