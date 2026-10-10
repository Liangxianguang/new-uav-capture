"""Synthetic loader/weight fixtures, NEVER native replay or training evidence."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audited_stream as stream


def toy_call(group='g', split='train', tail_invalid=False):
    anchor = np.zeros((8, 4, 3), dtype=np.float64)
    base = np.repeat(anchor[None], 3, 0)
    base[1, :, 0, 0] = .5; base[2, :, 0, 1] = 1.
    joint, pairs = stream.augment(base, anchor, 0)
    def labels(commands):
        return {'target': commands[:, 0].cumsum(0), 'defenders': commands.cumsum(0),
            'commanded': commands.copy(), 'executed': commands.copy(), 'valid': np.ones(8, dtype=bool),
            'branch_sign_label_only': np.zeros(8, dtype=np.int8), 'termination': np.full(8, 'running', dtype='U128')}
    v = {key: np.stack([labels(c)[key] for c in joint]) for key in labels(anchor)}
    v.update({'anchor_'+key: value for key, value in labels(anchor).items()})
    v.update(proposed=joint, anchor=anchor, prefix_pairs=pairs, base_candidate_count=np.asarray(3, dtype=np.int64),
        bounded_shared_indices=np.arange(3, dtype=np.int64), history=np.ones((8, 252)), relative=np.zeros((4, 6)),
        backbone=np.zeros((8, 3)), reference=np.zeros(3), velocity=np.zeros(3))
    if tail_invalid:
        v['valid'][3:, 7] = False; v['termination'][3:, 7] = 'after_terminal'
    row = {'group': group, 'split': split, 'agent': 0, 'base_candidate_count': 3,
        'augmented_candidate_count': len(joint), 'prefix_pair_count': len(pairs)}
    return {'record': row, 'values': v}


@pytest.fixture
def accepted_synthetic_corpus(tmp_path, monkeypatch):
    """Mock ONLY the already-native-audited preflight/manifest authority."""
    data, audit = tmp_path/'data', tmp_path/'audit'; data.mkdir(); audit.mkdir()
    (data/'summary.json').write_bytes(b'{"scope":"SYNTHETIC_NOT_NATIVE"}')
    population, records, costs, checked, manifest = [], [], [], [], {}
    for group in range(64):
        split = 'train' if group < 48 else 'development_validation'
        for axis in range(8):
            population.append({'episode_index': group*8+axis, 'mirror_group_id': str(group),
                'model_split': split, 'variant': str(axis)})
        call = toy_call(str(group), split, tail_invalid=group % 2 == 0)
        row = call['record']; index = group*8
        row.update(episode_index=index, step=2, ordinal=0, variant='0',
            arrays_path=f'calls/{index}.npz', context_path=f'calls/{index}.json')
        path = data/row['arrays_path']; path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path, **call['values']); path.with_suffix('.json').write_bytes(b'{}')
        row.update(arrays_sha256=stream.sha(path), context_sha256=stream.sha(path.with_suffix('.json')))
        manifest.update({row['arrays_path']: row['arrays_sha256'], row['context_path']: row['context_sha256']})
        records.append(row); identity = [index, 2, 0, 0]
        checked.append({'identity': identity, 'status': 'all_public_prefix_native_branches_original_costs_checked'})
        complete = np.flatnonzero(call['values']['valid'].all(1)).tolist()
        costs.append({'identity': identity, 'complete_indices': complete, 'true_costs': [1.]*len(complete), 'base_library_complete': True})
    report = {'status': 'complete_independent_fresh_geometry_prefix_public_branch_cost_audit_passed',
        'data_summary_sha256': stream.sha(data/'summary.json'), 'data_gate_passed': True,
        'training_authorized_by_this_report': True, 'enhanced_control_enabled': False, 'new_model_trained': False,
        'holdout_used': False, 'protocol': {'scope': 'SYNTHETIC'}, 'source_hashes': {},
        'checked_calls': 64, 'supported_calls': 64, 'complete_cost_calls': 64}
    for name, value in (('checked_calls.json', checked), ('full_label_cost_checks.json', costs), ('audited_data_manifest.json', manifest)):
        path = audit/name; path.write_text(json.dumps(value), encoding='utf8')
        report[{'checked_calls.json': 'checked_calls_sha256', 'full_label_cost_checks.json': 'full_label_cost_checks_sha256',
            'audited_data_manifest.json': 'audited_data_manifest_sha256'}[name]] = stream.sha(path)
    (audit/'summary.json').write_text(json.dumps(report), encoding='utf8')
    preflight = (report['protocol'], {'calls': 64}, population, records, [{}]*512)
    monkeypatch.setattr(stream.data_audit, 'preflight', lambda _: preflight)
    monkeypatch.setattr(stream.data_audit, 'evidence_manifest', lambda *args: manifest)
    return data, audit


def test_all_toy_calls_checked_but_no_physical_arrays_cached(accepted_synthetic_corpus):
    corpus = stream.AuditedCorpus(*accepted_synthetic_corpus)
    assert len(corpus) == 64 and [len(v) for v in corpus.indices.values()] == [48, 16]
    def no_array(value):
        assert not isinstance(value, np.ndarray)
        if isinstance(value, dict):
            for child in value.values(): no_array(child)
        elif isinstance(value, (list, tuple)):
            for child in value: no_array(child)
    no_array(corpus.__dict__)
    assert len(corpus.batch([0, 1, 2])) == 3
    corpus.reverify()


def test_mutation_after_audit_refused_on_each_load(accepted_synthetic_corpus):
    corpus = stream.AuditedCorpus(*accepted_synthetic_corpus)
    path = corpus.data/corpus.records[0]['arrays_path']; path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='arrays changed'): corpus.load(0)


def test_mutated_audit_certificate_refused(accepted_synthetic_corpus):
    corpus = stream.AuditedCorpus(*accepted_synthetic_corpus)
    (corpus.audit/'summary.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='certificate changed'): corpus.reverify()


@pytest.mark.parametrize('change', ['prefix', 'history', 'mask', 'base_indices'])
def test_native_loaded_call_contract_refuses_tampering(change):
    call = toy_call(); v = call['values']
    if change == 'prefix': v['target'][3, 0] += 1
    elif change == 'history': v['history'][0, 0] = np.nan
    elif change == 'mask': v['valid'] = v['valid'].astype(np.int64)
    else: v['bounded_shared_indices'] = np.arange(4, dtype=np.int64)
    with pytest.raises(ValueError): stream.validate_call(call['record'], v)


@pytest.mark.parametrize('ids', [[], list(range(17)), [0, 0], [True], [-1], [64]])
def test_only_unique_bounded16_exact_index_batches(accepted_synthetic_corpus, ids):
    corpus = stream.AuditedCorpus(*accepted_synthetic_corpus)
    with pytest.raises(ValueError): corpus.batch(ids)


def test_normalization_train_only_uniform_calls_not_probes(accepted_synthetic_corpus):
    corpus = stream.AuditedCorpus(*accepted_synthetic_corpus)
    original = corpus.load; visited = []
    def load(i):
        visited.append(i); call = original(i)
        if call['record']['split'] != 'train': pytest.fail('Development normalizer leakage')
        call['values']['history'][:] = i
        return call
    corpus.load = load
    mean, scale, frames = corpus.normalization()
    assert visited == list(range(48)) and frames == 384
    assert np.allclose(mean, np.arange(48).mean()) and np.allclose(scale, np.arange(48).std())


def test_base_view_does_not_slice_history_when_base_count_equals8():
    v = {'base_candidate_count': np.asarray(8, dtype=np.int64), 'target': np.zeros((15, 8, 3)),
        'proposed': np.zeros((15, 8, 4, 3)), 'history': np.zeros((8, 252)), 'backbone': np.zeros((8, 3))}
    base = stream.base_values(v)
    assert base['target'].shape == (8, 8, 3) and base['proposed'].shape == (8, 8, 4, 3)
    assert base['history'] is v['history'] and base['backbone'] is v['backbone']


def test_extra_probe_terminal_masks_do_not_remove_complete_base_cost_support(accepted_synthetic_corpus):
    corpus = stream.AuditedCorpus(*accepted_synthetic_corpus)
    weights, population = stream.population_weights(corpus)
    assert population['calls'] == 48 and population['groups'] == 48
    assert all(v == 1. for v in weights['cost'].values())
    assert all(v == 1. for v in weights['point'].values())
    assert any(not corpus.load(i)['values']['valid'].all() for i in corpus.indices['train'])


def test_unequal_group_call_and_augmented_point_denominators_are_fixed():
    from types import SimpleNamespace
    calls = [toy_call('a'), toy_call('a', tail_invalid=True), toy_call('b')]
    corpus = SimpleNamespace(indices={'train': [0, 1, 2]}, load=lambda i: calls[i])
    weights, population = stream.population_weights(corpus)
    assert population['group_calls'] == {'a': 2, 'b': 1}
    assert weights['call'] == {0: .75, 1: .75, 2: 1.5}
    assert weights['cost'] == {0: .75, 1: .75, 2: 1.5}
    assert population['group_valid_nonanchor_points'] == {'a': 92, 'b': 48}
    assert weights['point'] == {0: 1.5*48/92, 1: 1.5*44/92, 2: 1.5}


def test_missing_base_complete_support_stops_all_group_training_weights():
    from types import SimpleNamespace
    calls = [toy_call('a'), toy_call('b')]
    calls[1]['values']['valid'][:3, 7] = False
    corpus = SimpleNamespace(indices={'train': [0, 1]}, load=lambda i: calls[i])
    with pytest.raises(ValueError, match='ALL TRAIN families'): stream.population_weights(corpus)


@pytest.mark.parametrize('name', ['../outside', 'a/../b', '/outside', 'a\\b', 'a:b', './a', 'a//b'])
def test_unsafe_evidence_path_refused(tmp_path, name):
    with pytest.raises(ValueError): stream.safe_file(tmp_path, name)


def test_partial_audit_rejected_before_collection_reads(tmp_path, monkeypatch):
    (tmp_path/'summary.json').write_text('{"status":"partial"}', encoding='utf8')
    monkeypatch.setattr(stream.data_audit, 'preflight', lambda _: pytest.fail('No partial data accepted'))
    with pytest.raises(ValueError): stream.AuditedCorpus(tmp_path/'not-created', tmp_path)
