"""Synthetic training plumbing only; no native data optimizer/weights/results."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent/'cwm_v35')]
import train_streamed as entry
import cost_runtime
from test_audited_stream import toy_call
from audited_stream import base_values
from ranking_loss import ranking_cache


def normalizer(): return np.zeros((1, 1, 252)), np.ones((1, 1, 252))


def test_existing_output_refused_before_data_or_network_reads(tmp_path, monkeypatch):
    output = tmp_path/'existing'; output.mkdir(); (output/'marker').write_bytes(b'user-owned')
    monkeypatch.setattr(entry, 'AuditedCorpus', lambda *a: pytest.fail('No data access'))
    monkeypatch.setattr(entry, 'published_sources', lambda: pytest.fail('No network access'))
    with pytest.raises(ValueError): entry.train(tmp_path/'absent', tmp_path/'absent', output)
    assert (output/'marker').read_bytes() == b'user-owned'


def test_partial_audit_cannot_reach_model_output_normalizer_or_optimizer(tmp_path, monkeypatch):
    audit = tmp_path/'audit'; audit.mkdir(); (audit/'summary.json').write_text('{"status":"partial"}', encoding='utf8')
    monkeypatch.setattr(entry, 'published_sources', lambda: pytest.fail('No partial source authorization'))
    monkeypatch.setattr(torch.optim, 'Adam', lambda *a, **k: pytest.fail('No partial optimizer'))
    output = tmp_path/'never-created'
    with pytest.raises(ValueError): entry.train(tmp_path/'data', audit, output)
    assert not output.exists()


def test_independent_block_normalizer_reconstruction_and_development_exclusion():
    generator = np.random.default_rng(3601)
    calls = [toy_call(), toy_call(), toy_call(split='development_validation')]
    for call in calls: call['values']['history'] = generator.normal(size=(8, 252))
    visited = []
    def load(i):
        visited.append(i)
        assert calls[i]['record']['split'] == 'train'
        return calls[i]
    corpus = SimpleNamespace(indices={'train': [0, 1]}, load=load)
    from stream_normalization import stream_history_normalization
    mean, scale, count = stream_history_normalization(c['values']['history'] for c in calls[:2])
    entry.verify_train_normalizer(corpus, mean, scale, count)
    assert visited == [0, 1]
    mean = mean.copy(); mean[0, 0, 0] += .01
    with pytest.raises(ValueError): entry.verify_train_normalizer(corpus, mean, scale, count)


def test_identical_untrained_query_state_budget_and_optimizer_ownership():
    torch.set_num_threads(1)
    torch.manual_seed(3602); full = entry.PrefixQueryResponse(False)
    torch.manual_seed(3602); prefix = entry.PrefixQueryResponse(True)
    assert entry.state_digest(full) == entry.state_digest(prefix)
    common = entry.CalibratedOrigin(entry.LocalTwoHead('motion_only', 2.5, 2.5), 'cv')
    model = entry.FrozenOriginResponse(common, prefix)
    parameters = entry.checked_response_parameters(model)
    assert sum(p.numel() for p in parameters) == 31432
    assert not {id(p) for p in parameters} & {id(p) for p in model.common.parameters()}
    next(model.common.parameters()).requires_grad_(True)
    with pytest.raises(ValueError): entry.checked_response_parameters(model)


def test_synthetic_full80epoch_response_smoke_preserves_common_and_all_budget():
    """Toy optimizer plumbing; never a scientific V34 native training run."""
    torch.set_num_threads(1); torch.manual_seed(3603)
    calls = [toy_call('a'), toy_call('b')]
    corpus = SimpleNamespace(indices={'train': [0, 1]}, batch=lambda ids: [calls[i] for i in ids])
    common = entry.CalibratedOrigin(entry.LocalTwoHead('motion_only', 2.5, 2.5), 'cv')
    model = entry.FrozenOriginResponse(common, entry.PrefixQueryResponse(True))
    before = entry.state_digest(model.common)
    def caches(ids, supplied):
        return {i: ranking_cache(lambda target: target.square().sum((1, 2)), np.zeros((3, 8, 3)), base_values(call['values']))
            for i, call in zip(ids, supplied)}
    cache = SimpleNamespace(batch=caches); weights = {'point': {0: 1., 1: 1.}, 'cost': {0: 1., 1: 1.}}
    parameters = entry.checked_response_parameters(model); optimizer = torch.optim.Adam(parameters, lr=.001)
    history, rng = entry.run_epochs(model, optimizer, 'cv_prefix_rank_l2', corpus, *normalizer(), {}, weights, cache,
        {'motion_epochs': 80, 'response_epochs': 80, 'batch_calls': 16, 'seed': 3603})
    assert [r['epoch'] for r in history] == list(range(1, 81)) and rng['bit_generator'] == 'PCG64'
    assert entry.state_digest(model.common) == before
    assert all(not p.requires_grad and p.grad is None for p in model.common.parameters())
    assert not torch.equal(model.response_core.response_head.weight, torch.zeros_like(model.response_core.response_head.weight))


@pytest.mark.parametrize('config', [{'response_epochs': 1, 'batch_calls': 16, 'seed': 1},
    {'response_epochs': 80, 'batch_calls': 1, 'seed': 1}])
def test_smaller_budget_cannot_replace_fixed_scientific_entry(config):
    with pytest.raises(ValueError): entry.run_epochs(None, None, 'cv_prefix_rank_l2', None, None, None, None, None, None, config)


def test_complete_base_cost_permit_requires_every_family_and_axis():
    contract, _ = entry.load_contract(); records = []
    for group in range(64):
        for axis in contract['geometry_axes']: records.append({'group': str(group), 'variant': axis})
    corpus = SimpleNamespace(records=records, protocol={'variants': [{'name': a} for a in contract['geometry_axes']]})
    assert cost_runtime.require_cost_population(corpus, list(range(256)))['groups'] == 64
    with pytest.raises(ValueError): cost_runtime.require_cost_population(corpus, list(range(255)))


def test_motion_optimizer_permit_rejects_missing_or_fake_cost_audits(tmp_path):
    corpus = SimpleNamespace(audit_report={'training_authorized_by_this_report': True})
    with pytest.raises(ValueError): cost_runtime.require_motion_permit(corpus, {}, tmp_path)
    fake = {'gru': {'status': 'partial', 'groups': 64}, 'cv': {'status': 'partial', 'groups': 64}}
    with pytest.raises(ValueError): cost_runtime.require_motion_permit(corpus, fake, tmp_path)


def test_canonical_response_optimizer_permit_refuses_partial_population(tmp_path):
    cache = cost_runtime.CanonicalCache(SimpleNamespace(__len__=lambda: 1), None, tmp_path, {'status': 'partial'})
    with pytest.raises(ValueError): cache.require_response_permit()


def test_untrained_toy_evaluation_streams_all_predictions_and_base_decisions(tmp_path, monkeypatch):
    """Fake original score/context/cache authority; not native qualification."""
    import decision_gate
    contract, _ = entry.load_contract(); calls = []
    for group, split in (('a', 'train'), ('b', 'development_validation')):
        for axis in contract['geometry_axes']:
            call = toy_call(group, split); row, v = call['record'], call['values']
            row.update(episode_index=len(calls), step=2, ordinal=0, variant=axis, actual_choice_pool_index=0)
            v.update(actual_candidates=v['proposed'][:3, :, 0].copy(), original_local_costs=np.arange(3, dtype=np.float64),
                public_geometry_indices=np.arange(3, dtype=np.int64))
            calls.append(call)
    class ToyCorpus(SimpleNamespace):
        def __len__(self): return len(self.records)
    corpus = ToyCorpus(records=[c['record'] for c in calls], protocol={'variants': [{'name': a} for a in contract['geometry_axes']]},
        indices={'train': [0, 1, 2, 3], 'development_validation': [4, 5, 6, 7]}, batch=lambda ids: [calls[i] for i in ids])
    def load(index, call):
        return ranking_cache(lambda target: target.square().sum((1, 2)), np.zeros((3, 8, 3)), base_values(call['values']))
    cache = SimpleNamespace(corpus=corpus, configs=None, load=load,
        arrays=lambda i: {'reference_augmented': np.zeros((7, 8, 3))})
    monkeypatch.setattr(entry, 'context_for', lambda *args: {'scope': 'SYNTHETIC'})
    monkeypatch.setattr(decision_gate, 'original_scores', lambda ctx, actions, paths: np.arange(len(actions), dtype=np.float64)+paths[:, 0, 0])
    torch.set_num_threads(1)
    common = entry.CalibratedOrigin(entry.LocalTwoHead('motion_only', 2.5, 2.5), 'cv')
    summary, decisions = entry.evaluate(common, corpus, *normalizer(), cache, tmp_path/'predictions')
    assert summary['train']['calls'] == summary['development_validation']['calls'] == 4
    assert len(list((tmp_path/'predictions').glob('*.npz'))) == 8
    assert all(rows.count == 4 for split in decisions.values() for rows in split.values())
    assert all(row['indices'] == [0, 1, 2] for row in decisions['train']['bounded_shared'])
    assert len(json.loads((tmp_path/'predictions/manifest.json').read_bytes())) == 8


def test_fixed80_motion_plumbing_repeats_exact_model_optimizer_RNG_history():
    """Two-call synthetic neural optimizers, NOT V34 scientific training."""
    from repeatability_identity import exact_tree
    torch.set_num_threads(1); calls = [toy_call('a'), toy_call('b')]
    corpus = SimpleNamespace(indices={'train': [0, 1]}, batch=lambda ids: [calls[i] for i in ids])
    outputs = []
    for _ in range(2):
        torch.manual_seed(3604)
        model = entry.CalibratedOrigin(entry.LocalTwoHead('motion_only', 2.5, 2.5), 'cv')
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        history, rng = entry.run_epochs(model, optimizer, 'cv_motion_only', corpus, *normalizer(), {'a': 1, 'b': 1}, None, None,
            {'motion_epochs': 80, 'response_epochs': 80, 'batch_calls': 16, 'seed': 3604})
        outputs.append({'weights': copy.deepcopy(model.state_dict()), 'optimizer': copy.deepcopy(optimizer.state_dict()),
            'history': history, 'sampler': rng, 'torch_rng': torch.get_rng_state()})
    assert exact_tree(outputs[0], outputs[1])
    assert len(outputs[0]['history']) == 80


def test_native_prefix_identity_conversion_collapse_refused():
    from input_precision import command_precision
    call = toy_call(); assert command_precision(call)['native_prefix_pairs'] == 4
    # Raw candidate differs from anchor, but both become1.0 in float32.
    call['values']['anchor'][0, 0, 0] = 1.
    call['values']['proposed'][:, 0, 0, 0] = 1.
    call['values']['proposed'][1, 0, 0, 0] = 1.+1e-10
    with pytest.raises(ValueError, match='hard public prefix-anchor identity'): command_precision(call)
