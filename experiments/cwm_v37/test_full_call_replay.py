"""Toy cold evaluation stream with independent fake score authority."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent/'cwm_v36'),
    str(Path(__file__).resolve().parent.parent/'cwm_v35')]
import reload_streamed as cold
import train_streamed as training
import decision_gate
from test_audited_stream import toy_call
from audited_stream import base_values
from ranking_loss import ranking_cache


@pytest.fixture
def saved_toy_run(tmp_path, monkeypatch):
    contract, plan = training.load_contract(); calls = []
    for group, split in (('a', 'train'), ('b', 'development_validation')):
        for axis in contract['geometry_axes']:
            call = toy_call(group, split); row, v = call['record'], call['values']
            row.update(episode_index=len(calls), step=2, ordinal=0, variant=axis, actual_choice_pool_index=0)
            v.update(actual_candidates=v['proposed'][:3, :, 0].copy(), original_local_costs=np.arange(3, dtype=np.float64),
                public_geometry_indices=np.arange(3, dtype=np.int64))
            calls.append(call)
    class Corpus:
        def __len__(self): return len(calls)
        def batch(self, ids): return [calls[i] for i in ids]
    corpus = Corpus(); corpus.records = [c['record'] for c in calls]
    corpus.protocol = {'variants': [{'name': a} for a in contract['geometry_axes']]}
    corpus.indices = {'train': [0, 1, 2, 3], 'development_validation': [4, 5, 6, 7]}
    def score(context, actions, paths): return np.arange(len(actions), dtype=np.float64)+paths[:, 0, 0]
    def cache_for(call, reference):
        return ranking_cache(lambda targets: torch.arange(len(targets), dtype=torch.float64)+targets[:, 0, 0],
            reference, base_values(call['values']))
    cache = type('Cache', (), {})(); cache.corpus = corpus; cache.configs = None
    cache.arrays = lambda i: {'reference_augmented': np.zeros((7, 8, 3))}
    cache.load = lambda i, call: cache_for(call, np.zeros((3, 8, 3)))
    monkeypatch.setattr(training, 'context_for', lambda *args: {'scope': 'SYNTHETIC'})
    monkeypatch.setattr(decision_gate, 'original_scores', score); monkeypatch.setattr(cold, 'original_scores', score)
    monkeypatch.setattr(cold, 'independent_score', lambda corpus, call, configs, ref: (cache_for(call, ref),
        {'scope': 'SYNTHETIC'}, {'scope': 'SYNTHETIC'}))
    torch.set_num_threads(1); torch.manual_seed(plan['training']['seeds'][0])
    model = training.CalibratedOrigin(training.LocalTwoHead('motion_only', 2.5, 2.5), 'cv')
    mean = np.zeros((1, 1, 252)); scale = np.ones_like(mean); identifier = 'cv_motion_only_seed1034101'
    directory = tmp_path/'run'; directory.mkdir()
    measured, _ = training.evaluate(model, corpus, mean, scale, cache, directory/'predictions'/identifier)
    cache_root = directory/'cache1034101'; cache_root.mkdir(); files = {}; audits = []
    for index, call in enumerate(calls):
        ranking = cache_for(call, np.zeros((3, 8, 3)))
        values = {'reference_augmented': np.zeros((7, 8, 3)), **{k: ranking[k].numpy() for k in cold.KEYS}}
        path = cache_root/f'{index}.npz'; np.savez_compressed(path, **values)
        files[str(index)] = {'path': path.name, 'sha256': cold.sha(path), 'complete_base': True}
        audits.append({'call_index': index, 'scope': 'SYNTHETIC'})
    (cache_root/'manifest.json').write_text(json.dumps({'files': files, 'complete_calls': 8, 'groups': 2,
        'family_axes': {'a': sorted(contract['geometry_axes']), 'b': sorted(contract['geometry_axes'])}}), encoding='utf8')
    (cache_root/'score_audit.json').write_text(json.dumps(audits), encoding='utf8')
    return model, directory, identifier, corpus, mean, scale, measured


def test_every_prediction_byte_original_cost_and_metric_reexecuted(saved_toy_run):
    model, directory, identifier, corpus, mean, scale, original = saved_toy_run
    measured, rows = cold.replay_model(model, directory, identifier, corpus, None, mean, scale)
    assert cold.exact_tree(measured, original)
    assert all(pop.count == 4 for split in rows.values() for pop in split.values())


@pytest.mark.parametrize('kind', ['prediction', 'decision', 'derivative', 'cache', 'extra_decision'])
def test_any_changed_declared_prediction_cost_derivative_or_cache_refused(saved_toy_run, kind):
    model, directory, identifier, corpus, mean, scale, _ = saved_toy_run
    base = directory/'predictions'/identifier
    if kind == 'prediction':
        manifest = json.loads((base/'manifest.json').read_bytes()); path = base/'0.npz'; values = cold.arrays(path)
        values['prediction'][0, 0, 0] += .1; np.savez_compressed(path, **values)
        manifest['0']['sha256'] = cold.sha(path); (base/'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    elif kind in ('decision', 'extra_decision'):
        manifest = json.loads((base/'decision_manifest.json').read_bytes()); row = manifest['train']['bounded_shared']
        path = base/row['path']; lines = path.read_text(encoding='utf8').splitlines()
        if kind == 'decision':
            value = json.loads(lines[0]); value['costs']['model'][0] += 1; lines[0] = json.dumps(value)
        else: lines.append(lines[-1]); row['calls'] += 1
        path.write_text('\n'.join(lines)+'\n', encoding='utf8'); row['sha256'] = cold.sha(path)
        (base/'decision_manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    elif kind == 'derivative':
        path = directory/'cache1034101/score_audit.json'; value = json.loads(path.read_bytes()); value[0]['scope'] = 'changed'
        path.write_text(json.dumps(value), encoding='utf8')
    else:
        root = directory/'cache1034101'; path = root/'0.npz'; values = cold.arrays(path); values['temperature'] += 1.
        np.savez_compressed(path, **values); manifest = json.loads((root/'manifest.json').read_bytes())
        manifest['files']['0']['sha256'] = cold.sha(path); (root/'manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    with pytest.raises(ValueError): cold.replay_model(model, directory, identifier, corpus, None, mean, scale)
