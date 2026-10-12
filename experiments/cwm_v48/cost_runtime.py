"""Streaming original score/derivative audits and detached BASE ranking cache."""
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent/'cwm_v35'), str(HERE.parent/'cwm_v28')]
from audited_stream import base_values, checked_json, safe_file, sha
from stream_objective import public_batch
from train_ranking_response import (restore_public_call, FrozenLocalScore, original_scores,
    contrast_cache, audit_deployed_score, ranking_cache, save_json)

from coordinate_audit import audit_deployed_score

TENSOR_KEYS = ('reference', 'label_costs', 'temperature', 'logq', 'q')


def context_for(corpus, call, configs):
    row, v = call['record'], call['values']
    snapshot = checked_json(corpus.data, row['context_path'], row['context_sha256'])
    return restore_public_call(snapshot, *configs, v['backbone'])


def make_audited_cache(corpus, call, configs, reference):
    row, v = call['record'], base_values(call['values']); n = len(v['proposed'])
    context = context_for(corpus, call, configs)
    score = FrozenLocalScore(context, v['proposed'][:, :, row['agent']])
    contrast = contrast_cache(score, reference, v)
    audit = audit_deployed_score(context, v, reference, score, contrast)
    cache = ranking_cache(score, reference, v)
    truth = original_scores(context, v['proposed'][:, :, row['agent']], v['target'])
    identity = tuple(row[k] for k in ('episode_index', 'step', 'ordinal', 'agent'))
    native = corpus.cost_records[identity]
    if (native['complete_indices'][:n] != list(range(n)) or
        not np.allclose(truth, native['true_costs'][:n], rtol=1e-10, atol=1e-8) or
        not np.allclose(truth, cache['label_costs'].numpy(), rtol=1e-10, atol=1e-8)):
        raise ValueError('Detached BASE labels differ from ALL native original full costs')
    return cache, audit


def require_cost_population(corpus, identities):
    supported = collections.defaultdict(set)
    for index in identities:
        row = corpus.records[index]; supported[row['group']].add(row['variant'])
    axes = {v['name'] for v in corpus.protocol['variants']}
    expected = {row['group'] for row in corpus.records}
    if set(supported) != expected or any(v != axes for v in supported.values()) or len(expected) != 64:
        raise ValueError('Complete original BASE cost support in ALL64 families/4axes required before optimizer')
    return {'complete_calls': len(identities), 'groups': len(supported),
        'family_axes': {g: sorted(v) for g, v in sorted(supported.items())}}


def audit_public_origins(corpus, configs, output):
    """No optimizer or learned motion invoked before BOTH whole audits finish."""
    certificates = {}
    for origin in ('gru', 'cv'):
        audits = []; supported = []
        for index in range(len(corpus)):
            call = corpus.load(index); v = base_values(call['values'])
            if not (v['valid'].all() and v['anchor_valid'].all()): continue
            path = v['backbone'] if origin == 'gru' else v['reference'][None]+.1*np.arange(1, 9)[:, None]*v['velocity'][None]
            _, result = make_audited_cache(corpus, call, configs, np.repeat(path[None], len(v['proposed']), 0))
            audits.append({'call_index': index, **result}); supported.append(index)
            if len(audits) % 256 == 0: print(json.dumps({'status': 'initial_original_score_derivative_audit', 'origin': origin, 'complete_calls': len(audits)}), flush=True)
        population = require_cost_population(corpus, supported)
        path = output/f'initial_{origin}_score_audit.json'; save_json(path, audits)
        certificates[origin] = {'status': 'all_native_BASE_original_score_derivatives_checked',
            **population, 'audit_path': path.name, 'audit_sha256': sha(path)}
    save_json(output/'initial_optimizer_prerequisites.json', certificates)
    return certificates


def require_motion_permit(corpus, certificates, output):
    if set(certificates) != {'gru', 'cv'}:
        raise ValueError('Both full GRU and CV audits required before motion optimizer')
    for value in certificates.values():
        if value['status'] != 'all_native_BASE_original_score_derivatives_checked' or value['groups'] != 64:
            raise ValueError('Complete native original-score audit required before optimizer')
        records = checked_json(output, value['audit_path'], value['audit_sha256'])
        if len(records) != value['complete_calls'] or len({r['call_index'] for r in records}) != len(records):
            raise ValueError('ALL complete initial score/derivative audit records required')
        if require_cost_population(corpus, [r['call_index'] for r in records])['family_axes'] != value['family_axes']:
            raise ValueError('Declared whole initial score population differs')
    if certificates['gru']['family_axes'] != certificates['cv']['family_axes'] or certificates['gru']['complete_calls'] != certificates['cv']['complete_calls']:
        raise ValueError('Identical whole original-score audit support required')
    if corpus.audit_report['training_authorized_by_this_report'] is not True:
        raise ValueError('Complete independent native data audit required')


class CanonicalCache:
    """All reference/cache arrays on disk, scores rebuilt per bounded batch."""
    def __init__(self, corpus, configs, directory, manifest):
        self.corpus, self.configs, self.directory, self.manifest = corpus, configs, directory, manifest

    @classmethod
    def build(cls, common, corpus, configs, mean, scale, directory):
        directory.mkdir(parents=True, exist_ok=False); files = {}; supported = []; audits = []
        common.eval().requires_grad_(False)
        for start in range(0, len(corpus), 16):
            ids = list(range(start, min(start+16, len(corpus))))
            calls = corpus.batch(ids); inputs, slices = public_batch(calls, mean, scale)
            with torch.no_grad(): canonical = common(*inputs)[0].numpy()
            for index, call, (a, b) in zip(ids, calls, slices):
                v = base_values(call['values']); reference = canonical[a:b].copy()
                if not np.array_equal(reference, np.repeat(reference[:1], len(reference), 0)):
                    raise ValueError('Frozen common reference became candidate-dependent')
                values = {'reference_augmented': reference}; complete = bool(v['valid'].all() and v['anchor_valid'].all())
                if complete:
                    cache, audit = make_audited_cache(corpus, call, configs, reference[:len(v['proposed'])])
                    values.update({k: cache[k].numpy() for k in TENSOR_KEYS})
                    audits.append({'call_index': index, **audit}); supported.append(index)
                path = directory/f'{index}.npz'; np.savez_compressed(path, **values)
                files[str(index)] = {'path': path.name, 'sha256': sha(path), 'complete_base': complete}
            if start % 256 == 0: print(json.dumps({'status': 'canonical_deployed_original_score_audit', 'cached_calls': start+len(ids)}), flush=True)
        population = require_cost_population(corpus, supported)
        save_json(directory/'score_audit.json', audits)
        manifest = {'status': 'all_canonical_references_and_complete_BASE_cost_derivatives_checked',
            'files': files, **population, 'calls': len(corpus), 'score_audit_sha256': sha(directory/'score_audit.json')}
        save_json(directory/'manifest.json', manifest)
        return cls(corpus, configs, directory, manifest)

    def arrays(self, index):
        row = self.manifest['files'][str(index)]
        path = safe_file(self.directory, row['path'])
        if sha(path) != row['sha256']: raise ValueError('Canonical cache identity differs')
        with np.load(path, allow_pickle=False) as file:
            values = {k: file[k].copy() for k in file.files}
        expected = {'reference_augmented', *TENSOR_KEYS} if row['complete_base'] else {'reference_augmented'}
        if set(values) != expected: raise ValueError('Exact canonical cache tensor inventory required')
        return values

    def load(self, index, call):
        values = self.arrays(index); v = base_values(call['values'])
        if not self.manifest['files'][str(index)]['complete_base']:
            if v['valid'].all() and v['anchor_valid'].all(): raise ValueError('Missing complete BASE cache')
            return None
        context = context_for(self.corpus, call, self.configs)
        score = FrozenLocalScore(context, v['proposed'][:, :, call['record']['agent']])
        cache = ranking_cache(score, values['reference'], v)
        for key in TENSOR_KEYS:
            actual = cache[key].numpy(); saved = values[key]
            if actual.dtype != saved.dtype or actual.shape != saved.shape or actual.tobytes() != saved.tobytes():
                raise ValueError('Detached canonical BASE ranking tensor differs: '+key)
        return cache

    def batch(self, ids, calls):
        result = {}
        for index, call in zip(ids, calls):
            cache = self.load(index, call)
            if cache is not None: result[index] = cache
        return result

    def require_response_permit(self):
        if (self.manifest['status'] != 'all_canonical_references_and_complete_BASE_cost_derivatives_checked' or
            self.manifest['calls'] != len(self.corpus) or self.manifest['groups'] != 64 or
            sha(self.directory/'score_audit.json') != self.manifest['score_audit_sha256'] or
            json.loads((self.directory/'manifest.json').read_bytes()) != self.manifest):
            raise ValueError('Complete canonical native score/gradient audits required before response optimizer')
        for index in range(len(self.corpus)): self.arrays(index)
