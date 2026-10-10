"""Bounded-memory V34 training prerequisites; no optimizer or deployment.

Only a COMPLETE independently audited V34 population can become a corpus.
The old audited scientific sources are never edited. At most a fixed16-call
batch is loaded; no physical arrays, contexts or tensors are cached in corpus.
This loader does not replace native replay or original-cost gradient audits.
"""
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent/'cwm_v34'))
import audit_prefix_data as data_audit
from prefix_library import augment, check_native_prefix
from stream_normalization import stream_history_normalization

sha = data_audit.sha
ARRAY_FIELDS = ('target', 'defenders', 'commanded', 'executed',
                'branch_sign_label_only', 'valid', 'termination', 'proposed')


def safe_file(root, name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name or
        Path(name).is_absolute() or any(p in ('', '.', '..') for p in name.split('/'))):
        raise ValueError('Canonical relative evidence file required')
    path = root/name
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
        raise ValueError('Regular evidence file must remain within its declared root')
    return path


def checked_json(root, name, digest):
    path = safe_file(root, name)
    if sha(path) != digest:
        raise ValueError('Complete audit evidence identity differs: '+name)
    return json.loads(path.read_bytes())


def validate_call(row, values):
    count = int(values['base_candidate_count'])
    joint, pairs = augment(values['proposed'][:count], values['anchor'], row['agent'])
    if not data_audit.equal(joint, values['proposed']) or not data_audit.equal(pairs, values['prefix_pairs']):
        raise ValueError('Complete fixed public augmentation differs')
    if check_native_prefix(values) != row['prefix_pair_count']:
        raise ValueError('Complete native prefix mapping differs')
    n = len(joint)
    for name, shape in (('history', (8, 252)), ('relative', (4, 6)), ('backbone', (8, 3)),
                        ('reference', (3,)), ('velocity', (3,)), ('target', (n, 8, 3)),
                        ('anchor_target', (8, 3))):
        value = values[name]
        if value.shape != shape or not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError('Finite audited public/label shape differs: '+name)
    if (values['valid'].dtype != np.bool_ or values['valid'].shape != (n, 8) or
        values['anchor_valid'].dtype != np.bool_ or values['anchor_valid'].shape != (8,) or
        values['bounded_shared_indices'].dtype != np.int64 or
        not np.array_equal(values['bounded_shared_indices'], np.arange(count, dtype=np.int64)) or
        row['base_candidate_count'] != count or row['augmented_candidate_count'] != n):
        raise ValueError('Audited base-only ranking population/masks differ')
    return count


def base_values(values):
    """Slice explicit candidate-axis arrays ONLY; history/backbone stay whole."""
    count = int(values['base_candidate_count'])
    return {key: value[:count] if key in ARRAY_FIELDS else value for key, value in values.items()}


class AuditedCorpus:
    def __init__(self, data, audit):
        self.data, self.audit = Path(data), Path(audit)
        report = json.loads(safe_file(self.audit, 'summary.json').read_bytes())
        if (report['status'] != 'complete_independent_fresh_geometry_prefix_public_branch_cost_audit_passed' or
            report['data_summary_sha256'] != sha(safe_file(self.data, 'summary.json')) or
            report['data_gate_passed'] is not True or report['training_authorized_by_this_report'] is not True or
            any(report[k] is not False for k in ('enhanced_control_enabled', 'new_model_trained', 'holdout_used'))):
            raise ValueError('COMPLETE independent native V34 audit required before training preparation')
        protocol, source_report, population, records, episodes = data_audit.preflight(self.data)
        if report['protocol'] != protocol:
            raise ValueError('Fixed collection/audit protocol differs')
        for name, digest in report['source_hashes'].items():
            if sha(safe_file(ROOT, name)) != digest or sha(safe_file(self.audit, 'source/'+name)) != digest:
                raise ValueError('Actual native auditor sources changed')
        checked = checked_json(self.audit, 'checked_calls.json', report['checked_calls_sha256'])
        costs = checked_json(self.audit, 'full_label_cost_checks.json', report['full_label_cost_checks_sha256'])
        manifest = checked_json(self.audit, 'audited_data_manifest.json', report['audited_data_manifest_sha256'])
        for name in manifest:
            safe_file(self.data, name)
        if (manifest != data_audit.evidence_manifest(self.data, source_report, population, records) or
            len(checked) != len(records) or report['checked_calls'] != len(records) or
            report['complete_cost_calls'] != len(costs)):
            raise ValueError('ALL declared native audit evidence/counts required')
        for record, evidence in zip(records, checked):
            identity = [record[k] for k in ('episode_index', 'step', 'ordinal', 'agent')]
            expected = 'all_public_prefix_native_branches_original_costs_checked' if 'arrays_path' in record else record['status']
            if evidence != {'identity': identity, 'status': expected}:
                raise ValueError('ALL native sequential calls must be audited in order')
        self.records = [record for record in records if 'arrays_path' in record]
        if not self.records or len(self.records) != report['supported_calls'] or len(self.records) != source_report['calls']:
            raise ValueError('ALL supported calls required, never a favorable subset')
        assignments = {s['episode_index']: s for s in population}
        if len(assignments) != 512 or len(episodes) != 512:
            raise ValueError('ALL512 fixed native episodes required')
        self.indices = {split: [] for split in ('train', 'development_validation')}
        for index, row in enumerate(self.records):
            scene = assignments[row['episode_index']]
            if (row['group'], row['split'], row['variant']) != (scene['mirror_group_id'], scene['model_split'], scene['variant']):
                raise ValueError('Layout/mirror/axis split leakage')
            if (manifest.get(row['arrays_path']) != row['arrays_sha256'] or
                manifest.get(row['context_path']) != row['context_sha256']):
                raise ValueError('Audited supported-call identity differs')
            self.indices[row['split']].append(index)
        groups = [{self.records[i]['group'] for i in self.indices[s]} for s in self.indices]
        if tuple(map(len, groups)) != (48, 16) or groups[0] & groups[1]:
            raise ValueError('Complete independent TRAIN48/development16 families required')
        self.manifest, self.protocol = manifest, protocol
        self.data_report, self.audit_report = source_report, report
        self.audit_summary_sha256 = sha(self.audit/'summary.json')
        self.cost_records = {tuple(row['identity']): row for row in costs}
        if len(self.cost_records) != len(costs):
            raise ValueError('Duplicate native full-cost call evidence')
        # This final streaming pass inspects EVERY call, not just the next batch.
        supported_costs = set()
        for index in range(len(self.records)):
            call = self.load(index); row, v = call['record'], call['values']
            complete = np.flatnonzero(v['valid'].all(1) & v['anchor_valid'].all())
            identity = tuple(row[k] for k in ('episode_index', 'step', 'ordinal', 'agent'))
            if len(complete):
                evidence = self.cost_records.get(identity)
                if (evidence is None or evidence['complete_indices'] != complete.tolist() or
                    len(evidence['true_costs']) != len(complete) or not np.isfinite(evidence['true_costs']).all() or
                    evidence['base_library_complete'] is not bool(v['valid'][:int(v['base_candidate_count'])].all() and v['anchor_valid'].all())):
                    raise ValueError('ALL complete native augmented-cost evidence required')
                supported_costs.add(identity)
            elif identity in self.cost_records:
                raise ValueError('Incomplete call cannot carry full-cost evidence')
        if supported_costs != set(self.cost_records):
            raise ValueError('Undeclared full-cost audit evidence')

    def __len__(self): return len(self.records)

    def load(self, index):
        if type(index) is not int or not 0 <= index < len(self.records):
            raise ValueError('Exact supported-call index required')
        row = self.records[index]
        path = safe_file(self.data, row['arrays_path'])
        if sha(path) != row['arrays_sha256']:
            raise ValueError('Native arrays changed after complete audit')
        values = data_audit.prior.arrays(path.read_bytes())
        validate_call(row, values)
        return {'record': row, 'values': values}

    def batch(self, indices):
        ids = list(indices)
        if not 1 <= len(ids) <= 16 or len(set(ids)) != len(ids):
            raise ValueError('Fixed unique bounded16-call batch required')
        return [self.load(index) for index in ids]

    def normalization(self):
        return stream_history_normalization(self.load(i)['values']['history'] for i in self.indices['train'])

    def reverify(self):
        if sha(self.audit/'summary.json') != self.audit_summary_sha256:
            raise ValueError('Native audit certificate changed')
        current = AuditedCorpus(self.data, self.audit)
        if current.manifest != self.manifest or current.records != self.records:
            raise ValueError('Complete audited population changed during preparation/training')


def population_weights(corpus):
    """Full TRAIN equal-group/augmented-point weights; cost on BASE only."""
    indices = corpus.indices['train']
    counts, points, complete = collections.Counter(), collections.Counter(), collections.Counter()
    per_call = {}
    for i in indices:
        call = corpus.load(i); row, v = call['record'], call['values']
        nonanchor = ~(v['proposed'] == v['anchor'][None]).all((1, 2, 3))
        n = int((v['valid'] & v['anchor_valid'][None] & nonanchor[:, None]).sum())
        full = bool(v['valid'][:int(v['base_candidate_count'])].all() and v['anchor_valid'].all())
        if not n: raise ValueError('Every TRAIN call must have nonanchor paired support')
        group = row['group']; counts[group] += 1; points[group] += n; complete[group] += full
        per_call[i] = (group, n, full)
    if not indices or set(counts) != set(complete) or any(not complete[g] for g in counts):
        raise ValueError('Complete BASE ranking support required in ALL TRAIN families')
    factor = len(indices)/len(counts)
    weights = {'call': {}, 'point': {}, 'cost': {}}
    for i, (g, n, full) in per_call.items():
        weights['call'][i] = factor/counts[g]
        weights['point'][i] = factor*n/points[g]
        weights['cost'][i] = factor/complete[g] if full else 0.
    if any(not np.isclose(math.fsum(v.values()), len(indices), rtol=0, atol=1e-9) for v in weights.values()):
        raise ValueError('Fixed full-population weights do not normalize')
    return weights, {'calls': len(indices), 'groups': len(counts), 'group_calls': dict(counts),
        'group_valid_nonanchor_points': dict(points), 'group_complete_cost_calls': dict(complete)}
