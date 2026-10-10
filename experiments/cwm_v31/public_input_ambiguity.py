"""Exact TRAIN input collisions; labels cannot affect key construction."""
import argparse
import collections
import functools
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent/'cwm_v28'), str(HERE.parent/'cwm_v10')]
from repeatability_identity import digest, exact_tree
from package_ranking_release import safe_name
from audit_ranking_data import validate_data_protocol
from train_two_head import normalization

REMOTE = 'https://github.com/Liangxianguang/new-uav-capture.git'
BRANCH = 'refs/heads/causal-world-model-v1-20261009'


def read(path):
    return json.loads(path.read_bytes())


def feature_bytes(values, candidate, mean, scale):
    """Only the neural feature construction already present in V28 plain core."""
    h = torch.as_tensor((values['history']-mean)/scale, dtype=torch.float32)
    relative = torch.as_tensor(values['relative'], dtype=torch.float32)
    proposed = torch.as_tensor(values['proposed'][candidate], dtype=torch.float32)
    anchor = torch.as_tensor(values['anchor'], dtype=torch.float32)
    backbone = torch.as_tensor(values['backbone'], dtype=torch.float32)
    if (h.shape != (1, 8, 252) or relative.shape != (4, 6) or
        proposed.shape != (8, 4, 3) or anchor.shape != proposed.shape or backbone.shape != (8, 3)):
        raise ValueError('Existing V28 public tensor shapes required')
    tensors = (h, relative/relative.new_tensor([10., 10., 10., 5., 5., 5.]),
        proposed/5., anchor/5., backbone/5.)
    if any(not torch.isfinite(v).all() for v in tensors):
        raise ValueError('Finite public features required')
    equal = bool(torch.equal(proposed, anchor))
    core = b''.join(v.contiguous().numpy().tobytes() for v in tensors)+bytes([equal])
    cv = np.asarray(values['reference'])[None]+.1*np.arange(1, 9)[:, None]*np.asarray(values['velocity'])[None]
    cv = np.asarray(cv, dtype=np.float64)
    if cv.shape != (8, 3) or not np.isfinite(cv).all():
        raise ValueError('Finite public analytic CV required')
    return {'response_core': core, 'whole_optional_model': core+cv.tobytes()}, equal


class ConditionalBuckets:
    """Welford weighted variance, merging only byte-identical public inputs."""
    def __init__(self):
        self.buckets = {}
        self.rows = 0

    def add(self, features, label, valid, weight, origin, load_features):
        if label.shape != (8, 3) or valid.shape != (8,) or not np.isfinite(label).all() or weight <= 0:
            raise ValueError('Finite positive-weight common-prefix label support required')
        key = hashlib.sha256(features).hexdigest()
        if key not in self.buckets:
            self.buckets[key] = {'origin': origin, 'rows': 0, 'count': np.zeros(8, dtype=np.int64),
                'weight': np.zeros(8), 'mean': np.zeros((8, 3)), 'm2': np.zeros(8),
                'first': label.copy(), 'first_valid': valid.copy(), 'max_distance_from_first_m': 0.}
        bucket = self.buckets[key]
        if bucket['rows'] and load_features(bucket['origin']) != features:
            raise ValueError('Hash collision or changed original public feature bytes')
        common = valid & bucket['first_valid']
        if common.any():
            distance = float(np.linalg.norm(label[common]-bucket['first'][common], axis=-1).max())
            bucket['max_distance_from_first_m'] = max(bucket['max_distance_from_first_m'], distance)
        delta = label-bucket['mean']
        added = valid.astype(np.float64)*weight
        new_weight = bucket['weight']+added
        ratio = np.divide(added, new_weight, out=np.zeros(8), where=new_weight > 0)
        new_mean = bucket['mean']+delta*ratio[:, None]
        bucket['m2'] += added*(delta*(label-new_mean)).sum(-1)
        bucket['mean'] = new_mean; bucket['weight'] = new_weight
        bucket['count'] += valid; bucket['rows'] += 1; self.rows += 1

    def result(self, threshold):
        values = list(self.buckets.values())
        mass = sum(float(v['weight'].sum()) for v in values)
        if mass <= 0:
            raise ValueError('Nonempty train response point support required')
        squared = sum(float(v['m2'].sum()) for v in values)
        if squared < -1e-12:
            raise ValueError('Negative conditional variance beyond roundoff')
        repeated = [v for v in values if v['rows'] > 1]
        observed_repeated_points = sum(int((v['count']*(v['count'] > 1)).sum()) for v in values)
        observed_points = sum(int(v['count'].sum()) for v in values)
        return {'nonanchor_candidate_rows': self.rows, 'unique_input_buckets': len(values),
            'duplicate_input_buckets': len(repeated),
            'candidate_rows_in_duplicate_buckets': sum(v['rows'] for v in repeated),
            'observed_common_prefix_points': observed_points,
            'points_with_repeated_observation_at_same_input_and_offset': observed_repeated_points,
            'single_observation_point_fraction': 1.-observed_repeated_points/observed_points,
            'duplicate_buckets_with_nonzero_shared_offset_variance': sum(bool((v['m2'] > 0).any()) for v in repeated),
            'duplicate_buckets_with_distance_from_first_over_5cm': sum(v['max_distance_from_first_m'] > threshold for v in repeated),
            'maximum_distance_from_first_response_m': max(v['max_distance_from_first_m'] for v in values),
            'group_point_equal_weight_mass': mass,
            'empirical_minimum_deterministic_coordinate_mse_m2': max(0., squared)/(3.*mass),
            'interpretation': 'Finite TRAIN exact-key empirical coordinateMSE lower bound; singleton zero contributions are not input sufficiency, generalization, vectorL2 or causal identification evidence.'}


def published_sources():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    commit = git('rev-parse', 'HEAD')
    remote = git('-c', 'http.sslBackend=schannel', '-c', 'http.version=HTTP/1.1', 'ls-remote', REMOTE, BRANCH)
    if not remote or remote.split()[0] != commit:
        raise ValueError('Publish exact source/protocol HEAD before diagnostic run')
    files = [HERE/'public_input_ambiguity.py', HERE/'input_audit_protocol.json']
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if subprocess.check_output(['git', 'show', commit+':'+relative], cwd=ROOT) != path.read_bytes():
            raise ValueError('Diagnostic source differs from exact published bytes')
    return {'commit': commit, 'remote': REMOTE, 'branch': BRANCH,
        'source_hashes': {p.relative_to(ROOT).as_posix(): digest(p) for p in files}}


def run(data, audit, primary, output):
    if output.exists():
        raise ValueError('Exclusive NEW diagnostic output required')
    published = published_sources()
    protocol = read(HERE/'input_audit_protocol.json')
    identities = {'data': digest(data/'summary.json'), 'audit': digest(audit/'summary.json'),
        'primary': digest(primary/'summary.json')}
    expected = {'data': protocol['source_data_summary_sha256'], 'audit': protocol['source_complete_data_audit_sha256'],
        'primary': protocol['source_primary_summary_sha256']}
    if identities != expected or protocol['split'] != 'train':
        raise ValueError('Pinned TRAIN diagnostic evidence differs')
    d, a, p = read(data/'summary.json'), read(audit/'summary.json'), read(primary/'summary.json')
    run_sources = {Path(module.__file__).resolve().relative_to(ROOT).as_posix(): digest(Path(module.__file__).resolve())
        for module in tuple(sys.modules.values()) if getattr(module, '__file__', None)
        and Path(module.__file__).suffix == '.py' and Path(module.__file__).resolve().is_relative_to(ROOT/'experiments')}
    run_sources.update(published['source_hashes'])
    validate_data_protocol(d['protocol'])
    if (a['status'] != 'independent_fresh_sequential_public_branch_cost_audit_passed' or
        a['data_summary_sha256'] != identities['data'] or a['data_gate_passed'] is not True or
        any(r['enhanced_control_enabled'] is not False or r['holdout_used'] is not False for r in (d, a, p))):
        raise ValueError('Completed independent data audit/default-off required')
    for folder, report in ((data, d), (audit, a), (primary, p)):
        for name, sha in report['source_hashes'].items():
            if digest(ROOT/safe_name(name)) != sha or digest(folder/'source'/name) != sha:
                raise ValueError('Frozen current/used source differs')
    manifest_path = audit/'audited_data_manifest.json'
    if digest(manifest_path) != a['audited_data_manifest_sha256']:
        raise ValueError('Completed full data manifest differs')
    manifest = read(manifest_path)
    for name, sha in manifest.items():
        if digest(data/safe_name(name)) != sha:
            raise ValueError('Complete prior audited data bytes differ')
    records = read(data/'records.json')
    if len(records) != a['checked_calls']:
        raise ValueError('Complete sequential data record support required')
    train = [r for r in records if r['split'] == 'train' and 'arrays_path' in r]
    if len(train) != protocol['expected_train_calls']:
        raise ValueError('Complete declared TRAIN calls required')

    @functools.lru_cache(maxsize=8)
    def values(path):
        if digest(data/safe_name(path)) != manifest[path]:
            raise ValueError('TRAIN input changed during audit')
        with np.load(data/path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}

    histories = []; point_counts = collections.Counter()
    for row in train:
        v = values(row['arrays_path']); histories.append({'values': {'history': v['history']}})
        nonanchor = ~(v['proposed'] == v['anchor'][None]).all((1, 2, 3))
        mask = v['valid'] & v['anchor_valid'][None] & nonanchor[:, None]
        point_counts[row['group']] += int(mask.sum())
    mean, scale = normalization(histories, list(range(len(histories))))
    del histories
    if len(point_counts) != protocol['expected_train_groups'] or not all(point_counts.values()):
        raise ValueError('Complete TRAIN group-point support required')
    saved_norm = np.load(primary/'normalization.npz', allow_pickle=False)
    if not np.array_equal(mean, saved_norm['mean']) or not np.array_equal(scale, saved_norm['scale']):
        raise ValueError('Original train-only normalizer recomputation differs')
    saved_norm.close()
    selected = p['selected_median_seeds'][protocol['primary_configuration']]
    model_row = next(r for r in p['models'] if (r['configuration'], r['seed']) == (protocol['primary_configuration'], selected))
    checkpoint_path = primary/f"{protocol['primary_configuration']}_seed{selected}.pt"
    normalizer_sha = digest(primary/'normalization.npz')
    if digest(checkpoint_path) != model_row['checkpoint_sha256']:
        raise ValueError('Fixed selected checkpoint file digest differs')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (not exact_tree(checkpoint['protocol'], p['protocol']) or
        not np.array_equal(checkpoint['normalizer_mean'].numpy(), mean) or
        not np.array_equal(checkpoint['normalizer_scale'].numpy(), scale)):
        raise ValueError('Checkpoint TRAIN normalization/protocol differs')
    del checkpoint
    output.mkdir(parents=True, exist_ok=False)
    tables = {name: ConditionalBuckets() for name in ('response_core', 'whole_optional_model')}
    collapsed = {'raw_nonanchor_rows_float32_equal_to_anchor': 0, 'observed_points': 0,
        'response_points_over_5cm': 0, 'maximum_response_m': 0.}
    for ordinal, row in enumerate(train):
        v = values(row['arrays_path'])
        weight = 1./(len(point_counts)*point_counts[row['group']])
        for candidate in range(len(v['proposed'])):
            if np.array_equal(v['proposed'][candidate], v['anchor']):
                continue
            mask = v['valid'][candidate] & v['anchor_valid']
            label = v['target'][candidate]-v['anchor_target']
            features, equal = feature_bytes(v, candidate, mean, scale)
            origin = (row['arrays_path'], candidate)
            for name, table in tables.items():
                loader = lambda key, name=name: feature_bytes(values(key[0]), key[1], mean, scale)[0][name]
                table.add(features[name], label, mask, weight, origin, loader)
            if equal:
                lengths = np.linalg.norm(label[mask], axis=-1)
                collapsed['raw_nonanchor_rows_float32_equal_to_anchor'] += 1
                collapsed['observed_points'] += len(lengths)
                collapsed['response_points_over_5cm'] += int((lengths > protocol['descriptive_response_threshold_m']).sum())
                if len(lengths): collapsed['maximum_response_m'] = max(collapsed['maximum_response_m'], float(lengths.max()))
        if (ordinal+1) % 128 == 0:
            print(json.dumps({'train_calls_audited': ordinal+1}), flush=True)
    values.cache_clear()
    results = {name: table.result(protocol['descriptive_response_threshold_m']) for name, table in tables.items()}
    if identities != {'data': digest(data/'summary.json'), 'audit': digest(audit/'summary.json'), 'primary': digest(primary/'summary.json')}:
        raise ValueError('Pinned summaries changed during audit')
    for name, sha in manifest.items():
        if digest(data/name) != sha: raise ValueError('Complete audited data changed during diagnostic run')
    for name, sha in published['source_hashes'].items():
        if digest(ROOT/name) != sha: raise ValueError('Diagnostic source changed during run')
    for name, sha in run_sources.items():
        if digest(ROOT/name) != sha: raise ValueError('Run-used diagnostic dependency changed')
    if digest(checkpoint_path) != model_row['checkpoint_sha256'] or digest(primary/'normalization.npz') != normalizer_sha:
        raise ValueError('Selected checkpoint/normalizer changed during audit')
    for name, sha in run_sources.items():
        destination = output/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream: stream.write((ROOT/name).read_bytes())
    result = {'status': 'complete_train_only_exact_public_input_ambiguity_audit', 'protocol': protocol,
        'published': published, 'source_hashes': run_sources, 'evidence_sha256': identities,
        'selected_checkpoint_sha256': model_row['checkpoint_sha256'], 'normalization_sha256': normalizer_sha,
        'train_calls': len(train), 'train_groups': len(point_counts), 'common_nonanchor_points_by_group': dict(sorted(point_counts.items())),
        'input_tables': results, 'float32_anchor_collapse': collapsed, 'primary_research_eligible': p['primary_research_eligible'],
        'enhanced_control_enabled': False, 'holdout_used': False, 'new_training_started': False,
        'development_arrays_loaded_into_diagnostics': False, 'scope': protocol['scope']}
    with (output/'summary.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'data-audit', 'primary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = run(args.data, args.data_audit, args.primary, args.output)
    print(json.dumps({'status': result['status'], 'input_tables': result['input_tables']}), flush=True)
