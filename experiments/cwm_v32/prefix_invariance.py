"""TRAIN-only proposed-suffix sensitivity; never an active qualification."""
import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE.parent/'cwm_v28'), str(HERE.parent/'cwm_v10')]
from release_ranking_training_v2 import model_from_checkpoint
from cost_origin_model import public_cost_inputs, normalization
from repeatability_identity import digest, exact_tree
from package_ranking_release import safe_name
from audit_ranking_data import validate_data_protocol

REMOTE = 'https://github.com/Liangxianguang/new-uav-capture.git'
BRANCH = 'refs/heads/causal-world-model-v1-20261009'


def read(path):
    return json.loads(path.read_bytes())


def splice_suffix(proposed, anchor, prefix):
    if (type(prefix) is not int or not 1 <= prefix <= 8 or proposed.ndim != 4 or
        proposed.shape[1:] != (8, 4, 3) or anchor.shape != (8, 4, 3) or
        proposed.dtype != anchor.dtype or not np.isfinite(proposed).all() or
        not np.isfinite(anchor).all()):
        raise ValueError('Finite unchanged eight-step public action contract required')
    result = proposed.copy()
    result[:, prefix:] = anchor[None, prefix:]
    return result


def score_prefix(model, values, mean, scale, prefix):
    """No labels/masks/costs used for public perturbation or forward features."""
    original = values['proposed']
    changed = splice_suffix(original, values['anchor'], prefix)
    inputs, _ = public_cost_inputs([{'values': values}], [0], mean, scale)
    repeated = [torch.cat([v, v, v], dim=0) for v in inputs]
    count = len(original)
    repeated[2][count:2*count] = torch.as_tensor(changed, dtype=inputs[2].dtype)
    with torch.no_grad():
        prediction, motion, response = model(*repeated)
    if any(not torch.isfinite(v).all() for v in (prediction, motion, response)):
        raise ValueError('Finite model output required')
    response = response.double().numpy()
    distance = np.linalg.norm(response[:count, :prefix]-response[count:2*count, :prefix], axis=-1)
    control = np.linalg.norm(response[:count, :prefix]-response[2*count:, :prefix], axis=-1)
    changed_rows = (original != changed).any((1, 2, 3))
    return distance, control, changed_rows


class PrefixStatistics:
    def __init__(self, precision, physical):
        self.precision, self.physical = precision, physical
        self.groups = collections.defaultdict(lambda: [0., 0])
        self.rows = self.changed_rows = self.nonanchor_rows = self.zero_valid_rows = 0
        self.points = self.precision_points = self.physical_points = 0
        self.maximum = self.control_maximum = 0.

    def add(self, group, distance, control, changed, mask, nonanchor):
        if (distance.ndim != 2 or control.shape != distance.shape or mask.shape != distance.shape or
            mask.dtype != np.bool_ or changed.shape != (len(distance),) or
            nonanchor.shape != changed.shape or changed.dtype != np.bool_ or
            nonanchor.dtype != np.bool_ or not np.isfinite(distance).all() or
            not np.isfinite(control).all() or (distance < 0).any() or (control < 0).any()):
            raise ValueError('Complete finite prefix sensitivity support required')
        self.rows += len(distance); self.changed_rows += int(changed.sum())
        self.nonanchor_rows += int(nonanchor.sum())
        valid = mask & nonanchor[:, None]
        self.zero_valid_rows += int((~valid.any(1)).sum())
        observed = distance[valid]
        self.points += observed.size
        self.precision_points += int((observed > self.precision).sum())
        self.physical_points += int((observed > self.physical).sum())
        self.maximum = max(self.maximum, float(observed.max()) if observed.size else 0.)
        # ALL rows/offsets retained for numerical identical-input control.
        self.control_maximum = max(self.control_maximum, float(control.max()))
        self.groups[group][0] += float(observed.sum()); self.groups[group][1] += observed.size

    def result(self):
        if not self.points or any(not count for _, count in self.groups.values()):
            raise ValueError('Observed nonanchor support required in every TRAIN group')
        return {'candidate_rows': self.rows, 'suffix_changed_candidate_rows': self.changed_rows,
            'nonanchor_candidate_rows': self.nonanchor_rows, 'zero_common_valid_rows': self.zero_valid_rows,
            'observed_nonanchor_prefix_points': self.points,
            'group_point_equal_mean_response_difference_m': float(np.mean([s/n for s, n in self.groups.values()])),
            'maximum_observed_response_difference_m': self.maximum,
            'point_fraction_over_precision_description': self.precision_points/self.points,
            'point_fraction_over_physical_description': self.physical_points/self.points,
            'maximum_identical_input_repeat_response_difference_m': self.control_maximum,
            'by_group': {g: {'points': n, 'mean_response_difference_m': s/n}
                for g, (s, n) in sorted(self.groups.items())}}


def published_sources():
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
    commit = git('rev-parse', 'HEAD')
    remote = git('-c', 'http.sslBackend=schannel', '-c', 'http.version=HTTP/1.1', 'ls-remote', REMOTE, BRANCH)
    if not remote or remote.split()[0] != commit:
        raise ValueError('Publish exact source/protocol HEAD before actual diagnostic')
    files = (HERE/'prefix_invariance.py', HERE/'prefix_audit_protocol.json')
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        if subprocess.check_output(['git', 'show', commit+':'+name], cwd=ROOT) != path.read_bytes():
            raise ValueError('Actual diagnostic source differs from published bytes')
    return {'commit': commit, 'remote': REMOTE, 'branch': BRANCH}


def prepare(data, audit, primary, protocol):
    folders = {'data': data, 'audit': audit, 'primary': primary}
    identities = {key: digest(folder/'summary.json') for key, folder in folders.items()}
    if identities != {'data': protocol['source_data_summary_sha256'],
        'audit': protocol['source_complete_data_audit_sha256'],
        'primary': protocol['source_primary_summary_sha256']}:
        raise ValueError('Fixed complete evidence identity differs')
    d, a, p = (read(folders[key]/'summary.json') for key in ('data', 'audit', 'primary'))
    validate_data_protocol(d['protocol'])
    if (a['status'] != 'independent_fresh_sequential_public_branch_cost_audit_passed' or
        a['data_gate_passed'] is not True or a['data_summary_sha256'] != identities['data'] or
        any(r['enhanced_control_enabled'] is not False or r['holdout_used'] is not False for r in (d, a, p))):
        raise ValueError('Independent complete audited data/default-off required')
    for key, report in (('data', d), ('audit', a), ('primary', p)):
        for name, sha in report['source_hashes'].items():
            if digest(ROOT/safe_name(name)) != sha or digest(folders[key]/'source'/name) != sha:
                raise ValueError('Frozen scientific sources differ')
    manifest_path = audit/'audited_data_manifest.json'
    if digest(manifest_path) != a['audited_data_manifest_sha256']:
        raise ValueError('Completed full data manifest differs')
    manifest = read(manifest_path)
    for name, sha in manifest.items():
        if digest(data/safe_name(name)) != sha:
            raise ValueError('Complete audited data differs')
    records = read(data/'records.json')
    if len(records) != a['checked_calls']:
        raise ValueError('Complete audited call population differs')
    train = [r for r in records if r['split'] == 'train' and 'arrays_path' in r]
    if len(train) != protocol['expected_train_calls'] or len({r['group'] for r in train}) != protocol['expected_train_groups']:
        raise ValueError('Complete TRAIN-only population required')
    calls = []
    for row in train:
        with np.load(data/safe_name(row['arrays_path']), allow_pickle=False) as archive:
            values = {name: archive[name] for name in archive.files}
        calls.append({'record': row, 'values': values})
    mean, scale = normalization(calls, list(range(len(calls))))
    with np.load(primary/'normalization.npz', allow_pickle=False) as norm:
        if not np.array_equal(mean, norm['mean']) or not np.array_equal(scale, norm['scale']):
            raise ValueError('Complete TRAIN normalization recomputation differs')
    models, model_hashes = {}, {}
    for config in protocol['configurations']:
        seed = protocol['fixed_seed']
        if p['selected_median_seeds'][config] != seed:
            raise ValueError('Original fixed ADE-median choice differs')
        model_row = next(r for r in p['models'] if (r['configuration'], r['seed']) == (config, seed))
        path = primary/f'{config}_seed{seed}.pt'
        if digest(path) != model_row['checkpoint_sha256']:
            raise ValueError('Fixed selected checkpoint file identity differs')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if (checkpoint['configuration'] != config or checkpoint['seed'] != seed or
            not exact_tree(checkpoint['protocol'], p['protocol']) or
            not exact_tree(checkpoint['source_hashes'], p['source_hashes']) or
            checkpoint['data_summary_sha256'] != identities['data'] or
            checkpoint['data_audit_summary_sha256'] != identities['audit'] or
            not np.array_equal(checkpoint['normalizer_mean'].numpy(), mean) or
            not np.array_equal(checkpoint['normalizer_scale'].numpy(), scale)):
            raise ValueError('Fixed checkpoint provenance/normalization differs')
        models[config] = model_from_checkpoint(checkpoint)
        model_hashes[path.name] = digest(path)
    return calls, mean, scale, models, identities, manifest, model_hashes


def run(data, audit, primary, output):
    if output.exists():
        raise ValueError('Exclusive NEW diagnostic output required')
    published = published_sources()
    protocol = read(HERE/'prefix_audit_protocol.json')
    calls, mean, scale, models, identities, manifest, model_hashes = prepare(data, audit, primary, protocol)
    sources = {Path(m.__file__).resolve().relative_to(ROOT).as_posix(): digest(Path(m.__file__).resolve())
        for m in tuple(sys.modules.values()) if getattr(m, '__file__', None)
        and Path(m.__file__).suffix == '.py' and Path(m.__file__).resolve().is_relative_to(ROOT/'experiments')}
    sources[(HERE/'prefix_audit_protocol.json').relative_to(ROOT).as_posix()] = digest(HERE/'prefix_audit_protocol.json')
    normalization_sha = digest(primary/'normalization.npz')
    manifest_sha = read(audit/'summary.json')['audited_data_manifest_sha256']
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    tables = {config: {k: PrefixStatistics(protocol['precision_description_m'], protocol['physical_description_m'])
        for k in protocol['prefix_lengths']} for config in models}
    for ordinal, call in enumerate(calls):
        v = call['values']; nonanchor = ~(v['proposed'] == v['anchor'][None]).all((1, 2, 3))
        for config, model in models.items():
            for prefix, table in tables[config].items():
                distance, control, changed = score_prefix(model, v, mean, scale, prefix)
                mask = v['valid'][:, :prefix] & v['anchor_valid'][None, :prefix]
                table.add(call['record']['group'], distance, control, changed, mask, nonanchor)
        if (ordinal+1) % 128 == 0:
            print(json.dumps({'complete_train_calls_scored': ordinal+1}), flush=True)
    for key, folder in (('data', data), ('audit', audit), ('primary', primary)):
        if digest(folder/'summary.json') != identities[key]:
            raise ValueError('Evidence changed during actual diagnostic')
    for name, sha in manifest.items():
        if digest(data/name) != sha:
            raise ValueError('Complete data changed during actual diagnostic')
    for name, sha in model_hashes.items():
        if digest(primary/name) != sha:
            raise ValueError('Fixed model changed during actual diagnostic')
    if digest(primary/'normalization.npz') != normalization_sha:
        raise ValueError('Normalization changed during actual diagnostic')
    if digest(audit/'audited_data_manifest.json') != manifest_sha:
        raise ValueError('Completed audit manifest changed during actual diagnostic')
    for name, sha in sources.items():
        if digest(ROOT/name) != sha:
            raise ValueError('Actual diagnostic source changed')
        destination = output/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write((ROOT/name).read_bytes())
    result = {'status': 'complete_train_only_fixed_model_proposed_suffix_sensitivity_diagnostic',
        'protocol': protocol, 'published': published, 'evidence_sha256': identities,
        'source_hashes': sources, 'selected_checkpoint_hashes': model_hashes,
        'normalization_sha256': normalization_sha, 'train_calls': len(calls),
        'audited_data_manifest_sha256': manifest_sha,
        'train_groups': len({c['record']['group'] for c in calls}),
        'models': {config: {str(k): table.result() for k, table in values.items()} for config, values in tables.items()},
        'enhanced_control_enabled': False, 'holdout_used': False, 'new_model_trained': False,
        'native_branch_reexecuted': False, 'prior_causal_gate_still_failed': True,
        'scope': protocol['scope']}
    with (output/'summary.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': result['status'], 'summary_sha256': digest(output/'summary.json')}), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data', 'audit', 'primary', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.data, args.audit, args.primary, args.output)
