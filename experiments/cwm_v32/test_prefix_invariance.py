"""Synthetic prefix probes; no experiment or causal efficacy evidence."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prefix_invariance as audit


def public_values():
    actions = np.zeros((3, 8, 4, 3), dtype=np.float64)
    actions[1, :, 0, 0] = 1.
    actions[2, :, 0, 1] = 2.
    return {'history': np.zeros((8, 252)), 'relative': np.zeros((4, 6)),
        'proposed': actions, 'anchor': actions[0].copy(), 'backbone': np.zeros((8, 3)),
        'reference': np.zeros(3), 'velocity': np.zeros(3),
        'valid': np.ones((3, 8), dtype=bool), 'anchor_valid': np.ones(8, dtype=bool)}


class ToyModel(torch.nn.Module):
    def __init__(self, causal=True):
        super().__init__(); self.causal = causal

    def forward(self, history, relative, proposed, anchor, backbone, cv):
        delta = (proposed-anchor).sum((2, 3))
        signal = delta.cumsum(1) if self.causal else delta.sum(1)[:, None].expand(-1, 8)
        response = signal[:, :, None].expand(-1, -1, 3)
        return cv+response.double(), torch.zeros_like(response), response


@pytest.mark.parametrize('prefix', range(1, 9))
def test_every_splice_preserves_exact_prefix_and_source_inputs(prefix):
    v = public_values(); original = v['proposed'].copy(); anchor = v['anchor'].copy()
    changed = audit.splice_suffix(original, anchor, prefix)
    assert changed[:, :prefix].tobytes() == original[:, :prefix].tobytes()
    assert np.array_equal(changed[:, prefix:], np.repeat(anchor[None, prefix:], 3, 0))
    assert v['proposed'].tobytes() == original.tobytes()
    assert v['anchor'].tobytes() == anchor.tobytes()
    assert np.linalg.norm(changed, axis=-1).max() <= 5.


@pytest.mark.parametrize('prefix', [0, 9, True, 1.0])
def test_invalid_prefix_refused(prefix):
    v = public_values()
    with pytest.raises(ValueError): audit.splice_suffix(v['proposed'], v['anchor'], prefix)


@pytest.mark.parametrize('kind', ['shape', 'nonfinite', 'dtype'])
def test_invalid_action_contract_refused(kind):
    v = public_values()
    if kind == 'shape': v['proposed'] = v['proposed'][:, :7]
    elif kind == 'nonfinite': v['anchor'][0, 0, 0] = np.nan
    else: v['anchor'] = v['anchor'].astype(np.float32)
    with pytest.raises(ValueError): audit.splice_suffix(v['proposed'], v['anchor'], 4)


@pytest.mark.parametrize('prefix', range(1, 9))
def test_causal_toy_unchanged_prefix_and_identical_input_controls(prefix):
    v = public_values()
    distance, control, changed = audit.score_prefix(ToyModel(), v, np.zeros((1, 1, 252)), np.ones((1, 1, 252)), prefix)
    assert np.array_equal(distance, np.zeros((3, prefix)))
    assert np.array_equal(control, distance)
    assert changed.tolist() == ([False]*3 if prefix == 8 else [False, True, True])


def test_noncausal_toy_sensitive_to_future_with_full_horizon_self_control():
    v = public_values(); mean = np.zeros((1, 1, 252)); scale = np.ones_like(mean)
    distance, control, _ = audit.score_prefix(ToyModel(False), v, mean, scale, 3)
    assert (distance[1:] > .05).all(); assert not control.any()
    distance, control, changed = audit.score_prefix(ToyModel(False), v, mean, scale, 8)
    assert not distance.any() and not control.any() and not changed.any()


def test_private_fields_cannot_influence_intervention_or_forward():
    v = public_values(); mean = np.zeros((1, 1, 252)); scale = np.ones_like(mean)
    expected = audit.score_prefix(ToyModel(False), v, mean, scale, 4)
    for name in ('target', 'executed', 'commanded', 'costs', 'branch_sign_label_only'):
        v[name] = object()
    v['valid'] = object(); v['anchor_valid'] = object()
    actual = audit.score_prefix(ToyModel(False), v, mean, scale, 4)
    assert all(np.array_equal(x, y) for x, y in zip(expected, actual))


def test_group_point_equal_masking_and_global_repeat_control():
    table = audit.PrefixStatistics(1e-6, .05)
    table.add('a', np.array([[100., 100.], [.1, .3]]), np.ones((2, 2))*.00001,
        np.array([False, True]), np.array([[True, True], [True, False]]), np.array([False, True]))
    table.add('b', np.array([[.3, .5]]), np.zeros((1, 2)),
        np.array([False]), np.ones((1, 2), dtype=bool), np.array([True]))
    result = table.result()
    assert result['group_point_equal_mean_response_difference_m'] == pytest.approx(.25)
    assert result['maximum_observed_response_difference_m'] == .5
    assert result['observed_nonanchor_prefix_points'] == 3
    assert result['suffix_changed_candidate_rows'] == 1
    assert result['maximum_identical_input_repeat_response_difference_m'] == .00001
    assert result['point_fraction_over_physical_description'] == 1.


def test_missing_group_support_refused():
    table = audit.PrefixStatistics(1e-6, .05)
    table.add('empty', np.zeros((1, 1)), np.zeros((1, 1)), np.array([False]),
        np.zeros((1, 1), dtype=bool), np.array([True]))
    with pytest.raises(ValueError): table.result()


def test_existing_output_is_never_overwritten(tmp_path, monkeypatch):
    destination = tmp_path/'existing'; destination.mkdir()
    marker = destination/'marker'; marker.write_bytes(b'user-owned')
    monkeypatch.setattr(audit, 'published_sources', lambda: pytest.fail('Must refuse before remote lookup'))
    with pytest.raises(ValueError): audit.run(tmp_path, tmp_path, tmp_path, destination)
    assert marker.read_bytes() == b'user-owned'


def test_unpublished_sources_refused_before_output_creation(tmp_path, monkeypatch):
    destination = tmp_path/'never-created'
    def refuse(): raise ValueError('Unpublished')
    monkeypatch.setattr(audit, 'published_sources', refuse)
    with pytest.raises(ValueError): audit.run(tmp_path, tmp_path, tmp_path, destination)
    assert not destination.exists()


def test_protocol_keeps_fixed_failed_models_and_all_prefixes():
    protocol = json.loads((Path(__file__).resolve().parent/'prefix_audit_protocol.json').read_bytes())
    assert protocol['configurations'] == ['cv_l2', 'cv_rank_l2']
    assert protocol['fixed_seed'] == 994101
    assert protocol['prefix_lengths'] == list(range(1, 9))
    for name in ('new_training_started', 'native_branch_reexecuted', 'enhanced_control_enabled',
                 'holdout_used', 'old_qualification_override_allowed'):
        assert protocol[name] is False


@pytest.fixture
def complete_fixture(tmp_path, monkeypatch):
    """Toy preflight files, never a real scientific report."""
    folders = {k: tmp_path/k for k in ('data', 'audit', 'primary')}
    for folder in folders.values(): folder.mkdir()
    data, independent, primary = (folders[k] for k in ('data', 'audit', 'primary'))
    rows = []
    for index, split in enumerate(('train', 'train', 'development_validation')):
        row = {'split': split, 'group': f'g{index}', 'arrays_path': f'call{index}.npz'}
        np.savez(data/row['arrays_path'], **public_values()); rows.append(row)
    (data/'records.json').write_text(json.dumps(rows), encoding='utf8')
    manifest = {name: audit.digest(data/name) for name in ('records.json', 'call0.npz', 'call1.npz', 'call2.npz')}
    (independent/'audited_data_manifest.json').write_text(json.dumps(manifest), encoding='utf8')
    common = {'source_hashes': {}, 'enhanced_control_enabled': False, 'holdout_used': False}
    d = {**common, 'protocol': {'scope': 'Synthetic fixture'}}
    (data/'summary.json').write_text(json.dumps(d), encoding='utf8')
    a = {**common, 'status': 'independent_fresh_sequential_public_branch_cost_audit_passed',
        'data_gate_passed': True, 'data_summary_sha256': audit.digest(data/'summary.json'),
        'audited_data_manifest_sha256': audit.digest(independent/'audited_data_manifest.json'), 'checked_calls': 3}
    (independent/'summary.json').write_text(json.dumps(a), encoding='utf8')
    mean = np.zeros((1, 1, 252)); scale = np.ones_like(mean)*.01
    np.savez(primary/'normalization.npz', mean=mean, scale=scale)
    models = []
    for config in ('cv_l2', 'cv_rank_l2'):
        checkpoint = {'configuration': config, 'seed': 994101, 'protocol': {'toy': True},
            'source_hashes': {}, 'data_summary_sha256': a['data_summary_sha256'],
            'data_audit_summary_sha256': audit.digest(independent/'summary.json'),
            'normalizer_mean': torch.from_numpy(mean), 'normalizer_scale': torch.from_numpy(scale)}
        path = primary/f'{config}_seed994101.pt'; torch.save(checkpoint, path)
        models.append({'configuration': config, 'seed': 994101, 'checkpoint_sha256': audit.digest(path)})
    p = {**common, 'protocol': {'toy': True}, 'selected_median_seeds': {'cv_l2': 994101, 'cv_rank_l2': 994101}, 'models': models}
    (primary/'summary.json').write_text(json.dumps(p), encoding='utf8')
    protocol = {'source_data_summary_sha256': audit.digest(data/'summary.json'),
        'source_primary_summary_sha256': audit.digest(primary/'summary.json'),
        'source_complete_data_audit_sha256': audit.digest(independent/'summary.json'),
        'expected_train_calls': 2, 'expected_train_groups': 2, 'fixed_seed': 994101,
        'configurations': ['cv_l2', 'cv_rank_l2']}
    monkeypatch.setattr(audit, 'validate_data_protocol', lambda _: None)
    monkeypatch.setattr(audit, 'model_from_checkpoint', lambda _: ToyModel())
    return data, independent, primary, protocol


def test_complete_preflight_never_opens_development_arrays(complete_fixture, monkeypatch):
    data, independent, primary, protocol = complete_fixture
    original_load = audit.np.load; opened = []
    def guarded_load(path, *args, **kwargs):
        opened.append(Path(path).name)
        assert Path(path).name != 'call2.npz'
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(audit.np, 'load', guarded_load)
    files_before = {p: p.read_bytes() for folder in (data, independent, primary) for p in folder.iterdir()}
    calls, mean, scale, models, _, _, _ = audit.prepare(data, independent, primary, protocol)
    assert [c['record']['split'] for c in calls] == ['train', 'train']
    assert set(models) == {'cv_l2', 'cv_rank_l2'}
    assert set(opened) == {'call0.npz', 'call1.npz', 'normalization.npz'}
    assert all(p.read_bytes() == contents for p, contents in files_before.items())


@pytest.mark.parametrize('change', ['data_summary', 'audit_summary', 'primary_summary',
    'train_array', 'development_array', 'manifest', 'normalizer', 'checkpoint'])
def test_complete_preflight_tampering_refused(complete_fixture, change):
    data, independent, primary, protocol = complete_fixture
    paths = {'data_summary': data/'summary.json', 'audit_summary': independent/'summary.json',
        'primary_summary': primary/'summary.json', 'train_array': data/'call0.npz',
        'development_array': data/'call2.npz', 'manifest': independent/'audited_data_manifest.json',
        'checkpoint': primary/'cv_rank_l2_seed994101.pt'}
    if change == 'normalizer':
        np.savez(primary/'normalization.npz', mean=np.ones((1, 1, 252)), scale=np.ones((1, 1, 252)))
    else:
        path = paths[change]; path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError): audit.prepare(data, independent, primary, protocol)


@pytest.mark.parametrize('mutation', [None, 'train_array', 'manifest', 'normalizer', 'checkpoint', 'protocol'])
def test_synthetic_complete_workflow_rechecks_inputs_before_summary(complete_fixture, monkeypatch, mutation):
    data, independent, primary, protocol = complete_fixture
    here = data.parent/'diagnostic'; here.mkdir()
    protocol.update({'prefix_lengths': list(range(1, 9)), 'precision_description_m': 1e-6,
        'physical_description_m': .05, 'scope': 'SYNTHETIC TOY FIXTURE WITH MOCKED PUBLICATION'})
    (here/'prefix_audit_protocol.json').write_text(json.dumps(protocol), encoding='utf8')
    monkeypatch.setattr(audit, 'HERE', here); monkeypatch.setattr(audit, 'ROOT', data.parent)
    monkeypatch.setattr(audit, 'published_sources', lambda: {'fixture_only': True})
    original_score = audit.score_prefix; done = False
    paths = {'train_array': data/'call0.npz', 'manifest': independent/'audited_data_manifest.json',
        'normalizer': primary/'normalization.npz', 'checkpoint': primary/'cv_rank_l2_seed994101.pt',
        'protocol': here/'prefix_audit_protocol.json'}
    def mutate_after_score(*args, **kwargs):
        nonlocal done
        result = original_score(*args, **kwargs)
        if mutation and not done:
            path = paths[mutation]; path.write_bytes(path.read_bytes()+b' '); done = True
        return result
    monkeypatch.setattr(audit, 'score_prefix', mutate_after_score)
    output = data.parent/'toy-output'
    if mutation:
        with pytest.raises(ValueError): audit.run(data, independent, primary, output)
        assert not (output/'summary.json').exists()
    else:
        result = audit.run(data, independent, primary, output)
        assert result['published'] == {'fixture_only': True}
        assert result['protocol']['scope'].startswith('SYNTHETIC')
        assert result['train_calls'] == 2
        assert result['native_branch_reexecuted'] is False
        assert result['models']['cv_rank_l2']['8']['maximum_observed_response_difference_m'] == 0.
        assert (output/'summary.json').exists()


@pytest.mark.parametrize('mode', ['git', 'api', 'wrong_api_head'])
def test_api_transport_preserves_exact_prepublication_requirement(tmp_path, monkeypatch, mode):
    here = tmp_path/'experiments/cwm_v32'; here.mkdir(parents=True)
    publisher = tmp_path/'scripts/publish_exact_github_api.py'; publisher.parent.mkdir()
    for path in (here/'prefix_invariance.py', here/'prefix_audit_protocol.json', publisher):
        path.write_bytes(b'fixture committed bytes\n')
    monkeypatch.setattr(audit, 'ROOT', tmp_path); monkeypatch.setattr(audit, 'HERE', here)
    sha = 'a'*40
    def git_output(args, **kwargs):
        if args[1:3] == ['rev-parse', 'HEAD']: return sha+'\n'
        if 'ls-remote' in args:
            if mode != 'git': raise audit.subprocess.CalledProcessError(1, args)
            return sha+'\t'+audit.BRANCH+'\n'
        assert args[1] == 'show'
        return (tmp_path/args[2].split(':', 1)[1]).read_bytes()
    monkeypatch.setattr(audit.subprocess, 'check_output', git_output)
    monkeypatch.setattr(audit, 'credential_token', lambda: 'SYNTHETIC_SECRET')
    class FakeAPI:
        def __init__(self, token): assert token == 'SYNTHETIC_SECRET'
        def head(self): return 'b'*40 if mode == 'wrong_api_head' else sha
    monkeypatch.setattr(audit, 'GitHubAPI', FakeAPI)
    if mode == 'wrong_api_head':
        with pytest.raises(ValueError): audit.published_sources()
    else:
        result = audit.published_sources()
        assert result['commit'] == sha
        assert result['verification'] == ('git_ls_remote_exact_branch' if mode == 'git'
            else 'authenticated_github_git_data_api_exact_branch')
