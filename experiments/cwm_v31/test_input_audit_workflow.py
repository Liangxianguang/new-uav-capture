"""Synthetic complete IO workflow; publication alone is mocked, not evidence."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import public_input_ambiguity as entry
from test_public_input_ambiguity import public


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf8')


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    root = tmp_path/'repo'; here = root/'experiments/cwm_v31'; here.mkdir(parents=True)
    paths = {name: tmp_path/name for name in ('data', 'audit', 'primary')}
    for folder in paths.values(): folder.mkdir()
    published_data_protocol = entry.read(HERE.parent/'cwm_v28/data_protocol.json')
    records = []; arrays = {}
    for i, split in enumerate(('train', 'train', 'development_validation')):
        values = public()
        values.update({'valid': np.ones((2, 8), bool), 'anchor_valid': np.ones(8, bool),
            'target': np.ones((2, 8, 3))*i*2., 'anchor_target': np.zeros((8, 3))})
        path = f'call{i}.npz'; arrays[path] = values
        np.savez_compressed(paths['data']/path, **values)
        records.append({'split': split, 'arrays_path': path, 'group': f'g{i}'})
    write_json(paths['data']/'records.json', records)
    data = {'protocol': published_data_protocol, 'source_hashes': {},
        'enhanced_control_enabled': False, 'holdout_used': False}
    write_json(paths['data']/'summary.json', data)
    manifest = {name: entry.digest(paths['data']/name) for name in ('records.json', 'summary.json', *arrays)}
    write_json(paths['audit']/'audited_data_manifest.json', manifest)
    audit = {'status': 'independent_fresh_sequential_public_branch_cost_audit_passed',
        'data_summary_sha256': entry.digest(paths['data']/'summary.json'),
        'audited_data_manifest_sha256': entry.digest(paths['audit']/'audited_data_manifest.json'),
        'data_gate_passed': True, 'checked_calls': 3, 'source_hashes': {},
        'enhanced_control_enabled': False, 'holdout_used': False}
    write_json(paths['audit']/'summary.json', audit)
    mean, scale = entry.normalization([{'values': v} for v in list(arrays.values())[:2]], [0, 1])
    np.savez_compressed(paths['primary']/'normalization.npz', mean=mean, scale=scale)
    training = {'fixture_only': True}
    checkpoint = paths['primary']/'cv_rank_l2_seed994101.pt'
    torch.save({'protocol': training, 'normalizer_mean': torch.from_numpy(mean),
        'normalizer_scale': torch.from_numpy(scale)}, checkpoint)
    primary = {'protocol': training, 'source_hashes': {}, 'enhanced_control_enabled': False,
        'holdout_used': False, 'primary_research_eligible': False,
        'selected_median_seeds': {'cv_rank_l2': 994101},
        'models': [{'configuration': 'cv_rank_l2', 'seed': 994101, 'checkpoint_sha256': entry.digest(checkpoint)}]}
    write_json(paths['primary']/'summary.json', primary)
    protocol = {'split': 'train', 'expected_train_calls': 2, 'expected_train_groups': 2,
        'primary_configuration': 'cv_rank_l2', 'descriptive_response_threshold_m': .05,
        'source_data_summary_sha256': entry.digest(paths['data']/'summary.json'),
        'source_complete_data_audit_sha256': entry.digest(paths['audit']/'summary.json'),
        'source_primary_summary_sha256': entry.digest(paths['primary']/'summary.json'),
        'scope': 'Synthetic fixture only; NOT complete V28 data or a scientific result'}
    write_json(here/'input_audit_protocol.json', protocol)
    (here/'public_input_ambiguity.py').write_bytes((HERE/'public_input_ambiguity.py').read_bytes())
    sources = {p.relative_to(root).as_posix(): entry.digest(p) for p in here.iterdir()}
    monkeypatch.setattr(entry, 'ROOT', root); monkeypatch.setattr(entry, 'HERE', here)
    monkeypatch.setattr(entry, 'published_sources', lambda: {
        'commit': 'fixture-publication-mocked', 'source_hashes': sources})
    return paths, here, tmp_path/'output'


def execute(workflow):
    paths, here, output = workflow
    return entry.run(paths['data'], paths['audit'], paths['primary'], output)


def test_complete_workflow_only_loads_train_arrays_and_preserves_inputs(workflow, monkeypatch):
    paths, here, output = workflow
    before = {str(p): entry.digest(p) for folder in paths.values() for p in folder.iterdir()}
    original_load = entry.np.load; loaded = []
    def spy(path, *args, **kwargs):
        loaded.append(Path(path).resolve())
        if Path(path).name == 'call2.npz': pytest.fail('Development arrays cannot enter TRAIN statistics')
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(entry.np, 'load', spy)
    report = execute(workflow)
    assert report['train_calls'] == 2 and report['train_groups'] == 2
    assert report['input_tables']['whole_optional_model']['empirical_minimum_deterministic_coordinate_mse_m2'] == 1.
    assert report['development_arrays_loaded_into_diagnostics'] is False
    assert report['enhanced_control_enabled'] is False and report['holdout_used'] is False
    assert (output/'summary.json').exists()
    assert before == {str(p): entry.digest(p) for folder in paths.values() for p in folder.iterdir()}
    assert paths['data']/'call2.npz' not in loaded


@pytest.mark.parametrize('change', ['manifest', 'array', 'normalizer', 'checkpoint', 'model_hash', 'pinned_summary'])
def test_bad_input_never_creates_a_completed_summary(workflow, change):
    paths, here, output = workflow
    if change == 'manifest':
        (paths['audit']/'audited_data_manifest.json').write_bytes(b'{}')
    elif change == 'array':
        (paths['data']/'call0.npz').write_bytes(b'tampered')
    elif change == 'normalizer':
        np.savez_compressed(paths['primary']/'normalization.npz', mean=np.ones((1, 1, 252)), scale=np.ones((1, 1, 252)))
    elif change == 'checkpoint':
        (paths['primary']/'cv_rank_l2_seed994101.pt').write_bytes(b'tampered')
    else:
        report = entry.read(paths['primary']/'summary.json')
        if change == 'model_hash': report['models'][0]['checkpoint_sha256'] = '0'*64
        else: report['primary_research_eligible'] = True
        write_json(paths['primary']/'summary.json', report)
    with pytest.raises(ValueError): execute(workflow)
    assert not (output/'summary.json').exists()


@pytest.mark.parametrize('kind', ['source', 'train', 'development', 'checkpoint'])
def test_mutation_during_analysis_never_signs_completion(workflow, monkeypatch, kind):
    paths, here, output = workflow
    original = entry.feature_bytes; mutated = False
    def mutate(*args, **kwargs):
        nonlocal mutated
        if not mutated:
            if kind == 'source': target = here/'public_input_ambiguity.py'
            elif kind == 'checkpoint': target = paths['primary']/'cv_rank_l2_seed994101.pt'
            else: target = paths['data']/('call0.npz' if kind == 'train' else 'call2.npz')
            target.write_bytes(target.read_bytes()+b'mutated')
            mutated = True
        return original(*args, **kwargs)
    monkeypatch.setattr(entry, 'feature_bytes', mutate)
    with pytest.raises(ValueError): execute(workflow)
    assert not (output/'summary.json').exists()


def test_existing_output_is_not_overwritten_even_before_publication(workflow, monkeypatch):
    paths, here, output = workflow
    output.mkdir(); (output/'user-owned').write_bytes(b'keep')
    monkeypatch.setattr(entry, 'published_sources', lambda: pytest.fail('Must refuse before publication checks'))
    with pytest.raises(ValueError): execute(workflow)
    assert (output/'user-owned').read_bytes() == b'keep'
