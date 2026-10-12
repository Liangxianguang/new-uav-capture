"""Synthetic archive/failure orchestration only; no native training or efficacy."""
import copy
import hashlib
import json
import stat
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import archive_io as io
import package_streamed_release as release


def fixture_zip(path, members, rows=None):
    inventory = {name: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                 for name, raw in members.items()}
    with zipfile.ZipFile(path, 'x') as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        archive.writestr(io.MANIFEST, json.dumps({'schema': io.SCHEMA, 'members': inventory if rows is None else rows}))


@pytest.mark.parametrize('name', ['../x', '/x', 'a/../b', 'a//b', './x', 'C:/x',
    'a\\x', 'a\x00b', 'a/CON.txt', 'NUL', 'x/LPT9.npy', 'x.', 'x ', 'a:b', 'a?b'])
def test_unsafe_archive_refused_before_extraction(tmp_path, name):
    path = tmp_path/'unsafe.zip'
    fixture_zip(path, {name: b'fixture'})
    with pytest.raises(ValueError): io.extract_archive(path, tmp_path/'extracted')
    assert not (tmp_path/'extracted').exists()


def test_streaming_integrity_and_exclusive_extraction(tmp_path):
    path = tmp_path/'fixture.zip'
    fixture_zip(path, {'data/one.json': b'{}', 'primary/toy.pt': b'not scientific weights'})
    result = io.extract_archive(path, tmp_path/'extracted')
    assert result['sha256'] == io.sha(path) and result['members'] == 2
    assert (tmp_path/'extracted/primary/toy.pt').read_bytes() == b'not scientific weights'
    with pytest.raises(ValueError): io.extract_archive(path, tmp_path/'extracted')


@pytest.mark.parametrize('kind', ['omission', 'digest', 'length', 'bool_length', 'extra_field', 'case_alias', 'symlink', 'directory'])
def test_archive_manifest_or_member_mutation_refused(tmp_path, kind):
    path = tmp_path/'bad.zip'; members = {'data/x': b'x'}
    rows = {'data/x': {'bytes': 1, 'sha256': hashlib.sha256(b'x').hexdigest()}}
    if kind == 'omission': rows = {}
    elif kind == 'digest': rows['data/x']['sha256'] = '0'*64
    elif kind == 'length': rows['data/x']['bytes'] = 2
    elif kind == 'bool_length': rows['data/x']['bytes'] = True
    elif kind == 'extra_field': rows['data/x']['ignored'] = True
    elif kind == 'case_alias': members['DATA/X'] = b'x'
    if kind in ('symlink', 'directory'):
        with zipfile.ZipFile(path, 'x') as archive:
            info = zipfile.ZipInfo('data/x'); info.create_system = 3
            info.external_attr = ((stat.S_IFLNK if kind == 'symlink' else stat.S_IFDIR) | 0o777) << 16
            archive.writestr(info, b'x')
            archive.writestr(io.MANIFEST, json.dumps({'schema': io.SCHEMA, 'members': rows}))
    else:
        fixture_zip(path, members, rows)
    with pytest.raises(ValueError): io.verify_archive(path)


@pytest.mark.parametrize('raw', [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'])
def test_ambiguous_or_nonfinite_json_refused(raw):
    with pytest.raises(ValueError): io.decode_json(raw)


def test_npz_compare_checks_all_fields_dtype_signed_zero(tmp_path):
    a, b = tmp_path/'a.npz', tmp_path/'b.npz'
    np.savez_compressed(a, field=np.zeros((2, 3), np.float64))
    np.savez(b, field=np.zeros((2, 3), np.float64))
    assert io.npz_payloads(a) == io.npz_payloads(b)
    changed = np.zeros((2, 3), np.float64); changed[0, 0] = -0.
    np.savez(b, field=changed)
    assert io.npz_payloads(a) != io.npz_payloads(b)
    np.savez(b, field=np.zeros((2, 3), np.float32))
    assert io.npz_payloads(a) != io.npz_payloads(b)


def test_full512_trajectory_inventory_does_not_treat_axes_as_families():
    episodes = [{'episode_index': i, 'unhooked_original_trace_equal': True} for i in range(512)]
    assert len(release.trajectory_names(episodes)) == 1024
    for changed in (episodes[:-1], episodes[:-1]+[episodes[0]],
                    episodes[:-1]+[{'episode_index': 511, 'unhooked_original_trace_equal': False}]):
        with pytest.raises(ValueError): release.trajectory_names(changed)


def test_partial_native_data_cannot_create_archive_or_replay(tmp_path):
    archive, output = tmp_path/'never.zip', tmp_path/'never'
    with pytest.raises((FileNotFoundError, ValueError)):
        release.package(*[tmp_path/n for n in ('data', 'audit', 'primary', 'repeated', 'reload')], archive, output)
    assert not archive.exists() and not output.exists()


@pytest.mark.parametrize('kind', ['existing', 'same', 'nested', 'input', 'parts'])
def test_outputs_refused_before_scientific_reads(tmp_path, monkeypatch, kind):
    archive, output, parts = tmp_path/'a.zip', tmp_path/'replay', None
    data = tmp_path/'data'
    if kind == 'existing': archive.write_bytes(b'user-owned')
    elif kind == 'same': output = archive
    elif kind == 'nested': archive = output/'inside.zip'
    elif kind == 'input': archive = data/'inside.zip'
    else:
        parts = tmp_path/'transport.parts.json'; (tmp_path/'transport.part001').write_bytes(b'user-owned')
    monkeypatch.setattr(release, 'inventory', lambda *a, **k: pytest.fail('Reject outputs first'))
    with pytest.raises(ValueError): release.package(data, *[tmp_path/n for n in ('audit', 'primary', 'repeated', 'reload')], archive, output, parts)
    if kind == 'existing': assert archive.read_bytes() == b'user-owned'
    if kind == 'parts': assert (tmp_path/'transport.part001').read_bytes() == b'user-owned'


def toy_reload(tmp_path):
    directories = {stage: tmp_path/stage for stage in ('data', 'audit', 'primary', 'repeated')}
    for d in directories.values():
        d.mkdir(); (d/'summary.json').write_bytes(b'{}')
    qualification = {'cv_prefix_rank_l2': {'research_eligible': False, 'cost_gain': -.1}}
    report = {'status': 'two_complete_streamed_runs_all18models_cold_reloaded_original_costs_verified',
        'models_per_run': 9, 'complete_runs': 2, 'all12_checkpoint_contents_and_history_equal': True,
        'all_predictions_original_costs_gates_reexecuted': True, 'all_optimizer_sampler_torch_rng_actually_restored': True,
        'all_declared_native_evidence_unchanged': True, 'enhanced_control_enabled': False, 'holdout_used': False,
        'new_training_performed_by_reload': False, 'artifact_packaged_and_replayed': False,
        'identical_serialized_checkpoint_pairs': 7, 'different_serialized_checkpoint_pairs': 5,
        'primary_research_eligible': False, 'qualification': qualification,
        'runs': [{'models': 9, 'all_predictions_original_costs_gates_reexecuted': True, 'primary_research_eligible': False}]*2,
        'reload_source_hashes': {p.relative_to(release.ROOT).as_posix(): io.sha(p) for p in (release.HERE.parent/'cwm_v48').glob('*.py')}}
    for stage, d in directories.items():
        report[stage+'_summary_sha256' if stage != 'audit' else 'data_audit_summary_sha256'] = io.sha(d/'summary.json')
    reports = [{'qualification': qualification, 'primary_research_eligible': False}]*2
    return report, directories, reports


@pytest.mark.parametrize('kind', ['promotion', 'holdout', 'training', 'incomplete', 'identity', 'gates', 'counts', 'sources', 'typed_gate'])
def test_reload_certificate_tampering_refused(tmp_path, kind):
    report, directories, reports = toy_reload(tmp_path)
    release.validate_reload(report, directories, reports)
    changed = copy.deepcopy(report)
    if kind == 'promotion': changed['enhanced_control_enabled'] = True
    elif kind == 'holdout': changed['holdout_used'] = True
    elif kind == 'training': changed['new_training_performed_by_reload'] = True
    elif kind == 'incomplete': changed['all12_checkpoint_contents_and_history_equal'] = False
    elif kind == 'identity': changed['primary_summary_sha256'] = '0'*64
    elif kind == 'gates': changed['qualification']['cv_prefix_rank_l2']['cost_gain'] = .1
    elif kind == 'counts': changed['different_serialized_checkpoint_pairs'] = 4
    elif kind == 'sources': changed['reload_source_hashes'] = {}
    else: changed['all_predictions_original_costs_gates_reexecuted'] = 1
    with pytest.raises(ValueError): release.validate_reload(changed, directories, reports)


def mock_replay(tmp_path, monkeypatch):
    # No native evaluator is involved: tests only enforce ordering and refusal.
    root = tmp_path/'repo'; root.mkdir(); (root/'entry.py').write_bytes(b'fixture source')
    previous = {'primary_research_eligible': False, 'qualification': {'primary': {'research_eligible': False}}}
    monkeypatch.setattr(release, 'ROOT', root)
    monkeypatch.setattr(release, 'inventory', lambda **k: ({'verification_source/entry.py': root/'entry.py'}, previous))
    path = tmp_path/'toy.zip'
    fixture_zip(path, {'verification_source/entry.py': (root/'entry.py').read_bytes(),
        'release_contract.json': json.dumps(release.release_contract(previous)).encode()})
    entries = []
    def child(entry, arguments, output):
        entries.append(entry)
        if entry == release.DATA_ENTRY:
            dest = output/'extracted/data'; dest.mkdir(parents=True); io.write_json(dest/'episodes.json', [])
    monkeypatch.setattr(release, 'run_child', child)
    monkeypatch.setattr(release, 'compare_data_audit', lambda *a: entries.append('compare_all_native_data'))
    monkeypatch.setattr(release, 'compare_model_reload', lambda *a: entries.append('compare_all18_models'))
    (tmp_path/'reload-summary').write_bytes(b'{}')
    real_sha = release.sha
    monkeypatch.setattr(release, 'sha', lambda p: real_sha(tmp_path/'reload-summary') if p.name == 'summary.json' else real_sha(p))
    return path, previous, entries


def test_replay_requires_both_real_entries_in_order(tmp_path, monkeypatch):
    path, previous, entries = mock_replay(tmp_path, monkeypatch)
    report = release.replay_archive(path, tmp_path/'replay')
    assert entries == [release.DATA_ENTRY, 'compare_all_native_data', release.MODEL_ENTRY, 'compare_all18_models']
    assert report['primary_research_eligible'] is False
    assert report['enhanced_control_enabled'] is False and report['holdout_used'] is False
    assert report['closed_loop_capture_safety_latency_qualified'] is False
    assert report['active_model_fault_fallback_qualified'] is False
    assert report['qualification'] == previous['qualification']
    assert (tmp_path/'replay/summary.json').is_file()


@pytest.mark.parametrize('stage', ['data', 'data_compare', 'models', 'models_compare', 'mutation', 'omission'])
def test_any_replay_failure_has_no_pass_certificate(tmp_path, monkeypatch, stage):
    path, _, _ = mock_replay(tmp_path, monkeypatch)
    original_child = release.run_child
    def child(entry, arguments, output):
        if (entry == release.DATA_ENTRY and stage == 'data') or (entry == release.MODEL_ENTRY and stage == 'models'):
            raise subprocess.CalledProcessError(1, entry)
        original_child(entry, arguments, output)
        if entry == release.MODEL_ENTRY and stage == 'mutation':
            (output/'extracted/verification_source/entry.py').write_bytes(b'tampered source')
    monkeypatch.setattr(release, 'run_child', child)
    def fail(*args): raise ValueError('Synthetic comparison refusal')
    if stage == 'data_compare': monkeypatch.setattr(release, 'compare_data_audit', fail)
    if stage == 'models_compare': monkeypatch.setattr(release, 'compare_model_reload', fail)
    if stage == 'omission':
        monkeypatch.setattr(release, 'inventory', lambda **k: ({'verification_source/entry.py': release.ROOT/'entry.py',
            'data/missing': release.ROOT/'entry.py'}, {'primary_research_eligible': False, 'qualification': {}}))
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        release.replay_archive(path, tmp_path/'replay')
    assert not (tmp_path/'replay/summary.json').exists()


def test_data_replay_compares_typed_costs_all_arrays_and_sources(tmp_path, monkeypatch):
    old, new = tmp_path/'old', tmp_path/'new'
    monkeypatch.setattr(release, 'trajectory_names', lambda e: {'1.npz', '1.commands.npz'})
    monkeypatch.setattr(release, 'check_sources', lambda *a: None)
    for d in (old, new):
        (d/'trajectories').mkdir(parents=True)
        io.write_json(d/'summary.json', {'source_hashes': {}, 'checked_calls': 1, 'gain': 0.})
        for name in ('checked_calls.json', 'full_label_cost_checks.json', 'audited_data_manifest.json'):
            io.write_json(d/name, {'native_cost': 0.})
        np.savez_compressed(d/'trajectories/1.npz', target=np.zeros((2, 3)))
        np.savez_compressed(d/'trajectories/1.commands.npz', commanded=np.zeros((2, 4, 3)))
    release.compare_data_audit(old, new, [])
    (new/'full_label_cost_checks.json').write_bytes(b'{"native_cost":0}')
    with pytest.raises(ValueError): release.compare_data_audit(old, new, [])
    (new/'full_label_cost_checks.json').write_bytes((old/'full_label_cost_checks.json').read_bytes())
    np.savez_compressed(new/'trajectories/1.commands.npz', commanded=np.ones((2, 4, 3)))
    with pytest.raises(ValueError): release.compare_data_audit(old, new, [])


def test_all18_restoration_reports_required_and_typed(tmp_path, monkeypatch):
    old, new = tmp_path/'old', tmp_path/'new'; previous = {'gates': {'primary': False}}
    monkeypatch.setattr(release, 'check_sources', lambda *a: None)
    for d in (old, new):
        d.mkdir(); io.write_json(d/'summary.json', previous)
        for stage in ('primary', 'repeated'):
            (d/stage).mkdir()
            for i in range(9): io.write_json(d/stage/f'model{i}.json', {'actual_adam_steps': 80, 'rng_restored': True})
    release.compare_model_reload(old, new, previous)
    (new/'repeated/model8.json').write_bytes(b'{"actual_adam_steps":80.0,"rng_restored":true}')
    with pytest.raises(ValueError): release.compare_model_reload(old, new, previous)
    (new/'repeated/model8.json').unlink()
    with pytest.raises(ValueError): release.compare_model_reload(old, new, previous)


def test_curated_inventory_includes_all512_traces_both_runs_and18_reload_reports(tmp_path, monkeypatch):
    root = tmp_path/'repo'; here = root/'experiments/cwm_v48'
    here.mkdir(parents=True); (here/'entry.py').write_bytes(b'toy source')
    transport = root/'experiments/cwm_v21/artifact_transport.py'
    transport.parent.mkdir(); transport.write_bytes(b'toy transport')
    monkeypatch.setattr(release, 'ROOT', root); monkeypatch.setattr(release, 'HERE', here)
    dirs = {s: tmp_path/s for s in ('data', 'audit', 'primary', 'repeated', 'reload')}
    for d in dirs.values(): d.mkdir(); io.write_json(d/'summary.json', {})
    episodes = [{'episode_index': i, 'unhooked_original_trace_equal': True} for i in range(512)]
    for name in release.trajectory_names(episodes):
        path = dirs['audit']/'trajectories'/name; path.parent.mkdir(exist_ok=True); path.write_bytes(b'toy NPY evidence')
    for name in ('checked_calls.json', 'full_label_cost_checks.json', 'audited_data_manifest.json'):
        io.write_json(dirs['audit']/name, {})
    models = [{'configuration': m, 'seed': s} for m in ('motion', 'full', 'prefix') for s in (1, 2, 3)]
    report = {'source_hashes': {}, 'models': models}
    previous = {'reload_source_hashes': {}, 'primary_research_eligible': False, 'qualification': {'primary': False}}
    (dirs['reload']/'summary.json').write_text(json.dumps(previous), encoding='utf8')
    for stage in ('primary', 'repeated'):
        (dirs['reload']/stage).mkdir()
        for m in models: io.write_json(dirs['reload']/stage/f"{m['configuration']}_seed{m['seed']}.json", {'toy': True})
        (dirs[stage]/'all_call_cache.npz').write_bytes(b'toy all-call cache')
        (dirs[stage]/'user-extra.zip').write_bytes(b'out of scope user artifact')
    corpus = SimpleNamespace(manifest={'summary.json': io.sha(dirs['data']/'summary.json')},
        audit_report={'source_hashes': {}}, data_report={'source_hashes': {}, 'episodes': episodes}, protocol={})
    monkeypatch.setattr(release, 'AuditedCorpus', lambda *a: corpus)
    monkeypatch.setattr(release.run_evidence, 'summary', lambda *a: report)
    monkeypatch.setattr(release.run_evidence, 'inventory', lambda d, *a: {n: io.sha(d/n) for n in ('summary.json', 'all_call_cache.npz')})
    monkeypatch.setattr(release.run_evidence, 'require_same_summary', lambda *a: None)
    monkeypatch.setattr(release, 'validate_reload', lambda *a: None)
    monkeypatch.setattr(release, 'DEPENDENCIES', {})
    members, saved = release.inventory(**dirs)
    assert saved == previous
    assert len([n for n in members if n.startswith('audit/trajectories/')]) == 1024
    assert len([n for n in members if n.startswith(('reload/primary/', 'reload/repeated/'))]) == 18
    assert {'primary/all_call_cache.npz', 'repeated/all_call_cache.npz'} <= set(members)
    assert not any('user-extra' in n for n in members)
    (dirs['reload']/'repeated/prefix_seed3.json').unlink()
    with pytest.raises(ValueError): release.inventory(**dirs)


def test_fresh_child_uses_original_runtime_exact_entry_and_fails_on_nonzero(tmp_path, monkeypatch):
    root = tmp_path/'repo'; root.mkdir(); (root/'audit.py').write_bytes(b'fixture entry')
    monkeypatch.setattr(release, 'ROOT', root)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.CalledProcessError(2, command)
    monkeypatch.setattr(release.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError): release.run_child('audit.py', ['--data', tmp_path/'data'], tmp_path)
    command, kwargs = calls[0]
    assert command == [sys.executable, str(root/'audit.py'), '--data', str(tmp_path/'data')]
    assert kwargs['check'] is True and kwargs['cwd'] == root
    assert (tmp_path/'audit.stdout.log').is_file() and (tmp_path/'audit.stderr.log').is_file()
