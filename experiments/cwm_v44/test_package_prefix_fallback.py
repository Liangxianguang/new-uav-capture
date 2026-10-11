"""Synthetic evidence refusal/orchestration tests, never native1280 proof."""
import copy
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import package_prefix_fallback as release


def toy_zip(path, members):
    rows = {n: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} for n, raw in members.items()}
    with zipfile.ZipFile(path, 'x') as out:
        for name, raw in members.items(): out.writestr(name, raw)
        out.writestr(release.MANIFEST, json.dumps({'schema': release.SCHEMA, 'members': rows}))


def test_wrong_version_refused_before_extraction(tmp_path):
    path = tmp_path/'old.zip'
    with zipfile.ZipFile(path, 'x') as out:
        out.writestr(release.MANIFEST, '{"schema":"cwm_v30","members":{}}')
    with pytest.raises(ValueError): release.extract(path, tmp_path/'extracted')
    assert not (tmp_path/'extracted').exists()


def test_live_partial_proof_refused_before_reference_validation(tmp_path, monkeypatch):
    proof = tmp_path/'partial'; proof.mkdir()
    monkeypatch.setattr(release.native, 'check_reference', lambda *a: pytest.fail('Partial proof refuses first'))
    with pytest.raises(ValueError): release.inventory(None, None, None, proof)


@pytest.mark.parametrize('kind', ['unsafe', 'alias', 'duplicate_key', 'wrong_digest', 'typed_length'])
def test_archive_inventory_tampering_refused(tmp_path, kind):
    path = tmp_path/'bad.zip'; raw = b'toy'
    name = '../x' if kind == 'unsafe' else 'proof/x'
    rows = {name: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}
    if kind == 'wrong_digest': rows[name]['sha256'] = '0'*64
    if kind == 'typed_length': rows[name]['bytes'] = 3.0
    with zipfile.ZipFile(path, 'x') as out:
        out.writestr(name, raw)
        if kind == 'alias':
            out.writestr('PROOF/X', raw); rows['PROOF/X'] = rows[name]
        manifest = json.dumps({'schema': release.SCHEMA, 'members': rows})
        if kind == 'duplicate_key': manifest = manifest[:-1]+',"schema":"duplicate"}'
        out.writestr(release.MANIFEST, manifest)
    with pytest.raises(ValueError): release.extract(path, tmp_path/'extracted')
    assert not (tmp_path/'extracted').exists()


@pytest.mark.parametrize('kind', ['existing', 'same', 'nested', 'input', 'input_parent', 'parts'])
def test_output_collision_refused_before_publication_or_inventory(tmp_path, monkeypatch, kind):
    inputs = [tmp_path/n for n in ('old.zip', 'certificate.json', 'reference', 'proof')]
    artifact, replay, parts = tmp_path/'new.zip', tmp_path/'replay', None
    if kind == 'existing': artifact.write_bytes(b'user owned')
    elif kind == 'same': replay = artifact
    elif kind == 'nested': artifact = replay/'x.zip'
    elif kind == 'input': artifact = inputs[-1]/'x.zip'
    elif kind == 'input_parent': replay = tmp_path
    else:
        parts = tmp_path/'output.parts.json'; (tmp_path/'output.part001').write_bytes(b'user owned')
    monkeypatch.setattr(release.native, 'published_sources', lambda: pytest.fail('Validate outputs first'))
    monkeypatch.setattr(release, 'inventory', lambda *a: pytest.fail('Validate outputs first'))
    with pytest.raises(ValueError): release.package(*inputs, artifact, replay, parts)
    if kind == 'existing': assert artifact.read_bytes() == b'user owned'
    if kind == 'parts': assert (tmp_path/'output.part001').read_bytes() == b'user owned'


def wrapper_fixture(tmp_path, monkeypatch):
    root, folder = tmp_path/'repo', tmp_path/'proof'
    (folder/'native').mkdir(parents=True)
    (folder/'native/summary.json').write_bytes(b'{"toy_native":true}')
    monkeypatch.setattr(release, 'ROOT', root)
    sources = {}
    for name in release.native.ENTRY_SOURCES:
        raw = ('toy source '+name).encode()
        for path in (root/name, folder/'native/source'/name):
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(raw)
        sources[name] = hashlib.sha256(raw).hexdigest()
    monkeypatch.setattr(release.subprocess, 'check_output', lambda command, **kw: (root/command[-1].split(':', 1)[1]).read_bytes())
    integrity = {'sha256': release.native.ARCHIVE_SHA, 'bytes': 95, 'members': 10}
    certificate = {'original8Level_equal_macro': {'safe_capture': .99}}
    audit = {'source_hashes': sources, 'modes': {'off': {'episodes': 1280}, 'refusal': {'episodes': 1280}}}
    prereg = {'commit': 'a'*40, 'remote': 'https://github.com/Liangxianguang/new-uav-capture.git',
              'branch': 'refs/heads/causal-world-model-v1-20261009', 'exact_pushed_sources_verified': True}
    report = {'status': release.STATUS, **{k: False for k in release.FALSE_FLAGS},
        'all1280_each_mode_trajectories_plans_CBF_commands_equal': True,
        'all_nontiming_episode_step_diagnostics_equal': True, 'original_archive': integrity,
        **certificate, 'modes': audit['modes'], 'native_used_source_hashes': sources, 'preregistration': prereg,
        'original_certificate_sha256': release.native.CERTIFICATE_SHA,
        'native_audit_summary_sha256': release.sha(folder/'native/summary.json')}
    release.write_json(folder/'summary.json', report)
    release.write_json(folder/'preregistration.json', prereg)
    release.write_json(folder/'reference_binding.json', {'artifact': integrity,
        'certificate_sha256': release.native.CERTIFICATE_SHA, 'native_prefix_fallback_replay_completed': False,
        'enhanced_control_enabled': False, 'holdout_used': False})
    return folder, integrity, certificate, audit, report


@pytest.mark.parametrize('kind', ['partial', 'enhanced', 'typed', 'counts', 'certificate', 'native_sha',
                                 'source_omission', 'source_mutation', 'commit_source', 'prereg'])
def test_false_complete_wrapper_rejected(tmp_path, monkeypatch, kind):
    folder, integrity, certificate, audit, report = wrapper_fixture(tmp_path, monkeypatch)
    release.validate_wrapper(folder, integrity, certificate, audit)
    changed = copy.deepcopy(report)
    if kind == 'partial': changed['status'] = 'partial'
    elif kind == 'enhanced': changed['enhanced_control_enabled'] = True
    elif kind == 'typed': changed['holdout_used'] = 0
    elif kind == 'counts': changed['modes']['off']['episodes'] = 64
    elif kind == 'certificate': changed['original_certificate_sha256'] = '0'*64
    elif kind == 'native_sha': changed['native_audit_summary_sha256'] = '0'*64
    elif kind == 'source_omission': changed['native_used_source_hashes'] = {}
    elif kind == 'source_mutation': (folder/'native/source'/release.native.ENTRY_SOURCES[0]).write_bytes(b'changed')
    elif kind == 'commit_source': monkeypatch.setattr(release.subprocess, 'check_output', lambda *a, **kw: b'wrong published source')
    else: changed['preregistration']['exact_pushed_sources_verified'] = 1
    (folder/'summary.json').write_text(json.dumps(changed), encoding='utf8')
    with pytest.raises(ValueError): release.validate_wrapper(folder, integrity, certificate, audit)


@pytest.mark.parametrize('kind', ['missing_mode', 'typed_count', 'partial_count'])
def test_wrapper_and_audit_agreeing_on_partial_modes_is_still_rejected(tmp_path, monkeypatch, kind):
    folder, integrity, certificate, audit, report = wrapper_fixture(tmp_path, monkeypatch)
    if kind == 'missing_mode': audit['modes'].pop('refusal')
    else: audit['modes']['off']['episodes'] = 1280.0 if kind == 'typed_count' else 64
    report['modes'] = audit['modes']
    (folder/'summary.json').write_text(json.dumps(report), encoding='utf8')
    with pytest.raises(ValueError): release.validate_wrapper(folder, integrity, certificate, audit)


def mocked_replay(tmp_path, monkeypatch):
    root = tmp_path/'repo'; root.mkdir(); (root/'entry.py').write_bytes(b'fixture source')
    monkeypatch.setattr(release, 'ROOT', root)
    previous = {'original_archive': {}, 'original8Level_equal_macro': {'collision': .01}}
    path = tmp_path/'toy.zip'
    toy_zip(path, {'verification_source/entry.py': b'fixture source',
                  release.ORIGINAL: b'toy zip', release.CERTIFICATE: b'{}',
                  'release_contract.json': json.dumps(release.contract(previous)).encode()})
    monkeypatch.setattr(release, 'inventory', lambda *a: ({'verification_source/entry.py': root/'entry.py',
        release.ORIGINAL: path, release.CERTIFICATE: path}, previous))
    real_extract = release.extract
    def extract(path, dest, verifier=release.verify_archive):
        if verifier is release.native.original_release.verify_archive:
            dest.mkdir(); return {}
        return real_extract(path, dest, verifier)
    monkeypatch.setattr(release, 'extract', extract)
    calls = []
    def child(archive, certificate, reference, output, logs):
        calls.append('actual_child')
        output.mkdir(); release.write_json(output/'summary.json', {'toy': True})
    monkeypatch.setattr(release, 'run_child', child)
    monkeypatch.setattr(release, 'compare_native', lambda *a: calls.append('compare_complete'))
    return path, calls


def test_archive_qualification_only_after_child_and_complete_comparison(tmp_path, monkeypatch):
    path, calls = mocked_replay(tmp_path, monkeypatch)
    # Retained prior summary is necessary even when orchestration is mocked.
    original_extract = release.extract
    def extract(path, dest, verifier=release.verify_archive):
        result = original_extract(path, dest, verifier)
        if dest.name == 'e':
            (dest/'proof').mkdir(); release.write_json(dest/'proof/summary.json', {'toy': True})
        return result
    monkeypatch.setattr(release, 'extract', extract)
    result = release.replay_archive(path, tmp_path/'replay')
    assert calls == ['actual_child', 'compare_complete']
    assert result['enhanced_control_enabled'] is False and result['holdout_used'] is False
    assert result['latency_qualified'] is False and result['active_model_faults_exercised'] is False
    assert (tmp_path/'replay/summary.json').exists()


@pytest.mark.parametrize('kind', ['child', 'comparison', 'mutation', 'omission', 'original_input_mutation'])
def test_replay_failure_never_issues_success(tmp_path, monkeypatch, kind):
    path, _ = mocked_replay(tmp_path, monkeypatch)
    real_child = release.run_child
    def child(*args):
        if kind == 'child': raise subprocess.CalledProcessError(1, 'toy child')
        real_child(*args)
        if kind == 'mutation': (args[-1]/'e/verification_source/entry.py').write_bytes(b'changed')
    monkeypatch.setattr(release, 'run_child', child)
    if kind == 'comparison':
        def fail(*args): raise ValueError('Synthetic complete comparison failed')
        monkeypatch.setattr(release, 'compare_native', fail)
    if kind == 'omission': monkeypatch.setattr(release, 'inventory', lambda *a: ({}, {}))
    def recheck():
        if kind == 'original_input_mutation': raise ValueError('Synthetic original input mutation')
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        release.replay_archive(path, tmp_path/'replay', input_recheck=recheck)
    assert not (tmp_path/'replay/summary.json').exists()


def test_fresh_child_is_exact_V43_in_original_runtime(tmp_path, monkeypatch):
    root = tmp_path/'repo'; path = root/release.ENTRY
    path.parent.mkdir(parents=True); path.write_bytes(b'fixture source')
    monkeypatch.setattr(release, 'ROOT', root)
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs)); raise subprocess.CalledProcessError(2, command)
    monkeypatch.setattr(release.subprocess, 'run', run)
    with pytest.raises(subprocess.CalledProcessError):
        release.run_child(*[tmp_path/n for n in ('original.zip', 'cert.json', 'reference', 'fresh')], tmp_path)
    command, kwargs = calls[0]
    assert command[:2] == [sys.executable, str(path)]
    assert kwargs['check'] is True and kwargs['cwd'] == root
    assert (tmp_path/'stdout.log').is_file() and (tmp_path/'stderr.log').is_file()
