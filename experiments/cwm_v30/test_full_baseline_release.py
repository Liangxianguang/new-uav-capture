"""Archive integrity/refusal tests; no fixture claims native replay success."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import package_full_baseline as release


def tiny_archive(path, fault=None):
    raw, name = b'fixture-only-original-evidence', 'baseline/evidence.json'
    if fault == 'traversal':
        name = '../outside'
    elif fault == 'windows_alias':
        name = 'baseline/con.txt'
    members = {name: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}
    if fault == 'length':
        members[name]['bytes'] += 1
    elif fault == 'hash':
        members[name]['sha256'] = '0'*64
    schema = 'wrong' if fault == 'schema' else release.SCHEMA
    with zipfile.ZipFile(path, 'x') as archive:
        if fault == 'symlink':
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (0o120777 << 16)
            archive.writestr(info, raw)
        else:
            archive.writestr(name, raw)
        if fault == 'case_alias':
            archive.writestr(name.upper(), raw)
            members[name.upper()] = members[name]
        if fault == 'extra':
            archive.writestr('unrelated.bin', b'unrelated')
        if fault != 'missing_manifest':
            archive.writestr(release.MANIFEST, json.dumps({'schema': schema, 'members': members}))
    return raw


def test_streaming_crc_lengths_hashes_inventory_only_not_scientific_certificate(tmp_path):
    path = tmp_path/'fixture.zip'
    tiny_archive(path)
    report = release.verify_archive(path)
    assert report['members'] == 1
    assert report['sha256'] == release.sha(path)
    assert 'artifact_packaged_and_replayed' not in report
    assert 'status' not in report


@pytest.mark.parametrize('fault', ['traversal', 'windows_alias', 'symlink', 'case_alias',
    'length', 'hash', 'schema', 'extra', 'missing_manifest'])
def test_unsafe_or_incomplete_archive_refused(tmp_path, fault):
    path = tmp_path/'fixture.zip'
    tiny_archive(path, fault)
    with pytest.raises(ValueError):
        release.verify_archive(path)


def test_honestly_hashed_archive_cannot_substitute_toy_for_full_native_evidence(tmp_path):
    path = tmp_path/'toy.zip'
    tiny_archive(path)
    # Integrity is valid but no ALL1280 original report/data/audit exists.
    with pytest.raises((FileNotFoundError, ValueError)):
        release.replay_archive(path, tmp_path/'replay')
    assert not (tmp_path/'replay/summary.json').exists()


def test_package_requires_complete_baseline_audit_before_creating_archive(tmp_path):
    artifact, replay = tmp_path/'archive.zip', tmp_path/'replay'
    with pytest.raises((FileNotFoundError, ValueError)):
        release.package(tmp_path/'incomplete_baseline', tmp_path/'incomplete_audit', artifact, replay)
    assert not artifact.exists() and not replay.exists()


@pytest.mark.parametrize('existing', ['archive', 'replay', 'parts'])
def test_existing_scientific_evidence_not_overwritten(tmp_path, existing):
    artifact, replay, parts = tmp_path/'archive.zip', tmp_path/'replay', tmp_path/'archive.parts.json'
    target = {'archive': artifact, 'replay': replay, 'parts': parts}[existing]
    target.write_bytes(b'user-evidence')
    with pytest.raises(ValueError, match='Exclusive'):
        release.package(tmp_path/'baseline', tmp_path/'audit', artifact, replay, parts)
    assert target.read_bytes() == b'user-evidence'


@pytest.mark.parametrize('fault', ['truncated', 'active_flag', 'changed_timing_exclusion', 'changed_protocol'])
def test_terminal_like_audit_cannot_forge_complete_or_disabled_scope(tmp_path, monkeypatch, fault):
    baseline = {'protocol': {'fixture': True}, 'original8Level_equal_macro': {'fixture': 0.}}
    hashes = {'summary.json': 'fixture'}
    monkeypatch.setattr(release, 'validate_saved', lambda *_: (baseline, [], [], {}, hashes))
    report = {'status': release.STATUS, 'protocol': baseline['protocol'], 'baseline_summary_sha256': 'fixture',
        'original8Level_equal_macro': baseline['original8Level_equal_macro'], 'modes': {'off': {}, 'refusal': {}},
        'excluded_nondeterministic_step_fields': sorted(release.STEP_TIMING),
        'excluded_nondeterministic_episode_fields': sorted(release.EPISODE_TIMING),
        **{key: False for key in release.FALSE_FLAGS}}
    if fault == 'truncated':
        report['modes'].pop('refusal')
    elif fault == 'active_flag':
        report['active_model_faults_exercised'] = True
    elif fault == 'changed_timing_exclusion':
        report['excluded_nondeterministic_step_fields'].append('safety_solver_success')
    else:
        report['protocol'] = {'fake': True}
    release.write_json(tmp_path/'summary.json', report)
    with pytest.raises(ValueError, match='scope differs'):
        release.audit_inventory(tmp_path/'baseline', tmp_path, tmp_path/'capsule.zip')
