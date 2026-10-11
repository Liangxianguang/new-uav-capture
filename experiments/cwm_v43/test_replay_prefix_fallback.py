"""Native replay wrapper boundaries only; mocked audit is not native evidence."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_prefix_fallback as entry


def proof():
    published = entry.read_json(entry.ROOT/'experiments/cwm_v30/reports/completed_native_original_archive_replayed_20261011.json')
    return copy.deepcopy(published['native_summary'])


@pytest.mark.parametrize('change', ['count', 'timing', 'partial', 'promoted', 'holdout', 'artifact', 'numeric_flag'])
def test_original_native_certificate_full_modes_and_scope_required(change):
    report = proof(); integrity = copy.deepcopy(report['artifact'])
    entry.validate_certificate(report, integrity)
    if change == 'count': report['episodes_each_mode'] = 2
    elif change == 'timing': report['latency_qualified'] = True
    elif change == 'partial': report['modes'] = ['off']
    elif change == 'promoted': report['enhanced_control_enabled'] = True
    elif change == 'holdout': report['holdout_used'] = True
    elif change == 'artifact': integrity['sha256'] = 'a'*64
    else: report['all_full_original_trajectories_plans_cbf_commands_equal'] = 1
    with pytest.raises(ValueError): entry.validate_certificate(report, integrity)


@pytest.mark.parametrize('mode,options', [('research_eval', {}), ('shadow', {}), ('off', {'bundle': object()}),
    ('refusal', {'score': 'cv_prefix_rank_l2'})])
def test_wrapper_never_accepts_active_models_or_options(mode, options):
    boundary = entry.FallbackOnlyEntry.__new__(entry.FallbackOnlyEntry)
    with pytest.raises(ValueError): boundary.configure(mode, **options)


def test_reference_wrong_bytes_refuses_before_inventory_or_simulation(tmp_path, monkeypatch):
    archive = tmp_path/'invalid.zip'; certificate = tmp_path/'invalid.json'
    archive.write_bytes(b'fake'); certificate.write_bytes(b'{}')
    monkeypatch.setattr(entry.original_release, 'verify_archive', lambda *a: pytest.fail('Refuse identity first'))
    with pytest.raises(ValueError): entry.check_reference(archive, certificate, tmp_path/'e')


@pytest.mark.parametrize('failure', [True, False])
def test_reuses_full_auditor_but_restores_old_module_entry_even_on_native_failure(tmp_path, monkeypatch, failure):
    expected = proof(); original_class = entry.audit.SequentialResearchEntry
    monkeypatch.setattr(entry, 'published_sources', lambda: {'commit': 'SYNTHETIC_WRAPPER_ONLY'})
    monkeypatch.setattr(entry, 'check_reference', lambda *a: (expected['artifact'], expected))
    called = []
    def native(reference, output):
        called.append((reference, output))
        assert entry.audit.SequentialResearchEntry is entry.FallbackOnlyEntry
        if failure: raise ValueError('Synthetic native failure')
        output.mkdir(); (output/'summary.json').write_bytes(b'{}')
    monkeypatch.setattr(entry.audit, 'run', native)
    monkeypatch.setattr(entry, 'validate_result', lambda *a: {'source_hashes': {}, 'modes': {}})
    output = tmp_path/'out'
    if failure:
        with pytest.raises(ValueError): entry.run(None, None, tmp_path/'ref', output)
        assert not (output/'summary.json').exists()
    else:
        entry.run(None, None, tmp_path/'ref', output)
        report = entry.read_json(output/'summary.json')
        assert report['active_model_faults_exercised'] is False and report['latency_qualified'] is False
        assert report['artifact_packaged_and_replayed'] is False
    assert entry.audit.SequentialResearchEntry is original_class
    assert called == [(tmp_path/'ref/baseline', output/'native')]


def test_no_native_replay_before_exact_source_publication(tmp_path, monkeypatch):
    monkeypatch.setattr(entry, 'published_sources', lambda: (_ for _ in ()).throw(ValueError('not published')))
    monkeypatch.setattr(entry, 'check_reference', lambda *a: pytest.fail('No reference IO before source preregistration'))
    output = tmp_path/'out'
    with pytest.raises(ValueError): entry.run(None, None, None, output)
    assert not output.exists()


def test_native_result_requires_actual_new_prefix_entry_source_closure(tmp_path, monkeypatch):
    monkeypatch.setattr(entry.original_release, 'audit_inventory', lambda *a: (None, {'source_hashes': {}}, None, None))
    with pytest.raises(ValueError, match='source closure'): entry.validate_result(tmp_path/'ref', tmp_path/'out')
