"""Transport/orchestration fixtures are NOT real training/release evidence."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import package_ranking_release as release


def fixture_archive(path, members, rows=None):
    manifest = {name: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
                for name, raw in members.items()}
    if rows is not None:
        manifest = rows
    with zipfile.ZipFile(path, 'x') as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        archive.writestr(release.MANIFEST, json.dumps({'schema': release.SCHEMA, 'members': manifest}))


@pytest.mark.parametrize('name', ['../secret', '/root', 'a/../b', 'a//b', './a', 'C:/data',
    'a\\b', 'a\x00b', 'a/CON.txt', 'nul', 'a/LPT9.npz', 'a.', 'a ', 'a:b', 'a?b'])
def test_unsafe_paths_refused(name):
    with pytest.raises(ValueError):
        release.safe_name(name)


def test_integrity_and_exclusive_safe_extraction(tmp_path):
    archive = tmp_path / 'fixture.zip'
    fixture_archive(archive, {'data/calls/one.json': b'{}', 'primary/final.pt': b'fixture-not-a-model'})
    result = release.verify_archive(archive)
    assert result['members'] == 2
    assert result['sha256'] == release.file_digest(archive)
    output = tmp_path / 'extracted'
    assert release.extract_archive(archive, output) == result
    assert (output / 'primary/final.pt').read_bytes() == b'fixture-not-a-model'
    with pytest.raises(FileExistsError):
        release.extract_archive(archive, output)


@pytest.mark.parametrize('rows', [
    {}, {'data/one': {'bytes': 1, 'sha256': '0' * 64}},
    {'data/one': {'bytes': 99, 'sha256': hashlib.sha256(b'x').hexdigest()}}])
def test_missing_or_forged_member_digest_refused(tmp_path, rows):
    archive = tmp_path / 'fixture.zip'
    fixture_archive(archive, {'data/one': b'x'}, rows)
    with pytest.raises(ValueError):
        release.verify_archive(archive)


def test_case_alias_refused_before_extraction(tmp_path):
    archive = tmp_path / 'fixture.zip'
    fixture_archive(archive, {'data/one': b'x', 'DATA/ONE': b'y'})
    with pytest.raises(ValueError, match='case-aliased'):
        release.extract_archive(archive, tmp_path / 'extracted')
    assert not (tmp_path / 'extracted').exists()


def test_symlink_member_refused(tmp_path):
    archive = tmp_path / 'fixture.zip'
    with zipfile.ZipFile(archive, 'x') as out:
        info = zipfile.ZipInfo('link')
        info.create_system = 3
        info.external_attr = 0o120777 << 16
        out.writestr(info, b'target')
        out.writestr(release.MANIFEST, '{}')
    with pytest.raises(ValueError, match='Nonregular'):
        release.verify_archive(archive)


def test_traversal_refused_without_writing(tmp_path):
    archive = tmp_path / 'fixture.zip'
    fixture_archive(archive, {'../outside': b'x'})
    with pytest.raises(ValueError):
        release.extract_archive(archive, tmp_path / 'extracted')
    assert not (tmp_path / 'outside').exists()


def test_npz_payload_comparison_ignores_timestamp_not_data(tmp_path):
    paths = [tmp_path / f'{i}.npz' for i in range(3)]
    for i, path in enumerate(paths):
        with zipfile.ZipFile(path, 'x') as archive:
            info = zipfile.ZipInfo('field.npy', date_time=(2020 + i, 1, 1, 0, 0, 0))
            archive.writestr(info, b'same' if i < 2 else b'changed')
    assert release.npz_payloads(paths[0]) == release.npz_payloads(paths[1])
    assert release.npz_payloads(paths[0]) != release.npz_payloads(paths[2])


def test_missing_complete_data_refuses_before_artifact(tmp_path):
    archive = tmp_path / 'never.zip'
    with pytest.raises(FileNotFoundError):
        release.package(*[tmp_path / n for n in ('data', 'audit', 'primary', 'repeat', 'reload')],
                        archive, tmp_path / 'replay')
    assert not archive.exists()
    assert not (tmp_path / 'replay').exists()


def test_exclusive_outputs_refuse_before_any_inventory(tmp_path, monkeypatch):
    existing = tmp_path / 'exists.zip'
    existing.write_bytes(b'user-data')
    monkeypatch.setattr(release, 'inventory', lambda *a, **k: pytest.fail('Must refuse first'))
    with pytest.raises(ValueError, match='Exclusive'):
        release.package(*[tmp_path / str(i) for i in range(5)], existing, tmp_path / 'replay')
    assert existing.read_bytes() == b'user-data'


def mock_replay_setup(tmp_path, monkeypatch):
    """Only checks ordering/fail-closed, never executes a research evaluator."""
    previous = {'primary_research_eligible': False, 'qualification': {'cv_rank_l2': {'research_eligible': False}}}
    root = tmp_path / 'repo'
    root.mkdir()
    source = root / 'source.py'
    source.write_bytes(b'fixture-source')
    monkeypatch.setattr(release, 'ROOT', root)
    monkeypatch.setattr(release, 'inventory', lambda **roots: (
        {'verification_source/source.py': source}, previous))
    archive = tmp_path / 'fixture.zip'
    fixture_archive(archive, {'verification_source/source.py': source.read_bytes(),
        'release_contract.json': json.dumps(release.release_contract(previous), indent=2).encode()})
    return archive, previous


def test_real_entries_required_in_order_before_certificate(tmp_path, monkeypatch):
    archive, previous = mock_replay_setup(tmp_path, monkeypatch)
    entries = []
    def child(script, arguments, output):
        entries.append(script)
        if script == 'release_ranking_training.py':
            target = Path(arguments[-1])
            target.mkdir()
            release.write_json(target / 'summary.json', previous)
    monkeypatch.setattr(release, 'run_child', child)
    monkeypatch.setattr(release, 'compare_replayed_data', lambda old, new: entries.append('compare_data'))
    result = release.replay_archive(archive, tmp_path / 'replay')
    assert entries == ['audit_ranking_data.py', 'compare_data', 'release_ranking_training.py']
    assert result['primary_research_eligible'] is False
    assert result['enhanced_control_enabled'] is False
    assert result['holdout_used'] is False
    assert (tmp_path / 'replay/summary.json').exists()


@pytest.mark.parametrize('failed_stage', ['data', 'data_compare', 'reload', 'reload_compare'])
def test_replay_failures_never_emit_pass_certificate(tmp_path, monkeypatch, failed_stage):
    archive, previous = mock_replay_setup(tmp_path, monkeypatch)
    entries = []
    def child(script, arguments, output):
        entries.append(script)
        if (script == 'audit_ranking_data.py' and failed_stage == 'data') or (
            script == 'release_ranking_training.py' and failed_stage == 'reload'):
            raise subprocess.CalledProcessError(1, script)
        if script == 'release_ranking_training.py':
            target = Path(arguments[-1])
            target.mkdir()
            release.write_json(target / 'summary.json', previous if failed_stage != 'reload_compare' else {})
    def compare(old, new):
        if failed_stage == 'data_compare':
            raise ValueError('semantic mismatch')
    monkeypatch.setattr(release, 'run_child', child)
    monkeypatch.setattr(release, 'compare_replayed_data', compare)
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        release.replay_archive(archive, tmp_path / 'replay')
    assert not (tmp_path / 'replay/summary.json').exists()
    if failed_stage in ('data', 'data_compare'):
        assert entries == ['audit_ranking_data.py']


def test_rehashed_extra_member_refused_before_child(tmp_path, monkeypatch):
    archive, previous = mock_replay_setup(tmp_path, monkeypatch)
    bad = tmp_path / 'extra.zip'
    fixture_archive(bad, {'verification_source/source.py': b'fixture-source',
        'release_contract.json': json.dumps(release.release_contract(previous), indent=2).encode(),
        'unrelated_user_data': b'never-include'})
    monkeypatch.setattr(release, 'run_child', lambda *a: pytest.fail('Inventory must refuse first'))
    with pytest.raises(ValueError, match='omitted/added'):
        release.replay_archive(bad, tmp_path / 'replay')
    assert not (tmp_path / 'replay/summary.json').exists()


def test_matching_checkout_required_before_child(tmp_path, monkeypatch):
    archive, _ = mock_replay_setup(tmp_path, monkeypatch)
    (release.ROOT / 'source.py').write_bytes(b'changed-checkout')
    monkeypatch.setattr(release, 'run_child', lambda *a: pytest.fail('Source must refuse first'))
    with pytest.raises(ValueError, match='Matching Git checkout'):
        release.replay_archive(archive, tmp_path / 'replay')


def test_child_failure_retains_logs_and_uses_check(tmp_path, monkeypatch):
    def fail(command, **kwargs):
        assert command[0] == sys.executable
        assert kwargs['check'] is True and kwargs['cwd'] == release.ROOT
        kwargs['stderr'].write('fixture-failure')
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(release.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        release.run_child('audit_ranking_data.py', ['--data', 'fixture'], tmp_path)
    assert (tmp_path / 'audit_ranking_data.py.stderr.log').read_text() == 'fixture-failure'


def test_extracted_mutation_during_replay_refuses_certificate(tmp_path, monkeypatch):
    archive, previous = mock_replay_setup(tmp_path, monkeypatch)
    def child(script, arguments, output):
        if script == 'release_ranking_training.py':
            target = Path(arguments[-1])
            target.mkdir()
            release.write_json(target / 'summary.json', previous)
            (output / 'extracted/release_contract.json').write_bytes(b'mutated')
    monkeypatch.setattr(release, 'run_child', child)
    monkeypatch.setattr(release, 'compare_replayed_data', lambda *a: None)
    with pytest.raises(ValueError, match='Extracted evidence changed'):
        release.replay_archive(archive, tmp_path / 'replay')
    assert not (tmp_path / 'replay/summary.json').exists()


@pytest.mark.parametrize('value', [True, 0, None, 'false'])
def test_default_off_must_be_exact_false(value):
    with pytest.raises(ValueError):
        release.require_false({'enabled': value}, 'enabled')


@pytest.fixture
def complete_inventory_fixture(tmp_path, monkeypatch):
    """Synthetic64/9/2 inventory only, no optimizer or real qualification."""
    repo = tmp_path / 'repo'
    here = repo / 'experiments/cwm_v28'
    here.mkdir(parents=True)
    monkeypatch.setattr(release, 'ROOT', repo)
    monkeypatch.setattr(release, 'HERE', here)
    def raw(path, value=b'fixture'):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    def js(path, value):
        raw(path, json.dumps(value, indent=2).encode())
    for name in release.DEPENDENCIES:
        raw(repo / name)
    protocol = {'baseline_capsule_sha256': release.file_digest(repo / release.DEPENDENCIES[0]),
                'proposal_archive_sha256': release.file_digest(repo / release.DEPENDENCIES[1])}
    training = {'models': ['cv_motion_only', 'cv_l2', 'cv_rank_l2'], 'training': {'seeds': [994101, 994102, 994103]}}
    js(here / 'data_protocol.json', protocol)
    js(here / 'training_protocol.json', training)
    for name in ('package_ranking_release.py', 'release_ranking_training.py', 'audit_ranking_data.py'):
        raw(here / name)
    raw(repo / 'experiments/cwm_v21/artifact_transport.py')
    source_name = 'experiments/cwm_v28/used.py'
    raw(repo / source_name)
    hashes = {source_name: release.file_digest(repo / source_name)}
    roots = {n: tmp_path / n for n in ('data', 'audit', 'primary', 'repeated', 'reload')}
    for stage, path in roots.items():
        path.mkdir()
        if stage != 'reload':
            raw(path / 'source' / source_name)
    episodes = []
    records = []
    for episode in range(994010, 994074):
        for kind in ('observed', 'plain'):
            raw(roots['data'] / kind / f'{episode}.npz')
            raw(roots['data'] / kind / f'{episode}.commands.npz')
        for suffix in ('.npz', '.commands.npz', '.public252.npz'):
            raw(roots['audit'] / 'trajectories' / (str(episode) + suffix))
        episodes.append({'episode_index': episode, 'trajectory_commands_plans_equal': True,
            'observed_sha256': release.file_digest(roots['data'] / 'observed' / f'{episode}.npz'),
            'plain_sha256': release.file_digest(roots['data'] / 'plain' / f'{episode}.npz')})
        context, arrays = f'calls/{episode}.json', f'calls/{episode}.npz'
        raw(roots['data'] / context)
        raw(roots['data'] / arrays)
        records.append({'context_path': context, 'arrays_path': arrays})
    js(roots['data'] / 'episodes.json', episodes)
    js(roots['data'] / 'records.json', records)
    raw(roots['data'] / 'scenes.jsonl')
    js(roots['data'] / 'preregistration.json', {'fixture': True})
    d = {'status': 'fresh_bounded_sequential_data_collected_pending_independent_audit',
         'protocol': protocol, 'data_gate_passed': True, 'training_authorized_by_this_report': False,
         'enhanced_control_enabled': False, 'new_model_trained': False, 'holdout_used': False,
         'episodes': episodes, 'records': 64, 'calls': 64, 'source_hashes': hashes}
    js(roots['data'] / 'summary.json', d)
    names = ['summary.json', 'preregistration.json', 'scenes.jsonl', 'records.json', 'episodes.json', 'source/' + source_name]
    names.extend(n for record in records for n in (record['context_path'], record['arrays_path']))
    js(roots['audit'] / 'audited_data_manifest.json', {n: release.file_digest(roots['data'] / n) for n in names})
    js(roots['audit'] / 'checked_calls.json', [])
    js(roots['audit'] / 'full_label_cost_checks.json', [])
    a = {'status': 'independent_fresh_sequential_public_branch_cost_audit_passed', 'protocol': protocol,
         'data_summary_sha256': release.file_digest(roots['data'] / 'summary.json'),
         'data_gate_passed': True, 'training_authorized_by_this_report': True, 'enhanced_control_enabled': False,
         'new_model_trained': False, 'holdout_used': False, 'checked_calls': 64, 'supported_calls': 64,
         'source_hashes': hashes}
    for name, key in (('audited_data_manifest.json', 'audited_data_manifest_sha256'),
                      ('checked_calls.json', 'checked_calls_sha256'), ('full_label_cost_checks.json', 'full_label_cost_checks_sha256')):
        a[key] = release.file_digest(roots['audit'] / name)
    js(roots['audit'] / 'summary.json', a)
    models = []
    for seed in training['training']['seeds']:
        for model in training['models']:
            identifier = f'{model}_seed{seed}'
            models.append({'configuration': model, 'seed': seed, 'checkpoint_sha256': hashlib.sha256(b'fixture').hexdigest()})
            for stage in ('primary', 'repeated'):
                for suffix in ('.pt', '_history.json', '_decisions.json', '_library_contributions.json',
                               '_train_predictions.npz', '_development_validation_predictions.npz'):
                    raw(roots[stage] / (identifier + suffix))
        for stage in ('primary', 'repeated'):
            for tail in ('motion_stage.pt', 'motion_stage_history.json', 'deployed_score_audit.json', 'ranking_cache.npz'):
                raw(roots[stage] / f'cv_seed{seed}_{tail}')
    p = {'status': 'offline_fixed_fresh_ranking_training_finished_pending_two_run_release', 'protocol': training,
         'source_hashes': hashes, 'models': models, 'data_summary_sha256': a['data_summary_sha256'],
         'data_audit_summary_sha256': release.file_digest(roots['audit'] / 'summary.json'),
         'qualification': {'cv_rank_l2': {'research_eligible': False}}, 'primary_research_eligible': False,
         'enhanced_control_enabled': False, 'holdout_used': False, 'prior_gate_overridden': False,
         'common_motion_in_response_optimizer': False, 'two_complete_runs_verified': False}
    for stage in ('primary', 'repeated'):
        js(roots[stage] / 'summary.json', p)
        for name in ('normalization.npz', 'population_weights.json', 'initial_gru_score_audit.json', 'initial_cv_score_audit.json'):
            raw(roots[stage] / name)
    v = {'status': 'two_complete_fresh_ranking_runs_reloaded_original_costs_verified', 'models_per_run': 9,
         'complete_runs': 2, 'all_final_weights_optimizer_rng_histories_equal': True, 'all_prediction_bytes_reloaded': True,
         'train_only_normalization_verified': True, 'actual_geometry_shared_cost_contributions_recomputed': True,
         'data_summary_sha256': a['data_summary_sha256'], 'data_audit_summary_sha256': p['data_audit_summary_sha256'],
         'primary_summary_sha256': release.file_digest(roots['primary'] / 'summary.json'),
         'repeated_summary_sha256': release.file_digest(roots['repeated'] / 'summary.json'),
         'qualification': p['qualification'], 'primary_research_eligible': False,
         'enhanced_control_enabled': False, 'holdout_used': False, 'artifact_packaged_and_replayed': False}
    js(roots['reload'] / 'summary.json', v)
    return roots


def test_curated_inventory_includes_complete_evidence_not_worktrees_or_user_extras(complete_inventory_fixture):
    roots = complete_inventory_fixture
    for name in ('restored/private', '__pycache__/one.pyc', 'unrelated_user_file.zip'):
        path = roots['primary'] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'user-owned')
    members, report = release.inventory(**roots)
    assert report['primary_research_eligible'] is False
    assert len([n for n in members if n.startswith('data/observed/')]) == 128
    assert len([n for n in members if n.startswith('data/plain/')]) == 128
    assert len([n for n in members if n.startswith('audit/trajectories/')]) == 192
    assert len([n for n in members if n.startswith(('primary/', 'repeated/')) and n.endswith('.pt')]) == 24
    assert not any('restored' in n or '__pycache__' in n or 'unrelated_user' in n for n in members)
    assert all(path.is_file() for path in members.values())


@pytest.mark.parametrize('name', [
    'data/observed/994010.commands.npz', 'audit/trajectories/994073.public252.npz',
    'primary/cv_rank_l2_seed994102.pt', 'repeated/cv_seed994103_motion_stage.pt',
    'primary/initial_gru_score_audit.json', 'repeated/cv_seed994102_ranking_cache.npz'])
def test_missing_required_evidence_refuses(complete_inventory_fixture, name):
    stage, relative = name.split('/', 1)
    (complete_inventory_fixture[stage] / relative).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        release.inventory(**complete_inventory_fixture)


def test_modified_audited_arrays_refuses(complete_inventory_fixture):
    roots = complete_inventory_fixture
    (roots['data'] / 'calls/994010.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='Audited data changed'):
        release.inventory(**roots)


@pytest.mark.parametrize('change', ['incomplete', 'enabled', 'holdout', 'changed_qualification', 'missing_model'])
def test_incomplete_or_promoted_runs_refuse(complete_inventory_fixture, change):
    roots = complete_inventory_fixture
    report = release.read_json(roots['primary'] / 'summary.json')
    if change == 'incomplete':
        report['status'] = 'in_progress'
    elif change == 'enabled':
        report['enhanced_control_enabled'] = True
    elif change == 'holdout':
        report['holdout_used'] = True
    elif change == 'changed_qualification':
        report['qualification']['cv_rank_l2']['research_eligible'] = True
    else:
        report['models'] = report['models'][:-1]
    # Make coherent top-level hashes; semantic consistency must still refuse.
    for stage in ('primary', 'repeated'):
        (roots[stage] / 'summary.json').write_text(json.dumps(report))
    v = release.read_json(roots['reload'] / 'summary.json')
    for stage in ('primary', 'repeated'):
        v[stage + '_summary_sha256'] = release.file_digest(roots[stage] / 'summary.json')
    (roots['reload'] / 'summary.json').write_text(json.dumps(v))
    with pytest.raises(ValueError):
        release.inventory(**roots)


def test_complete_inventory_archive_and_lossless_transport_roundtrip(complete_inventory_fixture, tmp_path, monkeypatch):
    """Real ZIP/inventory/chunk roundtrip; scientific children are mocked."""
    roots = complete_inventory_fixture
    previous = release.read_json(roots['reload'] / 'summary.json')
    entries = []
    def child(script, arguments, output):
        entries.append(script)
        if script == 'release_ranking_training.py':
            target = Path(arguments[-1])
            target.mkdir()
            release.write_json(target / 'summary.json', previous)
    monkeypatch.setattr(release, 'run_child', child)
    monkeypatch.setattr(release, 'compare_replayed_data', lambda *a: None)
    archive, replay, parts = tmp_path / 'complete_fixture.zip', tmp_path / 'replay', tmp_path / 'fixture.parts.json'
    report = release.package(**roots, archive_path=archive, replay_output=replay, parts_output=parts)
    assert entries == ['audit_ranking_data.py', 'release_ranking_training.py']
    assert report['primary_research_eligible'] is False
    joined = release.resolve_artifact(parts, tmp_path / 'joined')
    assert release.file_digest(joined) == release.file_digest(archive)
    extracted_roots = {n: replay / 'extracted' / n for n in roots}
    assert set(release.inventory(**roots)[0]) == set(release.inventory(**extracted_roots)[0])
    with zipfile.ZipFile(archive) as stored:
        assert stored.read('primary/cv_rank_l2_seed994101.pt') == b'fixture'
        assert stored.read('dependencies/' + release.DEPENDENCIES[0]) == b'fixture'
    transport = release.read_json(replay / 'transport_verified.json')
    assert transport['transport_only_not_second_semantic_replay'] is True
    assert transport['semantic_replay_certificate_sha256'] == release.file_digest(replay / 'summary.json')
