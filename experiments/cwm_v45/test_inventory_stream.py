"""Synthetic inventory/entry plumbing only; no trained models or native rollout."""
import json
import sys
from contextlib import closing
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import inventory_stream as stream
import paired_prefix_streamed as entry


def fixture(root):
    names = ['preregistration.json', 'prefix_certificate.json', 'legacy_certificate.json',
        'scenes.json', 'benchmark_preload.json', 'source/experiments/synthetic.py']
    rows = {}
    for cohort, identifier in [('original', 1), ('fresh_geometry', 7)]:
        folder = f'{cohort}/{identifier}/synthetic_method'
        names.extend([f'{cohort}/episodes.jsonl', f'{cohort}/steps.jsonl',
            folder+'/trajectory.npz', folder+'/commands.npz', folder+'/calls/a.npz', folder+'/calls/a.json'])
        rows[cohort] = [{'trajectory_path': folder+'/trajectory.npz', 'commands_path': folder+'/commands.npz',
            'events_path': folder+'/events.json'}]
    for name in names:
        file = root/name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b'SYNTHETIC_NON_NATIVE_EVIDENCE')
    for cohort, population in rows.items():
        row = population[0]
        folder = Path(row['events_path']).parent
        event = {field+'_path': 'calls/a'+suffix for field, suffix in [('arrays', '.npz'), ('context', '.json')]}
        event.update({field+'_sha256': stream.sha(root/folder/event[field+'_path']) for field in ('arrays', 'context')})
        (root/row['events_path']).write_text(json.dumps([event, {'status': 'synthetic_missing_context'}]), encoding='utf8')
        for field in ('trajectory', 'commands', 'events'):
            row[field+'_sha256'] = stream.sha(root/row[field+'_path'])
    return rows, {'experiments/synthetic.py': stream.sha(root/names[5])}


def test_all_file_membership_and_manifest_BYTES_match_unchanged_native_builder(tmp_path):
    rows, sources = fixture(tmp_path)
    expected = entry.ORIGINAL_INVENTORY(tmp_path, rows, sources)
    with closing(stream.build_inventory(tmp_path, rows, sources)) as ledger:
        assert dict(ledger.items()) == expected
        assert ledger.count == len(expected)
        assert ledger.connection.execute('PRAGMA cache_size').fetchone() == (-4096,)
        assert ledger.connection.execute('PRAGMA temp_store').fetchone() == (1,)
        ledger.write(tmp_path/'evidence_manifest.json')
    entry.ORIGINAL_WRITER(tmp_path/'expected_manifest.json', expected)
    assert (tmp_path/'evidence_manifest.json').read_bytes() == (tmp_path/'expected_manifest.json').read_bytes()
    assert 'evidence_manifest_index.sqlite3' not in expected


@pytest.mark.parametrize('kind', ['events', 'arrays', 'context', 'trajectory', 'commands', 'missing', 'path'])
def test_tampered_missing_or_unsafe_full_evidence_never_produces_manifest(kind, tmp_path):
    rows, sources = fixture(tmp_path)
    row = rows['original'][0]
    if kind in ('events', 'trajectory', 'commands'):
        (tmp_path/row[kind+'_path']).write_bytes(b'CHANGED')
    elif kind in ('arrays', 'context'):
        name = 'a.npz' if kind == 'arrays' else 'a.json'
        (tmp_path/Path(row['events_path']).parent/'calls'/name).write_bytes(b'CHANGED')
    elif kind == 'missing':
        row['commands_path'] = 'absent.npz'
    else:
        file = tmp_path/row['events_path']
        events = json.loads(file.read_bytes())
        events[0]['arrays_path'] = '../outside.npz'
        file.write_text(json.dumps(events), encoding='utf8')
        row['events_sha256'] = stream.sha(file)
    with pytest.raises((ValueError, FileNotFoundError)):
        stream.build_inventory(tmp_path, rows, sources)
    assert not (tmp_path/'evidence_manifest.json').exists()


@pytest.mark.parametrize('name', ['../x', '/x', 'a//b', 'a/../b', 'a\\b', 'x:y', 'con', 'a.', 'a?'])
def test_unsafe_names_refused_before_regular_file_access(name, tmp_path):
    with closing(stream.DiskInventory(tmp_path)) as ledger:
        with pytest.raises(ValueError):
            ledger.add(name)
        assert ledger.count == 0


def test_duplicate_exact_path_dedup_alias_and_later_mutation_refused(tmp_path):
    (tmp_path/'A.txt').write_bytes(b'original')
    with closing(stream.DiskInventory(tmp_path)) as ledger:
        ledger.add('A.txt'); ledger.add('A.txt')
        assert ledger.count == 1
        if not (tmp_path/'a.txt').exists():
            (tmp_path/'a.txt').write_bytes(b'original')
        with pytest.raises(ValueError): ledger.add('a.txt')
        (tmp_path/'A.txt').write_bytes(b'changed')
        with pytest.raises(ValueError): ledger.add('A.txt')


def test_sealed_closed_existing_or_wrong_outputs_refused(tmp_path):
    with closing(stream.DiskInventory(tmp_path)) as ledger:
        with pytest.raises(ValueError): ledger.write(tmp_path/'evidence_manifest.json')
        ledger.seal()
        with pytest.raises(ValueError): ledger.add('anything')
        with pytest.raises(ValueError): ledger.write(tmp_path/'other.json')
        ledger.write(tmp_path/'evidence_manifest.json')
        assert (tmp_path/'evidence_manifest.json').read_bytes() == b'{}'
        with pytest.raises(FileExistsError): ledger.write(tmp_path/'evidence_manifest.json')
    with pytest.raises(ValueError): list(ledger.items())
    with pytest.raises(FileExistsError): stream.DiskInventory(tmp_path)


def test_512_member_ordering_unicode_and_canonical_serialization_without_aggregate_names(tmp_path):
    with closing(stream.DiskInventory(tmp_path)) as ledger:
        for index in range(512):
            name = f'part_{511-index:04}_公开.txt'
            (tmp_path/name).write_bytes(str(index).encode('ascii'))
            ledger.add(name)
        ledger.seal()
        assert ledger.count == 512
        ledger.write(tmp_path/'evidence_manifest.json')
    actual = json.loads((tmp_path/'evidence_manifest.json').read_bytes())
    assert list(actual) == sorted(actual) and len(actual) == 512
    entry.ORIGINAL_WRITER(tmp_path/'expected_manifest.json', actual)
    assert (tmp_path/'evidence_manifest.json').read_bytes() == (tmp_path/'expected_manifest.json').read_bytes()


def test_actual_native_callbacks_restored_on_success_and_preserve_regular_diagnostics(tmp_path):
    rows, sources = fixture(tmp_path)
    before = (entry.native.save_evidence_manifest, entry.native.write_json)
    with entry.post_rollout_inventory():
        entry.native.write_json(tmp_path/'diagnostic.json', {'value': float('inf')})
        ledger = entry.native.save_evidence_manifest(tmp_path, rows, sources)
        entry.native.write_json(tmp_path/'evidence_manifest.json', ledger)
        assert ledger.connection is not None
        with pytest.raises(ValueError): entry.native.save_evidence_manifest(tmp_path, rows, sources)
        with pytest.raises(ValueError):
            with entry.post_rollout_inventory(): pass
    assert (entry.native.save_evidence_manifest, entry.native.write_json) == before
    assert ledger.connection is None
    assert json.loads((tmp_path/'diagnostic.json').read_bytes()) == {'value': {'native_nonfinite_float': 'positive_infinity'}}


def test_actual_native_callbacks_restore_and_close_on_exception(tmp_path):
    rows, sources = fixture(tmp_path)
    before = (entry.native.save_evidence_manifest, entry.native.write_json)
    with pytest.raises(RuntimeError):
        with entry.post_rollout_inventory():
            ledger = entry.native.save_evidence_manifest(tmp_path, rows, sources)
            raise RuntimeError('controlled post-rollout failure')
    assert ledger.connection is None
    assert (entry.native.save_evidence_manifest, entry.native.write_json) == before
    assert not (tmp_path/'evidence_manifest.json').exists()


def test_prior_runtime_instrumentation_cannot_be_silently_overridden(monkeypatch):
    monkeypatch.setattr(entry.native, 'save_evidence_manifest', lambda *a: None)
    with pytest.raises(ValueError):
        with entry.post_rollout_inventory(): pass


def test_native_runner_remains_authority_for_all_five_unmodified_arguments(tmp_path, monkeypatch):
    calls = []
    values = (object(), object(), object(), object(), tmp_path/'new')
    def fake_run(*args):
        calls.append(args)
        assert entry.native.save_evidence_manifest is not entry.ORIGINAL_INVENTORY
        return 'SYNTHETIC_NOT_NATIVE'
    monkeypatch.setattr(entry.native, 'run', fake_run)
    assert entry.run(*values) == 'SYNTHETIC_NOT_NATIVE'
    assert calls == [values] and entry.native.save_evidence_manifest is entry.ORIGINAL_INVENTORY


def test_failed_primary_actual_entry_still_refuses_before_scenes_or_inventory(tmp_path, monkeypatch):
    from benchmark_loader import BenchmarkScorers
    monkeypatch.setattr(entry.native, 'published_sources', lambda: {'commit': 'NEGATIVE_FIXTURE_NOT_PUBLISHED'})
    monkeypatch.setattr(entry.native, 'load_benchmark', lambda *a, **k: BenchmarkScorers(status='prefix_primary_failed_no_legacy_access'))
    def forbidden(*a, **k): pytest.fail('No native scene/inventory before qualified primary')
    monkeypatch.setattr(entry.native, 'Baseline', forbidden)
    monkeypatch.setattr(entry.native, 'fresh_records', forbidden)
    monkeypatch.setattr(entry, 'build_inventory', forbidden)
    output = tmp_path/'new'
    entry.run(None, None, None, None, output)
    report = json.loads((output/'refusal.json').read_bytes())
    assert not report['research_actions_exercised'] and not report['fresh_scenes_generated']
    assert not (output/'evidence_manifest_index.sqlite3').exists()
    assert not (output/'scenes.json').exists()


def test_streaming_actual_source_included_in_native_publication_capture():
    assert HERE/'inventory_stream.py' in entry.native.used_paths()
    assert HERE/'paired_prefix_streamed.py' in entry.native.used_paths()
