"""Integrity/parser fixtures only, not qualified native policies or replay."""
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import manifest_reader as reader


def record():
    return {'bytes': 2, 'sha256': 'a'*64}


@pytest.mark.parametrize('chunk_size', [1, 2, 3, 7, 64, 65536])
def test_every_sorted_member_unicode_and_chunk_boundaries(tmp_path, chunk_size):
    expected = {f'{i:04}_公开.txt': record() for i in range(512)}
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(expected, indent=2), encoding='utf8')
    assert dict(reader.iter_manifest(path, chunk_size=chunk_size)) == expected


@pytest.mark.parametrize('raw', [
    '', '[ ]', '{', '{"a":', '{"a":{}}', '{"a":null}', '{"a":false}',
    '{"a":{"bytes":true,"sha256":"'+'a'*64+'"}}',
    '{"a":{"bytes":-1,"sha256":"'+'a'*64+'"}}',
    '{"a":{"bytes":2.0,"sha256":"'+'a'*64+'"}}',
    '{"a":{"bytes":2,"sha256":"'+'A'*64+'"}}',
    '{"a":{"bytes":2,"sha256":"'+'a'*63+'"}}',
    '{"a":{"bytes":2,"bytes":2,"sha256":"'+'a'*64+'"}}',
    '{"a":{"bytes":NaN,"sha256":"'+'a'*64+'"}}',
    '{"a":{"bytes":Infinity,"sha256":"'+'a'*64+'"}}',
    '{} {}', '{} garbage', '{"../a":'+json.dumps(record())+'}',
    '{"a":'+json.dumps(record())+',}',
    '{"a":'+json.dumps(record())+',"a":'+json.dumps(record())+'}',
    '{"b":'+json.dumps(record())+',"a":'+json.dumps(record())+'}',
])
def test_malformed_noncanonical_unsorted_duplicate_or_nonfinite_json_refused(raw, tmp_path):
    path = tmp_path/'manifest.json'; path.write_text(raw, encoding='utf8')
    with pytest.raises(ValueError): list(reader.iter_manifest(path, chunk_size=3))


def test_empty_whitespace_and_no_aggregate_size_limit(tmp_path):
    path = tmp_path/'manifest.json'
    path.write_text(' \r\n{} \t\n', encoding='utf8')
    assert list(reader.iter_manifest(path, chunk_size=1)) == []
    expected = {f'{i:06}': record() for i in range(1024)}
    path.write_text(json.dumps(expected), encoding='utf8')
    assert len(list(reader.iter_manifest(path, chunk_size=17, max_token=128))) == 1024


def test_individual_token_limit_and_buffer_parameters(tmp_path):
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps({'x'*512: record()}), encoding='utf8')
    with pytest.raises(ValueError): list(reader.iter_manifest(path, chunk_size=16, max_token=128))
    for chunk, limit in [(0, 128), (129, 128), (True, 128), (16, False)]:
        with pytest.raises(ValueError): list(reader.iter_manifest(path, chunk_size=chunk, max_token=limit))


def members(tmp_path):
    expected = {}
    for name in ('A.txt', 'B_公开.txt'):
        path = tmp_path/name; path.write_bytes(name.encode('utf8'))
        expected[name] = {'bytes': path.stat().st_size, 'sha256': reader.sha(path)}
    manifest = tmp_path/'evidence_manifest.json'
    manifest.write_text(json.dumps(expected, indent=2), encoding='utf8')
    return manifest, expected


def test_verification_reads_all_files_and_requires_complete_expected_membership(tmp_path):
    manifest, expected = members(tmp_path)
    assert reader.verify_members(tmp_path, manifest, iter(expected.items())) == 2
    with pytest.raises(ValueError): reader.verify_members(tmp_path, manifest, iter(list(expected.items())[:1]))
    extra = [*expected.items(), ('extra', record())]
    with pytest.raises(ValueError): reader.verify_members(tmp_path, manifest, iter(extra))
    with pytest.raises(ValueError): reader.verify_members(tmp_path, manifest, iter(reversed(list(expected.items()))))
    with pytest.raises(ValueError): reader.verify_members(tmp_path, manifest, [['A.txt', expected['A.txt']]])


@pytest.mark.parametrize('kind', ['length', 'same_length_bytes', 'missing', 'expected', 'manifest_location'])
def test_changed_actual_files_or_expectations_refused(kind, tmp_path):
    manifest, expected = members(tmp_path)
    if kind == 'length': (tmp_path/'A.txt').write_bytes(b'longer')
    elif kind == 'same_length_bytes': (tmp_path/'A.txt').write_bytes(b'other')
    elif kind == 'missing': (tmp_path/'A.txt').unlink()
    elif kind == 'expected': expected['A.txt']['sha256'] = '0'*64
    else:
        alternate = tmp_path/'alternate.json'; alternate.write_bytes(manifest.read_bytes()); manifest = alternate
    with pytest.raises(ValueError): reader.verify_members(tmp_path, manifest, iter(expected.items()))


def test_inventory_mutation_during_verification_refused(tmp_path):
    manifest, expected = members(tmp_path)
    def changed_expected():
        for index, item in enumerate(expected.items()):
            if index == 1: manifest.write_text(manifest.read_text()+' ', encoding='utf8')
            yield item
    with pytest.raises(ValueError, match='changed during'):
        reader.verify_members(tmp_path, manifest, changed_expected())


def test_exact_interoperation_with_actual_v45_writer(tmp_path):
    sys.path.insert(0, str(HERE.parent/'cwm_v45'))
    from inventory_stream import DiskInventory
    with_ledger = DiskInventory(tmp_path)
    try:
        for name in ('A.txt', 'B_公开.txt'):
            (tmp_path/name).write_bytes(name.encode('utf8')); with_ledger.add(name)
        with_ledger.seal(); with_ledger.write(tmp_path/'evidence_manifest.json')
        assert reader.verify_members(tmp_path, tmp_path/'evidence_manifest.json', with_ledger.items()) == 2
    finally:
        with_ledger.close()
