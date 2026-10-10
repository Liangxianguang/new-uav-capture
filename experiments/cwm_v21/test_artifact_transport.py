import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from artifact_transport import split_archive,resolve_artifact


def sample(tmp_path):
    raw = bytes(range(256))*3
    archive = tmp_path/'source.zip'
    archive.write_bytes(raw)
    manifest = tmp_path/'published/archive.parts.json'
    split_archive(archive,manifest,100)
    return raw,manifest


def test_lossless_join_and_cache_cannot_change_archive(tmp_path):
    raw,manifest = sample(tmp_path)
    result = resolve_artifact(manifest,tmp_path/'cache')
    assert result.read_bytes() == raw
    assert resolve_artifact(manifest,tmp_path/'cache') == result
    result.write_bytes(b'bad cached artifact')
    with pytest.raises(ValueError,match='Joined'):
        resolve_artifact(manifest,tmp_path/'cache')


@pytest.mark.parametrize('kind',['corrupt','missing','duplicate','path_escape','reordered'])
def test_corrupt_or_unsafe_parts_rejected(tmp_path,kind):
    raw,manifest = sample(tmp_path)
    report = json.loads(manifest.read_text())
    if kind == 'corrupt':
        (manifest.parent/report['parts'][0]['name']).write_bytes(b'0'*100)
    elif kind == 'missing':
        report['parts'] = report['parts'][1:]
    elif kind == 'duplicate':
        report['parts'][1] = report['parts'][0]
    elif kind == 'path_escape':
        report['parts'][0]['name'] = '../source.zip'
    else:
        report['parts'] = list(reversed(report['parts']))
    manifest.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        resolve_artifact(manifest,tmp_path/'cache')
