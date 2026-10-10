import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidate_cardinality_release import archive_integrity, audit_records, digest


def test_duplicate_archive_rejected(tmp_path):
    p = tmp_path / 'duplicate.zip'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('ARTIFACT_MANIFEST.json', '{}')
        with pytest.warns(UserWarning):
            z.writestr('ARTIFACT_MANIFEST.json', '{}')
    with pytest.raises(ValueError, match='Duplicate archive'):
        archive_integrity(p)


def test_incomplete_archive_rejected(tmp_path):
    p = tmp_path / 'incomplete.zip'
    with zipfile.ZipFile(p, 'w') as z:
        z.writestr('ARTIFACT_MANIFEST.json', '{}')
    with pytest.raises(ValueError, match='Incomplete or unexpected'):
        archive_integrity(p)


def test_incomplete_run_never_passes_saved_record_audit():
    protocol = json.loads((Path(__file__).resolve().parent / 'diagnostic_protocol.json').read_text())
    raw = b'[]'
    summary = {'status': 'posthoc_public_candidate_cardinality_finished_not_promoted', 'protocol': protocol,
               'data_archive_sha256': protocol['data_archive_sha256'], 'records_sha256': digest(raw), 'calls': 0}
    def read(name):
        return raw if name.endswith('records.json') else json.dumps(summary).encode()
    with pytest.raises(ValueError, match='Full fixed original-only'):
        audit_records(read)
