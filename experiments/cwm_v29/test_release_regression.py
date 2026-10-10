import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import release_fallback_regression as release


def test_exact_array_compare_checks_dtype_and_fields_not_just_values():
    def raw(dtype, name='a'):
        stream = io.BytesIO()
        np.savez_compressed(stream, **{name: np.ones((2, 3), dtype=dtype)})
        return stream.getvalue()
    assert release.same_arrays(raw('float64'), raw('float64'))
    assert not release.same_arrays(raw('float64'), raw('float32'))
    assert not release.same_arrays(raw('float64'), raw('float64', 'b'))


@pytest.mark.parametrize('field', ['enhanced_control_enabled', 'holdout_used', 'new_model_trained',
    'active_research_selector_exercised', 'actual_v28_checkpoint_load_failure_exercised', 'actual_v28_forward_failure_exercised'])
def test_disabled_regression_cannot_claim_active_models_or_fault_coverage(field):
    report = {'status': release.STATUS, 'protocol': json.loads((release.HERE/'protocol.json').read_bytes()),
        'enhanced_control_enabled': False, 'holdout_used': False, 'new_model_trained': False,
        'active_research_selector_exercised': False, 'actual_v28_checkpoint_load_failure_exercised': False,
        'actual_v28_forward_failure_exercised': False}
    report[field] = True
    with pytest.raises(ValueError, match='cannot qualify'):
        release.validate_report(lambda name: json.dumps(report).encode(), 'fixture')


def test_invalid_archive_never_launches_real_entry_or_emits_certificate(tmp_path, monkeypatch):
    archive = tmp_path/'bad.zip'
    with zipfile.ZipFile(archive, 'x') as stored:
        stored.writestr('../unsafe', b'x')
    monkeypatch.setattr(release.subprocess, 'run', lambda *a, **k: pytest.fail('Must refuse before real entry'))
    with pytest.raises(ValueError):
        release.verify(archive, tmp_path/'replay')
    assert not (tmp_path/'replay/summary.json').exists()


def test_exclusive_archive_never_overwrites_user_data(tmp_path):
    archive = tmp_path/'user.zip'
    archive.write_bytes(b'user')
    with pytest.raises(ValueError):
        release.package(tmp_path/'missingA', tmp_path/'missingB', archive, tmp_path/'replay')
    assert archive.read_bytes() == b'user'
