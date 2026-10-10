"""Semantic rejection checks on the actual small entry-fault replay archive."""
import json
import io
import sys
import zipfile
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_entry_audit import audit, ROOT


@pytest.fixture
def completed_members():
    artifact = Path(__file__).with_name('artifacts') / 'entry_fault_replay_20261010.zip'
    if not artifact.is_file():
        pytest.skip('Completed representative entry archive required')
    with zipfile.ZipFile(artifact) as z:
        return {name: z.read(name) for name in z.namelist()}


def test_completed_entry_replay_reaudited(completed_members):
    result = audit(completed_members.__getitem__, ROOT / 'experiments/cwm_v1/baseline/capsule.zip')
    assert result['two_run_arrays_exact'] and not result['enhanced_control_enabled']


@pytest.mark.parametrize('forgery,reason', [
    ('enabled', 'Not a completed'), ('support', 'Fixed replay support'),
    ('control', 'Fault injection must not control'), ('unexercised', 'Fault path was not exercised'),
    ('source', 'Two entry replay sources differ'),
])
def test_forged_completed_entry_summary_rejected(completed_members, forgery, reason):
    name = 'repeated/summary.json'
    summary = json.loads(completed_members[name])
    if forgery == 'enabled':
        summary['enhanced_control_enabled'] = True
    elif forgery == 'support':
        summary['modes']['off']['rows'][0]['episode_index'] += 1
    elif forgery == 'control':
        summary['modes']['refusal']['events'][0]['control_eligible'] = True
    elif forgery == 'unexercised':
        summary['modes']['prediction_failure']['prediction_calls'] = 0
    elif forgery == 'source':
        summary['source_hashes'] = {}
    completed_members[name] = json.dumps(summary).encode()
    with pytest.raises(ValueError, match=reason):
        audit(completed_members.__getitem__, ROOT / 'experiments/cwm_v1/baseline/capsule.zip')


@pytest.mark.parametrize('field,filename,reason', [
    ('target_positions', '.npz', 'Physical trajectory differs'),
    ('commanded', '.commands.npz', 'DN-MPC plan or post-CBF command differs'),
])
def test_forged_completed_entry_arrays_rejected(completed_members, field, filename, reason):
    summary = json.loads(completed_members['repeated/summary.json'])
    row = summary['modes']['off']['rows'][0]
    # Forge BOTH runs identically: cross-run consistency cannot be enough.
    for stage in ('primary', 'repeated'):
        name = f"{stage}/off/level{row['level']}_{row['episode_index']}{filename}"
        with np.load(io.BytesIO(completed_members[name]), allow_pickle=False) as data:
            values = {key: data[key].copy() for key in data.files}
        values[field].flat[0] += .01
        forged = io.BytesIO()
        np.savez_compressed(forged, **values)
        completed_members[name] = forged.getvalue()
    with pytest.raises(ValueError, match=reason):
        audit(completed_members.__getitem__, ROOT / 'experiments/cwm_v1/baseline/capsule.zip')
