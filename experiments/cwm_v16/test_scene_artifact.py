import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from scene_release import audit


@pytest.fixture(scope='module')
def evidence():
    with zipfile.ZipFile(ROOT/'artifacts/new_scene_diagnostic_20261010.zip') as archive:
        return {n:archive.read(n) for n in archive.namelist()}


def test_two_collections_and_independent_public_replay(evidence):
    result = audit(evidence.__getitem__)
    assert result['assigned_scenes'] == result['completed_episodes'] == 64
    assert result['windows'] == 256
    assert result['counterfactual_arrays_equal']
    assert result['public_joint_proposals_replayed']
    assert result['original_trajectories_byte_equal']
    assert not result['enhanced_control_enabled'] and not result['holdout_used']
    assert result['variants'][0]['eligible_for_fresh_confirmation']
    assert not any(r['eligible_for_fresh_confirmation'] for r in result['variants'][1:])


def test_failed_geometry_cannot_be_promoted_by_report(evidence):
    changed = dict(evidence)
    for stage in ('primary','repeated'):
        report = json.loads(changed[stage+'/summary.json'])
        report['variants'][1]['eligible_for_fresh_confirmation'] = True
        changed[stage+'/summary.json'] = json.dumps(report).encode()
    public = json.loads(changed['public/summary.json'])
    from scene_release import digest
    public['data_summary_sha256'] = digest(changed['primary/summary.json'])
    changed['public/summary.json'] = json.dumps(public).encode()
    with pytest.raises(ValueError,match='geometry/response gate'):
        audit(changed.__getitem__)


def test_identical_counterfactual_archive_is_not_a_public_replay(evidence):
    changed = dict(evidence)
    report = json.loads(changed['public/summary.json'])
    report['checked_windows'].pop()
    changed['public/summary.json'] = json.dumps(report).encode()
    with pytest.raises(ValueError,match='joint proposals'):
        audit(changed.__getitem__)
