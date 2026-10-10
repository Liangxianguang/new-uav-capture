import json
import sys
import zipfile
from pathlib import Path

import pytest
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from release_verify import load_protocol, verify_archive, _check_branch, _compare_runs


def test_protocol_remains_shadow_only():
    protocol = load_protocol()
    assert protocol["enhanced_control_enabled"] is False
    assert protocol["new_model_trained"] is False
    assert protocol["holdout_used"] is False


def test_archive_manifest_tamper_is_rejected(tmp_path):
    source = ROOT / "artifacts" / "public_local_geometry_pilot_audited_20261010.zip"
    if not source.exists():
        pytest.skip("V18 release archive is created after collection audit")
    target = tmp_path / source.name
    target.write_bytes(source.read_bytes())
    verify_archive(target)
    with zipfile.ZipFile(target) as original:
        files = {name: original.read(name) for name in original.namelist()}
    files["release_summary.json"] = files["release_summary.json"].replace(
        b"v18_public_geometry_pilot_release_audit_passed", b"tampered_release_status")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    with pytest.raises(ValueError):
        verify_archive(target)


def branch():
    return {'valid': np.ones(8, bool), 'termination': np.full(8, 'running', dtype='U128'),
            'target': np.zeros((8,3)), 'defenders': np.zeros((8,4,3)),
            'commanded': np.zeros((8,4,3)), 'executed': np.zeros((8,4,3))}


def test_branch_contract_rejects_executed_label_tamper():
    values = branch()
    _check_branch(values, '')
    values['executed'][0,0,0] = .01
    with pytest.raises(ValueError, match='commanded/executed'):
        _check_branch(values, '')


def test_branch_contract_rejects_future_padding_imputation():
    values = branch()
    values['valid'][4:] = False
    values['termination'][4:] = 'after_terminal'
    _check_branch(values, '')
    values['target'][5,0] = 1.
    with pytest.raises(ValueError, match='imputed'):
        _check_branch(values, '')


def test_branch_contract_rejects_nonsuffix_termination():
    values = branch()
    values['valid'][2] = False
    values['termination'][2] = 'after_terminal'
    with pytest.raises(ValueError, match='suffix'):
        _check_branch(values, '')


def test_repeat_accepts_json_key_order_but_not_value_change(tmp_path):
    first, second = tmp_path/'first', tmp_path/'second'
    for root in (first, second):
        root.mkdir()
        (root/'scenes.jsonl').write_text('{}\n')
        for name in ('records.json','decisions.json','episodes.json','summary.json'):
            (root/name).write_text(json.dumps({'a':1,'b':2} if root == first else {'b':2,'a':1}))
    _compare_runs(first, second)
    (second/'summary.json').write_text(json.dumps({'b':3,'a':1}))
    with pytest.raises(ValueError, match='summary'):
        _compare_runs(first, second)
