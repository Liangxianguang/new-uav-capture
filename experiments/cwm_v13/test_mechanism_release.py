"""Forgery and masking checks for the V13 TRAIN-only mechanism archive."""
import copy
import json
import sys
import io
from pathlib import Path
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import mechanism_release as release


@pytest.fixture(scope="module")
def evidence():
    with zipfile.ZipFile(ROOT / "artifacts/mechanism_diagnosis_20261010.zip") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(ROOT.parent / "cwm_v12/artifacts/task_effect_training_20261010.zip") as source:
        original = {name: source.read(name) for name in source.namelist()}
    return members, original


def test_independent_replays_and_private_labels_recompute(evidence):
    members, original = evidence
    result = release.audit_archive_from_members(members, original)
    assert result["calls"] == 716
    assert result["true_branch_flip_pairs"] == 44
    assert result["private_mechanism_labels_network_inputs"] is False


def test_summary_forgery_rejected(evidence):
    members, original = copy.deepcopy(evidence)
    summary = json.loads(members["primary/summary.json"])
    summary["true_branch_flip_pairs"] += 1
    members["primary/summary.json"] = json.dumps(summary).encode()
    with pytest.raises(ValueError, match="aggregate mismatch"):
        release.audit_archive_from_members(members, original)


def test_private_label_forgery_rejected(evidence):
    members, original = copy.deepcopy(evidence)
    record = json.loads(members["primary/records.json"])[0]
    name = "primary/" + record["mechanism_path"]
    values = release.arrays(members[name])
    values["branch_after_label_only"][0, 0] = 99
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    members[name] = buffer.getvalue()
    with pytest.raises(ValueError, match="digest mismatch"):
        release.audit_archive_from_members(members, original)


def test_original_arrays_cannot_contain_private_labels(evidence):
    members, original = copy.deepcopy(evidence)
    calls = json.loads(original["data/calls.json"])
    row = next(row for row in calls if row.get("arrays_path"))
    values = release.arrays(original["data/" + row["arrays_path"]])
    values["branch_after_label_only"] = np.zeros((2, 8), dtype=np.int64)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    original["data/" + row["arrays_path"]] = buffer.getvalue()
    with pytest.raises(ValueError, match="Private mechanism labels leaked"):
        release.audit_archive_from_members(members, original)


def test_control_or_holdout_activation_rejected(evidence):
    members, original = copy.deepcopy(evidence)
    summary = json.loads(members["primary/summary.json"])
    summary["enhanced_control_enabled"] = True
    members["primary/summary.json"] = json.dumps(summary).encode()
    with pytest.raises(ValueError, match="disabled-control"):
        release.audit_archive_from_members(members, original)


def test_duplicate_calls_rejected(evidence):
    members, original = copy.deepcopy(evidence)
    rows = json.loads(members["primary/records.json"])
    rows[-1] = rows[0]
    members["primary/records.json"] = json.dumps(rows).encode()
    with pytest.raises(ValueError, match="duplicate"):
        release.audit_archive_from_members(members, original)


def test_mechanism_source_closure_forgery_rejected(evidence):
    members, original = copy.deepcopy(evidence)
    members["primary/source/cwm_v13/mechanism_replay.py"] += b"# changed"
    with pytest.raises(ValueError, match="Source closure"):
        release.audit_archive_from_members(members, original)


def test_variable_candidates_nonanchor_exclusion_and_padding():
    n, h = 3, 8
    anchor = np.zeros((h, 4, 3))
    proposed = np.stack([anchor, anchor + .1, anchor + .2])
    valid = np.zeros((n, h), bool)
    valid[:, :2] = True
    target = np.zeros((n, h, 3))
    target[1, :2, 0] = .1
    target[2, :2, 0] = .2
    target[:, 2:] = 1e6  # future padding may not contribute response
    values = {"proposed": proposed, "anchor": anchor, "valid": valid, "anchor_valid": valid[0],
              "termination": np.array([["running", "safe_capture"] + ["after_terminal"] * 6] * n),
              "commanded": proposed.copy(), "executed": proposed.copy(), "target": target,
              "anchor_target": np.zeros((h, 3))}
    private = {"initial_branch_label_only": np.asarray(0), "branch_after_label_only": np.array([[1, 1] + [0] * 6, [-1, -1] + [0] * 6, [0] * 8]),
               "anchor_branch_after_label_only": np.array([1, 1] + [0] * 6)}
    protocol = json.loads((ROOT / "protocol.json").read_text())
    metrics = release.call_metrics(values, private, protocol)
    assert metrics["nonanchor_candidates"] == 2 and metrics["nonanchor_common_points"] == 4
    assert metrics["flipped_branch_response"]["count"] == 2
    assert metrics["pending_branch_response"]["count"] == 2
    assert metrics["branch_flip_pairs"] == 1 and metrics["full_nonanchor_pairs"] == 0
    assert metrics["response"]["maximum"] == .2
    assert metrics["observed_command_points"] == n * 2 * 4
    private["initial_branch_label_only"] = np.asarray(1)
    with pytest.raises(ValueError, match="Committed original target branch"):
        release.call_metrics(values, private, protocol)
