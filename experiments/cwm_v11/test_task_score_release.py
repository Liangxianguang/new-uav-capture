import copy
import json
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from task_score_release import verify_records


@pytest.fixture(scope="module")
def evidence():
    path = ROOT / "artifacts/full_local_score_audit_20261010.zip"
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    summary = json.loads(members["primary/summary.json"])
    for key in ("elapsed_seconds", "source_hashes", "protocol"):
        summary.pop(key)
    return members, json.loads(members["primary/records.json"]), summary


def test_published_records_and_summary_reconcile(evidence):
    members, records, summary = evidence
    measured = verify_records(members.__getitem__, "primary", records, summary)
    assert measured["calls"] == 768 and measured["gradient_checked_coordinates"] == 5019


def test_cost_gradient_claim_forgery_rejected(evidence):
    members, records, summary = copy.deepcopy(evidence)
    forged = json.loads(members["primary/records.json"])
    forged[0]["gradient_max_absolute_error"] = 0.5
    members["primary/records.json"] = json.dumps(forged).encode()
    with pytest.raises(ValueError, match="recomputation differs"):
        verify_records(members.__getitem__, "primary", records, summary)


def test_development_not_relabelled_as_training_effect_evidence(evidence):
    members, records, summary = copy.deepcopy(evidence)
    forged = json.loads(members["primary/records.json"])
    development = next(r for r in forged if r["split"] == "development_validation")
    development["train_diagnostic"] = next(r["train_diagnostic"] for r in forged if "train_diagnostic" in r)
    members["primary/records.json"] = json.dumps(forged).encode()
    with pytest.raises(ValueError, match="recomputation differs"):
        verify_records(members.__getitem__, "primary", records, summary)


def test_forged_positive_attribution_cannot_replace_computed_summary(evidence):
    members, records, summary = copy.deepcopy(evidence)
    forged = json.loads(members["primary/summary.json"])
    forged["training_positive_gain_calls"] = 403
    members["primary/summary.json"] = json.dumps(forged).encode()
    with pytest.raises(ValueError, match="summary differs"):
        verify_records(members.__getitem__, "primary", records, summary)


def test_run_used_source_closure_forgery_rejected(evidence):
    members, records, summary = copy.deepcopy(evidence)
    members["primary/source/cwm_v11/task_score.py"] += b"\n# forged\n"
    with pytest.raises(ValueError, match="source closure mismatch"):
        verify_records(members.__getitem__, "primary", records, summary)
