import json
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from decision_release import audit


@pytest.fixture(scope="module")
def evidence():
    with zipfile.ZipFile(ROOT / "artifacts/two_head_decision_20261010.zip") as archive:
        records = {n: archive.read(n) for n in archive.namelist()}
    with zipfile.ZipFile(ROOT / "artifacts/two_head_training_20261010.zip") as archive:
        source = {n: archive.read(n) for n in ("data/calls.json", "data/summary.json", "primary/summary.json")}
    return records, source


def test_actual_fixed_library_diagnostic_evidence(evidence):
    records, source = evidence
    result = audit(records.__getitem__, source.__getitem__)
    assert result["rankable_calls"] == 136 and result["development_calls"] == 192
    assert result["independent_records_byte_equal"] and not result["prior_gate_overridden"]


def test_diagnostic_cannot_silently_promote_a_failed_training_gate(evidence):
    records, source = evidence
    changed = dict(records)
    report = json.loads(changed["primary/summary.json"])
    report["prior_training_gate_overridden"] = True
    changed["primary/summary.json"] = json.dumps(report).encode()
    with pytest.raises(ValueError, match="gate contract"):
        audit(changed.__getitem__, source.__getitem__)


def test_response_attribution_is_recomputed_not_trusted(evidence):
    records, source = evidence
    changed = dict(records)
    report = json.loads(changed["primary/summary.json"])
    report["learned_response_increment"]["structured"]["gain"]["group_equal_mean"] += 1.
    changed["primary/summary.json"] = json.dumps(report).encode()
    with pytest.raises(ValueError, match="attribution"):
        audit(changed.__getitem__, source.__getitem__)
