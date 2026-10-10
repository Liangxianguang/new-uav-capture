import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from local_shadow_release import audit, frozen_configs


@pytest.fixture(scope="module")
def evidence():
    torch.set_num_threads(1)
    with zipfile.ZipFile(ROOT / "artifacts/local_shadow_20261010.zip") as archive:
        runs = {n: archive.read(n) for n in archive.namelist()}
    with zipfile.ZipFile(ROOT.parent / "cwm_v7/artifacts/paired_training_20261010.zip") as archive:
        source = {n: archive.read(n) for n in archive.namelist()}
    configs = frozen_configs(ROOT.parent / "cwm_v1/baseline/capsule.zip", ROOT.parent.parent / "results/cwm_v9/audit_restored")
    return runs, source, configs


def test_actual_local_shadow_reconstruction_and_fallbacks(evidence):
    runs, source, configs = evidence
    result = audit(runs.__getitem__, source.__getitem__, configs)
    assert result["shadow_calls"] == 320 and result["rankable_calls"] == 131
    assert result["missing_peer_calls"] == 128 and result["incomplete_support_calls"] == 61
    assert result["motion_research_signal"] and result["response_research_signal_given_motion"]
    assert result["local_candidates_and_scores_reconstructed"] and result["all_original_replay_arrays_equal"]
    assert result["default_off_no_optional_load_or_predict"] and not result["enhanced_control_enabled"]


def first_complete(runs):
    records = json.loads(runs["primary/calls.json"])
    row = next(r for r in records if r["all_candidates_full_horizon_valid"])
    return records, row


def test_release_reconstructs_cost_not_only_reported_ranks(evidence):
    runs, source, configs = evidence
    changed = dict(runs)
    records, row = first_complete(runs)
    row["costs"]["original_gru"][0] += .01
    changed["primary/calls.json"] = json.dumps(records).encode()
    with pytest.raises(ValueError, match="reconstructed local cost"):
        audit(changed.__getitem__, source.__getitem__, configs)


def test_release_rejects_changed_delayed_context_with_updated_digest(evidence):
    runs, source, configs = evidence
    changed = dict(runs)
    records, row = first_complete(runs)
    name = "primary/" + row["context_path"]
    snapshot = json.loads(changed[name])
    peer = next(iter(snapshot["known"]))
    snapshot["known"][peer]["position"]["array_values"][0] += 1.
    changed[name] = json.dumps(snapshot).encode()
    row["context_sha256"] = hashlib.sha256(changed[name]).hexdigest()
    changed["primary/calls.json"] = json.dumps(records).encode()
    with pytest.raises(ValueError):
        audit(changed.__getitem__, source.__getitem__, configs)


def test_release_rejects_false_physics_masks_with_updated_digest(evidence):
    runs, source, configs = evidence
    changed = dict(runs)
    records, row = first_complete(runs)
    name = "primary/" + row["arrays_path"]
    with np.load(io.BytesIO(changed[name]), allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files}
    values["valid"][0, 0] = False
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    changed[name] = stream.getvalue()
    row["arrays_sha256"] = hashlib.sha256(changed[name]).hexdigest()
    changed["primary/calls.json"] = json.dumps(records).encode()
    with pytest.raises(ValueError, match="physics/support"):
        audit(changed.__getitem__, source.__getitem__, configs)


def test_release_rejects_enhancement_gate_override(evidence):
    runs, source, configs = evidence
    changed = dict(runs)
    report = json.loads(changed["primary/summary.json"])
    report["prior_training_gate_overridden"] = True
    changed["primary/summary.json"] = json.dumps(report).encode()
    with pytest.raises(ValueError, match="qualification contract"):
        audit(changed.__getitem__, source.__getitem__, configs)


def test_release_rejects_optional_loading_in_default_off(evidence):
    runs, source, configs = evidence
    changed = dict(runs)
    report = json.loads(changed["off/summary.json"])
    report["providers"]["plain"]["load_calls"] = 1
    changed["off/summary.json"] = json.dumps(report).encode()
    with pytest.raises(ValueError, match="Fallback loader"):
        audit(changed.__getitem__, source.__getitem__, configs)
