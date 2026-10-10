import json
from pathlib import Path
import sys
import zipfile

import pytest
import torch

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from two_head_release import audit, audit_data
from local_shadow_release import frozen_configs


@pytest.fixture(scope="module")
def evidence():
    torch.set_num_threads(1)
    with zipfile.ZipFile(ROOT / "artifacts/two_head_training_20261010.zip") as archive:
        members = {n: archive.read(n) for n in archive.namelist()}
    with zipfile.ZipFile(ROOT.parent / "cwm_v7/artifacts/paired_training_20261010.zip") as archive:
        source = {n: archive.read(n) for n in archive.namelist()}
    restored = ROOT.parent.parent / "results/cwm_v10/audit_restored"
    configs = frozen_configs(ROOT.parent / "cwm_v1/baseline/capsule.zip", restored)
    return members, source, configs, restored


def test_fresh_public_replay_and_all_nine_models_reproduce(evidence):
    members, source, configs, restored = evidence
    result = audit(members.__getitem__, source.__getitem__, configs, restored)
    assert result["train_calls"] == 576 and result["development_calls"] == 192
    assert result["original_public_replay_calls"] == 1280
    assert result["weights_optimizer_rng_equal"] and result["reload_prediction_bytes_equal"]
    assert not result["development_gate_passed"] and not result["enhanced_control_enabled"]


def test_predeclared_gate_cannot_be_replaced_by_optimistic_report(evidence):
    members, source, configs, restored = evidence
    changed = dict(members)
    report = json.loads(changed["primary/summary.json"])
    report["development_gate_passed"] = True
    changed["primary/summary.json"] = json.dumps(report).encode()
    with pytest.raises(ValueError, match="development gate"):
        audit(changed.__getitem__, source.__getitem__, configs, restored)


def test_independent_public_frames_are_reencoded_not_just_hashed(evidence):
    import hashlib
    members, source, configs, restored = evidence
    changed = dict(members)
    report = json.loads(changed["public/summary.json"])
    row = report["episodes"][0]
    name = f"public/frames/{row['episode_index']}.json"
    frames = json.loads(changed[name])
    frames[0]["feature"]["array_values"][0] += 1.
    changed[name] = json.dumps(frames).encode()
    row["frames_sha256"] = hashlib.sha256(changed[name]).hexdigest()
    changed["public/summary.json"] = json.dumps(report).encode()
    with pytest.raises(ValueError, match="252 feature encoding"):
        audit_data(changed.__getitem__, source.__getitem__, configs, restored)
