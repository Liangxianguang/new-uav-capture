"""Training-artifact forgery checks; full data reaudited by CLI verifier."""
import copy
import json
from pathlib import Path
import sys
import zipfile

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import task_training_release as release


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    release.torch.set_num_threads(1)
    with zipfile.ZipFile(ROOT / "artifacts/task_effect_training_20261010.zip") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(ROOT.parent / "cwm_v7/artifacts/paired_training_20261010.zip") as source:
        original = {name: source.read(name) for name in source.namelist()}
    restored = tmp_path_factory.mktemp("v12release") / "restored"
    configs = release.frozen_configs(ROOT.parent / "cwm_v1/baseline/capsule.zip", restored)
    calls, report = release.audit_data(members.__getitem__, original.__getitem__, configs, restored)
    return members, original, configs, restored, calls, report


def training_audit(evidence, monkeypatch):
    members, original, configs, restored, calls, report = evidence
    # Isolate training-field forgery tests after the fixture's FULL original
    # data/public/local reconstruction. CLI archive --verify repeats everything.
    monkeypatch.setattr(release, "audit_data", lambda *args: (calls, report))
    return release.audit(members.__getitem__, original.__getitem__, configs, restored)


def test_full_artifact_reloads_all_models_and_costs(evidence, monkeypatch):
    measured = training_audit(evidence, monkeypatch)
    assert measured["models_per_run"] == 15 and measured["independent_training_runs"] == 2
    assert measured["reload_prediction_bytes_equal"] and measured["all_learned_full_cost_rankings_recomputed"]


def test_optimistic_qualification_forgery_rejected(evidence, monkeypatch):
    mutated = copy.deepcopy(evidence)
    members = mutated[0]
    report = json.loads(members["primary/summary.json"])
    report["qualification"]["structured_task"]["research_eligible"] = not report["qualification"]["structured_task"]["research_eligible"]
    members["primary/summary.json"] = json.dumps(report).encode()
    with pytest.raises(ValueError, match="qualification/median"):
        training_audit(mutated, monkeypatch)


def test_learned_original_mpc_cost_forgery_rejected(evidence, monkeypatch):
    mutated = copy.deepcopy(evidence)
    members = mutated[0]
    name = "primary/plain_task_seed974101_decisions.json"
    records = json.loads(members[name])
    records[0]["costs"]["model"][0] += .5
    members[name] = json.dumps(records).encode()
    with pytest.raises(ValueError, match="learned costs/rankings"):
        training_audit(mutated, monkeypatch)


def test_checkpoint_byte_forgery_rejected(evidence, monkeypatch):
    mutated = copy.deepcopy(evidence)
    members = mutated[0]
    members["primary/motion_only_seed974101.pt"] += b"forged"
    with pytest.raises(ValueError, match="checkpoint digest"):
        training_audit(mutated, monkeypatch)


def test_declared_partial_population_cannot_masquerade_as_complete(evidence, monkeypatch):
    mutated = copy.deepcopy(evidence)
    members = mutated[0]
    report = json.loads(members["primary/summary.json"])
    report["models"].pop()
    members["primary/summary.json"] = json.dumps(report).encode()
    with pytest.raises(ValueError, match="fifteen-model population"):
        training_audit(mutated, monkeypatch)
