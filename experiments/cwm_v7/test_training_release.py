import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from training_release import audit, audit_data, tensor_tree_equal


@pytest.fixture(scope="module")
def original_members():
    path = Path(__file__).parent / "artifacts/paired_training_20261010.zip"
    with zipfile.ZipFile(path) as archive:
        return {n: archive.read(n) for n in archive.namelist()}


def test_training_release_checks_actual_reload_and_retraining(original_members):
    torch.set_num_threads(1)
    result = audit(original_members.__getitem__)
    assert result["weights_optimizer_rng_equal"] and result["reload_prediction_arrays_byte_equal"]
    assert not result["development_gate_passed"] and not result["holdout_used"]
    assert result["dataset_states"] == 320


def test_training_release_rejects_optimistic_gate_report(original_members):
    members = dict(original_members)
    report = json.loads(members["primary/summary.json"])
    report["development_gate_passed"] = True
    report["status"] = "offline_development_gate_passed_requires_decision_holdout_validation"
    members["primary/summary.json"] = json.dumps(report).encode()
    with pytest.raises(ValueError, match="development gate"):
        audit(members.__getitem__)


def test_training_release_rejects_normalizer_changed_by_validation(original_members):
    members = dict(original_members)
    with np.load(io.BytesIO(members["primary/normalization.npz"]), allow_pickle=False) as archive:
        normalization = {k: archive[k] for k in archive.files}
    normalization["mean"] += 1.
    stream = io.BytesIO()
    np.savez_compressed(stream, **normalization)
    members["primary/normalization.npz"] = stream.getvalue()
    with pytest.raises(ValueError, match="Normalizer"):
        audit(members.__getitem__)


def test_training_collection_rejects_false_masks_with_updated_hash(original_members):
    members = dict(original_members)
    with np.load(io.BytesIO(members["data/pairs.npz"]), allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files}
    values["valid"][0, 0, 0] = False
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    members["data/pairs.npz"] = stream.getvalue()
    report = json.loads(members["data/summary.json"])
    report["dataset_sha256"] = hashlib.sha256(members["data/pairs.npz"]).hexdigest()
    members["data/summary.json"] = json.dumps(report).encode()
    with pytest.raises(ValueError, match="physics/validity"):
        audit_data(members.__getitem__)


def test_tensor_verifier_checks_optimizer_and_rng_not_only_model():
    a = {"model": torch.ones(3), "optimizer": {"momentum": torch.zeros(3)}, "rng": torch.tensor([1, 2])}
    b = {"model": torch.ones(3), "optimizer": {"momentum": torch.ones(3)}, "rng": torch.tensor([1, 2])}
    assert not tensor_tree_equal(a, b)
    b["optimizer"]["momentum"] = torch.zeros(3)
    assert tensor_tree_equal(a, b)
    b["rng"][0] += 1
    assert not tensor_tree_equal(a, b)
