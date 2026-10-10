import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from s4_release import audit


@pytest.fixture
def members():
    path = Path(__file__).parent / "artifacts/s4_mechanism_20261010.zip"
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_s4_release_recomputes_actual_data_and_reports_invalid_episodes(members):
    result = audit(members.__getitem__)
    assert result["stages"]["scout"]["target_invalid_episodes"] == 4
    assert result["stages"]["confirmation"]["target_invalid_episodes"] == 5
    assert not result["new_model_trained"]


def test_s4_release_rejects_falsified_response_statistics(members):
    report = json.loads(members["scout/summary.json"])
    report["statistics"]["horizons"]["8"]["branch_flip_pairs"] += 1
    members["scout/summary.json"] = json.dumps(report).encode()
    with pytest.raises(ValueError, match="Statistics"):
        audit(members.__getitem__)


def test_s4_release_checks_trajectory_arrays_not_only_updated_hashes(members):
    episodes = json.loads(members["scout/episodes.json"])
    name = f"scout/observed/{episodes[0]['episode_index']}.npz"
    with np.load(io.BytesIO(members[name]), allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["target_positions"][0, 0] += .1
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    members[name] = output.getvalue()
    episodes[0]["observed_sha256"] = hashlib.sha256(members[name]).hexdigest()
    report = json.loads(members["scout/summary.json"])
    report["episodes_results"] = episodes
    members["scout/summary.json"] = json.dumps(report).encode()
    members["scout/episodes.json"] = json.dumps(episodes).encode()
    with pytest.raises(ValueError, match="trajectory changed"):
        audit(members.__getitem__)


def test_s4_release_rejects_masks_despite_updated_dataset_hash(members):
    name = "scout/pairs.npz"
    with np.load(io.BytesIO(members[name]), allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    arrays["valid"][0, 0, 0] = False
    output = io.BytesIO()
    np.savez_compressed(output, **arrays)
    members[name] = output.getvalue()
    report = json.loads(members["scout/summary.json"])
    report["dataset_sha256"] = hashlib.sha256(members[name]).hexdigest()
    members["scout/summary.json"] = json.dumps(report).encode()
    with pytest.raises(ValueError, match="termination masks"):
        audit(members.__getitem__)
