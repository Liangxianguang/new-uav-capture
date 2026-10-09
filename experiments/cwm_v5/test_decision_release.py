import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from decision_value import METHODS, rank_metrics, summarize
spec = importlib.util.spec_from_file_location("cwm_v5_release", Path(__file__).with_name("release.py"))
release_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release_module)
audit = release_module.audit


def encode(value):
    return json.dumps(value).encode()


def npz(data):
    stream = io.BytesIO()
    np.savez_compressed(stream, **data)
    return stream.getvalue()


class Reference:
    def __init__(self, members):
        self.members = members

    def read(self, name):
        return self.members[name]


@pytest.fixture
def evidence():
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    trajectory = npz({"target_positions": np.zeros((2, 3)), "defender_positions": np.zeros((2, 4, 3))})
    outcomes = {"safe_capture_success": True, "collision": False, "boundary_violation": False, "timeout": False, "termination_reason": "safe_capture"}
    reference = Reference({"run/windows.json": encode([{"episode_index": 1, "step": 3, "group": "g"}]),
                           "run/pairs.npz": npz({"valid": np.ones((1, 13, 24), dtype=bool)}),
                           "run/selected_scenes.jsonl": b'{"episode_index":1}\n',
                           "run/episodes.json": encode([{"episode_index": 1, **outcomes}]), "run/observed/1.npz": trajectory})
    costs = {method: list(range(13)) for method in METHODS}
    windows = [{"episode_index": 1, "step": 3, "group": "g", "variant": "fast_target", "parent_integrity": True,
                "all_candidates_full_horizon_valid": True, "reference_score_matches_original_diagnostics": True,
                "costs": costs, "metrics": {k: rank_metrics(v, costs["action_specific_truth"]) for k, v in costs.items()}}]
    episodes = [{"episode_index": 1, "trajectory_byte_equal": True, "trajectory_sha256": hashlib.sha256(trajectory).hexdigest(), **outcomes}]
    summary = {"status": "offline_diagnostic_complete_not_promoted", "enhanced_control_enabled": False, "protocol": protocol,
               "source_archive_sha256": protocol["source_archive_sha256"], "source_hashes": {}, "capsule_sha256": "capsule",
               "statistics": summarize(windows, protocol), "episodes": episodes}
    members = {"source/cwm_v5/protocol.json": encode(protocol)}
    for stage in ("primary", "repeated"):
        members[f"{stage}/summary.json"] = encode(summary)
        members[f"{stage}/windows.json"] = encode(windows)
        members[f"{stage}/trajectories/1.npz"] = trajectory
    return members, reference, summary, windows


def test_actual_release_evidence(evidence):
    members, reference, _, _ = evidence
    assert audit(members.__getitem__, reference)["replay_windows_byte_equal"]


def test_release_rejects_false_rank_statistics(evidence):
    members, reference, _, windows = evidence
    windows[0]["metrics"]["original_gru"]["choice"] = 1
    members["primary/windows.json"] = encode(windows)
    with pytest.raises(ValueError, match="Rank metrics"):
        audit(members.__getitem__, reference)


def test_release_checks_actual_mask_not_rankable_boolean(evidence):
    members, reference, _, _ = evidence
    masks = np.ones((1, 13, 24), dtype=bool)
    masks[0, 1, 7] = False
    reference.members["run/pairs.npz"] = npz({"valid": masks})
    with pytest.raises(ValueError, match="terminal support"):
        audit(members.__getitem__, reference)


def test_release_does_not_trust_trajectory_boolean(evidence):
    members, reference, summary, _ = evidence
    altered = npz({"target_positions": np.ones((2, 3)), "defender_positions": np.zeros((2, 4, 3))})
    summary["episodes"][0]["trajectory_sha256"] = hashlib.sha256(altered).hexdigest()
    members["primary/summary.json"] = encode(summary)
    members["primary/trajectories/1.npz"] = altered
    with pytest.raises(ValueError, match="Actual original"):
        audit(members.__getitem__, reference)
