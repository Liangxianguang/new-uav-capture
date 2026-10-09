"""Negative tests for release provenance, raw trajectories and censoring."""
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from diagnose import commands, summarize
from analyze_latency import analyze
from release import audit


def encoded(value):
    return json.dumps(value).encode()


def npz_bytes(data):
    stream = io.BytesIO()
    np.savez_compressed(stream, **data)
    return stream.getvalue()


@pytest.fixture
def evidence():
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    protocol["variants"] = {"fast_target": 7}
    relative = np.ones((2, 4, 6))
    relative[:, :, :3] = np.arange(12).reshape(4, 3)
    proposed = np.stack([commands(np.zeros((8, 4, 3)), {"defender_positions": r[:, :3]}, np.zeros(3), protocol) for r in relative])
    data = {"history": np.zeros((2, 8, 252)), "target": np.zeros((2, 13, 24, 3)),
            "valid": np.ones((2, 13, 24), dtype=bool), "mode": np.full((2, 13, 24), "mode"),
            "termination": np.full((2, 13, 24), "running"), "executed": proposed.copy(),
            "sensor": np.zeros((2, 13, 24, 4, 3)), "variant": np.array(["fast_target"] * 2),
            "group": np.array(["a", "b"]), "step": np.array([3, 7]), "proposed": proposed,
            "relative": relative, "reference": np.zeros((2, 3))}
    original = [{"episode_index": i, "mirror_group_id": group, "variant": "fast_target"} for i, group in enumerate(["a", "b"])]
    selected = [{**r, "model_split": "scout_train_only", "level": 7} for r in original]
    members = {"source/cwm_v4/protocol.json": encoded(protocol),
               "inputs/train/scenes.jsonl": "".join(json.dumps(r) + "\n" for r in original).encode(),
               "inputs/excluded/0.jsonl": b'{"mirror_group_id":"excluded"}\n',
               "inputs/exclusion_manifest.json": encoded([{"original_path": "excluded.jsonl", "member": "inputs/excluded/0.jsonl"}]),
               "run/selected_scenes.jsonl": "".join(json.dumps(r) + "\n" for r in selected).encode(),
               "run/pairs.npz": npz_bytes(data),
               "run/latency_analysis.json": encoded(analyze(data, protocol)),
               "run/windows.json": encoded([{"episode_index": i, "group": g, "step": s, "parent_integrity": True, "repeat_exact": True}
                                            for i, (g, s) in enumerate(zip(["a", "b"], [3, 7]))])}
    trajectory = npz_bytes({"target_positions": np.zeros((2, 3)), "defender_positions": np.zeros((2, 4, 3))})
    episodes = []
    for i in range(2):
        members[f"run/observed/{i}.npz"] = trajectory
        members[f"run/unobserved/{i}.npz"] = trajectory
        episodes.append({"episode_index": i, "trajectories_byte_equal": True, "outcomes_equal": True,
                         "trajectory_sha256": hashlib.sha256(trajectory).hexdigest(), "unobserved_sha256": hashlib.sha256(trajectory).hexdigest()})
    members["run/episodes.json"] = encoded(episodes)
    summary = {"protocol": protocol, "enhanced_control_enabled": False, "excluded_group_overlap": 0,
               "source_hashes": {"cwm_v4/protocol.json": hashlib.sha256(members["source/cwm_v4/protocol.json"]).hexdigest()},
               "training_scenes_sha256": hashlib.sha256(members["inputs/train/scenes.jsonl"]).hexdigest(),
               "exclusion_sha256": {"excluded.jsonl": hashlib.sha256(members["inputs/excluded/0.jsonl"]).hexdigest()},
               "episodes": 2, "groups": 2, "states": 2, "dataset_sha256": hashlib.sha256(members["run/pairs.npz"]).hexdigest(),
               "variants": summarize(data, protocol), "episodes_results": episodes, "signal_candidates": [],
               "status": "data_adequacy_no_go_keep_baseline"}
    members["run/summary.json"] = encoded(summary)
    return members, summary, data


def test_release_recomputes_actual_evidence(evidence):
    members, _, _ = evidence
    assert audit(members.__getitem__)["statistics_recomputed"]


def test_release_rejects_unjustified_statistics(evidence):
    members, summary, _ = evidence
    summary["variants"]["fast_target"]["8"]["fraction_over_0_05m"] = .7
    members["run/summary.json"] = encoded(summary)
    with pytest.raises(ValueError, match="statistics"):
        audit(members.__getitem__)


def test_release_rejects_resumed_after_terminal_even_with_new_hash(evidence):
    members, summary, data = evidence
    data["valid"][0, 0, 3] = False
    members["run/pairs.npz"] = npz_bytes(data)
    summary["dataset_sha256"] = hashlib.sha256(members["run/pairs.npz"]).hexdigest()
    members["run/summary.json"] = encoded(summary)
    with pytest.raises(ValueError, match="Resumed"):
        audit(members.__getitem__)


def test_release_does_not_trust_trajectory_pass_boolean(evidence):
    members, summary, _ = evidence
    altered = npz_bytes({"target_positions": np.ones((2, 3)), "defender_positions": np.zeros((2, 4, 3))})
    members["run/unobserved/0.npz"] = altered
    summary["episodes_results"][0]["unobserved_sha256"] = hashlib.sha256(altered).hexdigest()
    members["run/episodes.json"] = encoded(summary["episodes_results"])
    members["run/summary.json"] = encoded(summary)
    with pytest.raises(ValueError, match="Actual baseline"):
        audit(members.__getitem__)


def test_release_rejects_duplicate_invariance_coverage(evidence):
    members, summary, _ = evidence
    summary["episodes_results"][1] = summary["episodes_results"][0].copy()
    members["run/episodes.json"] = encoded(summary["episodes_results"])
    members["run/summary.json"] = encoded(summary)
    with pytest.raises(ValueError, match="coverage"):
        audit(members.__getitem__)


def test_windows_recorded_source_keys_have_portable_archive_names(evidence):
    members, summary, _ = evidence
    summary["source_hashes"] = {k.replace("/", "\\"): v for k, v in summary["source_hashes"].items()}
    members["run/summary.json"] = encoded(summary)
    assert audit(members.__getitem__)["status"] == "passed"
