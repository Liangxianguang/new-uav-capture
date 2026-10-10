import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
from s4_value import METHODS, information_paths, rank_metrics, summarize
from s4_value_release import audit_runs


@pytest.fixture(scope="module")
def members():
    with zipfile.ZipFile(ROOT / "artifacts/s4_decision_value_20261010.zip") as archive:
        runs = {name: archive.read(name) for name in archive.namelist()}
    with zipfile.ZipFile(ROOT.parent / "cwm_v7/artifacts/paired_training_20261010.zip") as archive:
        source = {name: archive.read(name) for name in archive.namelist()}
    return runs, source


def test_s4_actual_value_release(members):
    runs, source = members
    result = audit_runs(runs.__getitem__, source.__getitem__)
    assert result["windows"] == 80 and result["rankable_windows"] == 69
    assert not result["motion_research_signal"] and not result["response_research_signal_given_motion"]
    assert not result["prior_training_gate_passed"] and not result["enhanced_control_enabled"]


@pytest.mark.parametrize("field", ["support", "ranking", "statistics", "gate"])
def test_s4_release_rejects_optimistic_reporting(members, field):
    runs, source = members
    forged = dict(runs)
    if field in ("support", "ranking"):
        records = json.loads(forged["primary/windows.json"])
        if field == "support":
            records[0]["all_candidates_full_horizon_valid"] = False
        else:
            records[0]["metrics"]["structured_learned"]["choice"] = 7
        forged["primary/windows.json"] = json.dumps(records).encode()
    else:
        summary = json.loads(forged["primary/summary.json"])
        if field == "statistics":
            summary["statistics"]["response_research_signal_given_motion"] = True
        else:
            summary["prior_training_gate_passed"] = True
        forged["primary/summary.json"] = json.dumps(summary).encode()
    with pytest.raises(ValueError):
        audit_runs(forged.__getitem__, source.__getitem__)


def test_s4_release_rejects_changed_trajectory_with_updated_digest(members):
    runs, source = members
    forged = dict(runs)
    summary = json.loads(forged["primary/summary.json"])
    row = summary["episodes"][0]
    path = f"primary/trajectories/{row['episode_index']}.npz"
    with np.load(io.BytesIO(forged[path]), allow_pickle=False) as archive:
        values = {k: archive[k] for k in archive.files}
    values["defender_positions"][0, 0, 0] += .01
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **values)
    forged[path] = buffer.getvalue()
    row["trajectory_sha256"] = hashlib.sha256(forged[path]).hexdigest()
    forged["primary/summary.json"] = json.dumps(summary).encode()
    with pytest.raises(ValueError, match="replay arrays"):
        audit_runs(forged.__getitem__, source.__getitem__)


def test_private_truth_never_changes_learned_or_public_paths():
    rng = np.random.default_rng(12)
    backbone = rng.normal(size=(8, 3))
    truth = rng.normal(size=(8, 8, 3))
    predictions = {kind: {"prediction": rng.normal(size=(8, 8, 3)), "response": rng.normal(size=(8, 8, 3))} for kind in ("plain", "structured")}
    a = information_paths(backbone, truth, np.ones(3), np.ones(3), predictions)
    b = information_paths(backbone, truth + 10., np.ones(3), np.ones(3), predictions)
    for method in ("original_gru", "constant_velocity", "plain_learned", "structured_learned"):
        assert a[method].tobytes() == b[method].tobytes()
    assert not np.array_equal(a["action_specific_truth"], b["action_specific_truth"])


def test_research_signal_needs_actual_choice_and_group_cost_gain():
    protocol = json.loads((ROOT / "protocol.json").read_text())
    records = []
    for group in ("a", "b"):
        costs = {k: [0., 1., 2., 3., 4., 5., 6., 7.] for k in METHODS}
        row = {"group": group, "all_candidates_full_horizon_valid": True,
               "metrics": {k: rank_metrics(v, costs["action_specific_truth"]) for k, v in costs.items()}}
        records.append(row)
    assert not summarize(records, protocol)["response_research_signal_given_motion"]
    for row in records:
        row["metrics"]["fixed_reference_truth"] = rank_metrics([1., 0., 2., 3., 4., 5., 6., 7.], costs["action_specific_truth"])
    result = summarize(records, protocol)
    assert result["response_research_signal_given_motion"]
    # Exploratory signal is not, and cannot silently become, controller promotion.
    assert "enhanced_control_enabled" not in result
    assert not protocol["prior_training_gate_override_allowed"]


def test_s4_release_requires_identical_independent_records(members):
    runs, source = members
    forged = dict(runs)
    forged["repeated/windows.json"] += b"\n"
    with pytest.raises(ValueError, match="record bytes"):
        audit_runs(forged.__getitem__, source.__getitem__)
