"""Independent verifier and packager for the TRAIN-only V13 mechanism audit."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cwm_v7"), str(ROOT.parent / "cwm_v10")]
from geometry_release import compare_arrays, sha
from mechanism_replay import call_metrics, summarize
from verify_release import check_archive

SOURCE_SHA = "a833ad7fe3302d32ce6b17237c7dcab19864826967dd9a099519169203405329"
CAPSULE_SHA = "c83a27a4360ad055b69dbced4c28370cb31a32ae89ea9fbab13523fa6218a329"


def arrays(raw):
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _json(raw):
    return json.loads(raw.decode() if isinstance(raw, bytes) else raw)


def _read_json(read, name):
    return _json(read(name))


def _same_json_without_elapsed(a, b):
    return {key: value for key, value in a.items() if key != "elapsed_seconds"} == {
        key: value for key, value in b.items() if key != "elapsed_seconds"
    }


def _validate_private_labels(read, stage, record, values, original_values):
    mechanism_name = f"{stage}/{record['mechanism_path']}"
    if hashlib.sha256(read(mechanism_name)).hexdigest() != record["mechanism_sha256"]:
        raise ValueError("Mechanism label digest mismatch")
    labels = set(values)
    if not labels or any(not key.endswith("label_only") for key in labels):
        raise ValueError("Mechanism archive contains non-private fields")
    if any(key.endswith("label_only") for key in original_values):
        raise ValueError("Private mechanism labels leaked into original arrays")
    count = len(original_values["proposed"])
    horizon = original_values["proposed"].shape[1]
    shapes = {"pre_target_position_label_only": (3,), "pre_target_velocity_label_only": (3,),
              "branch_before_label_only": (), "branch_after_label_only": (), "branch_commit_label_only": (),
              "branch_scores_label_only": (2,), "branch_scores_valid_label_only": (), "avoidance_exposure_label_only": (4,)}
    if labels != set(shapes) | {"anchor_" + key for key in shapes} | {"initial_branch_label_only"}:
        raise ValueError("Incomplete private mechanism label schema")
    for key, value in values.items():
        if key.startswith("anchor_"):
            candidate_key = key[len("anchor_"):]
            expected = (horizon,) + shapes[candidate_key]
        elif key == "initial_branch_label_only":
            expected = ()
        else:
            expected = (count, horizon) + shapes[key]
        if value.shape != expected or (value.dtype.kind in "fc" and not np.isfinite(value).all()):
            raise ValueError(f"Mechanism label shape mismatch: {key} {value.shape} != {expected}")
    termination = original_values["termination"]
    observed = termination != "after_terminal"
    if not np.array_equal(observed, np.arange(horizon)[None, :] < observed.sum(axis=1)[:, None]):
        raise ValueError("Candidate terminal padding is not a suffix")
    anchor_observed = original_values["anchor_termination"] != "after_terminal"
    if not np.array_equal(anchor_observed, np.arange(horizon) < int(anchor_observed.sum())):
        raise ValueError("Anchor terminal padding is not a suffix")
    for key, value in values.items():
        if key == "initial_branch_label_only" or value.ndim < 1:
            continue
        if key.startswith("anchor_"):
            mask = anchor_observed
            if value.dtype.kind in "biufc" and np.any(value[~mask] != 0):
                raise ValueError(f"Anchor terminal padding is counted in {key}")
        elif value.dtype.kind in "biufc":
            if np.any(value[~observed] != 0):
                raise ValueError(f"Terminal padding is counted in {key}")
    branches = [(values["branch_after_label_only"], values["branch_before_label_only"], values["branch_commit_label_only"], observed),
                (values["anchor_branch_after_label_only"][None], values["anchor_branch_before_label_only"][None], values["anchor_branch_commit_label_only"][None], anchor_observed[None])]
    for row, before, commits, mask in (item for args in branches for item in zip(*args)):
        if not mask.any() or before[0] != int(values["initial_branch_label_only"]):
            raise ValueError("Initial true branch provenance mismatch")
        if not np.isin(row[mask], [-1, 0, 1]).all() or not np.isin(before[mask], [-1, 0, 1]).all():
            raise ValueError("Invalid true branch label")
        if not np.array_equal(commits[mask], (before[mask] == 0) & (row[mask] != 0)):
            raise ValueError("Branch commitment label mismatch")
        if np.any((before[mask] != 0) & (row[mask] != before[mask])) or not np.array_equal(before[1:int(mask.sum())], row[:int(mask.sum())-1]):
            raise ValueError("Committed target branch changed after commitment")
        committed = np.flatnonzero((row != 0) & mask)
        if committed.size and np.any(row[committed[-1]:int(mask.sum())] != row[committed[-1]]):
            raise ValueError("Committed target branch changed after commitment")


def audit_stage(read, stage, protocol, source_read):
    summary = _read_json(read, f"{stage}/summary.json")
    records = _json(read(f"{stage}/records.json"))
    if summary["protocol"] != protocol or summary["status"] != "train_mechanism_replay_complete_not_promoted":
        raise ValueError("V13 stage protocol/status mismatch")
    if any(summary[key] for key in ("enhanced_control_enabled", "new_model_training_enabled", "holdout_used", "prior_gate_overridden", "private_mechanism_labels_network_inputs")):
        raise ValueError("V13 disabled-control contract was overridden")
    for name, digest in summary["source_hashes"].items():
        if hashlib.sha256(read(f"{stage}/source/{name}")).hexdigest() != digest:
            raise ValueError(f"Source closure mismatch: {name}")
    if _read_json(read, f"{stage}/source/cwm_v13/protocol.json") != protocol:
        raise ValueError("Recorded V13 protocol mismatch")
    if len(records) != protocol["eligible_calls"] or len({(r["episode_index"], r["step"], r["agent"]) for r in records}) != len(records):
        raise ValueError("Incomplete or duplicate V13 call population")
    original_calls = _json(source_read("data/calls.json"))
    rows = {(r["episode_index"], r["step"], r["agent"]): r for r in original_calls if r["split"] == "train" and "arrays_path" in r}
    if len(rows) != protocol["eligible_calls"]:
        raise ValueError("Pinned V12 TRAIN population mismatch")
    checked = []
    for record in records:
        key = (record["episode_index"], record["step"], record["agent"])
        if key not in rows or record["split"] != "train" or not record["original_branch_fields_replay_equal"] or not record["anchor_repeat_equal"]:
            raise ValueError("V13 record provenance mismatch")
        row = rows[key]
        if record["group"] != row["group"]:
            raise ValueError("V13 group provenance mismatch")
        original_values = arrays(source_read("data/" + row["arrays_path"]))
        private = arrays(read(f"{stage}/{record['mechanism_path']}"))
        _validate_private_labels(read, stage, record, private, original_values)
        expected_metrics = call_metrics(original_values, private, protocol)
        if expected_metrics != record["metrics"]:
            raise ValueError("Independent V13 mechanism metrics mismatch")
        checked.append(record)
    expected_summary = summarize(checked, protocol)
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise ValueError(f"V13 aggregate mismatch: {key}")
    episodes = summary["episodes"]
    scenes = [_json(line) for line in source_read("data/scenes.jsonl").decode().splitlines()]
    expected_episode_ids = {scene["episode_index"] for scene in scenes if scene["model_split"] == "train"}
    if len(episodes) != protocol["episodes"] or {row["episode_index"] for row in episodes} != expected_episode_ids:
        raise ValueError("Incomplete V13 episode population")
    for episode in episodes:
        name = f"{stage}/trajectories/{episode['episode_index']}.npz"
        replay = arrays(read(name))
        original = arrays(source_read(f"data/observed/{episode['episode_index']}.npz"))
        if not compare_arrays(replay, original) or not episode["original_arrays_equal"] or sha_bytes(read(name)) != episode["trajectory_sha256"]:
            raise ValueError("V13 original trajectory replay mismatch")
    if summary["calls"] != len(checked) or summary["groups"] != 24:
        raise ValueError("V13 population aggregate mismatch")
    return summary, records


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def audit_archive(path, source_path, capsule_path):
    if sha(source_path) != SOURCE_SHA or sha(capsule_path) != CAPSULE_SHA:
        raise ValueError("Pinned V12 source or baseline changed")
    integrity = check_archive(path, "ARTIFACT_MANIFEST.json", False)
    check_archive(source_path, "ARTIFACT_MANIFEST.json", False)
    protocol = _json((ROOT / "protocol.json").read_bytes())
    with zipfile.ZipFile(path) as archive, zipfile.ZipFile(source_path) as source:
        read = archive.read
        source_read = source.read
        primary, primary_records = audit_stage(read, "primary", protocol, source_read)
        repeated, repeated_records = audit_stage(read, "repeated", protocol, source_read)
        if not _same_json_without_elapsed(primary, repeated) or primary_records != repeated_records:
            raise ValueError("Independent V13 replays differ")
        for name in archive.namelist():
            if name.startswith("primary/"):
                twin = "repeated/" + name[len("primary/"):]
                if name.endswith("summary.json"):
                    continue
                if twin not in archive.namelist() or archive.read(name) != archive.read(twin):
                    raise ValueError(f"Independent replay bytes differ: {name}")
    return {"status": "passed_independent_train_mechanism_recomputation", "artifact_sha256": integrity["sha256"],
            "artifact_members_verified": integrity["members_verified"], "calls": primary["calls"], "episodes": primary["groups"] * 2,
            "true_branch_flip_pairs": primary["true_branch_flip_pairs"], "cbf_changed_command_points": primary["cbf_changed_command_points"],
            "private_mechanism_labels_network_inputs": False, "enhanced_control_enabled": False, "new_model_training_enabled": False,
            "holdout_used": False, "scope": "TRAIN-only mechanism diagnosis; no model qualification, generalization, safety certificate, or closed-loop control claim."}


def audit_archive_from_members(members, source_members):
    """Test-only in-memory equivalent of ``audit_archive`` without zip I/O."""
    protocol = _json((ROOT / "protocol.json").read_bytes())
    read = members.__getitem__
    source_read = source_members.__getitem__
    primary, primary_records = audit_stage(read, "primary", protocol, source_read)
    repeated, repeated_records = audit_stage(read, "repeated", protocol, source_read)
    if not _same_json_without_elapsed(primary, repeated) or primary_records != repeated_records:
        raise ValueError("Independent V13 replays differ")
    for name in members:
        if name.startswith("primary/") and not name.endswith("summary.json"):
            twin = "repeated/" + name[len("primary/"):]
            if twin not in members or members[name] != members[twin]:
                raise ValueError(f"Independent replay bytes differ: {name}")
    return {"status": "passed_independent_train_mechanism_recomputation", "calls": primary["calls"],
            "true_branch_flip_pairs": primary["true_branch_flip_pairs"],
            "private_mechanism_labels_network_inputs": False}


def package(primary_path, repeated_path, source_path, capsule_path):
    if sha(source_path) != SOURCE_SHA or sha(capsule_path) != CAPSULE_SHA:
        raise ValueError("Pinned V12 source or baseline changed")
    check_archive(source_path, "ARTIFACT_MANIFEST.json", False)
    members = {}
    for stage, run in (("primary", primary_path), ("repeated", repeated_path)):
        for file in run.rglob("*"):
            relative = file.relative_to(run)
            if file.is_file() and "restored" not in relative.parts:
                members[f"{stage}/{relative.as_posix()}"] = file.read_bytes()
    for file in ROOT.glob("*.py"):
        members[f"verification_source/{file.name}"] = file.read_bytes()
    audit_archive_from_members(members, _source_members(source_path))
    path = ROOT / "artifacts/mechanism_diagnosis_20261010.zip"
    path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        archive.writestr("ARTIFACT_MANIFEST.json", json.dumps({name: sha_bytes(raw) for name, raw in members.items()}, indent=2))
    result = audit_archive(path, source_path, capsule_path)
    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    with (report_dir / "release_manifest.json").open("x") as file:
        json.dump({**result, "artifact_bytes": path.stat().st_size}, file, indent=2)
    return result


def _source_members(source_path):
    if sha(source_path) != SOURCE_SHA:
        raise ValueError("Pinned V12 source changed")
    with zipfile.ZipFile(source_path) as source:
        return {name: source.read(name) for name in source.namelist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--repeated", type=Path)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "cwm_v12/artifacts/task_effect_training_20261010.zip")
    parser.add_argument("--capsule", type=Path, default=ROOT.parent / "cwm_v1/baseline/capsule.zip")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(audit_archive(args.verify, args.source, args.capsule), indent=2), flush=True)
        return
    if args.primary is None or args.repeated is None:
        parser.error("Completed primary and repeated V13 replays are required")
    print(json.dumps(package(args.primary, args.repeated, args.source, args.capsule), indent=2), flush=True)


if __name__ == "__main__":
    main()
