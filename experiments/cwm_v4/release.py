"""Audit actual asymmetric data and baseline trajectories, then archive evidence."""
import argparse
import hashlib
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np

from diagnose import ROOT, commands, summarize
from analyze_latency import analyze
from freeze_baseline import sha
from verify_release import check_archive


def rows(data):
    return [json.loads(line) for line in data.decode().splitlines() if line]


def audit(read):
    report = json.loads(read("run/summary.json"))
    protocol = json.loads(read("source/cwm_v4/protocol.json"))
    if report["enhanced_control_enabled"] or report["excluded_group_overlap"] or report["protocol"] != protocol:
        raise ValueError("Protocol or controller mismatch")
    for name, expected in report["source_hashes"].items():
        member = "source/" + name.replace("\\", "/")
        if hashlib.sha256(read(member)).hexdigest() != expected:
            raise ValueError("Experiment source mismatch")
    training_bytes = read("inputs/train/scenes.jsonl")
    if hashlib.sha256(training_bytes).hexdigest() != report["training_scenes_sha256"]:
        raise ValueError("Training pool hash mismatch")
    training = {r["episode_index"]: r for r in rows(training_bytes)}
    exclusions = json.loads(read("inputs/exclusion_manifest.json"))
    forbidden = set()
    for item in exclusions:
        raw = read(item["member"])
        if hashlib.sha256(raw).hexdigest() != report["exclusion_sha256"][item["original_path"]]:
            raise ValueError("Exclusion hash mismatch")
        forbidden |= {str(r["mirror_group_id"]) for r in rows(raw)}
    selected = rows(read("run/selected_scenes.jsonl"))
    groups = {str(r["mirror_group_id"]) for r in selected}
    if groups & forbidden or len(groups) != len(selected) or len(selected) != report["episodes"]:
        raise ValueError("Scene population or partition mismatch")
    selected_ids = {r["episode_index"] for r in selected}
    if len(selected_ids) != len(selected):
        raise ValueError("Duplicate selected episode")
    for variant in protocol["variants"]:
        if sum(r["variant"] == variant for r in selected) != protocol["groups_per_variant"]:
            raise ValueError("Predeclared per-variant population mismatch")
    for r in selected:
        origin = training[r["episode_index"]]
        if any(origin.get(k) != v for k, v in r.items() if k not in ("model_split", "level")):
            raise ValueError("Selected scene altered from original TRAIN")
        if r["model_split"] != "scout_train_only" or r["level"] != protocol["variants"][r["variant"]]:
            raise ValueError("Scene annotation mismatch")
    raw = read("run/pairs.npz")
    if hashlib.sha256(raw).hexdigest() != report["dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    n = report["states"]
    if data["target"].shape != (n, 13, 24, 3) or data["history"].shape != (n, 8, 252):
        raise ValueError("Data contract mismatch")
    for value in data.values():
        if len(value) != n or (np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all()):
            raise ValueError("Population or finite-value check failed")
    if set(data["group"].tolist()) != groups:
        raise ValueError("Dataset group mismatch")
    if data["proposed"].shape != (n, 13, 24, 4, 3) or np.linalg.norm(data["proposed"], axis=-1).max() > protocol["maximum_command_speed_mps"] + 1e-8:
        raise ValueError("Intervention command contract mismatch")
    for index in range(n):
        rebuilt = commands(data["proposed"][index, 0, :8],
                           {"defender_positions": data["relative"][index, :, :3] + data["reference"][index]},
                           data["reference"][index], protocol)
        if not np.allclose(rebuilt, data["proposed"][index], atol=1e-12, rtol=0.):
            raise ValueError("Commands differ from predeclared public intervention")
    if (data["valid"][:, :, 1:] & ~data["valid"][:, :, :-1]).any():
        raise ValueError("Resumed simulation after censoring")
    if (data["valid"] & (data["termination"] == "after_terminal")).any():
        raise ValueError("Post-terminal valid point")
    if summarize(data, protocol) != report["variants"]:
        raise ValueError("Reported statistics differ from actual arrays")
    if json.loads(read("run/latency_analysis.json")) != analyze(data, protocol):
        raise ValueError("Reported response latency differs from actual arrays")
    windows = json.loads(read("run/windows.json"))
    if len(windows) != n or not all(w["parent_integrity"] and w["repeat_exact"] for w in windows):
        raise ValueError("Window evidence mismatch")
    episode_to_group = {r["episode_index"]: str(r["mirror_group_id"]) for r in selected}
    for index, w in enumerate(windows):
        if w["step"] not in protocol["snapshot_steps"] or w["step"] != int(data["step"][index]) or w["group"] != episode_to_group[w["episode_index"]] or w["group"] != data["group"][index]:
            raise ValueError("Window provenance mismatch")
    episodes = json.loads(read("run/episodes.json"))
    if episodes != report["episodes_results"] or len(episodes) != len(selected):
        raise ValueError("Episode evidence mismatch")
    if {r["episode_index"] for r in episodes} != selected_ids:
        raise ValueError("Episode invariance coverage mismatch")
    for row in episodes:
        values = []
        for kind, key in (("observed", "trajectory_sha256"), ("unobserved", "unobserved_sha256")):
            raw = read(f"run/{kind}/{row['episode_index']}.npz")
            if hashlib.sha256(raw).hexdigest() != row[key]:
                raise ValueError("Trajectory archive hash mismatch")
            with np.load(io.BytesIO(raw), allow_pickle=False) as trajectory:
                values.append({k: trajectory[k] for k in ("target_positions", "defender_positions")})
        if not row["trajectories_byte_equal"] or not row["outcomes_equal"] or any(values[0][k].shape != values[1][k].shape or values[0][k].dtype != values[1][k].dtype or values[0][k].tobytes() != values[1][k].tobytes() for k in values[0]):
            raise ValueError("Actual baseline trajectory invariance failed")
    if len(groups) != report["groups"]:
        raise ValueError("Independent group count mismatch")
    candidates = []
    gate = protocol["minimum_next_stage_signal"]
    for variant, horizons in report["variants"].items():
        for horizon, metrics in horizons.items():
            if (metrics["fraction_over_0_05m"] or 0) >= gate["effect_over_0_05m_valid_point_fraction"] and len(metrics["signal_groups"]) >= gate["independent_scene_groups_with_effect_over_0_05m"]:
                candidates.append({"variant": variant, "horizon": int(horizon)})
    if candidates != report["signal_candidates"] or report["status"] != ("signal_requires_independent_confirmation" if candidates else "data_adequacy_no_go_keep_baseline"):
        raise ValueError("Screen decision mismatch")
    return {"status": "passed", "states": n, "episodes": len(selected), "groups": len(groups),
            "actual_trajectory_byte_equality": True, "original_train_provenance": True, "exclusion_overlap": 0,
            "statistics_recomputed": True, "scope": "Exploratory TRAIN diagnostic and observer isolation, not closed-loop enhancement safety."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--training-scenes", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        integrity = check_archive(args.verify, "ARTIFACT_MANIFEST.json", False)
        with zipfile.ZipFile(args.verify) as archive:
            result = audit(archive.read)
            recorded_capsule = json.loads(archive.read("run/summary.json"))["capsule_sha256"]
        capsule = ROOT.parent / "cwm_v1/baseline/capsule.zip"
        if sha(capsule) != recorded_capsule:
            raise ValueError("Protected baseline capsule differs from experiment")
        print(json.dumps({**result, "archive_sha256": integrity["sha256"]}))
        return
    if args.run is None or args.training_scenes is None:
        parser.error("--run and --training-scenes required to package")
    report = json.loads((args.run / "summary.json").read_text())
    capsule = ROOT.parent / "cwm_v1/baseline/capsule.zip"
    if sha(capsule) != report["capsule_sha256"]:
        raise ValueError("Protected capsule changed")
    members = {f"run/{p.relative_to(args.run).as_posix()}": p.read_bytes()
               for p in args.run.rglob("*") if p.is_file() and "restored" not in p.relative_to(args.run).parts and p.name != "pairs_partial.npz"}
    members["inputs/train/scenes.jsonl"] = args.training_scenes.read_bytes()
    exclusions = []
    for index, path in enumerate(report["exclusion_sha256"]):
        member = f"inputs/excluded/{index}.jsonl"
        members[member] = Path(path).read_bytes()
        exclusions.append({"original_path": path, "member": member})
    members["inputs/exclusion_manifest.json"] = json.dumps(exclusions, indent=2).encode()
    for name in report["source_hashes"]:
        normalized = name.replace("\\", "/")
        members[f"source/{normalized}"] = (ROOT.parent / normalized).read_bytes()
    for path in ROOT.glob("*.py"):
        members[f"source/cwm_v4/{path.name}"] = path.read_bytes()
    members["source/cwm_v1/verify_release.py"] = (ROOT.parent / "cwm_v1/verify_release.py").read_bytes()
    result = audit(members.__getitem__)
    artifact = ROOT / "artifacts/asymmetric_diagnostic_20261009.zip"
    artifact.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(artifact, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        archive.writestr("ARTIFACT_MANIFEST.json", json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in members.items()}, indent=2))
    check_archive(artifact, "ARTIFACT_MANIFEST.json", False)
    with zipfile.ZipFile(artifact) as archive:
        audit(archive.read)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    if (reports / "diagnostic_summary.json").exists():
        raise FileExistsError("Never overwrite a published milestone")
    shutil.copy2(args.run / "summary.json", reports / "diagnostic_summary.json")
    with (reports / "release_manifest.json").open("x") as stream:
        json.dump({**result, "diagnostic_status": report["status"], "enhanced_control_enabled": False,
                   "artifact_sha256": sha(artifact), "artifact_bytes": artifact.stat().st_size,
                   "baseline_capsule_sha256": sha(capsule)}, stream, indent=2)
    print(json.dumps({**result, "diagnostic_status": report["status"], "artifact_bytes": artifact.stat().st_size}))


if __name__ == "__main__":
    main()
