"""Audit development support, raw ranks and two unchanged-control replays."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cwm_v7")]
from s4_value import METHODS, rank_metrics, summarize
from training_release import audit as audit_training, audit_data
from geometry_release import BASELINE_SHA, arrays, compare_arrays, sha, target_bad
from verify_release import check_archive


def audit_runs(read, source_read):
    data, selected, _, data_report = audit_data(source_read)
    indices = selected["development_validation"]
    source_windows = json.loads(source_read("data/windows.json"))
    scenes = [json.loads(line) for line in source_read("data/selected_scenes.jsonl").decode().splitlines()]
    scenes = {r["episode_index"]: r for r in scenes if r["model_split"] == "development_validation"}
    expected = {(source_windows[i]["episode_index"], source_windows[i]["step"]): int(i) for i in indices}
    source_episodes = {r["episode_index"]: r for r in data_report["episodes_results"]}
    training = json.loads(source_read("primary/summary.json"))
    summaries = []
    if len(indices) != 80 or len(scenes) != 16 or len(expected) != len(indices):
        raise ValueError("Actual development population mismatch")
    for stage in ("primary", "repeated"):
        summary = json.loads(read(stage + "/summary.json"))
        protocol = summary["protocol"]
        if (summary["capsule_sha256"] != BASELINE_SHA or summary["source_archive_sha256"] != protocol["source_archive_sha256"]
                or summary["enhanced_control_enabled"] or summary["holdout_used"] or summary["prior_training_gate_overridden"]
                or protocol["enhanced_control_enabled"] or protocol["holdout_used"] or protocol["new_model_training_enabled"]
                or protocol["prior_training_gate_override_allowed"] or protocol["split"] != "development_validation"
                or protocol["horizon_steps"] != 8 or protocol["candidate_count"] != 8
                or protocol["information_conditions"] != list(METHODS)):
            raise ValueError("Frozen development/disabled-control contract mismatch")
        if (summary["prior_training_gate_passed"] != training["development_gate_passed"]
                or protocol["selected_median_seeds"] != training["chosen_median_seed_per_architecture"]
                or not summary["chosen_model_outputs_reloaded_byte_equal"]):
            raise ValueError("Prior gate/model selection mismatch")
        for name, digest in summary["source_hashes"].items():
            if hashlib.sha256(read(f"{stage}/source/{name}")).hexdigest() != digest:
                raise ValueError("Actual value source digest mismatch")
        if json.loads(read(stage + "/source/cwm_v8/protocol.json")) != protocol:
            raise ValueError("Actual value protocol mismatch")
        records = json.loads(read(stage + "/windows.json"))
        keys = [(r["episode_index"], r["step"]) for r in records]
        if len(records) != len(expected) or len(set(keys)) != len(keys) or set(keys) != set(expected):
            raise ValueError("Actual development window population mismatch")
        for row in records:
            index = expected[row["episode_index"], row["step"]]
            scene = scenes[row["episode_index"]]
            valid = bool(data["valid"][index, :, :8].all())
            if (row["all_candidates_full_horizon_valid"] != valid or row["group"] != scene["mirror_group_id"]
                    or row["variant"] != scene["variant"] or not row["parent_integrity"]
                    or not row["model_public_context_matches_current_replay"]):
                raise ValueError("Actual support/provenance mismatch")
            if not valid:
                if "metrics" in row or "costs" in row:
                    raise ValueError("Incomplete window was scored")
                continue
            if (not row["reference_score_matches_original_diagnostics"] or set(row["costs"]) != set(METHODS)
                    or set(row["metrics"]) != set(METHODS)):
                raise ValueError("Incomplete information condition")
            truth_cost = np.asarray(row["costs"]["action_specific_truth"])
            for method in METHODS:
                cost = np.asarray(row["costs"][method])
                if cost.shape != (8,) or not np.isfinite(cost).all():
                    raise ValueError("Actual finite cost contract mismatch")
                if rank_metrics(cost, truth_cost, protocol["tie_absolute_cost_tolerance"]) != row["metrics"][method]:
                    raise ValueError("Actual cost/ranking mismatch")
        if summarize(records, protocol) != summary["statistics"]:
            raise ValueError("Actual group statistics/signal mismatch")
        episodes = summary["episodes"]
        if len(episodes) != len(scenes) or {r["episode_index"] for r in episodes} != set(scenes):
            raise ValueError("Actual replay episode population mismatch")
        outcomes = ("safe_capture_success", "collision", "boundary_violation", "timeout", "target_invalid_episode", "termination_reason")
        for row in episodes:
            index = row["episode_index"]
            raw = read(f"{stage}/trajectories/{index}.npz")
            if hashlib.sha256(raw).hexdigest() != row["trajectory_sha256"]:
                raise ValueError("Actual replay digest mismatch")
            a, b = arrays(raw), arrays(source_read(f"data/observed/{index}.npz"))
            if not row["trajectory_byte_equal"] or not compare_arrays(a, b):
                raise ValueError("Actual original replay arrays differ")
            if bool(target_bad(a["target_positions"], scenes[index]["scenario"]).any()) != row["target_invalid_episode"]:
                raise ValueError("Actual replay physics mismatch")
            if any(row[k] != source_episodes[index][k] for k in outcomes):
                raise ValueError("Actual original replay outcomes differ")
        summaries.append(summary)
    if read("primary/windows.json") != read("repeated/windows.json"):
        raise ValueError("Independent value record bytes differ")
    for key in ("status", "protocol", "statistics", "source_hashes", "source_archive_sha256", "capsule_sha256", "limitations",
                "prior_training_gate_passed", "prior_training_gate_overridden", "chosen_model_outputs_reloaded_byte_equal"):
        if summaries[0][key] != summaries[1][key]:
            raise ValueError("Independent value summary differs")
    stats = summaries[0]["statistics"]
    return {"status": "passed", "independent_runs": 2, "development_episodes": 16, "development_groups": 8,
            "windows": 80, "rankable_windows": stats["rankable_windows"], "ranking_records_byte_equal": True,
            "original_replay_arrays_byte_equal": True, "motion_research_signal": stats["motion_research_signal"],
            "response_research_signal_given_motion": stats["response_research_signal_given_motion"],
            "prior_training_gate_passed": training["development_gate_passed"], "holdout_used": False, "enhanced_control_enabled": False,
            "scope": "Recompute ranks/group statistics from recorded full-MPC costs, verify actual archived support/trajectories and identical independent replay; not independent recalculation of MPC scores, closed-loop gains or universal zero value."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "cwm_v7/artifacts/paired_training_20261010.zip")
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--repeated", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if sha(args.source) != protocol["source_archive_sha256"] or sha(ROOT.parent / "cwm_v1/baseline/capsule.zip") != BASELINE_SHA:
        raise ValueError("Protected source/baseline changed")
    check_archive(args.source, "ARTIFACT_MANIFEST.json", False)
    with zipfile.ZipFile(args.source) as source:
        training_result = audit_training(source.read)
        if args.verify:
            integrity = check_archive(args.verify, "ARTIFACT_MANIFEST.json", False)
            with zipfile.ZipFile(args.verify) as archive:
                result = audit_runs(archive.read, source.read)
            print(json.dumps({**result, "source_training_audit": training_result, "artifact_sha256": integrity["sha256"]}))
            return
        if args.primary is None or args.repeated is None:
            parser.error("Both completed value runs required")
        members = {}
        for stage, run in (("primary", args.primary), ("repeated", args.repeated)):
            for path in run.rglob("*"):
                relative = path.relative_to(run)
                if path.is_file() and "restored" not in relative.parts:
                    members[f"{stage}/{relative.as_posix()}"] = path.read_bytes()
        for path in ROOT.glob("*.py"):
            members["verification_source/" + path.name] = path.read_bytes()
        result = audit_runs(members.__getitem__, source.read)
        path = ROOT / "artifacts/s4_decision_value_20261010.zip"
        path.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in members.items():
                archive.writestr(name, raw)
            archive.writestr("ARTIFACT_MANIFEST.json", json.dumps({n: hashlib.sha256(v).hexdigest() for n, v in members.items()}, indent=2))
        integrity = check_archive(path, "ARTIFACT_MANIFEST.json", False)
        with zipfile.ZipFile(path) as archive:
            audit_runs(archive.read, source.read)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    with (reports / "value_summary.json").open("xb") as file:
        file.write((args.primary / "summary.json").read_bytes())
    with (reports / "value_release_manifest.json").open("x") as file:
        json.dump({**result, "source_training_audit": training_result, "artifact_sha256": integrity["sha256"], "artifact_bytes": path.stat().st_size}, file, indent=2)
    print(json.dumps({**result, "artifact_sha256": integrity["sha256"], "artifact_bytes": path.stat().st_size}))


if __name__ == "__main__":
    main()
