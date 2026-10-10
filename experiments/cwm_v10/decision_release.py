"""Verify paired descriptive diagnostics against the fixed V10 training archive."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cwm_v8"))
from s4_value import descriptive_bootstrap, rank_metrics
from geometry_release import sha
from verify_release import check_archive

METHODS = ("original_gru", "constant_velocity", "trained_motion_only", "plain_motion_component", "structured_motion_component", "plain_two_head", "structured_two_head", "reference_truth", "action_specific_truth")


def recompute(records):
    groups = [r["group"] for r in records]
    result = {}
    for method in METHODS:
        result[method] = {"mean_diagnostic_regret": float(np.mean([r["metrics"][method]["regret_in_diagnostic_cost"] for r in records]))}
        for name, reference in (("gru", "original_gru"), ("cv", "constant_velocity"), ("trained_motion_only", "trained_motion_only")):
            values = [r["metrics"][reference]["regret_in_diagnostic_cost"] - r["metrics"][method]["regret_in_diagnostic_cost"] for r in records]
            result[method]["gain_vs_" + name] = descriptive_bootstrap(values, groups, 2000, 970102)
    response = {}
    for kind in ("plain", "structured"):
        values = [r["metrics"][kind + "_motion_component"]["regret_in_diagnostic_cost"] - r["metrics"][kind + "_two_head"]["regret_in_diagnostic_cost"] for r in records]
        response[kind] = {"choice_changes": sum(r["metrics"][kind + "_motion_component"]["choice"] != r["metrics"][kind + "_two_head"]["choice"] for r in records),
                          "gain": descriptive_bootstrap(values, groups, 2000, 970102)}
    return result, response


def audit(read, training_read):
    data_records = json.loads(training_read("data/calls.json"))
    development = [r for r in data_records if r["split"] == "development_validation" and "arrays_path" in r]
    expected = {(r["episode_index"], r["step"], r["agent"]): r for r in development if r["all_candidates_full_horizon_valid"]}
    training = json.loads(training_read("primary/summary.json"))
    summaries = []
    for stage in ("primary", "repeated"):
        summary = json.loads(read(stage + "/summary.json"))
        if (summary["enhanced_control_enabled"] or summary["holdout_used"] or summary["prior_training_gate_overridden"]
                or summary["prior_training_gate_passed"] != training["development_gate_passed"]
                or summary["selected_median_seeds"] != training["chosen_median_seed_per_architecture"]
                or summary["data_summary_sha256"] != hashlib.sha256(training_read("data/summary.json")).hexdigest()
                or summary["training_summary_sha256"] != hashlib.sha256(training_read("primary/summary.json")).hexdigest()):
            raise ValueError("Frozen diagnostic training/gate contract mismatch")
        records = json.loads(read(stage + "/records.json"))
        keys = [(r["episode_index"], r["step"], r["agent"]) for r in records]
        if len(records) != len(expected) or len(set(keys)) != len(keys) or set(keys) != set(expected):
            raise ValueError("Actual diagnostic joint support population mismatch")
        for row, key in zip(records, keys):
            original = expected[key]
            if row["group"] != original["group"] or set(row["costs"]) != set(METHODS):
                raise ValueError("Actual diagnostic information/group mismatch")
            for method in METHODS:
                cost = np.asarray(row["costs"][method])
                if cost.shape != (original["candidate_count"],) or not np.isfinite(cost).all():
                    raise ValueError("Actual finite candidate-cost contract mismatch")
                if rank_metrics(cost, row["costs"]["action_specific_truth"]) != row["metrics"][method]:
                    raise ValueError("Actual diagnostic ranking mismatch")
            for method in ("original_gru", "constant_velocity", "action_specific_truth"):
                if not np.allclose(row["costs"][method], original["costs"][method], rtol=1e-10, atol=1e-8):
                    raise ValueError("Original frozen reference diagnostic costs changed")
        methods, response = recompute(records)
        if summary["methods"] != methods or summary["learned_response_increment"] != response or summary["development_calls"] != len(development) or summary["rankable_calls"] != len(records):
            raise ValueError("Actual diagnostic summary/attribution mismatch")
        summaries.append(summary)
    if read("primary/records.json") != read("repeated/records.json") or read("primary/source.py") != read("repeated/source.py"):
        raise ValueError("Independent diagnostic records/source differ")
    for key in summaries[0].keys() - {"elapsed_seconds"}:
        if summaries[0][key] != summaries[1][key]:
            raise ValueError("Independent diagnostic summary differs")
    return {"status": "passed", "rankable_calls": len(expected), "development_calls": len(development), "independent_records_byte_equal": True,
            "prior_gate_overridden": False, "enhanced_control_enabled": False, "holdout_used": False,
            "scope": "Recompute ranks/attribution from paired recorded scores on actual fixed support and compare frozen reference scores. New-head scores are reproduced by the two diagnostic runs, not independently recomputed by this archive verifier."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, default=ROOT / "artifacts/two_head_training_20261010.zip")
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--repeated", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if sha(args.training) != "4c0f138df0797c89dab7c5132ceb8b91664df40ac98f3ef7dd8fca748ac9356e":
        raise ValueError("Frozen V10 source archive changed")
    check_archive(args.training, "ARTIFACT_MANIFEST.json", False)
    with zipfile.ZipFile(args.training) as training:
        if args.verify:
            integrity = check_archive(args.verify, "ARTIFACT_MANIFEST.json", False)
            with zipfile.ZipFile(args.verify) as archive:
                result = audit(archive.read, training.read)
            print(json.dumps({**result, "artifact_sha256": integrity["sha256"]}))
            return
        if args.primary is None or args.repeated is None:
            parser.error("Two frozen completed diagnostics required")
        members = {f"{stage}/{name}": (run / name).read_bytes() for stage, run in (("primary", args.primary), ("repeated", args.repeated)) for name in ("records.json", "summary.json", "source.py")}
        members["verification_source.py"] = Path(__file__).read_bytes()
        result = audit(members.__getitem__, training.read)
        path = ROOT / "artifacts/two_head_decision_20261010.zip"
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in members.items():
                archive.writestr(name, raw)
            archive.writestr("ARTIFACT_MANIFEST.json", json.dumps({n: hashlib.sha256(v).hexdigest() for n, v in members.items()}, indent=2))
        check_archive(path, "ARTIFACT_MANIFEST.json", False)
    with (ROOT / "reports/decision_summary.json").open("xb") as file:
        file.write((args.primary / "summary.json").read_bytes())
    with (ROOT / "reports/decision_release_manifest.json").open("x") as file:
        json.dump({**result, "artifact_sha256": sha(path), "artifact_bytes": path.stat().st_size}, file, indent=2)
    print(json.dumps({**result, "artifact_sha256": sha(path), "artifact_bytes": path.stat().st_size}))


if __name__ == "__main__":
    main()
