"""Publish/verify full recomputation of the score audit, not stored claims."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from audit_task_score import analyze, frozen_configs, sha, check_archive


def verify_records(read, stage, records, summary):
    stored = json.loads(read(stage + "/records.json"))
    measured = json.loads(read(stage + "/summary.json"))
    if stored != records:
        raise ValueError("Independent full cost/gradient/train-effect recomputation differs")
    for key, value in summary.items():
        if measured.get(key) != value:
            raise ValueError("Independent score audit summary differs: " + key)
    if any(measured[k] for k in ("enhanced_control_enabled", "new_model_training_enabled", "holdout_used", "prior_gate_overridden")):
        raise ValueError("Offline audit contract overridden")
    for name, digest in measured["source_hashes"].items():
        captured = read(stage + "/source/" + name)
        if hashlib.sha256(captured).hexdigest() != digest or sha(ROOT.parent / name) != digest:
            raise ValueError("Run-used source closure mismatch")
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "cwm_v10/artifacts/two_head_training_20261010.zip")
    parser.add_argument("--capsule", type=Path, default=ROOT.parent / "cwm_v1/baseline/capsule.zip")
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--repeated", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if sha(args.source) != protocol["training_archive_sha256"] or sha(args.capsule) != protocol["baseline_capsule_sha256"]:
        raise ValueError("Pinned protected audit inputs differ")
    if any(protocol[k] for k in ("new_model_training_enabled", "enhanced_control_enabled", "holdout_used", "prior_gate_override_allowed")):
        raise ValueError("Offline fixed audit contract mismatch")
    check_archive(args.source, "ARTIFACT_MANIFEST.json", False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configs = frozen_configs(args.capsule, ROOT.parent.parent / "results/cwm_v11/verify_restored")
    with zipfile.ZipFile(args.source) as source:
        records, summary = analyze(source, configs, protocol)
    if args.verify:
        integrity = check_archive(args.verify, "ARTIFACT_MANIFEST.json", False)
        with zipfile.ZipFile(args.verify) as archive:
            primary = verify_records(archive.read, "primary", records, summary)
            repeated = verify_records(archive.read, "repeated", records, summary)
            if primary["protocol"] != protocol or repeated["protocol"] != protocol:
                raise ValueError("Predeclared audit protocol differs")
        print(json.dumps({"status": "passed_independent_full_recomputation", "calls": len(records), "artifact_sha256": integrity["sha256"]}))
        return
    if args.primary is None or args.repeated is None:
        parser.error("Two completed independent audit processes required")
    members = {}
    for stage, run in (("primary", args.primary), ("repeated", args.repeated)):
        for name in ("records.json", "summary.json"):
            members[stage + "/" + name] = (run / name).read_bytes()
        measured = json.loads(members[stage + "/summary.json"])
        if measured["protocol"] != protocol:
            raise ValueError("Run-used protocol differs")
        for name in measured["source_hashes"]:
            members[stage + "/source/" + name] = (run / "source" / name).read_bytes()
        verify_records(members.__getitem__, stage, records, summary)
    if members["primary/records.json"] != members["repeated/records.json"]:
        raise ValueError("Repeated full audits differ")
    members["verification_source.py"] = Path(__file__).read_bytes()
    path = ROOT / "artifacts/full_local_score_audit_20261010.zip"
    path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        archive.writestr("ARTIFACT_MANIFEST.json", json.dumps({name: hashlib.sha256(raw).hexdigest() for name, raw in members.items()}, indent=2))
    check_archive(path, "ARTIFACT_MANIFEST.json", False)
    report = {"status": "passed_independent_full_recomputation", "calls": len(records), "repeated_records_byte_equal": True,
              "artifact_sha256": sha(path), "artifact_bytes": path.stat().st_size,
              "scope": "Recompute seven full-cost methods and original finite-difference gradients on all V10 calls, and TRAIN ONLY exact/linear/learned response attribution from pinned data and reloaded weights. No control qualification."}
    (ROOT / "reports").mkdir(exist_ok=True)
    with (ROOT / "reports/summary.json").open("xb") as file:
        file.write(members["primary/summary.json"])
    with (ROOT / "reports/release_manifest.json").open("x") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
