"""Archive completed scout data and hash-matched source; reject unverified releases."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "cwm_v1"))
from freeze_baseline import sha


def matched_source(root, name, expected):
    paths = [root / name, root.parent / "cwm_v1" / name]
    for path in paths:
        if path.is_file():
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() == expected:
                return data
            if name == "scout.py":
                # Scout #1 predates the three diagnostic-only metadata fields.
                lines = data.splitlines(keepends=True)
                old = b"".join(line for line in lines if not any(token in line for token in
                    (b'"private_target_mode_diagnostic_only":', b'"private_target_route_diagnostic_only":',
                     b'"steps_since_target_replan_diagnostic_only":')))
                if hashlib.sha256(old).hexdigest() == expected:
                    return old
    raise ValueError(f"Cannot reproduce recorded source bytes: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scout", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--tap-audit", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    audit = json.loads((args.tap_audit / "summary.json").read_text())
    if audit["status"] != "passed":
        raise ValueError("Backbone tap invariance not verified")
    reports_data, content, known_groups = {}, {}, set()
    for stage, run in (("scout", args.scout), ("confirmation", args.confirmation)):
        summary = json.loads((run / "summary.json").read_text())
        if summary["status"] != "scout_complete_not_online_promoted" or summary["original_validation_and_v1_group_overlap"] != 0:
            raise ValueError("Scout audit incomplete")
        if sha(run / "pairs.npz") != summary["dataset_sha256"]:
            raise ValueError("Dataset integrity failure")
        if sha(root.parent / "cwm_v1/baseline/capsule.zip") != summary["capsule_sha256"]:
            raise ValueError("Baseline capsule changed")
        with np.load(run / "pairs.npz", allow_pickle=False) as data:
            if len(data["history"]) != summary["states"]:
                raise ValueError("Dataset population mismatch")
            for key in data.files:
                if np.issubdtype(data[key].dtype, np.number) and not np.isfinite(data[key]).all():
                    raise ValueError("Nonfinite scout data")
            groups = set(data["group"].tolist())
        if groups & known_groups:
            raise ValueError("Confirmation group overlap")
        known_groups |= groups
        selected = [json.loads(line) for line in (run / "selected_scenes.jsonl").read_text().splitlines() if line]
        if len(selected) != summary["episodes"] or {str(r["mirror_group_id"]) for r in selected} != groups:
            raise ValueError("Scene provenance mismatch")
        reports_data[stage] = summary
        for name in ("pairs.npz", "summary.json", "selected_scenes.jsonl", "windows.json"):
            content[f"{stage}/{name}"] = (run / name).read_bytes()
        for name, expected in summary["source_hashes"].items():
            content[f"{stage}/source/{name}"] = matched_source(root, name, expected)
    content["tap_audit.json"] = json.dumps(audit, indent=2).encode()
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    destination = artifacts / "level7_8_scout_20261009.zip"
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in content.items():
            archive.writestr(name, data)
        archive.writestr("ARTIFACT_MANIFEST.json", json.dumps({k: hashlib.sha256(v).hexdigest() for k, v in content.items()}, indent=2))
    for stage, summary in reports_data.items():
        with (reports / f"{stage}_summary.json").open("x") as stream:
            json.dump(summary, stream, indent=2)
    with (reports / "tap_regression.json").open("x") as stream:
        json.dump(audit, stream, indent=2)
    adequate_variants = []
    for variant, value in reports_data["confirmation"]["variants"].items():
        thresholds = reports_data["confirmation"]["protocol"]["minimum_next_stage_signal"]
        if value["fraction_over_0_05m"] >= thresholds["effect_over_0_05m_valid_point_fraction"] and len(value["signal_groups"]) >= thresholds["independent_scene_groups_with_effect_over_0_05m"]:
            adequate_variants.append(variant)
    release = {"status": "adequacy_pass_not_promoted" if adequate_variants else "data_adequacy_no_go_keep_baseline",
               "adequate_variants": adequate_variants, "enhanced_control_enabled": False,
               "artifact_sha256": sha(destination), "artifact_bytes": destination.stat().st_size,
               "episodes": sum(s["episodes"] for s in reports_data.values()),
               "states": sum(s["states"] for s in reports_data.values()),
               "groups": len(known_groups), "source_snapshot_hashes_verified": True,
               "baseline_capsule_sha256": reports_data["confirmation"]["capsule_sha256"]}
    with (reports / "release_manifest.json").open("x") as stream:
        json.dump(release, stream, indent=2)
    print(json.dumps(release))


if __name__ == "__main__":
    main()
