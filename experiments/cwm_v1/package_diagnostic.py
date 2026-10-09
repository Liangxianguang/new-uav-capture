"""Publish verified diagnostic evidence without modifying previous releases."""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from freeze_baseline import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    summary = json.loads((args.run / "summary.json").read_text())
    windows = json.loads((args.run / "windows.json").read_text())
    selected = [json.loads(line) for line in (args.run / "selected_scenes.jsonl").read_text().splitlines() if line]
    if summary["status"] != "completed_no_control_promotion" or not summary["baseline_and_parent_integrity_pass"] or summary["target_truth_is_input"]:
        raise ValueError("Diagnostic contract failure")
    if len(windows) != summary["states"] or len(selected) != summary["episodes"] or any(row["model_split"] != "train" for row in selected):
        raise ValueError("Diagnostic population mismatch")
    capsule = root / "baseline/capsule.zip"
    if sha(capsule) != summary["capsule_sha256"]:
        raise ValueError("Baseline capsule changed")
    with zipfile.ZipFile(capsule) as archive:
        forbidden = {str(json.loads(line)["mirror_group_id"]) for line in archive.read("results/phase91_zero_cbf_curriculum_v5/validation/scenes.jsonl").decode().splitlines() if line}
    if forbidden & {str(row["mirror_group_id"]) for row in selected}:
        raise ValueError("Original validation group overlap")
    destination = root / "artifacts/interaction_diagnostic_20261009.zip"
    files = {f"diagnostic/{name}": args.run / name for name in ("summary.json", "windows.json", "selected_scenes.jsonl")}
    files.update({f"source/{name}": root / name for name in ("diagnose_interactions.py", "baseline.py", "collect_pairs.py", "freeze_baseline.py", "package_diagnostic.py")})
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in files.items():
            archive.write(path, name)
        archive.writestr("ARTIFACT_MANIFEST.json", json.dumps({name: sha(path) for name, path in files.items()}, indent=2))
    shutil.copy2(args.run / "summary.json", root / "reports/interaction_diagnostic_summary.json")
    release = {"status": "diagnostic_only_no_promotion", "enhanced_control_enabled": False,
               "artifact_sha256": sha(destination), "artifact_bytes": destination.stat().st_size,
               "episodes": len(selected), "states": len(windows), "original_validation_group_overlap": 0,
               "baseline_capsule_sha256": sha(capsule), "source_hashes": {name: sha(path) for name, path in files.items() if name.startswith("source/")}}
    with (root / "reports/interaction_diagnostic_release.json").open("x") as stream:
        json.dump(release, stream, indent=2)
    print(json.dumps(release))


if __name__ == "__main__":
    main()
