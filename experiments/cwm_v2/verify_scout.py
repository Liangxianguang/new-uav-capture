"""Read-only release, scene separation, and backbone integrity verification."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "cwm_v1"))
from freeze_baseline import sha
from verify_release import check_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    root = args.root
    release = json.loads((root / "reports/release_manifest.json").read_text())
    capsule = root.parent / "cwm_v1/baseline/capsule.zip"
    if sha(capsule) != release["baseline_capsule_sha256"] or release["enhanced_control_enabled"]:
        raise ValueError("Baseline or disabled enhancement mismatch")
    path = root / "artifacts/level7_8_scout_20261009.zip"
    archive_check = check_archive(path, "ARTIFACT_MANIFEST.json", False)
    if archive_check["sha256"] != release["artifact_sha256"]:
        raise ValueError("Release archive mismatch")
    with zipfile.ZipFile(capsule) as archive:
        validation_groups = {str(json.loads(line)["mirror_group_id"]) for line in archive.read("results/phase91_zero_cbf_curriculum_v5/validation/scenes.jsonl").decode().splitlines() if line}
    groups = set()
    states = episodes = 0
    with zipfile.ZipFile(path) as archive:
        for stage in ("scout", "confirmation"):
            summary = json.loads(archive.read(f"{stage}/summary.json"))
            if json.loads((root / f"reports/{stage}_summary.json").read_text()) != summary:
                raise ValueError("Published summary differs from archive")
            if hashlib.sha256(archive.read(f"{stage}/pairs.npz")).hexdigest() != summary["dataset_sha256"]:
                raise ValueError("Paired dataset mismatch")
            selected = [json.loads(line) for line in archive.read(f"{stage}/selected_scenes.jsonl").decode().splitlines() if line]
            stage_groups = {str(r["mirror_group_id"]) for r in selected}
            if stage_groups & (groups | validation_groups) or any(r["model_split"] != "scout_train_only" for r in selected):
                raise ValueError("Scout partition leakage")
            groups |= stage_groups
            states += summary["states"]
            episodes += len(selected)
            for name, expected in summary["source_hashes"].items():
                if hashlib.sha256(archive.read(f"{stage}/source/{name}")).hexdigest() != expected:
                    raise ValueError("Recorded source hash mismatch")
        audit = json.loads(archive.read("tap_audit.json"))
        if audit["status"] != "passed" or any(not r["trajectories_byte_equal"] or not r["outcomes_equal"] for r in audit["episodes"]):
            raise ValueError("Tap invariance failure")
    if (states, episodes, len(groups)) != (release["states"], release["episodes"], release["groups"]):
        raise ValueError("Release population mismatch")
    print(json.dumps({"status": "passed", "states": states, "episodes": episodes,
                      "independent_groups": len(groups), "original_validation_overlap": 0,
                      "archives": archive_check, "enhanced_control_enabled": False}))


if __name__ == "__main__":
    main()
