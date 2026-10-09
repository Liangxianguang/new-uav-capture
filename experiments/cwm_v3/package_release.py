"""Release verified bounded-response diagnostics with a mandatory NO-GO status."""
import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "cwm_v1"))
from freeze_baseline import sha
from verify_release import check_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    summary = json.loads((args.run / "summary.json").read_text())
    reproduction = json.loads((args.run / "reproduction.json").read_text())
    shadow = json.loads((args.shadow / "summary.json").read_text())
    if reproduction["status"] != "passed" or shadow["status"] != "passed" or summary["enhanced_control_enabled"] or summary["source_adequacy_gate_passed"]:
        raise ValueError("Diagnostic publication gate failed")
    for row in reproduction["results"]:
        if not all(v for k, v in row.items() if k != "name"):
            raise ValueError("Reproduction details do not pass")
    if shadow["errors"] or any(not r["trajectories_byte_equal"] or not r["outcomes_equal"] for r in shadow["episodes"]):
        raise ValueError("Shadow details do not pass")
    if sha(args.run / "dataset.npz") != summary["dataset_sha256"]:
        raise ValueError("Dataset changed")
    for name, expected in summary["source_hashes"].items():
        if sha(root / name) != expected:
            raise ValueError("Training source changed")
    for row in summary["results"]:
        if sha(args.run / f"{row['kind']}_seed{row['seed']}.pt") != row["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed")
    source_archive = root.parent / "cwm_v2/artifacts/level7_8_scout_20261009.zip"
    if sha(source_archive) != summary["source_archive_sha256"]:
        raise ValueError("Original scout archive changed")
    baseline = root.parent / "cwm_v1/baseline/capsule.zip"
    baseline_expected = json.loads((root.parent / "cwm_v1/reports/baseline_release_manifest.json").read_text())["baseline_capsule_sha256"]
    if sha(baseline) != baseline_expected:
        raise ValueError("Protected baseline changed")
    members = {f"training/{p.name}": p for p in args.run.iterdir() if p.suffix in (".json", ".pt", ".npz")}
    members.update({f"source/{p.name}": p for p in root.glob("*.py")})
    members["source/protocol.json"] = root / "protocol.json"
    members["shadow/summary.json"] = args.shadow / "summary.json"
    members["shadow/events.json"] = args.shadow / "events.json"
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    destination = artifacts / "anchored_diagnostic_20261009.zip"
    with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in members.items():
            archive.write(path, name)
        archive.writestr("ARTIFACT_MANIFEST.json", json.dumps({name: sha(path) for name, path in members.items()}, indent=2))
    check_archive(destination, "ARTIFACT_MANIFEST.json", False)
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    for name, path in (("training_summary.json", args.run / "summary.json"), ("reproduction.json", args.run / "reproduction.json"),
                       ("split_manifest.json", args.run / "split_manifest.json"), ("shadow_invariance.json", args.shadow / "summary.json")):
        if (reports / name).exists():
            raise FileExistsError("Never replace a milestone report")
        shutil.copy2(path, reports / name)
    control = summary["controls"]["development_validation"]["frozen_backbone_zero_response"]
    averages = {}
    for kind in ("plain", "structured"):
        rows = [r["development_validation"] for r in summary["results"] if r["kind"] == kind]
        averages[kind] = {key: sum(r[key] for r in rows) / len(rows) for key in ("ade_m", "paired_effect_error_m")}
    release = {"status": "no_go_keep_original_controller", "trained_models": len(summary["results"]),
               "enhanced_control_enabled": False, "data_adequacy_gate_passed": False,
               "artifact_sha256": sha(destination), "artifact_bytes": destination.stat().st_size,
               "baseline_capsule_sha256": baseline_expected, "source_archive_sha256": sha(source_archive),
               "development_means": averages, "zero_response_control": control,
               "interpretation": "Tiny trajectory differences are not meaningful evidence. Neither learned response beats zero-response; no closed-loop gain established."}
    with (reports / "release_manifest.json").open("x") as stream:
        json.dump(release, stream, indent=2)
    print(json.dumps(release))


if __name__ == "__main__":
    main()
