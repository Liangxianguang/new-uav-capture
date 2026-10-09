"""Make a portable, byte-verified capsule without editing the source checkout."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import sys
import zipfile
from pathlib import Path

REFERENCE = "results/phase91_dn_mpc_cbf_capability_20261005_original_margin"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def script_closure(source: Path, seeds: list[str]) -> set[Path]:
    pending = [source / "scripts" / name for name in seeds]
    found: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = []
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                dependency = source / "scripts" / (name.split(".")[0] + ".py")
                if dependency.is_file() and dependency not in found:
                    pending.append(dependency)
    return found


def freeze(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Never overwrite a capsule: {output}")
    reference = source / REFERENCE
    provenance = json.loads((reference / "manifest.json").read_text())
    failures = []
    for name, expected in provenance["sha256"].items():
        path = source / name.replace("\\", "/")
        if not path.is_file() or sha(path) != expected:
            failures.append(name)
    if failures:
        raise ValueError(f"Historical baseline hash mismatch: {failures}")
    files = set((source / "src").rglob("*.py"))
    files |= script_closure(source, [
        "evaluate_phase91_dn_mpc_cbf_capability.py", "evaluate_minimax_mpc.py",
        "evaluate_rule_teacher_phase87.py", "summarize_phase91_dn_mpc_cbf_capability.py",
    ])
    for name in [
        "configs/phase91_mappo_pursuit_first_v9.yaml",
        "configs/phase85_target_contract_repaired_environment.yaml",
        "configs/capture_radius_pursuit_central_v4_flee.yaml", "configs/phase4_dn_mpc.yaml",
        "models/formal_phase2_v3/gru_baseline_seed745101.pt",
        "models/formal_phase2_v3/manifest.json", "environment.yml",
        "results/phase91_zero_cbf_curriculum_v5/validation/scenes.jsonl",
    ]:
        files.add(source / name)
    for name in ["manifest.json", "summary.json", "analysis.json", "REPORT.md", "scenes.jsonl", "episodes.jsonl"]:
        files.add(reference / name)
    files |= set((reference / "trajectories").glob("*.npz"))
    for path in files:
        if not path.is_file() or not path.resolve().is_relative_to(source):
            raise ValueError(f"Invalid capsule member: {path}")
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    manifest = {
        "schema_version": 1, "name": "phase91_dn_mpc_local_cbf_20261005",
        "status": "historical_hashes_verified_replay_pending", "historical_hash_checks": len(provenance["sha256"]),
        "reference_directory": REFERENCE,
        "contract": {"capture_radius_m": 0.8, "boundary_margin_m": 0.3, "cbf_margin_m": 0.35,
                     "dt_seconds": 0.1, "max_steps": 250, "predictor_input_dim": 252,
                     "predictor_seed": 745101, "prediction_samples": 1, "planner_horizon": 8},
        "runtime": {"python": sys.version, "platform": platform.platform(), "packages": dict(sorted(packages.items()))},
        "files": {path.relative_to(source).as_posix(): sha(path) for path in sorted(files)},
        "limitations": ["Wall-clock latency is not deterministic.", "Exact replay must be checked on this runtime; other platforms need revalidation.",
                         "Local CBF remains an empirical filter, not a formal invariance certificate."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in manifest["files"]:
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 9, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, (source / name).read_bytes())
        archive.writestr("CAPSULE_MANIFEST.json", json.dumps(manifest, indent=2))
    metadata = {"capsule_sha256": sha(output), "capsule_bytes": output.stat().st_size, **manifest}
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def restore(capsule: Path, destination: Path) -> dict:
    destination = destination.resolve()
    with zipfile.ZipFile(capsule) as archive:
        manifest = json.loads(archive.read("CAPSULE_MANIFEST.json"))
        for name in archive.namelist():
            target = (destination / name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Unsafe archive path: {name}")
        if destination.exists():
            verify(destination, manifest)
        else:
            destination.mkdir(parents=True)
            archive.extractall(destination)
            verify(destination, manifest)
    return manifest


def verify(root: Path, manifest: dict) -> None:
    for name, expected in manifest["files"].items():
        path = root / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"Capsule integrity failure: {name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.source, args.output)
    print(json.dumps({key: result[key] for key in ["name", "historical_hash_checks", "capsule_bytes", "capsule_sha256"]}))
