"""Read-only integrity checks for portable milestone archives."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath


def check_archive(path: Path, member_manifest: str, nested: bool):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for name in names:
            parsed = PurePosixPath(name.replace("\\", "/"))
            if parsed.is_absolute() or ".." in parsed.parts or ":" in name:
                raise ValueError("Unsafe archive member")
        document = json.loads(archive.read(member_manifest))
        hashes = document["files"] if nested else document
        if set(names) != set(hashes) | {member_manifest}:
            raise ValueError("Unlisted or missing archive member")
        for name, expected in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Archive hash mismatch: {name}")
    return {"file": path.name, "members_verified": len(hashes),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    results = []
    for stage, relative, manifest, nested in [
        ("baseline", "baseline/capsule.zip", "CAPSULE_MANIFEST.json", True),
        ("pilot", "artifacts/pilot_20261009.zip", "ARTIFACT_MANIFEST.json", False),
    ]:
        row = check_archive(args.root / relative, manifest, nested)
        release = json.loads((args.root / f"reports/{stage}_release_manifest.json").read_text())
        expected = release["baseline_capsule_sha256"] if stage == "baseline" else release["artifact_sha256"]
        if row["sha256"] != expected or release["enhanced_control_enabled"]:
            raise ValueError("Release integrity or disabled-control contract mismatch")
        results.append(row)
    print(json.dumps({"status": "passed", "archives": results}))


if __name__ == "__main__":
    main()
