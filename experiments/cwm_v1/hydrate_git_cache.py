"""Recover missing remote Git blobs ONLY from SHA-matching local archive files.

Useful when a partial Git fetch succeeds but the subsequent blob download fails.
Never copies unverified working-tree content into the remote base revision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="FETCH_HEAD")
    parser.add_argument("--cache", type=Path, action="append", required=True)
    args = parser.parse_args()
    env = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}
    tree = subprocess.check_output(["git", "ls-tree", "-r", "-z", args.revision], env=env)
    restored, missing = [], []
    for item in tree.split(b"\0"):
        if not item:
            continue
        metadata, filename = item.split(b"\t", 1)
        _, kind, expected = metadata.split()
        if kind != b"blob":
            continue
        name = filename.decode("utf-8")
        if subprocess.run(["git", "cat-file", "-e", expected.decode()], env=env, capture_output=True).returncode == 0:
            continue
        for root in args.cache:
            path = root / name
            if not path.is_file():
                continue
            data = path.read_bytes()
            variants = [data]
            if b"\r\n" in data and b"\0" not in data:
                variants.append(data.replace(b"\r\n", b"\n"))
            for value in variants:
                digest = hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()
                if digest == expected.decode():
                    actual = subprocess.check_output(["git", "hash-object", "-w", "--stdin"], input=value, env=env).decode().strip()
                    if actual != digest:
                        raise AssertionError("Git cache hydration mismatch")
                    restored.append(name)
                    break
            else:
                continue
            break
        else:
            missing.append({"path": name, "sha": expected.decode()})
    print(json.dumps({"restored": len(restored), "missing": missing}))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
