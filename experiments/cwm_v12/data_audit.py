"""Reuse the complete V10 public/local data audit with the V12 protocol path."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cwm_v10"))
from two_head_release import audit_data as v10_audit_data


def audit_data(read, source_read, configs, restored):
    protocol = json.loads((ROOT / "data_protocol.json").read_text())
    report = json.loads(read("data/summary.json"))
    if report["protocol"] != protocol:
        raise ValueError("Predeclared fresh V12 data protocol mismatch")
    scenes = [json.loads(line) for line in read("data/scenes.jsonl").decode().splitlines()]
    if {r["episode_seed"] for r in scenes} != set(range(973010, 973074)) or {r["layout_seed"] for r in scenes} != set(range(1973010, 1973042)):
        raise ValueError("Actual fresh V12 seeds/population mismatch")
    def protocol_path_read(name):
        if name == "data/source/cwm_v10/data_protocol.json":
            return read("data/source/cwm_v12/data_protocol.json")
        return read(name)
    # All geometry, public reencoding, historical trajectory equality,
    # local-candidate/cost reconstruction, masks, split and data-gate checks
    # remain in the unmodified V10 verifier; only protocol location changes.
    return v10_audit_data(protocol_path_read, source_read, configs, restored)
