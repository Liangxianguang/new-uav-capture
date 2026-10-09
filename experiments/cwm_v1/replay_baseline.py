"""Replay saved episodes and compare outcomes AND true trajectories to the capsule."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from baseline import Baseline
from freeze_baseline import REFERENCE, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Verify saved trajectories and resume an interrupted audit")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = Baseline(args.capsule, args.output / "restored")
    records = base.records()
    if not args.all:
        failures = {11388, 11389, 11585, 12206, 12207, 14612}
        selected = records[:2] + [r for r in records if r["level"] == 6][:2]
        selected += [r for r in records if r["episode_index"] in failures]
        records = list({r["episode_index"]: r for r in selected}.values())
    references = {r["episode_index"]: r for r in
                  map(json.loads, (base.root / REFERENCE / "episodes.jsonl").read_text().splitlines())}
    comparisons = []
    if args.resume:
        previous = json.loads((args.output / "summary.json").read_text())
        comparisons = previous["comparisons"]
        if previous["selected"] != len(records) or any(not row["passed"] for row in comparisons):
            raise ValueError("Resume requires a matching, previously passing prefix")
        for checked in comparisons:
            name = f"level{checked['level']}_{checked['episode_index']}.npz"
            with np.load(base.root / REFERENCE / "trajectories" / name) as original, np.load(args.output / "trajectories" / name) as replay:
                if any(not np.array_equal(original[field], replay[field]) for field in ["defender_positions", "target_positions"]):
                    raise ValueError("Saved replay integrity failure")
    completed = {checked["episode_index"] for checked in comparisons}
    for record in records:
        key = record["episode_index"]
        if key in completed:
            continue
        trajectory = args.output / "trajectories" / f"level{record['level']}_{key}.npz"
        row, _ = base.run(record, trajectory)
        expected = references[key]
        mismatches = {}
        for field in ["safe_capture_success", "capture_event", "collision", "boundary_violation", "timeout",
                      "target_invalid_episode", "target_boundary_violation", "target_obstacle_violation", "termination_reason"]:
            if row[field] != expected[field]:
                mismatches[field] = [expected[field], row[field]]
        for field in ["capture_time_seconds", "min_clearance_m"]:
            same = (row[field] is None and expected[field] is None) if row[field] is None or expected[field] is None else np.isclose(row[field], expected[field], atol=1e-9, rtol=0, equal_nan=True)
            if not same:
                mismatches[field] = [expected[field], row[field]]
        errors = {}
        exact = True
        reference_path = base.root / REFERENCE / "trajectories" / trajectory.name
        with np.load(reference_path) as original, np.load(trajectory) as replay:
            for field in ["defender_positions", "target_positions"]:
                a, b = original[field], replay[field]
                same = a.shape == b.shape and np.array_equal(a, b)
                exact &= same
                errors[field] = (float(np.max(np.abs(a - b))) if a.shape == b.shape else None)
                if not same:
                    mismatches[field] = {"original_shape": list(a.shape), "replay_shape": list(b.shape), "max_abs_error": errors[field]}
        comparisons.append({"episode_index": key, "level": record["level"], "passed": not mismatches,
                            "bitwise_trajectory_equal": exact, "errors": errors, "mismatches": mismatches})
        summary = {"status": "running", "selected": len(records), "completed": len(comparisons),
                   "passed": sum(r["passed"] for r in comparisons), "comparisons": comparisons}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(comparisons[-1]), flush=True)
    verify(base.root, base.capsule_manifest)
    summary = {"status": "passed" if all(r["passed"] for r in comparisons) else "failed",
               "selected": len(records), "completed": len(comparisons),
               "passed": sum(r["passed"] for r in comparisons), "comparisons": comparisons}
    summary["source_and_artifact_integrity_after_replay"] = True
    summary["limitations"] = "Exact trajectories apply to this recorded CPU/runtime. Latency and other platforms are not guaranteed identical."
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    if summary["status"] != "passed":
        raise SystemExit("Baseline replay mismatch; do not promote any enhancement")


if __name__ == "__main__":
    main()
