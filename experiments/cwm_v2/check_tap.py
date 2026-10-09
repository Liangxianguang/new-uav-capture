"""Exact regression check for the read-only backbone tap on frozen references."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scout import BackboneTap
from freeze_baseline import REFERENCE, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = BackboneTap(args.capsule, args.output / "restored")
    records = base.records()
    selected = records[:1] + [r for r in records if r["level"] == 6][:1]
    expected = {r["episode_index"]: r for r in map(json.loads, (base.root / REFERENCE / "episodes.jsonl").read_text().splitlines())}
    results = []
    for record in selected:
        path = args.output / "trajectories" / f"level{record['level']}_{record['episode_index']}.npz"
        row, _ = base.run(record, path)
        with np.load(path) as actual, np.load(base.root / REFERENCE / "trajectories" / path.name) as original:
            exact = all(actual[k].dtype == original[k].dtype and actual[k].shape == original[k].shape
                        and actual[k].tobytes() == original[k].tobytes() for k in ("target_positions", "defender_positions"))
        outcomes = all(row[k] == expected[record["episode_index"]][k] for k in ("safe_capture_success", "capture_event", "collision", "boundary_violation", "timeout", "target_invalid_episode", "termination_reason"))
        results.append({"episode_index": record["episode_index"], "trajectories_byte_equal": exact, "outcomes_equal": outcomes})
    verify(base.root, base.capsule_manifest)
    report = {"status": "passed" if all(r["trajectories_byte_equal"] and r["outcomes_equal"] for r in results) else "failed",
              "episodes": results, "scope": "Two regression scenes for new tap only; full 256-episode baseline audit remains separate."}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if report["status"] != "passed":
        raise SystemExit("Backbone tap changed baseline")


if __name__ == "__main__":
    main()
