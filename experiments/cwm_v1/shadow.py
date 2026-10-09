"""Check that optional offline shadow inference leaves baseline trajectories exact."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from baseline import Baseline
from collect_pairs import interventions
from freeze_baseline import REFERENCE, sha, verify
from model import ResponseModel
from plugin import ResponsePlugin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("off", "shadow"), default="off")
    parser.add_argument("--run", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = Baseline(args.capsule, args.output / "restored")
    models = []
    if args.mode == "shadow":
        if args.run is None:
            raise ValueError("Shadow requires a verified training run")
        summary = json.loads((args.run / "summary.json").read_text())
        for row in summary["results"]:
            if row["kind"] != "structured":
                continue
            path = args.run / f"structured_seed{row['seed']}.pt"
            if sha(path) != row["checkpoint_sha256"]:
                raise ValueError("Checkpoint mismatch")
            state = torch.load(path, map_location="cpu", weights_only=True)
            model = ResponseModel("structured")
            model.load_state_dict(state["model_state"])
            model.eval()
            models.append((model, state["normalizer_mean"], state["normalizer_scale"]))
    history, step_rows, rows = [], [], []

    def predictor(values):
        predictions = []
        with torch.no_grad():
            for model, mean, scale in models:
                h = torch.as_tensor(values["history"], dtype=torch.float32)
                x = ((h[None] - mean) / scale).repeat(4, 1, 1)
                relative = torch.as_tensor(values["relative"], dtype=torch.float32)[None].repeat(4, 1, 1)
                actions = torch.as_tensor(values["proposed"], dtype=torch.float32)
                velocity = torch.as_tensor(values["velocity"], dtype=torch.float32)[None].repeat(4, 1)
                predictions.append(model(x, relative, actions, velocity).numpy())
        return np.mean(predictions, axis=0)

    plugin = ResponsePlugin(mode=args.mode, predictor=predictor if models else None)

    def observer(env, observation, actions, sequence):
        history.append(base.evaluator.policy_observations(env, observation).reshape(-1).copy())
        reference, velocity = base.evaluator.belief_reference(observation, base.planner_config)
        relative = np.concatenate([observation["defender_positions"] - reference,
                                   observation["defender_velocities"] - velocity], axis=-1)
        padded = [history[0]] * max(0, 8 - len(history)) + history[-8:]
        values = {"history": np.stack(padded), "relative": relative,
                  "proposed": interventions(sequence[:8]), "velocity": velocity}
        returned = plugin.observe(actions, values)
        if not np.array_equal(returned, actions):
            raise AssertionError("Shadow altered baseline command")
        step_rows.append({"step": int(env.step_count), **plugin.last_diagnostics})

    records = base.records()
    selected = records[:2] + [r for r in records if r["level"] == 6][:2]
    reference_rows = {r["episode_index"]: r for r in map(json.loads, (base.root / REFERENCE / "episodes.jsonl").read_text().splitlines())}
    for record in selected:
        history.clear()
        trajectory = args.output / "trajectories" / f"level{record['level']}_{record['episode_index']}.npz"
        row, _ = base.run(record, trajectory, observer=observer if args.mode == "shadow" else None)
        same = True
        with np.load(trajectory) as actual, np.load(base.root / REFERENCE / "trajectories" / trajectory.name) as expected:
            same = all(np.array_equal(actual[key], expected[key]) for key in ("defender_positions", "target_positions"))
        expected_row = reference_rows[record["episode_index"]]
        outcomes = all(row[key] == expected_row[key] for key in ("safe_capture_success", "collision", "boundary_violation", "timeout"))
        rows.append({"episode_index": record["episode_index"], "bitwise_trajectory_equal": same, "outcomes_equal": outcomes})
    verify(base.root, base.capsule_manifest)
    latencies = [r.get("elapsed_ms", 0.) for r in step_rows]
    errors = sum(r["status"] == "shadow_error" for r in step_rows)
    report = {"status": "passed" if all(r["bitwise_trajectory_equal"] and r["outcomes_equal"] for r in rows) and errors == 0 else "failed",
              "mode": args.mode, "episodes": rows, "steps": len(step_rows), "errors": errors,
              "shadow_extra_latency_ms": {"p50": float(np.quantile(latencies, .5)), "p95": float(np.quantile(latencies, .95)),
                                          "p99": float(np.quantile(latencies, .99))} if latencies else {},
              "limitations": "Offline shadow invariance sanity check only; no enhanced action, no guarded mode, no control-performance improvement claim."}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    (args.output / "steps.json").write_text(json.dumps(step_rows, indent=2))
    print(json.dumps(report))
    if report["status"] != "passed":
        raise SystemExit("Shadow isolation failure")


if __name__ == "__main__":
    main()
