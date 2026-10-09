"""Read-only anchored inference with unchanged historical DN-MPC/CBF actions."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "cwm_v2"))
from scout import BackboneTap, commands
from freeze_baseline import REFERENCE, sha, verify
from plugin import ResponsePlugin
from residual_model import AnchoredResponse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = BackboneTap(args.capsule, args.output / "restored")
    summary = json.loads((args.run / "summary.json").read_text())
    models = []
    for row in summary["results"]:
        if row["kind"] == "structured":
            path = args.run / f"structured_seed{row['seed']}.pt"
            if sha(path) != row["checkpoint_sha256"]:
                raise ValueError("Checkpoint mismatch")
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state["online_promoted"]:
                raise ValueError("Not an offline checkpoint")
            model = AnchoredResponse("structured", state["protocol"]["training"]["response_scale_m"])
            model.load_state_dict(state["model_state"])
            model.eval()
            models.append((model, state["normalizer_mean"], state["normalizer_scale"]))
    if len(models) != 3:
        raise ValueError("Expected three-seed structured ensemble")
    history, events = [], []

    def predict(values):
        responses = []
        with torch.no_grad():
            for model, mean, scale in models:
                h = (torch.as_tensor(values["history"])[None] - mean) / scale
                relative = torch.as_tensor(values["relative"])[None].repeat(4, 1, 1)
                proposed = torch.as_tensor(values["proposed"])
                anchor = proposed[0:1].repeat(4, 1, 1, 1)
                backbone = torch.as_tensor(values["backbone"])[None].repeat(4, 1, 1)
                prediction, response = model(h.repeat(4, 1, 1), relative, proposed, anchor, backbone)
                if prediction[0].numpy().tobytes() != values["backbone"].tobytes():
                    raise ValueError("Anchor changed")
                responses.append(response.numpy())
        combined = values["backbone"][None] + np.mean(responses, axis=0)
        combined[0] = values["backbone"]
        return combined

    plugin = ResponsePlugin(mode="shadow", predictor=predict)

    def observer(env, observation, baseline_action, sequence):
        history.append(base.evaluator.policy_observations(env, observation).reshape(-1).copy())
        reference, velocity = base.evaluator.belief_reference(observation, base.planner_config)
        padded = [history[0]] * max(0, 8 - len(history)) + history[-8:]
        relative = np.concatenate([observation["defender_positions"] - reference,
                                   observation["defender_velocities"] - velocity], axis=-1)
        values = {"history": np.stack(padded).astype(np.float32), "relative": relative.astype(np.float32),
                  "proposed": commands(sequence, observation, reference).astype(np.float32),
                  "backbone": (base.last_backbone[0] - reference).astype(np.float32)}
        action = plugin.observe(baseline_action, values)
        if action.dtype != baseline_action.dtype or action.tobytes() != baseline_action.tobytes():
            raise AssertionError("Shadow changed command")
        events.append({"episode_index": current_record["episode_index"], "step": env.step_count, **plugin.last_diagnostics})

    all_records = base.records()
    selected = all_records[:1] + [r for r in all_records if r["level"] == 6][:1]
    selected += [r for r in all_records if r["episode_index"] in (11388, 14612)]
    references = {r["episode_index"]: r for r in map(json.loads, (base.root / REFERENCE / "episodes.jsonl").read_text().splitlines())}
    comparisons = []
    for current_record in selected:
        history.clear()
        path = args.output / "trajectories" / f"level{current_record['level']}_{current_record['episode_index']}.npz"
        row, _ = base.run(current_record, path, observer=observer)
        with np.load(path) as actual, np.load(base.root / REFERENCE / "trajectories" / path.name) as original:
            exact = all(actual[k].dtype == original[k].dtype and actual[k].shape == original[k].shape and actual[k].tobytes() == original[k].tobytes() for k in ("target_positions", "defender_positions"))
        outcomes = all(row[k] == references[current_record["episode_index"]][k] for k in
            ("safe_capture_success", "capture_event", "collision", "boundary_violation", "timeout", "target_invalid_episode", "termination_reason"))
        comparisons.append({"episode_index": current_record["episode_index"], "trajectories_byte_equal": exact, "outcomes_equal": outcomes})
    verify(base.root, base.capsule_manifest)
    errors = sum(row["status"] == "shadow_error" for row in events)
    latency = [row["elapsed_ms"] for row in events if "elapsed_ms" in row]
    report = {"status": "passed" if errors == 0 and all(r["trajectories_byte_equal"] and r["outcomes_equal"] for r in comparisons) else "failed",
              "episodes": comparisons, "calls": len(events), "errors": errors,
              "extra_latency_ms": {"p50": float(np.quantile(latency, .5)), "p95": float(np.quantile(latency, .95)), "p99": float(np.quantile(latency, .99))} if latency else {},
              "enhanced_control_enabled": False,
              "scope": "Offline shadow sanity including historical failures; not an enhanced closed-loop performance or safety claim."}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    (args.output / "events.json").write_text(json.dumps(events, indent=2))
    print(json.dumps(report))
    if report["status"] != "passed":
        raise SystemExit("Shadow invariance failure")


if __name__ == "__main__":
    main()
