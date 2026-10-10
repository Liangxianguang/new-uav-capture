"""Exploratory fixed-library decision attribution for frozen median V10 models."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cwm_v9"), str(ROOT.parent / "cwm_v8")]
from train_two_head import LocalTwoHead, load_calls, evaluate
from local_shadow import decode_public, restore_public_call, local_cost, rank_metrics
from local_shadow_release import frozen_configs
from s4_value import descriptive_bootstrap
from geometry_release import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    calls, data_report = load_calls(args.data)
    indices = [i for i, c in enumerate(calls) if c["record"]["split"] == "development_validation"]
    training = json.loads((args.training / "summary.json").read_text())
    if training["enhanced_control_enabled"] or training["holdout_used"]:
        raise ValueError("No control activation/holdout in diagnostic")
    configs = frozen_configs(args.capsule, args.output / "restored")
    predictions = {}
    for kind, seed in training["chosen_median_seed_per_architecture"].items():
        checkpoint = torch.load(args.training / f"{kind}_seed{seed}.pt", map_location="cpu", weights_only=True)
        model = LocalTwoHead(kind, checkpoint["protocol"]["training"]["motion_scale_m"], checkpoint["protocol"]["training"]["response_scale_m"])
        model.load_state_dict(checkpoint["model_state"], strict=True)
        _, predictions[kind] = evaluate(model, calls, indices, checkpoint["normalizer_mean"].numpy(), checkpoint["normalizer_scale"].numpy())
    methods = ("original_gru", "constant_velocity", "trained_motion_only", "plain_motion_component", "structured_motion_component", "plain_two_head", "structured_two_head", "reference_truth", "action_specific_truth")
    records = []
    started = time.perf_counter()
    for position, index in enumerate(indices):
        row, values = calls[index]["record"], calls[index]["values"]
        if not row["all_candidates_full_horizon_valid"]:
            continue
        snapshot = json.loads((args.data / row["context_path"]).read_text())
        call = restore_public_call(snapshot, *configs, values["backbone"])
        count = len(values["proposed"])
        backbone = np.repeat(values["backbone"][None], count, 0)
        cv = values["reference"][None] + .1 * np.arange(1, 9)[:, None] * values["velocity"][None]
        paths = {"original_gru": backbone, "constant_velocity": np.repeat(cv[None], count, 0),
                 "trained_motion_only": predictions["motion_only"][position]["prediction"],
                 "plain_motion_component": backbone + predictions["plain"][position]["motion"],
                 "structured_motion_component": backbone + predictions["structured"][position]["motion"],
                 "plain_two_head": predictions["plain"][position]["prediction"], "structured_two_head": predictions["structured"][position]["prediction"],
                 "reference_truth": np.repeat(values["anchor_target"][None], count, 0), "action_specific_truth": values["target"]}
        costs = {method: local_cost(call, values["proposed"][:, :, row["agent"]], paths[method]) for method in methods}
        if not np.allclose(costs["action_specific_truth"], row["costs"]["action_specific_truth"], rtol=1e-10, atol=1e-8):
            raise AssertionError("Independent full local truth costs differ from collection")
        metrics = {method: rank_metrics(cost, costs["action_specific_truth"]) for method, cost in costs.items()}
        records.append({"episode_index": row["episode_index"], "step": row["step"], "agent": row["agent"], "group": row["group"],
                        "costs": {k: v.tolist() for k, v in costs.items()}, "metrics": metrics})
    groups = [r["group"] for r in records]
    summary = {}
    for method in methods:
        def gain_against(reference):
            gains = [r["metrics"][reference]["regret_in_diagnostic_cost"] - r["metrics"][method]["regret_in_diagnostic_cost"] for r in records]
            return descriptive_bootstrap(gains, groups, 2000, 970102)
        summary[method] = {"mean_diagnostic_regret": float(np.mean([r["metrics"][method]["regret_in_diagnostic_cost"] for r in records])),
                           "gain_vs_gru": gain_against("original_gru"), "gain_vs_cv": gain_against("constant_velocity"),
                           "gain_vs_trained_motion_only": gain_against("trained_motion_only")}
    response_increment = {}
    for kind in ("plain", "structured"):
        increments = [r["metrics"][kind + "_motion_component"]["regret_in_diagnostic_cost"] - r["metrics"][kind + "_two_head"]["regret_in_diagnostic_cost"] for r in records]
        response_increment[kind] = {"choice_changes": sum(r["metrics"][kind + "_motion_component"]["choice"] != r["metrics"][kind + "_two_head"]["choice"] for r in records),
                                    "gain": descriptive_bootstrap(increments, groups, 2000, 970102)}
    report = {"status": "exploratory_fixed_library_complete_not_promoted", "development_calls": len(indices), "rankable_calls": len(records),
              "methods": summary, "learned_response_increment": response_increment, "selected_median_seeds": training["chosen_median_seed_per_architecture"],
              "prior_training_gate_passed": training["development_gate_passed"], "prior_training_gate_overridden": False,
              "enhanced_control_enabled": False, "holdout_used": False, "data_summary_sha256": sha(args.data / "summary.json"),
              "training_summary_sha256": sha(args.training / "summary.json"), "elapsed_seconds": time.perf_counter() - started,
              "limitations": "Fresh V10 development already inspected by training metrics; descriptive, not independent test. Same frozen V9-generated pool, not new-model self-generation or sequential/closed-loop control. Raw forecasts and label-only truth have no dynamics/safety certificate. Failed training gate cannot be overridden by this diagnostic."}
    (args.output / "records.json").write_text(json.dumps(records, indent=2))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    (args.output / "source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
