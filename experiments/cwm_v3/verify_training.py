"""Verify saved weights and optional deterministic retraining without promotion."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train_residual import AnchoredResponse, evaluate, sha


def same_bytes(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rerun", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    summary = json.loads((args.run / "summary.json").read_text())
    if sha(args.run / "dataset.npz") != summary["dataset_sha256"] or summary["enhanced_control_enabled"] or summary["source_adequacy_gate_passed"]:
        raise ValueError("Diagnostic contract mismatch")
    with np.load(args.run / "dataset.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    assigned = {}
    for group, split in zip(data["group"], data["split"]):
        if str(group) in assigned and assigned[str(group)] != split:
            raise ValueError("Group split leakage")
        assigned[str(group)] = split
    rerun_summary = None
    if args.rerun:
        rerun_summary = json.loads((args.rerun / "summary.json").read_text())
        if rerun_summary["protocol"] != summary["protocol"] or rerun_summary["source_hashes"] != summary["source_hashes"]:
            raise ValueError("Rerun protocol/source changed")
        with np.load(args.rerun / "dataset.npz") as rerun_data:
            if any(not same_bytes(value, rerun_data[key]) for key, value in data.items()):
                raise ValueError("Rerun data changed")
    rows = []
    for row in summary["results"]:
        name = f"{row['kind']}_seed{row['seed']}"
        path = args.run / f"{name}.pt"
        if sha(path) != row["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["online_promoted"] or checkpoint["baseline_weights_included"] or checkpoint["dataset_sha256"] != summary["dataset_sha256"]:
            raise ValueError("Checkpoint provenance mismatch")
        model = AnchoredResponse(row["kind"], checkpoint["protocol"]["training"]["response_scale_m"])
        model.load_state_dict(checkpoint["model_state"])
        compared = {"name": name, "reload_metrics_equal": True, "reload_predictions_byte_equal": True,
                    "anchor_backbone_byte_equal": True, "response_bound_pass": True}
        for split in ("calibration", "development_validation"):
            indices = np.flatnonzero(data["split"] == split)
            metrics, prediction, response = evaluate(model, data, indices,
                checkpoint["normalizer_mean"].numpy(), checkpoint["normalizer_scale"].numpy())
            with np.load(args.run / f"{name}_{split}.npz") as saved:
                compared["reload_predictions_byte_equal"] &= same_bytes(prediction, saved["prediction"]) and same_bytes(response, saved["response"])
            compared["reload_metrics_equal"] &= metrics == row[split]
            compared["anchor_backbone_byte_equal"] &= same_bytes(prediction[:, 0], data["backbone"][indices])
            compared["response_bound_pass"] &= bool(np.abs(response).max() <= model.response_scale_m + 1e-8)
            if args.rerun:
                with np.load(args.rerun / f"{name}_{split}.npz") as other:
                    compared.setdefault("retrain_predictions_byte_equal", True)
                    compared["retrain_predictions_byte_equal"] &= same_bytes(prediction, other["prediction"]) and same_bytes(response, other["response"])
        if args.rerun:
            other_state = torch.load(args.rerun / f"{name}.pt", map_location="cpu", weights_only=True)
            compared["retrain_weights_byte_equal"] = all(same_bytes(value.numpy(), other_state["model_state"][key].numpy()) for key, value in checkpoint["model_state"].items())
            compared["retrain_history_equal"] = json.loads((args.run / f"{name}_history.json").read_text()) == json.loads((args.rerun / f"{name}_history.json").read_text())
            other_row = next(r for r in rerun_summary["results"] if (r["kind"], r["seed"]) == (row["kind"], row["seed"]))
            compared["retrain_metrics_equal"] = all(row[k] == other_row[k] for k in ("calibration", "development_validation"))
        rows.append(compared)
    report = {"status": "passed" if all(all(v for k, v in row.items() if k != "name") for row in rows) else "failed",
              "results": rows, "groups": len(assigned), "enhanced_control_enabled": False,
              "scope": "Recorded local CPU/runtime; numerical/model bytes, not wall-clock or checkpoint-container bytes."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report))
    if report["status"] != "passed":
        raise SystemExit("Residual reproduction failure")


if __name__ == "__main__":
    main()
