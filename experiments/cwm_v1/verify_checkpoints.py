"""Reload every pilot checkpoint safely and reproduce its saved predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from freeze_baseline import sha
from model import ResponseModel
from train import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    summary = json.loads((args.run / "summary.json").read_text())
    if sha(args.dataset) != summary["dataset_sha256"]:
        raise ValueError("Dataset hash mismatch")
    with np.load(args.dataset, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    selected = np.flatnonzero(data["split"] == "validation")
    results = []
    for row in summary["results"]:
        name = f"{row['kind']}_seed{row['seed']}"
        checkpoint = args.run / (name + ".pt")
        if sha(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError("Checkpoint integrity failure")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["online_promoted"] or state["dataset_sha256"] != summary["dataset_sha256"]:
            raise ValueError("Invalid pilot provenance")
        model = ResponseModel(state["kind"])
        model.load_state_dict(state["model_state"])
        metrics, predictions = evaluate(model, data, selected,
                                        state["normalizer_mean"].numpy(), state["normalizer_scale"].numpy())
        with np.load(args.run / (name + "_validation.npz"), allow_pickle=False) as saved:
            exact = np.array_equal(saved["prediction"], predictions)
            error = float(np.max(np.abs(saved["prediction"] - predictions)))
        results.append({"name": name, "bitwise_prediction_equal": exact, "max_abs_error": error,
                        "metrics_equal": metrics == row["validation"]})
    report = {"status": "passed" if all(r["bitwise_prediction_equal"] and r["metrics_equal"] for r in results) else "failed",
              "results": results, "runtime_scope": "Recorded CPU/PyTorch runtime only"}
    (args.run / "reload_verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if report["status"] != "passed":
        raise SystemExit("Reload reproduction failed")


if __name__ == "__main__":
    main()
