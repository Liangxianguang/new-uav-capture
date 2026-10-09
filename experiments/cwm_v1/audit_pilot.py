"""Audit deterministic retraining and training-only intervention strength."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from freeze_baseline import REFERENCE, sha


def stats(values):
    values = np.asarray(values)
    return {"count": int(values.size), "mean": float(values.mean()),
            "p50": float(np.quantile(values, .5)), "p95": float(np.quantile(values, .95)),
            "max": float(values.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--rerun", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--original-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads((args.reference / "summary.json").read_text())
    rerun = json.loads((args.rerun / "summary.json").read_text())
    if not reference["dataset_sha256"] == rerun["dataset_sha256"] == sha(args.dataset):
        raise ValueError("Training dataset mismatch")
    comparisons = []
    for left, right in zip(reference["results"], rerun["results"], strict=True):
        if (left["kind"], left["seed"]) != (right["kind"], right["seed"]):
            raise ValueError("Matched run mismatch")
        name = f"{left['kind']}_seed{left['seed']}"
        with np.load(args.reference / f"{name}_validation.npz") as a, np.load(args.rerun / f"{name}_validation.npz") as b:
            predictions_equal = all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape
                                    and a[k].tobytes() == b[k].tobytes() for k in a.files)
        weights = [torch.load(root / f"{name}.pt", map_location="cpu", weights_only=True)["model_state"]
                   for root in (args.reference, args.rerun)]
        weights_equal = weights[0].keys() == weights[1].keys() and all(
            weights[0][key].numpy().tobytes() == weights[1][key].numpy().tobytes() for key in weights[0])
        histories_equal = json.loads((args.reference / f"{name}_history.json").read_text()) == json.loads((args.rerun / f"{name}_history.json").read_text())
        comparisons.append({"name": name, "predictions_byte_equal": predictions_equal,
                            "weights_byte_equal": weights_equal, "training_histories_equal": histories_equal,
                            "metrics_equal": all(left[k] == right[k] for k in ("calibration", "validation"))})
    with np.load(args.dataset, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    with zipfile.ZipFile(args.capsule) as archive:
        validation = [json.loads(line) for line in archive.read("results/phase91_zero_cbf_curriculum_v5/validation/scenes.jsonl").decode().splitlines() if line]
        forbidden = {str(row["mirror_group_id"]) for row in validation}
        if forbidden & set(data["group"]):
            raise ValueError("Full original validation group overlap")
        provenance = json.loads(archive.read(f"{REFERENCE}/manifest.json"))
    historical_checks = None
    if args.original_source is not None:
        historical_checks = 0
        for name, expected in provenance["sha256"].items():
            if sha(args.original_source / name.replace("\\", "/")) != expected:
                raise ValueError("Original provenance changed")
            historical_checks += 1
    train = data["split"] == "train"
    mask = data["mask"][train]
    common = mask[:, 1:] & mask[:, :1]
    diagnostic = {"scope": "original training pool / model train split only",
                  "states": int(train.sum()), "branches": {}}
    for index, branch in enumerate(("left_pressure", "right_pressure", "half_speed"), start=1):
        valid = common[:, index - 1]
        row = {}
        for key in ("proposed", "commanded", "executed"):
            actions = data[key][train]
            difference = np.linalg.norm(actions[:, index] - actions[:, 0], axis=-1)
            row[f"{key}_difference_mps"] = stats(difference[valid])
        labels = data["labels"][train]
        effect = np.linalg.norm(labels[:, index] - labels[:, 0], axis=-1)[valid]
        row["target_response_m"] = stats(effect)
        row["effect_over_0_1m_fraction"] = float(np.mean(effect > .1))
        diagnostic["branches"][branch] = row
    nearest = np.linalg.norm(data["relative"][train, :, :3], axis=-1).min(axis=-1)
    diagnostic["nearest_defender_to_public_reference_m"] = stats(nearest)
    passed = all(all(row[k] for k in ("predictions_byte_equal", "weights_byte_equal", "training_histories_equal", "metrics_equal")) for row in comparisons)
    report = {"status": "passed" if passed else "failed", "retraining": comparisons,
              "original_provenance_hashes_reverified": historical_checks,
              "full_original_validation_group_overlap": 0, "training_only_diagnostic": diagnostic,
              "limitations": "Recorded CPU runtime only; command differences are not defender-geometry effects or identified causal mechanisms."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))
    if not passed:
        raise SystemExit("Retraining reproduction mismatch")


if __name__ == "__main__":
    main()
