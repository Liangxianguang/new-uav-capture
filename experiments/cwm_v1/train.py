"""Train small matched pilots; fixed final epoch, no online control promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from freeze_baseline import sha
from model import ResponseModel, group_loss


def inputs(data, indices, mean, scale):
    history = torch.as_tensor((data["history"][indices] - mean) / scale, dtype=torch.float32)
    relative = torch.as_tensor(data["relative"][indices], dtype=torch.float32)
    commands = torch.as_tensor(data["proposed"][indices], dtype=torch.float32)
    velocity = torch.as_tensor(data["reference_velocity"][indices], dtype=torch.float32)
    branches = commands.shape[1]
    return (history.repeat_interleave(branches, dim=0), relative.repeat_interleave(branches, dim=0),
            commands.flatten(0, 1), velocity.repeat_interleave(branches, dim=0))


def evaluate(model, data, indices, mean, scale):
    if not len(indices):
        raise ValueError("Empty evaluation split")
    model.eval()
    with torch.no_grad():
        prediction = model(*inputs(data, indices, mean, scale)).reshape(len(indices), 4, 8, 3).numpy()
    labels, mask = data["labels"][indices], data["mask"][indices]
    error = np.linalg.norm(prediction - labels, axis=-1)
    effect_error = np.linalg.norm((prediction[:, 1:] - prediction[:, :1]) - (labels[:, 1:] - labels[:, :1]), axis=-1)
    common = mask[:, 1:] & mask[:, :1]
    terminal = mask[..., -1]
    report = {"states": len(indices), "ade_m": float(np.mean(error[mask])),
              "fde_m": float(np.mean(error[..., -1][terminal])) if terminal.any() else None,
              "paired_effect_error_m": float(np.mean(effect_error[common])),
              "paired_effect_p95_m": float(np.quantile(effect_error[common], .95)),
              "effect_valid_points": int(np.sum(common)), "full_horizon_branches": int(np.sum(terminal))}
    return report, prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    config = protocol["training"]
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with np.load(args.dataset, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    selected = {name: np.flatnonzero(data["split"] == name) for name in ("train", "calibration", "validation")}
    groups = {name: set(data["group"][indices]) for name, indices in selected.items()}
    if any(groups[a] & groups[b] for a, b in [("train", "validation"), ("train", "calibration"), ("calibration", "validation")]):
        raise ValueError("Group leakage")
    if any(not len(value) for value in selected.values()):
        raise ValueError("Empty split")
    mean = data["history"][selected["train"]].mean(axis=(0, 1), keepdims=True)
    scale = np.maximum(data["history"][selected["train"]].std(axis=(0, 1), keepdims=True), .01)
    results = []
    source_hashes = {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name("model.py"), args.protocol]}
    for seed in config["seeds"]:
        for kind in ("plain", "structured"):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = ResponseModel(kind)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            count = sum(parameter.numel() for parameter in model.parameters())
            history = []
            started = time.perf_counter()
            for epoch in range(config["epochs"]):
                model.train()
                losses = []
                order = rng.permutation(selected["train"])
                for offset in range(0, len(order), config["batch_states"]):
                    batch = order[offset:offset + config["batch_states"]]
                    prediction = model(*inputs(data, batch, mean, scale)).reshape(len(batch), 4, 8, 3)
                    label = torch.as_tensor(data["labels"][batch], dtype=torch.float32)
                    mask = torch.as_tensor(data["mask"][batch], dtype=torch.bool)
                    loss, reconstruction, effect = group_loss(prediction, label, mask, config["effect_loss_weight"])
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite loss")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    losses.append([float(loss.detach()), float(reconstruction.detach()), float(effect.detach())])
                row = {"epoch": epoch + 1, "loss": np.mean(losses, axis=0).tolist()}
                history.append(row)
                print(json.dumps({"kind": kind, "seed": seed, **row}), flush=True)
            calibration, _ = evaluate(model, data, selected["calibration"], mean, scale)
            validation, predictions = evaluate(model, data, selected["validation"], mean, scale)
            path = args.output / f"{kind}_seed{seed}.pt"
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "kind": kind, "seed": seed, "epochs": config["epochs"], "normalizer_mean": torch.from_numpy(mean.copy()),
                        "normalizer_scale": torch.from_numpy(scale.copy()), "dataset_sha256": sha(args.dataset),
                        "source_hashes": source_hashes, "protocol": protocol,
                        "torch_rng_state": torch.get_rng_state(), "sampler_rng_state": rng.bit_generator.state,
                        "online_promoted": False}, path)
            np.savez_compressed(args.output / f"{kind}_seed{seed}_validation.npz", prediction=predictions,
                                labels=data["labels"][selected["validation"]], mask=data["mask"][selected["validation"]],
                                groups=data["group"][selected["validation"]])
            (args.output / f"{kind}_seed{seed}_history.json").write_text(json.dumps(history, indent=2))
            results.append({"kind": kind, "seed": seed, "parameters": count, "elapsed_seconds": time.perf_counter() - started,
                            "checkpoint_sha256": sha(path), "calibration": calibration, "validation": validation})
            (args.output / "progress.json").write_text(json.dumps({"completed_models": len(results), "results": results}, indent=2))
    cv = np.arange(1, 9)[None, None, :, None] * .1 * data["reference_velocity"][selected["validation"]][:, None, None, :]
    cv_errors = np.linalg.norm(cv - data["labels"][selected["validation"]], axis=-1)
    effects = np.linalg.norm(data["labels"][selected["validation"]][:, 1:] - data["labels"][selected["validation"]][:, :1], axis=-1)
    masks = data["mask"][selected["validation"]]
    summary = {"status": "pilot_training_complete_not_online_promoted", "dataset_sha256": sha(args.dataset),
               "source_hashes": source_hashes, "selection": "fixed final epoch; no validation-based checkpoint selection",
               "results": results, "zero_response_effect_error_m": float(np.mean(effects[masks[:, 1:] & masks[:, :1]])),
               "constant_velocity_ade_m": float(np.mean(cv_errors[masks])),
               "limitations": ["Small training-pool-disjoint pilot, not the formal Level5/6 or Level7/8 evaluation.",
                                "No SCM identifiability, safety, novelty, or closed-loop gain is established.",
                                "Calibration errors are descriptive; no guard was calibrated or enabled."]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
