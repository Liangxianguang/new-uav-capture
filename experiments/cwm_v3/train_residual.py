"""Train tiny anchored diagnostics; data-adequacy gate remains NO-GO."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "cwm_v1"))
from freeze_baseline import sha
from verify_release import check_archive
from residual_model import AnchoredResponse, paired_loss


def split_groups(data, seed):
    rng = np.random.default_rng(seed)
    assigned = {}
    for variant in sorted(set(data["variant"].tolist())):
        groups = sorted(set(data["group"][data["variant"] == variant].tolist()))
        if len(groups) != 4:
            raise ValueError("Diagnostic expects four independent groups per variant")
        for group, split in zip(rng.permutation(groups), ("train", "train", "calibration", "development_validation"), strict=True):
            if str(group) in assigned:
                raise ValueError("Group shared across variants")
            assigned[str(group)] = split
    values = np.asarray([assigned[str(g)] for g in data["group"]])
    return values, assigned


def inputs(data, indices, mean, scale):
    commands = torch.as_tensor(data["proposed"][indices], dtype=torch.float32)
    count = commands.shape[1]
    history = torch.as_tensor((data["history"][indices] - mean) / scale, dtype=torch.float32)
    relative = torch.as_tensor(data["relative"][indices], dtype=torch.float32)
    backbone = torch.as_tensor(data["backbone"][indices], dtype=torch.float32)
    return (history.repeat_interleave(count, 0), relative.repeat_interleave(count, 0), commands.flatten(0, 1),
            commands[:, 0].repeat_interleave(count, 0), backbone.repeat_interleave(count, 0))


def describe(prediction, response, labels, mask):
    common = mask[:, 1:] & mask[:, :1]
    if not common.any() or not mask.any():
        raise ValueError("No valid evaluation labels")
    effect = np.linalg.norm((response[:, 1:] - response[:, :1]) - (labels[:, 1:] - labels[:, :1]), axis=-1)
    error = np.linalg.norm(prediction - labels, axis=-1)
    full = common.all(axis=-1)
    terminal = mask[..., -1]
    return {"states": len(labels), "ade_m": float(error[mask].mean()),
            "fde_m": float(error[..., -1][terminal].mean()) if terminal.any() else None,
            "paired_effect_error_m": float(effect[common].mean()),
            "paired_effect_p95_m": float(np.quantile(effect[common], .95)),
            "full_horizon_effect_error_m": float(effect[full].mean()) if full.any() else None,
            "full_horizon_pairs": int(full.sum()), "paired_valid_points": int(common.sum()),
            "response_max_norm_m": float(np.linalg.norm(response, axis=-1).max()),
            "anchor_response_exact_zero": bool(np.array_equal(response[:, 0], np.zeros_like(response[:, 0])))}


def evaluate(model, data, indices, mean, scale):
    if not len(indices):
        raise ValueError("Empty evaluation split")
    model.eval()
    with torch.no_grad():
        prediction, response = model(*inputs(data, indices, mean, scale))
    prediction = prediction.reshape(len(indices), 4, 8, 3).numpy()
    response = response.reshape(len(indices), 4, 8, 3).numpy()
    metrics = describe(prediction, response, data["labels"][indices], data["mask"][indices])
    if not np.array_equal(prediction[:, 0], data["backbone"][indices]):
        raise AssertionError("Model changed anchor backbone")
    return metrics, prediction, response


def load_scout_archive(path):
    check_archive(path, "ARTIFACT_MANIFEST.json", False)
    pieces = []
    with zipfile.ZipFile(path) as archive:
        for stage in ("scout", "confirmation"):
            summary = json.loads(archive.read(f"{stage}/summary.json"))
            raw = archive.read(f"{stage}/pairs.npz")
            if hashlib.sha256(raw).hexdigest() != summary["dataset_sha256"]:
                raise ValueError("Scout data hash mismatch")
            with archive.open(f"{stage}/pairs.npz") as file, np.load(file, allow_pickle=False) as data:
                pieces.append({key: data[key] for key in data.files})
    if pieces[0].keys() != pieces[1].keys():
        raise ValueError("Scout schemas differ")
    return {key: np.concatenate([part[key] for part in pieces]) for key in pieces[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("protocol.json"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads(args.protocol.read_text())
    if protocol["enhanced_control_enabled"] or protocol["guarded_mode_allowed"]:
        raise ValueError("Diagnostic cannot authorize control")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data = load_scout_archive(args.archive)
    splits, assigned = split_groups(data, protocol["split_seed"])
    selected = {s: np.flatnonzero(splits == s) for s in ("train", "calibration", "development_validation")}
    if any(not len(v) for v in selected.values()):
        raise ValueError("Empty diagnostic split")
    data["split"] = splits
    np.savez_compressed(args.output / "dataset.npz", **data)
    dataset_sha = sha(args.output / "dataset.npz")
    (args.output / "split_manifest.json").write_text(json.dumps({"groups": assigned, "states": {s: len(v) for s, v in selected.items()},
         "all_source_groups": "original train only; aggregate summaries previously inspected"}, indent=2))
    mean = data["history"][selected["train"]].mean(axis=(0, 1), keepdims=True)
    scale = np.maximum(data["history"][selected["train"]].std(axis=(0, 1), keepdims=True), .01)
    source_hashes = {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("residual_model.py"), args.protocol)}
    config = protocol["training"]
    results = []
    for seed in config["seeds"]:
        for kind in ("plain", "structured"):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = AnchoredResponse(kind, config["response_scale_m"])
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            started = time.perf_counter()
            history = []
            for epoch in range(config["epochs"]):
                model.train()
                batches = []
                order = rng.permutation(selected["train"])
                for offset in range(0, len(order), config["batch_states"]):
                    indices = order[offset:offset + config["batch_states"]]
                    _, response = model(*inputs(data, indices, mean, scale))
                    loss, effect, penalty = paired_loss(response.reshape(len(indices), 4, 8, 3),
                        torch.as_tensor(data["labels"][indices]), torch.as_tensor(data["mask"][indices]), config["residual_penalty"])
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite diagnostic loss")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    batches.append([float(loss.detach()), float(effect.detach()), float(penalty.detach())])
                history.append({"epoch": epoch + 1, "losses": np.mean(batches, axis=0).tolist()})
            metrics = {}
            for split in ("calibration", "development_validation"):
                value, prediction, response = evaluate(model, data, selected[split], mean, scale)
                metrics[split] = value
                np.savez_compressed(args.output / f"{kind}_seed{seed}_{split}.npz", prediction=prediction, response=response)
            checkpoint = args.output / f"{kind}_seed{seed}.pt"
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "kind": kind,
                        "seed": seed, "protocol": protocol, "normalizer_mean": torch.from_numpy(mean),
                        "normalizer_scale": torch.from_numpy(scale), "dataset_sha256": dataset_sha,
                        "source_archive_sha256": sha(args.archive), "source_hashes": source_hashes,
                        "torch_rng_state": torch.get_rng_state(), "sampler_rng_state": rng.bit_generator.state,
                        "online_promoted": False, "baseline_weights_included": False}, checkpoint)
            (args.output / f"{kind}_seed{seed}_history.json").write_text(json.dumps(history, indent=2))
            row = {"kind": kind, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
                   "elapsed_seconds": time.perf_counter() - started, "checkpoint_sha256": sha(checkpoint), **metrics}
            results.append(row)
            print(json.dumps(row), flush=True)
    controls = {}
    for split in ("calibration", "development_validation"):
        indices = selected[split]
        zeros = np.zeros_like(data["labels"][indices])
        predicted = np.repeat(data["backbone"][indices, None], 4, axis=1)
        cv = .1 * np.arange(1, 9)[None, None, :, None] * data["reference_velocity"][indices, None, None]
        controls[split] = {"frozen_backbone_zero_response": describe(predicted, zeros, data["labels"][indices], data["mask"][indices]),
                           "constant_velocity_ade_m": float(np.linalg.norm(cv - data["labels"][indices], axis=-1)[data["mask"][indices]].mean())}
    summary = {"status": "diagnostic_trained_no_go_data_adequacy_unchanged", "protocol": protocol,
               "results": results, "controls": controls, "source_hashes": source_hashes,
               "dataset_sha256": dataset_sha, "source_archive_sha256": sha(args.archive),
               "enhanced_control_enabled": False, "source_adequacy_gate_passed": False}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
