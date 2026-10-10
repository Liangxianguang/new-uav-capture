"""Fixed-budget public-input paired training on preassigned, qualified S4 groups."""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from geometry import ROOT, sha
from response_model import ResponseModel, loss_terms


def inputs(data, indices, mean, scale):
    proposed = torch.as_tensor(data["proposed"][indices, :, :8], dtype=torch.float32)
    branches = proposed.shape[1]
    history = torch.as_tensor((data["history"][indices] - mean) / scale, dtype=torch.float32)
    relative = torch.as_tensor(data["relative"][indices], dtype=torch.float32)
    backbone = torch.as_tensor(data["backbone"][indices], dtype=torch.float64)
    return (history.repeat_interleave(branches, 0), relative.repeat_interleave(branches, 0), proposed.flatten(0, 1),
            proposed[:, 0].repeat_interleave(branches, 0), backbone.repeat_interleave(branches, 0))


def metrics(prediction, response, logits, data, indices):
    target, valid, signs = data["target"][indices, :, :8], data["valid"][indices, :, :8], data["branch_sign_label_only"][indices, :, :8]
    common = valid[:, 1:] & valid[:, :1]
    effect_error = np.linalg.norm((response[:, 1:] - response[:, :1]) - (target[:, 1:] - target[:, :1]), axis=-1)
    motion_error = np.linalg.norm(prediction - target, axis=-1)
    groups = data["group"][indices]
    committed = valid & (signs != 0)
    by_group = {}
    for group in sorted(set(groups.tolist())):
        chosen = groups == group
        cm, vm = common[chosen], valid[chosen]
        if not cm.any() or not vm.any():
            raise ValueError("Evaluation group has no valid paired labels")
        by_group[str(group)] = {"paired_effect_error_m": float(effect_error[chosen][cm].mean()),
                                "ade_m": float(motion_error[chosen][vm].mean()), "states": int(chosen.sum())}
    return {"states": len(indices), "groups": len(by_group), "by_group": by_group,
            "group_equal_paired_effect_error_m": float(np.mean([v["paired_effect_error_m"] for v in by_group.values()])),
            "group_equal_ade_m": float(np.mean([v["ade_m"] for v in by_group.values()])),
            "paired_effect_error_m": float(effect_error[common].mean()), "paired_effect_p95_m": float(np.quantile(effect_error[common], .95)),
            "ade_m": float(motion_error[valid].mean()), "paired_valid_points": int(common.sum()),
            "committed_branch_accuracy": float(((logits > 0) == (signs > 0))[committed].mean()) if committed.any() and logits is not None else None,
            "anchor_exact_zero_response": bool((response[:, 0] == 0).all()),
            "anchor_backbone_exact": bool(np.array_equal(prediction[:, 0], data["backbone"][indices])),
            "maximum_response_norm_m": float(np.linalg.norm(response, axis=-1).max())}


def evaluate(model, data, indices, mean, scale):
    model.eval()
    with torch.no_grad():
        prediction, response, logits = model(*inputs(data, indices, mean, scale))
    predictions = {"prediction": prediction.numpy().reshape(len(indices), 8, 8, 3),
                   "response": response.numpy().reshape(len(indices), 8, 8, 3),
                   "branch_logits": logits.numpy().reshape(len(indices), 8, 8)}
    result = metrics(predictions["prediction"], predictions["response"], predictions["branch_logits"], data, indices)
    if not result["anchor_exact_zero_response"] or not result["anchor_backbone_exact"]:
        raise AssertionError("Model changed original anchor trajectory")
    return result, predictions


def validate_splits(data, scenes, protocol):
    assigned = {r["mirror_group_id"]: r["model_split"] for r in scenes}
    if len(assigned) != protocol["groups"] or set(assigned.values()) != {"train", "development_validation"}:
        raise ValueError("Preassigned group population mismatch")
    for scene in scenes:
        group_index = scene["layout_seed"] - protocol["layout_seed_start"]
        if not 0 <= group_index < protocol["groups"] or scene["model_split"] != protocol["variant_cycles_split"][group_index // 4]:
            raise ValueError("Scene split/seed differs from predeclared protocol")
        member = 0 if scene["mirror_pair_member"] == "upper" else 1 if scene["mirror_pair_member"] == "lower" else -1
        if member < 0 or scene["episode_seed"] != protocol["seed_start"] + group_index * 2 + member:
            raise ValueError("Unexpected episode or mirror seed")
        if scene["model_split"] != assigned[scene["mirror_group_id"]]:
            raise ValueError("Mirror group split leakage")
        for low, high in protocol["excluded_development_layout_seed_blocks"]:
            if low <= scene["layout_seed"] <= high:
                raise ValueError("Geometry-development scene reused as model data")
    if any(str(g) not in assigned or str(s) != assigned[str(g)] for g, s in zip(data["group"], data["split"])):
        raise ValueError("Actual dataset split differs from scene protocol")
    if any(str(s) == "holdout" for s in data["split"]):
        raise ValueError("Holdout may not enter training")
    if set(data["group"].tolist()) != set(assigned):
        raise ValueError("Actual dataset misses preassigned groups")
    return {split: np.flatnonzero(data["split"] == split) for split in ("train", "development_validation")}, assigned


def normalization(data, train_indices):
    mean = data["history"][train_indices].mean((0, 1), keepdims=True)
    scale = np.maximum(data["history"][train_indices].std((0, 1), keepdims=True), .01)
    return mean, scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "training_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads((args.data / "summary.json").read_text())
    protocol = json.loads(args.protocol.read_text())
    if not source["training_eligible"] or source["enhanced_control_enabled"] or source["holdout_collected"] or protocol["enhanced_control_enabled"] or protocol["guarded_mode_allowed"]:
        raise ValueError("Training eligibility / disabled-control contract failed")
    if sha(args.data / "pairs.npz") != source["dataset_sha256"]:
        raise ValueError("Actual paired dataset digest mismatch")
    with np.load(args.data / "pairs.npz", allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    scenes = [json.loads(line) for line in (args.data / "selected_scenes.jsonl").read_text().splitlines()]
    selected, assigned = validate_splits(data, scenes, source["protocol"])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    mean, scale = normalization(data, selected["train"])
    np.savez_compressed(args.output / "normalization.npz", mean=mean, scale=scale)
    source_hashes = {p.name: sha(p) for p in (Path(__file__), ROOT / "response_model.py", args.protocol.resolve())}
    for name in source_hashes:
        p = ROOT / name
        destination = args.output / "source" / name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(p.read_bytes())
    group_counts = {group: int((data["group"][selected["train"]] == group).sum()) for group in set(data["group"][selected["train"]])}
    config, runs = protocol["training"], []
    for seed in config["seeds"]:
        for kind in protocol["models"]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            sampler = np.random.default_rng(seed)
            model = ResponseModel(kind, config["response_scale_m"])
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            history, started = [], time.perf_counter()
            for epoch in range(config["epochs"]):
                model.train()
                values = []
                order = sampler.permutation(selected["train"])
                for offset in range(0, len(order), config["batch_states"]):
                    indices = order[offset:offset + config["batch_states"]]
                    _, response, logits = model(*inputs(data, indices, mean, scale))
                    weights = torch.as_tensor([1. / group_counts[g] for g in data["group"][indices]], dtype=torch.float32)
                    terms = loss_terms(response.reshape(len(indices), 8, 8, 3), logits.reshape(len(indices), 8, 8),
                                       torch.as_tensor(data["target"][indices, :, :8], dtype=torch.float32), torch.as_tensor(data["valid"][indices, :, :8]),
                                       torch.as_tensor(data["branch_sign_label_only"][indices, :, :8]), weights, config)
                    if not torch.isfinite(terms[0]):
                        raise ValueError("Nonfinite paired loss")
                    optimizer.zero_grad()
                    terms[0].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
                    optimizer.step()
                    values.append([float(v.detach()) for v in terms])
                history.append({"epoch": epoch + 1, "mean_batch_loss_terms": np.mean(values, 0).tolist()})
            measured, predictions = evaluate(model, data, selected["development_validation"], mean, scale)
            name = f"{kind}_seed{seed}"
            np.savez_compressed(args.output / f"{name}_development_predictions.npz", **predictions)
            torch.save({"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "seed": seed, "kind": kind,
                        "protocol": protocol, "dataset_sha256": source["dataset_sha256"], "source_hashes": source_hashes,
                        "normalizer_mean": torch.from_numpy(mean), "normalizer_scale": torch.from_numpy(scale),
                        "sampler_rng_state": sampler.bit_generator.state, "torch_rng_state": torch.get_rng_state(),
                        "online_promoted": False, "baseline_weights_included": False}, args.output / f"{name}.pt")
            (args.output / f"{name}_history.json").write_text(json.dumps(history, indent=2))
            row = {"kind": kind, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()), "development_validation": measured,
                   "checkpoint_sha256": sha(args.output / f"{name}.pt"), "elapsed_seconds": time.perf_counter() - started}
            runs.append(row)
            print(json.dumps(row), flush=True)
    indices = selected["development_validation"]
    zeros = np.zeros_like(data["target"][indices, :, :8])
    backbone = np.repeat(data["backbone"][indices, None], 8, axis=1)
    control = metrics(backbone, zeros, None, data, indices)
    cv = data["reference"][indices, None, None] + .1 * np.arange(1, 9)[None, None, :, None] * data["reference_velocity"][indices, None, None]
    cv = np.repeat(cv, 8, axis=1)
    cv_control = metrics(cv, zeros, None, data, indices)
    errors = {kind: [r["development_validation"]["group_equal_paired_effect_error_m"] for r in runs if r["kind"] == kind] for kind in protocol["models"]}
    gate = protocol["development_gate"]
    eligible = all(v <= control["group_equal_paired_effect_error_m"] * gate["maximum_structured_to_zero_group_error_ratio_each_seed"] for v in errors["structured"])
    eligible &= np.median(errors["structured"]) <= np.median(errors["plain"]) * gate["maximum_structured_to_plain_median_group_error_ratio"]
    chosen = {kind: sorted([r for r in runs if r["kind"] == kind], key=lambda r: (r["development_validation"]["group_equal_paired_effect_error_m"], r["seed"]))[1]["seed"] for kind in protocol["models"]}
    if any(sha(ROOT / name) != expected for name, expected in source_hashes.items()):
        raise AssertionError("Training source changed during training")
    summary = {"status": "offline_development_gate_passed_requires_decision_holdout_validation" if eligible else "offline_trained_not_promoted_development_gate_failed",
               "protocol": protocol, "dataset_sha256": source["dataset_sha256"], "data_summary_sha256": sha(args.data / "summary.json"),
               "source_hashes": source_hashes, "split_groups": assigned, "split_states": {k: len(v) for k, v in selected.items()},
               "models": runs, "controls": {"frozen_gru_zero_response": control, "constant_velocity_zero_response": cv_control},
               "chosen_median_seed_per_architecture": chosen, "development_gate_passed": bool(eligible),
               "enhanced_control_enabled": False, "holdout_used": False, "baseline_weights_included": False}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": summary["status"], "errors": errors, "zero_response": control["group_equal_paired_effect_error_m"], "chosen": chosen}), flush=True)


if __name__ == "__main__":
    main()
