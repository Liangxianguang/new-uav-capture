"""Fixed-budget training on fresh delayed-local calls, never baseline weights."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cwm_v7")]
from two_head_model import LocalTwoHead
from geometry_release import arrays, sha


def load_calls(directory):
    report = json.loads((directory / "summary.json").read_text())
    if not report["training_eligible"] or report["enhanced_control_enabled"] or report["holdout_collected"]:
        raise ValueError("Actual local data are not eligible")
    if sha(directory / "calls.json") != report["calls_sha256"] or sha(directory / "scenes.jsonl") != report["scenes_sha256"]:
        raise ValueError("Data provenance mismatch")
    records = json.loads((directory / "calls.json").read_text())
    scenes = [json.loads(line) for line in (directory / "scenes.jsonl").read_text().splitlines()]
    assignments = {r["mirror_group_id"]: r["model_split"] for r in scenes}
    if len(assignments) != 32 or set(assignments.values()) != {"train", "development_validation"}:
        raise ValueError("Fresh preassigned split mismatch")
    calls = []
    for row in records:
        if row["split"] != assignments[row["group"]]:
            raise ValueError("Local mirror group leakage")
        if "arrays_path" not in row:
            continue
        path = directory / row["arrays_path"]
        if sha(path) != row["arrays_sha256"]:
            raise ValueError("Actual local arrays changed")
        calls.append({"record": row, "values": arrays(path.read_bytes())})
    if not calls:
        raise ValueError("No local data")
    return calls, report


def normalization(calls, indices):
    histories = np.stack([calls[i]["values"]["history"] for i in indices])
    return histories.mean((0, 1), keepdims=True), np.maximum(histories.std((0, 1), keepdims=True), .01)


def pack_inputs(calls, indices, mean, scale):
    values, slices, offset = [[] for _ in range(5)], [], 0
    for index in indices:
        row = calls[index]["values"]
        count = len(row["proposed"])
        values[0].append(np.repeat(row["history"][None], count, 0))
        values[1].append(np.repeat(row["relative"][None], count, 0))
        values[2].append(row["proposed"])
        values[3].append(np.repeat(row["anchor"][None], count, 0))
        values[4].append(np.repeat(row["backbone"][None], count, 0))
        slices.append((offset, offset + count))
        offset += count
    history, relative, proposed, anchor, backbone = [np.concatenate(v) for v in values]
    tensors = (torch.as_tensor((history - mean) / scale, dtype=torch.float32), torch.as_tensor(relative, dtype=torch.float32),
               torch.as_tensor(proposed, dtype=torch.float32), torch.as_tensor(anchor, dtype=torch.float32), torch.as_tensor(backbone, dtype=torch.float64))
    return tensors, slices


def batch_loss(model, calls, indices, mean, scale, group_counts, config):
    tensors, slices = pack_inputs(calls, indices, mean, scale)
    _, motion, response = model(*tensors)
    terms, weights = [], []
    for index, (start, end) in zip(indices, slices):
        values = calls[index]["values"]
        anchor_valid = torch.as_tensor(values["anchor_valid"])
        target = torch.as_tensor(values["target"], dtype=torch.float32)
        anchor_target = torch.as_tensor(values["anchor_target"], dtype=torch.float32)
        nonanchor = ~(values["proposed"] == values["anchor"][None]).all((1, 2, 3))
        common = torch.as_tensor(values["valid"] & values["anchor_valid"][None] & nonanchor[:, None])
        motion_prediction = tensors[-1][start] + motion[start].to(torch.float64)
        motion_error = (((motion_prediction - anchor_target) ** 2).sum(-1) * anchor_valid).sum() / (3 * anchor_valid.sum().clamp_min(1))
        response_error = (((response[start:end] - (target - anchor_target[None])) ** 2).sum(-1) * common).sum() / (3 * common.sum().clamp_min(1))
        if model.kind == "motion_only":
            response_error = response_error * 0.
        energy = ((motion[start] ** 2).sum(-1) * anchor_valid).sum() / (3 * anchor_valid.sum().clamp_min(1))
        energy = energy + (((response[start:end] ** 2).sum(-1) * common).sum() / (3 * common.sum().clamp_min(1)))
        terms.append(torch.stack([motion_error, response_error, energy]))
        weights.append(1. / group_counts[calls[index]["record"]["group"]] if anchor_valid.any() else 0.)
    weights = torch.as_tensor(weights, dtype=torch.float64)
    if weights.sum() <= 0:
        raise ValueError("No valid local labels in batch")
    means = (torch.stack(terms) * weights[:, None]).sum(0) / weights.sum()
    loss = config["motion_weight"] * means[0] + config["response_weight"] * means[1] + config["residual_penalty"] * means[2]
    return loss, means


def metrics(calls, indices, outputs):
    grouped = {}
    anchor_zero = True
    for index, output in zip(indices, outputs):
        row, values = calls[index]["record"], calls[index]["values"]
        group = grouped.setdefault(row["group"], {"ade": [], "reference_ade": [], "response": []})
        common = values["valid"] & values["anchor_valid"][None]
        equal = (values["proposed"] == values["anchor"][None]).all((1, 2, 3))
        nonanchor_common = common & ~equal[:, None]
        group["ade"].extend(np.linalg.norm(output["prediction"] - values["target"], axis=-1)[values["valid"]].tolist())
        group["reference_ade"].extend(np.linalg.norm(values["backbone"] + output["motion"][0] - values["anchor_target"], axis=-1)[values["anchor_valid"]].tolist())
        group["response"].extend(np.linalg.norm(output["response"] - (values["target"] - values["anchor_target"][None]), axis=-1)[nonanchor_common].tolist())
        anchor_zero &= bool((output["response"][equal] == 0).all())
    if any(not all(values.values()) for values in grouped.values()):
        raise ValueError("Metric group has no valid label support")
    by_group = {g: {"ade_m": float(np.mean(v["ade"])), "reference_ade_m": float(np.mean(v["reference_ade"])),
                    "paired_response_error_m": float(np.mean(v["response"])), "target_points": len(v["ade"]), "response_points": len(v["response"])} for g, v in grouped.items()}
    return {"calls": len(indices), "groups": len(by_group), "by_group": by_group,
            "group_equal_ade_m": float(np.mean([v["ade_m"] for v in by_group.values()])),
            "group_equal_reference_ade_m": float(np.mean([v["reference_ade_m"] for v in by_group.values()])),
            "group_equal_paired_response_error_m": float(np.mean([v["paired_response_error_m"] for v in by_group.values()])),
            "anchor_response_exact_zero": anchor_zero}


def evaluate(model, calls, indices, mean, scale):
    model.eval()
    tensors, slices = pack_inputs(calls, indices, mean, scale)
    with torch.no_grad():
        values = [value.numpy() for value in model(*tensors)]
    outputs = [{key: value[start:end].copy() for key, value in zip(("prediction", "motion", "response"), values)} for start, end in slices]
    return metrics(calls, indices, outputs), outputs


def control_metrics(calls, indices):
    controls = {}
    for name in ("frozen_gru", "constant_velocity", "v7_plain_response", "v7_structured_response"):
        outputs = []
        for index in indices:
            values = calls[index]["values"]
            count = len(values["proposed"])
            backbone = np.repeat(values["backbone"][None], count, 0)
            motion, response = np.zeros_like(backbone), np.zeros_like(backbone)
            if name == "constant_velocity":
                cv = values["reference"][None] + .1 * np.arange(1, 9)[:, None] * values["velocity"][None]
                motion = np.repeat((cv - values["backbone"])[None], count, 0)
                prediction = np.repeat(cv[None], count, 0)
            elif name.startswith("v7_"):
                kind = name.split("_")[1]
                prediction, response = values[kind + "_prediction"], values[kind + "_response"]
            else:
                prediction = backbone
            outputs.append({"prediction": prediction, "motion": motion, "response": response})
        controls[name] = metrics(calls, indices, outputs)
    return controls


def gate_and_selection(models, controls, protocol):
    gate = protocol["development_gate"]
    structured = [r for r in models if r["kind"] == "structured"]
    plain = [r for r in models if r["kind"] == "plain"]
    eligible = all(r["development"]["group_equal_ade_m"] <= controls["frozen_gru"]["group_equal_ade_m"] * gate["maximum_structured_to_gru_ade_ratio_each_seed"] for r in structured)
    eligible &= np.median([r["development"]["group_equal_ade_m"] for r in structured]) <= controls["constant_velocity"]["group_equal_ade_m"] * gate["maximum_structured_to_cv_median_ade_ratio"]
    eligible &= np.median([r["development"]["group_equal_paired_response_error_m"] for r in structured]) <= np.median([r["development"]["group_equal_paired_response_error_m"] for r in plain]) * gate["maximum_structured_to_plain_median_response_error_ratio"]
    chosen = {kind: sorted([r for r in models if r["kind"] == kind], key=lambda r: (r["development"]["group_equal_ade_m"], r["seed"]))[1]["seed"] for kind in protocol["models"]}
    return bool(eligible), chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "training_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol["original_gru_in_optimizer"] or protocol["enhanced_control_enabled"] or protocol["holdout_used"] or protocol["qualification_gate_override_allowed"]:
        raise ValueError("Frozen original/offline contract mismatch")
    calls, data_report = load_calls(args.data)
    indices = {split: [i for i, c in enumerate(calls) if c["record"]["split"] == split] for split in ("train", "development_validation")}
    mean, scale = normalization(calls, indices["train"])
    group_counts = {}
    for i in indices["train"]:
        group = calls[i]["record"]["group"]
        group_counts[group] = group_counts.get(group, 0) + 1
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "normalization.npz", mean=mean, scale=scale)
    sources = [Path(__file__), ROOT / "two_head_model.py", args.protocol.resolve()]
    source_hashes = {p.name: sha(p) for p in sources}
    for path in sources:
        destination = args.output / "source" / path.name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(path.read_bytes())
    config, runs = protocol["training"], []
    controls = control_metrics(calls, indices["development_validation"])
    for seed in config["seeds"]:
        for kind in protocol["models"]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            sampler = np.random.default_rng(seed)
            model = LocalTwoHead(kind, config["motion_scale_m"], config["response_scale_m"])
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"])
            history, started = [], time.perf_counter()
            for epoch in range(1, config["epochs"] + 1):
                model.train()
                values = []
                order = sampler.permutation(indices["train"])
                for offset in range(0, len(order), config["batch_calls"]):
                    batch = order[offset:offset + config["batch_calls"]]
                    loss, terms = batch_loss(model, calls, batch, mean, scale, group_counts, config)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite two-head training loss")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
                    optimizer.step()
                    values.append([float(loss.detach()), *[float(v.detach()) for v in terms]])
                history.append({"epoch": epoch, "mean_batch_loss_motion_response_energy": np.mean(values, 0).tolist()})
            measured, outputs = evaluate(model, calls, indices["development_validation"], mean, scale)
            name = f"{kind}_seed{seed}"
            np.savez_compressed(args.output / f"{name}_development_predictions.npz", **{f"call{j}_{key}": v for j, output in enumerate(outputs) for key, v in output.items()})
            torch.save({"kind": kind, "seed": seed, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "protocol": protocol,
                        "normalizer_mean": torch.from_numpy(mean), "normalizer_scale": torch.from_numpy(scale), "torch_rng_state": torch.get_rng_state(),
                        "sampler_rng_state": sampler.bit_generator.state, "data_summary_sha256": sha(args.data / "summary.json"),
                        "source_hashes": source_hashes, "online_promoted": False, "baseline_weights_included": False}, args.output / f"{name}.pt")
            (args.output / f"{name}_history.json").write_text(json.dumps(history, indent=2))
            row = {"kind": kind, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()), "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "development": measured, "checkpoint_sha256": sha(args.output / f"{name}.pt"), "elapsed_seconds": time.perf_counter() - started}
            runs.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "development"} | {"ade_m": measured["group_equal_ade_m"], "response_error_m": measured["group_equal_paired_response_error_m"]}), flush=True)
    eligible, chosen = gate_and_selection(runs, controls, protocol)
    if any(sha(ROOT / name) != digest for name, digest in source_hashes.items()):
        raise AssertionError("Training sources changed during run")
    report = {"status": "offline_development_gate_passed_requires_decision_qualification" if eligible else "offline_trained_development_gate_failed_keep_baseline",
              "protocol": protocol, "source_hashes": source_hashes, "data_summary_sha256": sha(args.data / "summary.json"), "data_calls_sha256": data_report["calls_sha256"],
              "split_calls": {k: len(v) for k, v in indices.items()}, "split_groups": {split: sorted({calls[i]["record"]["group"] for i in chosen_indices}) for split, chosen_indices in indices.items()},
              "models": runs, "controls": controls, "development_gate_passed": eligible, "chosen_median_seed_per_architecture": chosen,
              "baseline_weights_included": False, "enhanced_control_enabled": False, "holdout_used": False}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "selected_median_seeds": chosen}), flush=True)


if __name__ == "__main__":
    main()
