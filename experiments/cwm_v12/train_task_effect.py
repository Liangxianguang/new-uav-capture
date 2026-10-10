"""Preregistered fifteen-model fresh task-effect experiment; offline only."""
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

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cwm_v11"), str(ROOT.parent / "cwm_v10"), str(ROOT.parent / "cwm_v9"), str(ROOT.parent / "cwm_v8")]
from task_score import FrozenLocalScore, original_scores
from train_two_head import LocalTwoHead, load_calls, normalization, pack_inputs, batch_loss as physical_loss, evaluate, control_metrics
from local_shadow import restore_public_call, rank_metrics
from local_shadow_release import frozen_configs
from geometry_release import sha
from verify_release import check_archive
from s4_value import descriptive_bootstrap
from data_audit import audit_data


def architecture(name):
    if name == "motion_only":
        return name
    if name in ("plain_mse", "plain_task", "structured_mse", "structured_task"):
        return name.split("_")[0]
    raise ValueError("Unexpected experiment configuration")


class PairedTaskLabel:
    """Detached cached full-support labels; no original predictor/motion input."""
    def __init__(self, score, values):
        if not values["valid"].all() or not values["anchor_valid"].all():
            raise ValueError("Full valid terminal support required for task supervision")
        count = len(values["proposed"])
        reference = torch.from_numpy(np.repeat(values["anchor_target"][None], count, 0)).requires_grad_(True)
        cost = score(reference)
        derivative = torch.autograd.grad(cost.sum(), reference)[0]
        self.score, self.reference, self.reference_cost = score, reference.detach(), cost.detach()
        self.scale = derivative.detach().square().sum((1, 2)).clamp_min(1.)
        self.truth_cost = score(torch.from_numpy(values["target"])).detach()
        self.effect = (self.truth_cost - self.reference_cost).detach()

    def loss(self, response):
        estimated = self.score(self.reference + response.double()) - self.reference_cost
        return ((estimated - self.effect).square() / self.scale).mean()


def prepare_scores_read(calls, read, configs):
    scores, labels = {}, {}
    for index, item in enumerate(calls):
        row, values = item["record"], item["values"]
        context = read(row["context_path"])
        if hashlib.sha256(context).hexdigest() != row["context_sha256"]:
            raise ValueError("Actual delayed public context changed")
        call = restore_public_call(json.loads(context), *configs, values["backbone"])
        score = FrozenLocalScore(call, values["proposed"][:, :, row["agent"]])
        full = bool(values["valid"].all() and values["anchor_valid"].all())
        if full != row["all_candidates_full_horizon_valid"]:
            raise ValueError("Stored full-horizon flag differs from valid labels")
        scores[index] = score
        if full:
            label = PairedTaskLabel(score, values)
            independent = original_scores(call, score.actions, values["target"])
            if not np.allclose(independent, label.truth_cost.numpy(), rtol=1e-10, atol=1e-8) or not np.allclose(independent, row["costs"]["action_specific_truth"], rtol=1e-10, atol=1e-8):
                raise ValueError("Original paired task-cost labels differ")
            labels[index] = label
    return scores, labels


def prepare_scores(calls, directory, configs):
    return prepare_scores_read(calls, lambda name: (directory / name).read_bytes(), configs)


def batch_loss(model, name, calls, indices, mean, scale, group_counts, config, labels):
    base, terms = physical_loss(model, calls, indices, mean, scale, group_counts, config)
    task = base * 0.
    if name.endswith("_task"):
        inputs, slices = pack_inputs(calls, indices, mean, scale)
        _, _, responses = model(*inputs)
        weighted, weights = [], []
        for index, (start, end) in zip(indices, slices):
            values = calls[index]["values"]
            weight = 1. / group_counts[calls[index]["record"]["group"]] if values["anchor_valid"].any() else 0.
            # Missing full support contributes zero auxiliary term, not a
            # fabricated horizon; physical masked supervision still remains.
            term = labels[index].loss(responses[start:end]) if index in labels else responses[start:end].sum() * 0.
            weighted.append(term * weight)
            weights.append(weight)
        task = torch.stack(weighted).sum() / sum(weights)
    return base + config["task_effect_weight"] * task, torch.cat([terms, task[None]])


def effect_metrics(calls, indices, outputs, labels):
    grouped = {}
    for index, output in zip(indices, outputs):
        if index not in labels:
            continue
        value = float(labels[index].loss(torch.from_numpy(output["response"])).detach())
        grouped.setdefault(calls[index]["record"]["group"], []).append(value)
    expected = {calls[i]["record"]["group"] for i in indices}
    if set(grouped) != expected:
        raise ValueError("Not every development group has complete task-effect support")
    by_group = {key: float(np.mean(value)) for key, value in grouped.items()}
    return {"group_equal_normalized_task_effect_loss": float(np.mean(list(by_group.values()))),
            "by_group": by_group, "full_support_calls": sum(len(v) for v in grouped.values())}


def decision_records(calls, indices, outputs, scores, labels):
    records = []
    for index, output in zip(indices, outputs):
        if index not in labels:
            continue
        score, label = scores[index], labels[index]
        values, row = calls[index]["values"], calls[index]["record"]
        backbone = np.repeat(values["backbone"][None], len(values["proposed"]), 0)
        # All full-cost rankings below are computed by the original engine,
        # not the differentiable loss implementation or a substitute objective.
        learned = original_scores(score.call, score.actions, output["prediction"])
        motion = original_scores(score.call, score.actions, backbone + output["motion"])
        truth = row["costs"]["action_specific_truth"]
        costs = {"model": learned.tolist(), "own_motion_component": motion.tolist(), "constant_velocity": row["costs"]["constant_velocity"], "original_gru": row["costs"]["original_gru"], "action_specific_truth": truth}
        records.append({"episode_index": row["episode_index"], "step": row["step"], "agent": row["agent"], "group": row["group"],
                        "costs": costs, "metrics": {key: rank_metrics(cost, truth) for key, cost in costs.items()}})
    return records


def median_selection(runs, protocol):
    result = {}
    for name in protocol["models"]:
        chosen = sorted([r for r in runs if r["configuration"] == name], key=lambda r: (r["development"]["group_equal_ade_m"], r["seed"]))
        if len(chosen) != 3:
            raise ValueError("Fixed three-seed population required")
        result[name] = chosen[1]["seed"]
    return result


def qualification(runs, controls, decisions, protocol):
    selected, gate = median_selection(runs, protocol), protocol["development_gate"]
    fixed_motion = decisions[f"motion_only_seed{selected['motion_only']}"]
    result = {}
    for kind in ("plain", "structured"):
        name = kind + "_task"
        task_runs = [r for r in runs if r["configuration"] == name]
        mse_runs = [r for r in runs if r["configuration"] == kind + "_mse"]
        chosen = decisions[f"{name}_seed{selected[name]}"]
        keys = lambda rows: [(r["episode_index"], r["step"], r["agent"], r["group"]) for r in rows]
        if keys(chosen) != keys(fixed_motion) or not chosen:
            raise ValueError("Selected decision comparisons require same full support")
        checks = {
            "each_seed_ade_vs_gru": all(r["development"]["group_equal_ade_m"] <= controls["frozen_gru"]["group_equal_ade_m"] * gate["maximum_task_to_gru_ade_ratio_each_seed"] for r in task_runs),
            "median_ade_vs_cv": np.median([r["development"]["group_equal_ade_m"] for r in task_runs]) <= controls["constant_velocity"]["group_equal_ade_m"] * gate["maximum_task_to_cv_median_ade_ratio"],
            "median_response_error_vs_zero": np.median([r["development"]["group_equal_paired_response_error_m"] for r in task_runs]) <= controls["frozen_gru"]["group_equal_paired_response_error_m"] * gate["maximum_task_to_zero_response_median_error_ratio"],
            "median_effect_loss_vs_same_arch_mse": np.median([r["task_effect"]["group_equal_normalized_task_effect_loss"] for r in task_runs]) <= np.median([r["task_effect"]["group_equal_normalized_task_effect_loss"] for r in mse_runs]) * gate["maximum_task_to_same_arch_mse_median_effect_loss_ratio"]}
        gains = {}
        for reference, threshold_key in (("own_motion_component", "motion_component"), ("constant_velocity", "cv"), ("trained_motion_only", "trained_motion_only")):
            reference_regrets = [r["metrics"]["model"]["regret_in_diagnostic_cost"] for r in fixed_motion] if reference == "trained_motion_only" else [r["metrics"][reference]["regret_in_diagnostic_cost"] for r in chosen]
            values = [a - r["metrics"]["model"]["regret_in_diagnostic_cost"] for a, r in zip(reference_regrets, chosen)]
            measured = descriptive_bootstrap(values, [r["group"] for r in chosen], protocol["decision_bootstrap"]["draws"], protocol["decision_bootstrap"]["seed"])
            measured["interpretation"] = protocol["decision_bootstrap"]["interpretation"]
            gains[reference] = measured
            checks["selected_gain_vs_" + reference] = measured["percentile_95_interval"][0] > gate["minimum_selected_group_gain_interval_lower_bound_vs_" + threshold_key]
        result[name] = {"research_eligible": bool(all(checks.values())), "checks": {k: bool(v) for k, v in checks.items()}, "selected_seed": selected[name], "selected_gains": gains}
    return selected, result


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--capsule", type=Path, default=ROOT.parent / "cwm_v1/baseline/capsule.zip")
    parser.add_argument("--audit", type=Path, default=ROOT.parent / "cwm_v11/artifacts/full_local_score_audit_20261010.zip")
    parser.add_argument("--source", type=Path, default=ROOT.parent / "cwm_v7/artifacts/paired_training_20261010.zip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "training_protocol.json").read_text())
    data_protocol = json.loads((ROOT / "data_protocol.json").read_text())
    if any(protocol[key] for key in ("original_gru_in_optimizer", "enhanced_control_enabled", "holdout_used", "prior_gate_override_allowed")):
        raise ValueError("Frozen original/offline training contract mismatch")
    if sha(args.audit) != protocol["audit_archive_sha256"] or sha(args.capsule) != "c83a27a4360ad055b69dbced4c28370cb31a32ae89ea9fbab13523fa6218a329":
        raise ValueError("Protected score-audit/baseline source mismatch")
    check_archive(args.audit, "ARTIFACT_MANIFEST.json", False)
    if sha(args.source) != data_protocol["source_archive_sha256"]:
        raise ValueError("Frozen candidate-library source mismatch")
    check_archive(args.source, "ARTIFACT_MANIFEST.json", False)
    calls, data_report = load_calls(args.data)
    if data_report["protocol"] != data_protocol:
        raise ValueError("Only fresh V12 preassigned data may train this stage")
    public_report = json.loads((args.public / "summary.json").read_text())
    if (public_report["data_summary_sha256"] != sha(args.data / "summary.json") or public_report["enhanced_control_enabled"] or public_report["holdout_used"]
            or public_report["status"] != "public_history_and_backbone_replay_equal" or public_report["checked_calls"] != len(json.loads((args.data / "calls.json").read_text()))):
        raise ValueError("Completed independent original public replay required")
    indices = {split: [i for i, c in enumerate(calls) if c["record"]["split"] == split] for split in ("train", "development_validation")}
    full_counts = {k: sum(calls[i]["record"]["all_candidates_full_horizon_valid"] for i in selected) for k, selected in indices.items()}
    if full_counts["train"] < protocol["additional_data_gate"]["minimum_full_support_train_calls"] or full_counts["development_validation"] < protocol["additional_data_gate"]["minimum_full_support_development_calls"]:
        raise ValueError("Insufficient actual complete paired task support")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True, exist_ok=False)
    configs = frozen_configs(args.capsule, args.output / "restored")
    def dataset_read(name):
        stage, relative = name.split("/", 1)
        directories = {"data": args.data, "public": args.public}
        return (directories[stage] / relative).read_bytes()
    with zipfile.ZipFile(args.source) as source:
        audited_calls, audited_report = audit_data(dataset_read, source.read, configs, args.output / "restored")
    if audited_report != data_report or len(audited_calls) != len(calls):
        raise ValueError("Independent fresh/public/local data audit differs")
    scores, labels = prepare_scores(calls, args.data, configs)
    sources = sorted({Path(module.__file__).resolve() for module in list(sys.modules.values()) if getattr(module, "__file__", None)
                      and Path(module.__file__).resolve().is_relative_to(ROOT.parent)} | {ROOT / "data_protocol.json", ROOT / "training_protocol.json"})
    source_hashes = {p.relative_to(ROOT.parent).as_posix(): sha(p) for p in sources}
    for source in sources:
        destination = args.output / "source" / source.relative_to(ROOT.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    mean, scale = normalization(calls, indices["train"])
    np.savez_compressed(args.output / "normalization.npz", mean=mean, scale=scale)
    group_counts = {group: sum(calls[i]["record"]["group"] == group for i in indices["train"]) for group in {calls[i]["record"]["group"] for i in indices["train"]}}
    controls, runs, decisions = control_metrics(calls, indices["development_validation"]), [], {}
    config = protocol["training"]
    for seed in config["seeds"]:
        for name in protocol["models"]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            sampler = np.random.default_rng(seed)
            model = LocalTwoHead(architecture(name), config["motion_scale_m"], config["response_scale_m"])
            initial = state_digest(model)
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"])
            history, started = [], time.perf_counter()
            for epoch in range(1, config["epochs"] + 1):
                model.train()
                rows = []
                order = sampler.permutation(indices["train"])
                for offset in range(0, len(order), config["batch_calls"]):
                    chosen_indices = order[offset:offset + config["batch_calls"]]
                    loss, terms = batch_loss(model, name, calls, chosen_indices, mean, scale, group_counts, config, labels)
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite task-effect training loss")
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
                    optimizer.step()
                    rows.append([float(loss.detach()), *[float(v.detach()) for v in terms]])
                history.append({"epoch": epoch, "mean_batch_loss_motion_response_energy_task": np.mean(rows, axis=0).tolist()})
                if epoch % 20 == 0:
                    print(json.dumps({"configuration": name, "seed": seed, "epoch": epoch}), flush=True)
            measured, outputs = evaluate(model, calls, indices["development_validation"], mean, scale)
            effect = effect_metrics(calls, indices["development_validation"], outputs, labels)
            identifier = f"{name}_seed{seed}"
            decisions[identifier] = decision_records(calls, indices["development_validation"], outputs, scores, labels)
            (args.output / f"{identifier}_decisions.json").write_text(json.dumps(decisions[identifier], indent=2))
            np.savez_compressed(args.output / f"{identifier}_development_predictions.npz", **{f"call{j}_{key}": value for j, output in enumerate(outputs) for key, value in output.items()})
            torch.save({"configuration": name, "architecture_kind": architecture(name), "seed": seed, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "protocol": protocol, "normalizer_mean": torch.from_numpy(mean), "normalizer_scale": torch.from_numpy(scale), "torch_rng_state": torch.get_rng_state(),
                        "sampler_rng_state": sampler.bit_generator.state, "data_summary_sha256": sha(args.data / "summary.json"), "public_summary_sha256": sha(args.public / "summary.json"),
                        "source_hashes": source_hashes, "initial_state_digest": initial, "online_promoted": False, "baseline_weights_included": False}, args.output / f"{identifier}.pt")
            (args.output / f"{identifier}_history.json").write_text(json.dumps(history, indent=2))
            row = {"configuration": name, "architecture_kind": architecture(name), "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "initial_state_digest": initial, "development": measured, "task_effect": effect,
                   "checkpoint_sha256": sha(args.output / f"{identifier}.pt"), "elapsed_seconds": time.perf_counter() - started}
            runs.append(row)
            print(json.dumps({"configuration": name, "seed": seed, "ade_m": measured["group_equal_ade_m"], "response_error_m": measured["group_equal_paired_response_error_m"],
                              "normalized_effect_loss": effect["group_equal_normalized_task_effect_loss"]}), flush=True)
    selected, gates = qualification(runs, controls, decisions, protocol)
    if any(sha(ROOT.parent / name) != digest for name, digest in source_hashes.items()):
        raise AssertionError("Run-used training source changed")
    for seed in config["seeds"]:
        for kind in ("plain", "structured"):
            pair = [r["initial_state_digest"] for r in runs if r["seed"] == seed and r["configuration"] in (kind + "_mse", kind + "_task")]
            if len(pair) != 2 or pair[0] != pair[1]:
                raise AssertionError("MSE/task pair initializations differ")
    report = {"status": "offline_research_gate_passed_requires_further_qualification" if any(g["research_eligible"] for g in gates.values()) else "offline_task_effect_gate_failed_keep_baseline",
              "protocol": protocol, "source_hashes": source_hashes, "data_summary_sha256": sha(args.data / "summary.json"), "public_summary_sha256": sha(args.public / "summary.json"),
              "data_calls_sha256": data_report["calls_sha256"], "split_calls": {k: len(v) for k, v in indices.items()}, "full_support_calls": full_counts,
              "split_groups": {k: sorted({calls[i]["record"]["group"] for i in v}) for k, v in indices.items()}, "models": runs, "controls": controls,
              "selected_median_seeds": selected, "qualification": gates, "baseline_weights_included": False, "enhanced_control_enabled": False, "holdout_used": False,
              "prior_gate_overridden": False, "same_architecture_initializations_verified": True}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "selected_median_seeds": selected, "qualification": gates}), flush=True)


if __name__ == "__main__":
    main()
