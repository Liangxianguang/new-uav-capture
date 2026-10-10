"""Audit exact scores/gradients; train-only effect mechanism diagnosis."""
import argparse
import hashlib
import json
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cwm_v10"), str(ROOT.parent / "cwm_v9"), str(ROOT.parent / "cwm_v8")]
from task_score import FrozenLocalScore, original_scores, task_effect_loss
from local_shadow import restore_public_call
from local_shadow_release import frozen_configs
from geometry_release import arrays, sha
from verify_release import check_archive
from s4_value import descriptive_bootstrap
from two_head_model import LocalTwoHead
from train_two_head import pack_inputs


def fixed_predictions(source, calls):
    import io
    training = json.loads(source.read("primary/summary.json"))
    result = {}
    for kind in ("plain", "structured"):
        seed = training["chosen_median_seed_per_architecture"][kind]
        checkpoint = torch.load(io.BytesIO(source.read(f"primary/{kind}_seed{seed}.pt")), map_location="cpu", weights_only=True)
        with torch.random.fork_rng(devices=[]):
            model = LocalTwoHead(kind)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        inputs, slices = pack_inputs(calls, list(range(len(calls))), checkpoint["normalizer_mean"].numpy(), checkpoint["normalizer_scale"].numpy())
        with torch.no_grad():
            predictions, _, responses = model(*inputs)
        result[kind] = [{"prediction": predictions[start:end].numpy(), "response": responses[start:end].numpy()} for start, end in slices]
    return result


def analyze(source, configs, protocol, progress=False):
    all_rows = json.loads(source.read("data/calls.json"))
    calls = [{"record": row, "values": arrays(source.read("data/" + row["arrays_path"]))} for row in all_rows if "arrays_path" in row]
    predictions = fixed_predictions(source, calls)
    records = []
    for position, item in enumerate(calls):
        row, values = item["record"], item["values"]
        snapshot = json.loads(source.read("data/" + row["context_path"]))
        call = restore_public_call(snapshot, *configs, values["backbone"])
        before = hashlib.sha256(pickle.dumps(call["planner"].__dict__, protocol=5)).hexdigest()
        actions = values["proposed"][:, :, row["agent"]]
        score = FrozenLocalScore(call, actions)
        count = len(actions)
        repeated = lambda path: np.repeat(path[None], count, axis=0)
        reference = repeated(values["anchor_target"])
        cv = values["reference"][None] + .1 * np.arange(1, 9)[:, None] * values["velocity"][None]
        # Numerical audit only; future paths/terminal padding are never used as inputs.
        rng = np.random.default_rng(protocol["stress_seed"] + position)
        paths = {"original_gru": repeated(values["backbone"]), "constant_velocity": repeated(cv),
                 "reference_truth": reference, "action_specific_truth": values["target"],
                 "plain_two_head": predictions["plain"][position]["prediction"],
                 "structured_two_head": predictions["structured"][position]["prediction"],
                 "stress": repeated(values["backbone"]) + rng.uniform(-protocol["stress_offset_m"], protocol["stress_offset_m"], (count, 8, 3))}
        errors, costs = {}, {}
        for method, path in paths.items():
            original = original_scores(call, actions, path)
            calculated = score(torch.from_numpy(path)).detach().numpy()
            if not np.allclose(original, calculated, rtol=protocol["score_rtol"], atol=protocol["score_atol"]):
                raise ValueError(f"Full local score mismatch {position}/{method}")
            errors[method] = float(np.max(np.abs(original - calculated)))
            costs[method] = original
        reference_tensor = torch.from_numpy(reference).requires_grad_(True)
        gradient = torch.autograd.grad(score(reference_tensor).sum(), reference_tensor)[0].numpy()
        if not np.isfinite(gradient).all():
            raise ValueError("Nonfinite local derivative")
        gradient_checks, skipped = [], 0
        step = protocol["finite_difference_step_m"]
        role_gap = score.role_gap(reference_tensor).detach().numpy()
        for candidate in sorted({0, count // 2, count - 1}):
            for t, xyz in protocol["finite_difference_coordinates"]:
                plus, minus = reference[candidate:candidate + 1].copy(), reference[candidate:candidate + 1].copy()
                plus[0, t, xyz] += step
                minus[0, t, xyz] -= step
                high = original_scores(call, actions[candidate:candidate + 1], plus)[0]
                low = original_scores(call, actions[candidate:candidate + 1], minus)[0]
                midpoint = costs["reference_truth"][candidate]
                # Skip local nonsmooth points (role switches, norm/abs kinks),
                # explicitly counted, not silently treated as gradient agreement.
                second = abs(high + low - 2 * midpoint) / step
                if role_gap[candidate] <= 4 * step or second > protocol["nondifferentiable_second_difference_per_step_threshold"]:
                    skipped += 1
                    continue
                numerical = (high - low) / (2 * step)
                automatic = float(gradient[candidate, t, xyz])
                if not np.isclose(numerical, automatic, rtol=protocol["gradient_rtol"], atol=protocol["gradient_atol"]):
                    raise ValueError(f"Original gradient mismatch {position}/{candidate}/{t}/{xyz}: {numerical}/{automatic}")
                gradient_checks.append(abs(numerical - automatic))
        record = {"episode_index": row["episode_index"], "step": row["step"], "agent": row["agent"], "group": row["group"], "split": row["split"],
                  "candidate_count": count, "score_max_absolute_errors": errors, "gradient_checked_coordinates": len(gradient_checks),
                  "gradient_skipped_nonsmooth_coordinates": skipped, "gradient_max_absolute_error": max(gradient_checks, default=0.),
                  "full_horizon_valid": row["all_candidates_full_horizon_valid"]}
        if row["split"] == "train" and row["all_candidates_full_horizon_valid"]:
            truth, base = costs["action_specific_truth"], costs["reference_truth"]
            effect = values["target"] - reference
            linear_effect = (gradient * effect).sum((1, 2))
            exact_effect = truth - base
            base_choice, truth_choice = int(np.argmin(base)), int(np.argmin(truth))
            record["train_diagnostic"] = {"reference_choice": base_choice, "truth_choice": truth_choice,
                "oracle_response_gain": float(truth[base_choice] - truth.min()), "exact_cost_effect": exact_effect.tolist(),
                "first_order_cost_effect": linear_effect.tolist(), "mean_response_m": float(np.linalg.norm(effect, axis=-1).mean()),
                "maximum_response_m": float(np.linalg.norm(effect, axis=-1).max()),
                "role_switch_candidates": int((score.roles(torch.from_numpy(values["target"])) != score.roles(reference_tensor)).sum()),
                "effect_linearization_max_error": float(np.abs(exact_effect - linear_effect).max()),
                "first_order_choice": int(np.argmin(base + linear_effect)),
                "first_order_regret": float(truth[np.argmin(base + linear_effect)] - truth.min()),
                "gradient_norm_median": float(np.median(np.linalg.norm(gradient.reshape(count, -1), axis=-1))),
                "response_label_points": count * 8}
            for kind in ("plain", "structured"):
                response = torch.from_numpy(predictions[kind][position]["response"])
                effect_cost = score(torch.from_numpy(reference) + response).detach().numpy()
                physical_error = float(np.linalg.norm(response.numpy() - effect, axis=-1).mean())
                record["train_diagnostic"][kind] = {"regret_with_true_reference": float(truth[int(np.argmin(effect_cost))] - truth.min()),
                    "normalized_task_effect_loss": float(task_effect_loss(score, response, torch.from_numpy(values["target"]), torch.from_numpy(values["anchor_target"]), torch.from_numpy(values["valid"]), torch.from_numpy(values["anchor_valid"])).detach()),
                    "paired_response_error_m_including_anchor": physical_error}
        after = hashlib.sha256(pickle.dumps(call["planner"].__dict__, protocol=5)).hexdigest()
        if before != after:
            raise ValueError("Score audit mutated original planner state")
        records.append(record)
        if progress and (position + 1) % 64 == 0:
            print(json.dumps({"audited_calls": position + 1, "total_calls": len(calls)}), flush=True)
    diagnostics = [r for r in records if "train_diagnostic" in r]
    if not diagnostics or any("train_diagnostic" in r for r in records if r["split"] != "train"):
        raise ValueError("Train-only diagnosis support mismatch")
    groups = [r["group"] for r in diagnostics]
    values = [r["train_diagnostic"]["oracle_response_gain"] for r in diagnostics]
    group_equal = lambda key: float(np.mean([np.mean([r["train_diagnostic"][key] for r in diagnostics if r["group"] == group]) for group in sorted(set(groups))]))
    gain_summary = descriptive_bootstrap(values, groups, protocol["bootstrap_draws"], protocol["bootstrap_seed"])
    gain_summary["interpretation"] = "Training groups only; descriptive mechanism evidence, not generalization or untouched validation/test."
    summary = {"status": "full_local_score_and_piecewise_gradient_verified_not_promoted", "calls": len(records),
        "score_methods": list(records[0]["score_max_absolute_errors"]), "candidates": sum(r["candidate_count"] for r in records),
        "score_max_absolute_error": max(max(r["score_max_absolute_errors"].values()) for r in records),
        "gradient_checked_coordinates": sum(r["gradient_checked_coordinates"] for r in records),
        "gradient_skipped_nonsmooth_coordinates": sum(r["gradient_skipped_nonsmooth_coordinates"] for r in records),
        "gradient_max_absolute_error": max(r["gradient_max_absolute_error"] for r in records),
        "training_calls": sum(r["split"] == "train" for r in records), "training_full_support_calls": len(diagnostics), "training_groups": len(set(groups)),
        "training_oracle_response_gain": gain_summary,
        "training_choice_changes": sum(r["train_diagnostic"]["reference_choice"] != r["train_diagnostic"]["truth_choice"] for r in diagnostics),
        "training_positive_gain_calls": sum(v > 1e-9 for v in values),
        "training_sub5cm_max_response_positive_gain_calls": sum(r["train_diagnostic"]["maximum_response_m"] <= protocol["strong_response_threshold_m"] and r["train_diagnostic"]["oracle_response_gain"] > 1e-9 for r in diagnostics),
        "training_role_switch_candidates": sum(r["train_diagnostic"]["role_switch_candidates"] for r in diagnostics),
        "training_group_equal": {key: group_equal(key) for key in ("mean_response_m", "maximum_response_m", "effect_linearization_max_error", "first_order_regret", "gradient_norm_median")},
        "training_learned_with_truth_reference": {kind: {key: float(np.mean([np.mean([r["train_diagnostic"][kind][key] for r in diagnostics if r["group"] == g]) for g in sorted(set(groups))]))
                                                  for key in ("regret_with_true_reference", "normalized_task_effect_loss", "paired_response_error_m_including_anchor")} for kind in ("plain", "structured")},
        "enhanced_control_enabled": False, "new_model_training_enabled": False, "holdout_used": False, "prior_gate_overridden": False,
        "limitations": "K=1 frozen non-QDR/non-FC/no escape-gap/reachability objective only. Exact constant original safety/control terms, not a safety certificate or executable adapter. Local derivatives omit discrete role/kink derivatives; skips counted. TRAIN ONLY mechanism diagnosis; no new accuracy/generalization/control claim. Raw paths including invalid/padded labels used solely for numerical cost audit, never full-horizon task labels. V10 failed gate unchanged."}
    return records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "cwm_v10/artifacts/two_head_training_20261010.zip")
    parser.add_argument("--capsule", type=Path, default=ROOT.parent / "cwm_v1/baseline/capsule.zip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if sha(args.source) != protocol["training_archive_sha256"] or sha(args.capsule) != protocol["baseline_capsule_sha256"]:
        raise ValueError("Pinned source/capsule mismatch")
    if any(protocol[k] for k in ("new_model_training_enabled", "enhanced_control_enabled", "holdout_used", "prior_gate_override_allowed")):
        raise ValueError("Diagnostic contract mismatch")
    check_archive(args.source, "ARTIFACT_MANIFEST.json", False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    configs = frozen_configs(args.capsule, args.output / "restored")
    sources = sorted({Path(module.__file__).resolve() for module in list(sys.modules.values())
                      if getattr(module, "__file__", None) and Path(module.__file__).resolve().is_relative_to(ROOT.parent)}
                     | {ROOT / "protocol.json"})
    digests = {p.relative_to(ROOT.parent).as_posix(): sha(p) for p in sources}
    for p in sources:
        destination = args.output / "source" / p.relative_to(ROOT.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())
    started = time.perf_counter()
    with zipfile.ZipFile(args.source) as source:
        records, summary = analyze(source, configs, protocol, True)
    if any(sha(ROOT.parent / name) != digest for name, digest in digests.items()):
        raise ValueError("Audit sources changed during process")
    summary.update(protocol=protocol, source_hashes=digests, elapsed_seconds=time.perf_counter() - started)
    (args.output / "records.json").write_text(json.dumps(records, indent=2))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
