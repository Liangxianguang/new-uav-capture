"""Audit actual collection, reload all weights, and compare deterministic retraining."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch

from geometry_release import ROOT, arrays, compare_arrays, target_bad, BASELINE_SHA, sha
from s4_collect import s4_statistics
from train_response import ResponseModel, evaluate, metrics, normalization, validate_splits
from verify_release import check_archive


def tensor_tree_equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(tensor_tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) == type(b) and len(a) == len(b) and all(tensor_tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def audit_data(read):
    report = json.loads(read("data/summary.json"))
    protocol = report["protocol"]
    if report["capsule_sha256"] != BASELINE_SHA or report["enhanced_control_enabled"] or report["new_model_trained"] or report["holdout_collected"]:
        raise ValueError("Collection baseline/disabled/holdout contract mismatch")
    raw = read("data/pairs.npz")
    if hashlib.sha256(raw).hexdigest() != report["dataset_sha256"]:
        raise ValueError("Actual collection digest mismatch")
    data = arrays(raw)
    scenes = [json.loads(line) for line in read("data/selected_scenes.jsonl").decode().splitlines()]
    episodes = json.loads(read("data/episodes.json"))
    windows = json.loads(read("data/windows.json"))
    selected, assigned = validate_splits(data, scenes, protocol)
    if len(scenes) != 64 or len(assigned) != 32 or len(episodes) != len(scenes) or episodes != report["episodes_results"]:
        raise ValueError("Actual collection population mismatch")
    for name, expected in report["source_hashes"].items():
        if hashlib.sha256(read("data/source/" + name)).hexdigest() != expected:
            raise ValueError("Recorded collection source mismatch")
    if json.loads(read("data/source/cwm_v7/training_data_protocol.json")) != protocol:
        raise ValueError("Actual collection protocol mismatch")
    by_id = {r["episode_index"]: r for r in scenes}
    expected_windows = set()
    for row in episodes:
        index = row["episode_index"]
        scene = by_id[index]
        a, b = [arrays(read(f"data/{kind}/{index}.npz")) for kind in ("observed", "unobserved")]
        if not compare_arrays(a, b) or not row["trajectory_byte_equal"] or not row["outcomes_equal"]:
            raise ValueError("Actual original-controller arrays changed")
        for kind in ("observed", "unobserved"):
            if hashlib.sha256(read(f"data/{kind}/{index}.npz")).hexdigest() != row[kind + "_sha256"]:
                raise ValueError("Actual collection trajectory digest mismatch")
        if bool(target_bad(a["target_positions"], scene["scenario"]).any()) != row["target_invalid_episode"] or row["target_invalid_episode"]:
            raise ValueError("Training scene is target-invalid")
        steps = a["target_positions"].shape[0] - 1
        expected_windows.update((index, t) for t in protocol["snapshot_steps"] if t < steps)
    if len(windows) != len(expected_windows) or {(w["episode_index"], w["step"]) for w in windows} != expected_windows:
        raise ValueError("Actual training snapshot population mismatch")
    for i, window in enumerate(windows):
        record = by_id[window["episode_index"]]
        if not window["parent_integrity"] or not window["repeat_exact"] or str(data["group"][i]) != record["mirror_group_id"] or str(data["split"][i]) != record["model_split"] or data["step"][i] != window["step"]:
            raise ValueError("Actual training window provenance mismatch")
        for probe in range(8):
            observed = data["termination"][i, probe] != "after_terminal"
            count = int(observed.sum())
            if not np.array_equal(observed, np.arange(16) < count) or count <= 0:
                raise ValueError("Paired terminal mask is not a contiguous prefix")
            bad = target_bad(data["target"][i, probe, :count], record["scenario"])
            expected = np.zeros(16, bool)
            expected[:count] = ~np.maximum.accumulate(bad)
            if not np.array_equal(expected, data["valid"][i, probe]):
                raise ValueError("Actual paired physics/validity mismatch")
    statistics = s4_statistics(data, protocol)
    if statistics != report["statistics"] or not report["training_eligible"] or not statistics["short_horizon_data_gate_passed"]:
        raise ValueError("Actual collection signal gate mismatch")
    if report["target_valid_original_episodes"] != 64 or report["states"] != len(windows) or report["episodes"] != 64 or report["groups"] != 32:
        raise ValueError("Actual dataset aggregate mismatch")
    return data, selected, assigned, report


def audit(read):
    data, selected, assigned, data_report = audit_data(read)
    mean, scale = normalization(data, selected["train"])
    summaries, checkpoint_sets = [], []
    for stage in ("primary", "retrained"):
        report = json.loads(read(stage + "/summary.json"))
        protocol = report["protocol"]
        if report["dataset_sha256"] != data_report["dataset_sha256"] or report["enhanced_control_enabled"] or report["holdout_used"] or report["baseline_weights_included"]:
            raise ValueError("Training source/disabled/holdout contract mismatch")
        if report["split_groups"] != assigned or report["split_states"] != {k: len(v) for k, v in selected.items()}:
            raise ValueError("Actual training split population mismatch")
        stored_norm = arrays(read(stage + "/normalization.npz"))
        if not np.array_equal(stored_norm["mean"], mean) or not np.array_equal(stored_norm["scale"], scale):
            raise ValueError("Normalizer differs from original train-only histories")
        if report["data_summary_sha256"] != hashlib.sha256(read("data/summary.json")).hexdigest():
            raise ValueError("Training collection provenance mismatch")
        for name, expected in report["source_hashes"].items():
            if hashlib.sha256(read(stage + "/source/" + name)).hexdigest() != expected:
                raise ValueError("Actual training source mismatch")
        if json.loads(read(stage + "/source/training_protocol.json")) != protocol:
            raise ValueError("Actual training protocol mismatch")
        expected_models = {(kind, seed) for kind in protocol["models"] for seed in protocol["training"]["seeds"]}
        if len(report["models"]) != len(expected_models) or {(r["kind"], r["seed"]) for r in report["models"]} != expected_models:
            raise ValueError("Incomplete trained model population")
        checkpoints = {}
        for row in report["models"]:
            name = f"{row['kind']}_seed{row['seed']}"
            raw = read(f"{stage}/{name}.pt")
            if hashlib.sha256(raw).hexdigest() != row["checkpoint_sha256"]:
                raise ValueError("Checkpoint digest mismatch")
            checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            if checkpoint["online_promoted"] or checkpoint["baseline_weights_included"] or checkpoint["protocol"] != protocol or checkpoint["dataset_sha256"] != report["dataset_sha256"] or checkpoint["source_hashes"] != report["source_hashes"]:
                raise ValueError("Checkpoint contract mismatch")
            if not np.array_equal(checkpoint["normalizer_mean"].numpy(), mean) or not np.array_equal(checkpoint["normalizer_scale"].numpy(), scale):
                raise ValueError("Checkpoint normalizer mismatch")
            model = ResponseModel(row["kind"], protocol["training"]["response_scale_m"])
            model.load_state_dict(checkpoint["model_state"], strict=True)
            actual_metrics, predictions = evaluate(model, data, selected["development_validation"], mean, scale)
            if actual_metrics != row["development_validation"] or row["parameters"] != sum(p.numel() for p in model.parameters()):
                raise ValueError("Reloaded development metrics differ from actual model")
            if not compare_arrays(predictions, arrays(read(f"{stage}/{name}_development_predictions.npz"))):
                raise ValueError("Reloaded prediction bytes differ from recorded arrays")
            history = json.loads(read(f"{stage}/{name}_history.json"))
            if [h["epoch"] for h in history] != list(range(1, protocol["training"]["epochs"] + 1)) or not np.isfinite([h["mean_batch_loss_terms"] for h in history]).all():
                raise ValueError("Fixed epoch/history contract mismatch")
            checkpoints[name] = checkpoint
        indices = selected["development_validation"]
        zeros = np.zeros_like(data["target"][indices, :, :8])
        backbone = np.repeat(data["backbone"][indices, None], 8, axis=1)
        zero_metrics = metrics(backbone, zeros, None, data, indices)
        cv = data["reference"][indices, None, None] + .1 * np.arange(1, 9)[None, None, :, None] * data["reference_velocity"][indices, None, None]
        cv = np.repeat(cv, 8, axis=1)
        if report["controls"] != {"frozen_gru_zero_response": zero_metrics, "constant_velocity_zero_response": metrics(cv, zeros, None, data, indices)}:
            raise ValueError("Actual zero-response controls mismatch")
        errors = {kind: [r["development_validation"]["group_equal_paired_effect_error_m"] for r in report["models"] if r["kind"] == kind] for kind in protocol["models"]}
        gate = protocol["development_gate"]
        eligible = all(v <= zero_metrics["group_equal_paired_effect_error_m"] * gate["maximum_structured_to_zero_group_error_ratio_each_seed"] for v in errors["structured"])
        eligible &= np.median(errors["structured"]) <= np.median(errors["plain"]) * gate["maximum_structured_to_plain_median_group_error_ratio"]
        chosen = {kind: sorted([r for r in report["models"] if r["kind"] == kind], key=lambda r: (r["development_validation"]["group_equal_paired_effect_error_m"], r["seed"]))[1]["seed"] for kind in protocol["models"]}
        expected_status = "offline_development_gate_passed_requires_decision_holdout_validation" if eligible else "offline_trained_not_promoted_development_gate_failed"
        if report["development_gate_passed"] != bool(eligible) or report["status"] != expected_status or report["chosen_median_seed_per_architecture"] != chosen:
            raise ValueError("Actual development gate/selection mismatch")
        summaries.append(report)
        checkpoint_sets.append(checkpoints)
    for key in ("protocol", "dataset_sha256", "data_summary_sha256", "source_hashes", "split_groups", "split_states", "controls", "chosen_median_seed_per_architecture", "development_gate_passed"):
        if summaries[0][key] != summaries[1][key]:
            raise ValueError("Independent retraining summary differs")
    for name, checkpoint in checkpoint_sets[0].items():
        if not tensor_tree_equal(checkpoint, checkpoint_sets[1][name]):
            raise ValueError("Actual deterministic retraining tensors/RNG/optimizer differ")
        if read(f"primary/{name}_history.json") != read(f"retrained/{name}_history.json"):
            raise ValueError("Actual deterministic loss history differs")
    return {"status": "passed", "models_per_run": 6, "independent_training_runs": 2, "weights_optimizer_rng_equal": True,
            "reload_prediction_arrays_byte_equal": True, "epochs": summaries[0]["protocol"]["training"]["epochs"],
            "dataset_states": len(data["group"]), "dataset_episodes": 64, "train_groups": 24, "development_groups": 8,
            "development_gate_passed": summaries[0]["development_gate_passed"], "holdout_used": False,
            "enhanced_control_enabled": False, "baseline_weights_included": False,
            "scope": "Offline paired-response training/reproducibility only; not DN-MPC decision value or closed-loop safety/capture improvement."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--primary", type=Path)
    parser.add_argument("--retrained", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if sha(ROOT.parent / "cwm_v1/baseline/capsule.zip") != BASELINE_SHA:
        raise ValueError("Protected baseline capsule changed")
    if args.verify:
        integrity = check_archive(args.verify, "ARTIFACT_MANIFEST.json", False)
        with zipfile.ZipFile(args.verify) as archive:
            result = audit(archive.read)
        print(json.dumps({**result, "artifact_sha256": integrity["sha256"]}))
        return
    if args.data is None or args.primary is None or args.retrained is None:
        parser.error("Complete collection and both training runs required")
    members = {}
    for stage, run in (("data", args.data), ("primary", args.primary), ("retrained", args.retrained)):
        for file in run.rglob("*"):
            relative = file.relative_to(run)
            if file.is_file() and "restored" not in relative.parts and file.name != "pairs_partial.npz":
                members[f"{stage}/{relative.as_posix()}"] = file.read_bytes()
    for file in ROOT.glob("*.py"):
        members["verification_source/" + file.name] = file.read_bytes()
    result = audit(members.__getitem__)
    path = ROOT / "artifacts/paired_training_20261010.zip"
    path.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)
        archive.writestr("ARTIFACT_MANIFEST.json", json.dumps({n: hashlib.sha256(v).hexdigest() for n, v in members.items()}, indent=2))
    check_archive(path, "ARTIFACT_MANIFEST.json", False)
    with zipfile.ZipFile(path) as archive:
        audit(archive.read)
    reports = ROOT / "reports"
    for stage, run in (("data", args.data), ("training", args.primary)):
        with (reports / f"{stage}_summary.json").open("xb") as file:
            file.write((run / "summary.json").read_bytes())
    with (reports / "training_release_manifest.json").open("x") as file:
        json.dump({**result, "artifact_sha256": sha(path), "artifact_bytes": path.stat().st_size}, file, indent=2)
    print(json.dumps({**result, "artifact_sha256": sha(path), "artifact_bytes": path.stat().st_size}))


if __name__ == "__main__":
    main()
