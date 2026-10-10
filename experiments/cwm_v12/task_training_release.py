"""Independent fresh-data, fifteen-model and full-cost qualification audit."""
import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from data_audit import audit_data
from train_task_effect import (LocalTwoHead, architecture, normalization, evaluate, control_metrics, effect_metrics,
                               decision_records, prepare_scores_read, qualification, state_digest, frozen_configs, sha, check_archive)
from geometry_release import arrays, compare_arrays
from training_release import tensor_tree_equal
from two_head_release import audit_sources


def audit(read, source_read, configs, restored):
    protocol = json.loads((ROOT / "training_protocol.json").read_text())
    calls, data_report = audit_data(read, source_read, configs, restored)
    indices = {split: [i for i, call in enumerate(calls) if call["record"]["split"] == split] for split in ("train", "development_validation")}
    counts = {split: sum(calls[i]["record"]["all_candidates_full_horizon_valid"] for i in selected) for split, selected in indices.items()}
    if counts["train"] < protocol["additional_data_gate"]["minimum_full_support_train_calls"] or counts["development_validation"] < protocol["additional_data_gate"]["minimum_full_support_development_calls"]:
        raise ValueError("Actual complete-support training data gate failed")
    mean, scale = normalization(calls, indices["train"])
    scores, labels = prepare_scores_read(calls, lambda name: read("data/" + name), configs)
    controls = control_metrics(calls, indices["development_validation"])
    summaries, checkpoints = [], []
    for stage in ("primary", "retrained"):
        report = json.loads(read(stage + "/summary.json"))
        audit_sources(read, stage, report)
        if report["protocol"] != protocol or json.loads(read(stage + "/source/cwm_v12/training_protocol.json")) != protocol:
            raise ValueError("Actual predeclared task training protocol mismatch")
        if any(report[key] for key in ("baseline_weights_included", "enhanced_control_enabled", "holdout_used", "prior_gate_overridden")):
            raise ValueError("Frozen original/offline contract mismatch")
        if report["data_summary_sha256"] != hashlib.sha256(read("data/summary.json")).hexdigest() or report["public_summary_sha256"] != hashlib.sha256(read("public/summary.json")).hexdigest():
            raise ValueError("Actual fresh-data/public provenance mismatch")
        if (report["data_calls_sha256"] != data_report["calls_sha256"] or report["controls"] != controls or report["full_support_calls"] != counts
                or report["split_calls"] != {k: len(v) for k, v in indices.items()}
                or report["split_groups"] != {k: sorted({calls[i]["record"]["group"] for i in v}) for k, v in indices.items()}):
            raise ValueError("Actual split/support/control mismatch")
        stored_normalizer = arrays(read(stage + "/normalization.npz"))
        if not np.array_equal(stored_normalizer["mean"], mean) or not np.array_equal(stored_normalizer["scale"], scale):
            raise ValueError("Actual train-only normalizer mismatch")
        expected_population = {(name, seed) for name in protocol["models"] for seed in protocol["training"]["seeds"]}
        if len(report["models"]) != 15 or {(r["configuration"], r["seed"]) for r in report["models"]} != expected_population:
            raise ValueError("Actual fifteen-model population mismatch")
        decision_set, checkpoint_set = {}, {}
        for row in report["models"]:
            name, seed = row["configuration"], row["seed"]
            identifier = f"{name}_seed{seed}"
            raw = read(stage + "/" + identifier + ".pt")
            if hashlib.sha256(raw).hexdigest() != row["checkpoint_sha256"]:
                raise ValueError("Actual task checkpoint digest mismatch")
            checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            if (checkpoint["configuration"] != name or checkpoint["seed"] != seed or checkpoint["architecture_kind"] != architecture(name)
                    or checkpoint["online_promoted"] or checkpoint["baseline_weights_included"] or checkpoint["protocol"] != protocol
                    or checkpoint["data_summary_sha256"] != report["data_summary_sha256"] or checkpoint["public_summary_sha256"] != report["public_summary_sha256"]
                    or checkpoint["source_hashes"] != report["source_hashes"]):
                raise ValueError("Actual task checkpoint contract mismatch")
            if not np.array_equal(checkpoint["normalizer_mean"].numpy(), mean) or not np.array_equal(checkpoint["normalizer_scale"].numpy(), scale):
                raise ValueError("Actual checkpoint normalizer mismatch")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                model = LocalTwoHead(architecture(name), protocol["training"]["motion_scale_m"], protocol["training"]["response_scale_m"])
            initial = state_digest(model)
            if row["initial_state_digest"] != initial or checkpoint["initial_state_digest"] != initial:
                raise ValueError("Actual matched initialization mismatch")
            model.load_state_dict(checkpoint["model_state"], strict=True)
            measured, outputs = evaluate(model, calls, indices["development_validation"], mean, scale)
            if measured != row["development"] or effect_metrics(calls, indices["development_validation"], outputs, labels) != row["task_effect"]:
                raise ValueError("Independent task predictions/effect metrics mismatch")
            expected_arrays = {f"call{j}_{key}": value for j, output in enumerate(outputs) for key, value in output.items()}
            if not compare_arrays(expected_arrays, arrays(read(f"{stage}/{identifier}_development_predictions.npz"))):
                raise ValueError("Independent task prediction bytes mismatch")
            calculated = decision_records(calls, indices["development_validation"], outputs, scores, labels)
            if calculated != json.loads(read(f"{stage}/{identifier}_decisions.json")):
                raise ValueError("Independent full original-engine learned costs/rankings mismatch")
            decision_set[identifier] = calculated
            if row["parameters"] != sum(p.numel() for p in model.parameters()) or row["trainable_parameters"] != sum(p.numel() for p in model.parameters() if p.requires_grad):
                raise ValueError("Actual matched parameter budget mismatch")
            history = json.loads(read(f"{stage}/{identifier}_history.json"))
            if [r["epoch"] for r in history] != list(range(1, 81)) or not np.isfinite([r["mean_batch_loss_motion_response_energy_task"] for r in history]).all():
                raise ValueError("Actual fixed eighty-epoch history mismatch")
            if not name.endswith("_task") and any(r["mean_batch_loss_motion_response_energy_task"][-1] != 0. for r in history):
                raise ValueError("MSE/motion-only control unexpectedly received task loss")
            checkpoint_set[identifier] = checkpoint
        selected, gates = qualification(report["models"], controls, decision_set, protocol)
        if report["selected_median_seeds"] != selected or report["qualification"] != gates:
            raise ValueError("Independent development qualification/median selection mismatch")
        expected_status = "offline_research_gate_passed_requires_further_qualification" if any(g["research_eligible"] for g in gates.values()) else "offline_task_effect_gate_failed_keep_baseline"
        if report["status"] != expected_status or not report["same_architecture_initializations_verified"]:
            raise ValueError("Actual task status/initialization claim mismatch")
        for seed in protocol["training"]["seeds"]:
            for kind in ("plain", "structured"):
                pair = [checkpoint_set[f"{kind}_{loss}_seed{seed}"] for loss in ("mse", "task")]
                if pair[0]["initial_state_digest"] != pair[1]["initial_state_digest"] or not tensor_tree_equal(pair[0]["sampler_rng_state"], pair[1]["sampler_rng_state"]):
                    raise ValueError("Actual paired initialization/sampling order differs")
        summaries.append(report)
        checkpoints.append(checkpoint_set)
    for key in summaries[0].keys() - {"models"}:
        if summaries[0][key] != summaries[1][key]:
            raise ValueError("Independent task-retraining summaries differ")
    for name, checkpoint in checkpoints[0].items():
        if not tensor_tree_equal(checkpoint, checkpoints[1][name]) or read(f"primary/{name}_history.json") != read(f"retrained/{name}_history.json"):
            raise ValueError("Independent task weights/optimizer/RNG/loss history differ")
    return {"status": "passed_independent_data_and_thirty_model_recomputation", "data_episodes": 64, "data_groups": 32,
            "train_calls": len(indices["train"]), "development_calls": len(indices["development_validation"]), "full_support_calls": counts,
            "models_per_run": 15, "independent_training_runs": 2, "weights_optimizer_rng_history_equal": True,
            "public252_histories_verified": True, "reload_prediction_bytes_equal": True, "all_learned_full_cost_rankings_recomputed": True,
            "qualification": summaries[0]["qualification"], "enhanced_control_enabled": False, "holdout_used": False,
            "scope": "Fresh offline fixed-library task-effect learning; not self-generated library, sequential/closed-loop, original eight Levels or safety/latency qualification."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in ("data", "public", "primary", "retrained"):
        parser.add_argument("--" + stage, type=Path)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "cwm_v7/artifacts/paired_training_20261010.zip")
    parser.add_argument("--capsule", type=Path, default=ROOT.parent / "cwm_v1/baseline/capsule.zip")
    parser.add_argument("--score-audit", type=Path, default=ROOT.parent / "cwm_v11/artifacts/full_local_score_audit_20261010.zip")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "training_protocol.json").read_text())
    data_protocol = json.loads((ROOT / "data_protocol.json").read_text())
    if sha(args.source) != data_protocol["source_archive_sha256"] or sha(args.score_audit) != protocol["audit_archive_sha256"] or sha(args.capsule) != "c83a27a4360ad055b69dbced4c28370cb31a32ae89ea9fbab13523fa6218a329":
        raise ValueError("Protected score/candidate/baseline sources changed")
    if any(protocol[key] for key in ("original_gru_in_optimizer", "enhanced_control_enabled", "holdout_used", "prior_gate_override_allowed")):
        raise ValueError("Fixed offline task-training protocol overridden")
    for path in (args.source, args.score_audit):
        check_archive(path, "ARTIFACT_MANIFEST.json", False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    restored = ROOT.parent.parent / "results/cwm_v12/audit_restored"
    configs = frozen_configs(args.capsule, restored)
    with zipfile.ZipFile(args.source) as source:
        if args.verify:
            integrity = check_archive(args.verify, "ARTIFACT_MANIFEST.json", False)
            with zipfile.ZipFile(args.verify) as archive:
                result = audit(archive.read, source.read, configs, restored)
            print(json.dumps({**result, "artifact_sha256": integrity["sha256"]}), flush=True)
            return
        members = {}
        for stage in ("data", "public", "primary", "retrained"):
            run = getattr(args, stage)
            if run is None:
                parser.error("Completed fresh data/public replay and two training runs required")
            for file in run.rglob("*"):
                relative = file.relative_to(run)
                if file.is_file() and "restored" not in relative.parts:
                    members[f"{stage}/{relative.as_posix()}"] = file.read_bytes()
        for file in ROOT.glob("*.py"):
            members["verification_source/" + file.name] = file.read_bytes()
        result = audit(members.__getitem__, source.read, configs, restored)
        path = ROOT / "artifacts/task_effect_training_20261010.zip"
        path.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in members.items():
                archive.writestr(name, raw)
            archive.writestr("ARTIFACT_MANIFEST.json", json.dumps({name: hashlib.sha256(raw).hexdigest() for name, raw in members.items()}, indent=2))
        integrity = check_archive(path, "ARTIFACT_MANIFEST.json", False)
        with zipfile.ZipFile(path) as archive:
            audit(archive.read, source.read, configs, restored)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    for stage, run in (("data", args.data), ("public", args.public), ("training", args.primary)):
        with (reports / f"{stage}_summary.json").open("xb") as file:
            file.write((run / "summary.json").read_bytes())
    with (reports / "release_manifest.json").open("x") as file:
        json.dump({**result, "artifact_sha256": integrity["sha256"], "artifact_bytes": path.stat().st_size}, file, indent=2)
    print(json.dumps({**result, "artifact_sha256": integrity["sha256"], "artifact_bytes": path.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
