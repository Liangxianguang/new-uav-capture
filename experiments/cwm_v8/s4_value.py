"""Frozen S4 forecast information value, with unchanged original control replay."""
import argparse
import copy
import hashlib
import io
import json
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent / "cwm_v5"), str(ROOT.parent / "cwm_v7")]
from decision_value import ScoreTap, rank_metrics, group_bootstrap
from freeze_baseline import sha, verify
from diagnose import fingerprint
from verify_release import check_archive
from train_response import ResponseModel, evaluate

METHODS = ("original_gru", "plain_learned", "structured_learned", "constant_velocity", "gru_plus_exact_response", "fixed_reference_truth",
           "reference_truth_plus_plain_response", "reference_truth_plus_structured_response", "action_specific_truth")


def descriptive_bootstrap(values, groups, draws, seed):
    result = group_bootstrap(values, groups, draws, seed)
    result["interpretation"] = "Previously inspected S4 development groups; descriptive bootstrap only, not untouched validation or test."
    return result


def information_paths(backbone, truth, reference, velocity, predictions):
    if np.asarray(backbone).shape != (8, 3) or np.asarray(truth).shape != (8, 8, 3):
        raise ValueError("Frozen horizon/probe forecast contract mismatch")
    paths = {"original_gru": np.repeat(backbone[None], 8, 0),
             "constant_velocity": np.repeat((reference[None] + .1 * np.arange(1, 9)[:, None] * velocity[None])[None], 8, 0),
             "gru_plus_exact_response": backbone[None] + truth - truth[:1],
             "fixed_reference_truth": np.repeat(truth[:1], 8, 0), "action_specific_truth": truth.copy()}
    for kind in ("plain", "structured"):
        prediction = predictions[kind]
        if prediction["prediction"].shape != (8, 8, 3) or prediction["response"].shape != (8, 8, 3):
            raise ValueError("Reloaded model output shape mismatch")
        paths[kind + "_learned"] = prediction["prediction"].copy()
        paths["reference_truth_plus_" + kind + "_response"] = truth[:1] + prediction["response"]
    if set(paths) != set(METHODS) or any(not np.isfinite(v).all() for v in paths.values()):
        raise ValueError("Incomplete or nonfinite information condition")
    return paths


def summarize(records, protocol):
    selected = [r for r in records if r["all_candidates_full_horizon_valid"]]
    groups = [r["group"] for r in selected]
    result = {"all_windows": len(records), "rankable_windows": len(selected), "excluded_incomplete_windows": len(records) - len(selected),
              "methods": {}, "motion_research_signal": False, "response_research_signal_given_motion": False}
    for method in METHODS:
        regret = [r["metrics"][method]["regret_in_diagnostic_cost"] for r in selected]
        gain = [r["metrics"]["original_gru"]["regret_in_diagnostic_cost"] - r["metrics"][method]["regret_in_diagnostic_cost"] for r in selected]
        result["methods"][method] = {"choice_changes_vs_original_gru": sum(r["metrics"][method]["choice"] != r["metrics"]["original_gru"]["choice"] for r in selected),
                                      "tie_optimal_choices": sum(r["metrics"][method]["realized_tie_optimal"] for r in selected),
                                      "mean_regret_in_diagnostic_cost": float(np.mean(regret)) if regret else None,
                                      "gain_vs_original_gru": descriptive_bootstrap(gain, groups, protocol["group_bootstrap_draws"], protocol["group_bootstrap_seed"])}
    result["comparisons"] = {}
    for name, first, second in (("motion_correction", "original_gru", "fixed_reference_truth"),
                                ("exact_response_on_original_gru", "original_gru", "gru_plus_exact_response"),
                                ("exact_response_given_motion_truth", "fixed_reference_truth", "action_specific_truth"),
                                ("plain_response_given_motion_truth", "fixed_reference_truth", "reference_truth_plus_plain_response"),
                                ("structured_response_given_motion_truth", "fixed_reference_truth", "reference_truth_plus_structured_response")):
        changes = sum(r["metrics"][first]["choice"] != r["metrics"][second]["choice"] for r in selected)
        gain = np.asarray([r["metrics"][first]["regret_in_diagnostic_cost"] - r["metrics"][second]["regret_in_diagnostic_cost"] for r in selected])
        g = np.asarray(groups)
        group_means = {str(group): float(gain[g == group].mean()) for group in sorted(set(groups))}
        positive = sum(v > protocol["research_signal"]["minimum_group_equal_diagnostic_cost_gain"] for v in group_means.values())
        bootstrap = descriptive_bootstrap(gain, groups, protocol["group_bootstrap_draws"], protocol["group_bootstrap_seed"])
        spec = protocol["research_signal"]
        signal = (changes >= spec["minimum_choice_changes"] and positive >= spec["minimum_positive_gain_groups"]
                  and (bootstrap["group_equal_mean"] or 0.) > spec["minimum_group_equal_diagnostic_cost_gain"])
        result["comparisons"][name] = {"choice_changes": changes, "positive_gain_groups": positive, "gain_by_group": group_means,
                                       "diagnostic_regret_gain": bootstrap, "exploratory_research_signal": signal}
    result["motion_research_signal"] = result["comparisons"]["motion_correction"]["exploratory_research_signal"]
    result["response_research_signal_given_motion"] = result["comparisons"]["exact_response_given_motion_truth"]["exploratory_research_signal"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    protocol = json.loads(args.protocol.read_text())
    if sha(args.archive) != protocol["source_archive_sha256"] or protocol["enhanced_control_enabled"] or protocol["holdout_used"] or protocol["new_model_training_enabled"]:
        raise ValueError("Frozen input / disabled-control contract mismatch")
    check_archive(args.archive, "ARTIFACT_MANIFEST.json", False)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with zipfile.ZipFile(args.archive) as archive:
        all_scenes = [json.loads(line) for line in archive.read("data/selected_scenes.jsonl").decode().splitlines()]
        scenes = [r for r in all_scenes if r["model_split"] == protocol["split"]]
        all_windows = json.loads(archive.read("data/windows.json"))
        with np.load(io.BytesIO(archive.read("data/pairs.npz")), allow_pickle=False) as raw:
            data = {k: raw[k] for k in raw.files}
        indices = np.flatnonzero(data["split"] == protocol["split"])
        index_positions = {int(index): position for position, index in enumerate(indices)}
        windows = [w for i, w in enumerate(all_windows) if i in index_positions]
        archived_trajectories = {r["episode_index"]: archive.read(f"data/observed/{r['episode_index']}.npz") for r in scenes}
        model_report = json.loads(archive.read("primary/summary.json"))
        if model_report["chosen_median_seed_per_architecture"] != protocol["selected_median_seeds"] or model_report["holdout_used"]:
            raise ValueError("Do not substitute model seeds or use holdout")
        predictions = {}
        for kind, seed in protocol["selected_median_seeds"].items():
            checkpoint = torch.load(io.BytesIO(archive.read(f"primary/{kind}_seed{seed}.pt")), map_location="cpu", weights_only=True)
            if checkpoint["online_promoted"] or checkpoint["baseline_weights_included"]:
                raise ValueError("Invalid model deployment contract")
            model = ResponseModel(kind, checkpoint["protocol"]["training"]["response_scale_m"])
            model.load_state_dict(checkpoint["model_state"], strict=True)
            _, prediction = evaluate(model, data, indices, checkpoint["normalizer_mean"].numpy(), checkpoint["normalizer_scale"].numpy())
            with np.load(io.BytesIO(archive.read(f"primary/{kind}_seed{seed}_development_predictions.npz")), allow_pickle=False) as recorded:
                if any(recorded[k].dtype != prediction[k].dtype or recorded[k].shape != prediction[k].shape or recorded[k].tobytes() != prediction[k].tobytes() for k in prediction):
                    raise AssertionError("Reloaded chosen model does not reproduce archived output bytes")
            predictions[kind] = prediction
    lookup = {(w["episode_index"], w["step"]): i for i, w in enumerate(all_windows) if i in index_positions}
    base = ScoreTap(args.capsule, args.output / "restored")
    from encirclement3d.minimax_mpc import ScenarioTrajectorySet, aggregate_scenario_costs
    sources = [Path(__file__), args.protocol] + [ROOT.parent / p for p in ("cwm_v5/decision_value.py", "cwm_v4/diagnose.py", "cwm_v2/scout.py",
              "cwm_v1/baseline.py", "cwm_v1/collect_pairs.py", "cwm_v1/diagnose_interactions.py", "cwm_v1/freeze_baseline.py", "cwm_v1/verify_release.py",
              "cwm_v7/train_response.py", "cwm_v7/response_model.py", "cwm_v7/geometry.py", "cwm_v6/s4_collect.py")]
    source_hashes = {p.relative_to(ROOT.parent).as_posix(): sha(p) for p in sources}
    for p in sources:
        path = args.output / "source" / p.relative_to(ROOT.parent)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(p.read_bytes())
    records, episodes = [], []
    started = time.perf_counter()
    for scene in scenes:
        public_history = []
        def observer(env, observation, actions, sequence):
            public_history.append(base.evaluator.policy_observations(env, observation).reshape(-1).copy())
            key = (scene["episode_index"], int(env.step_count))
            if key not in lookup:
                return
            index = lookup[key]
            position = index_positions[index]
            history = np.stack([public_history[0]] * max(0, 8 - len(public_history)) + public_history[-8:])
            reference, velocity = base.evaluator.belief_reference(observation, base.planner_config)
            relative = np.concatenate([observation["defender_positions"] - reference, observation["defender_velocities"] - velocity], -1)
            if not np.array_equal(history, data["history"][index]) or not np.array_equal(relative, data["relative"][index]):
                raise AssertionError("Model's archived public context differs from current original replay")
            if not np.array_equal(reference, data["reference"][index]) or not np.array_equal(velocity, data["reference_velocity"][index]):
                raise AssertionError("Public reference differs from collection")
            if sequence is None or sequence.tobytes() != data["proposed"][index, 0, :8].tobytes() or base.last_scenarios.trajectories[0].tobytes() != data["backbone"][index].tobytes():
                raise AssertionError("Frozen original plan/GRU differs from archived collection")
            if base.last_step_index != env.step_count:
                raise AssertionError("Planner timestamp mismatch")
            valid = bool(data["valid"][index, :, :8].all())
            row = {"episode_index": scene["episode_index"], "group": scene["mirror_group_id"], "variant": scene["variant"], "step": env.step_count,
                   "all_candidates_full_horizon_valid": valid, "model_public_context_matches_current_replay": True}
            before = fingerprint(env)
            planner_before = hashlib.sha256(pickle.dumps(base.last_planner.__dict__, protocol=5)).hexdigest()
            if valid:
                paths = information_paths(data["backbone"][index], data["target"][index, :, :8], reference, velocity,
                                          {kind: {k: v[position] for k, v in prediction.items()} for kind, prediction in predictions.items()})
                costs = {}
                for method in METHODS:
                    values = []
                    for proposed, trajectory in zip(data["proposed"][index, :, :8], paths[method]):
                        planner = copy.deepcopy(base.last_planner)
                        forecast = ScenarioTrajectorySet(trajectory[None], np.ones(1), dynamics_status="raw")
                        values.append(aggregate_scenario_costs(planner._team_scenario_costs(base.last_planner_observation, forecast, proposed, base.last_step_index),
                                                              forecast.normalized_weights, planner.config.risk_mode, planner.config.cvar_alpha))
                    costs[method] = np.asarray(values)
                if not np.isclose(costs["original_gru"][0], base.last_objective, rtol=1e-10, atol=1e-8):
                    raise AssertionError("Complete copied reference score differs from original DN-MPC diagnostic")
                row["costs"] = {k: v.tolist() for k, v in costs.items()}
                row["metrics"] = {k: rank_metrics(v, costs["action_specific_truth"], protocol["tie_absolute_cost_tolerance"]) for k, v in costs.items()}
                row["reference_score_matches_original_diagnostics"] = True
            if before != fingerprint(env) or planner_before != hashlib.sha256(pickle.dumps(base.last_planner.__dict__, protocol=5)).hexdigest():
                raise AssertionError("Diagnostic scoring changed original parent state")
            row["parent_integrity"] = True
            records.append(row)
            (args.output / "windows.json").write_text(json.dumps(records, indent=2))
            print(json.dumps({"episode": scene["episode_index"], "step": env.step_count, "rankable": valid}), flush=True)
        path = args.output / "trajectories" / f"{scene['episode_index']}.npz"
        result, _ = base.run(scene, path, observer)
        with np.load(path, allow_pickle=False) as actual, np.load(io.BytesIO(archived_trajectories[scene["episode_index"]]), allow_pickle=False) as original:
            if any(actual[k].dtype != original[k].dtype or actual[k].shape != original[k].shape or actual[k].tobytes() != original[k].tobytes() for k in original.files):
                raise AssertionError("Value diagnostic changed original trajectory arrays")
        episodes.append({"episode_index": scene["episode_index"], "trajectory_byte_equal": True, "trajectory_sha256": sha(path),
                         **{k: result[k] for k in ("safe_capture_success", "collision", "boundary_violation", "timeout", "target_invalid_episode", "termination_reason")}})
    if {(r["episode_index"], r["step"]) for r in records} != set(lookup):
        raise AssertionError("Incomplete original validation snapshot coverage")
    verify(base.root, base.capsule_manifest)
    if any(sha(ROOT.parent / n) != expected for n, expected in source_hashes.items()):
        raise AssertionError("Source changed during diagnostic")
    summary = {"status": "development_value_diagnostic_complete_not_promoted", "protocol": protocol, "statistics": summarize(records, protocol),
               "episodes": episodes, "source_hashes": source_hashes, "source_archive_sha256": sha(args.archive), "capsule_sha256": sha(args.capsule),
               "chosen_model_outputs_reloaded_byte_equal": True, "prior_training_gate_passed": model_report["development_gate_passed"],
               "prior_training_gate_overridden": False, "enhanced_control_enabled": False, "holdout_used": False,
               "elapsed_seconds": time.perf_counter() - started,
               "limitations": ["Previously inspected S4 development data, not original Levels or untouched holdout.",
                               "Ranks eight predefined proposed-command probes using copied full team scores; not sequential optimizer choices, actual CBF execution or realized capture/safety.",
                               "Oracle conditions are raw label-only information probes, not deployable inputs or formal bounds.",
                               "All information conditions share joint full8 support; no post-terminal scoring.",
                               "Research-signal flags do not authorize training-gate override or controller promotion."]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": summary["status"], "statistics": summary["statistics"]}), flush=True)


if __name__ == "__main__":
    main()
