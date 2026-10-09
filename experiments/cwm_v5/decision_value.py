"""Offline information-value probe using complete frozen distributed MPC scores."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cwm_v4"))
from diagnose import BackboneTap, fingerprint, sha, verify

METHODS = ("original_gru", "gru_plus_oracle_response", "fixed_reference_truth", "action_specific_truth")


def rank_metrics(costs, realized, tolerance=1e-9):
    costs, realized = np.asarray(costs), np.asarray(realized)
    if costs.ndim != 1 or costs.shape != realized.shape or not np.isfinite(costs).all() or not np.isfinite(realized).all():
        raise ValueError("Finite equal-length candidate cost vectors required")
    choice = int(np.argmin(costs))
    regret = float(realized[choice] - realized.min())
    return {"choice": choice, "regret_in_diagnostic_cost": regret,
            "realized_cost_at_choice": float(realized[choice]),
            "realized_tie_optimal": bool(regret <= tolerance)}


def group_bootstrap(values, groups, draws=2000, seed=950910):
    values, groups = np.asarray(values, dtype=np.float64), np.asarray(groups)
    unique = sorted(set(groups.tolist()))
    if not len(values) or len(values) != len(groups):
        return {"groups": 0, "group_equal_mean": None, "percentile_95_interval": None}
    means = np.asarray([values[groups == g].mean() for g in unique])
    rng = np.random.default_rng(seed)
    boot = means[rng.integers(0, len(means), size=(draws, len(means)))].mean(axis=1)
    return {"groups": len(unique), "group_equal_mean": float(means.mean()),
            "percentile_95_interval": [float(v) for v in np.quantile(boot, [.025, .975])],
            "interpretation": "Descriptive group bootstrap; few exploratory TRAIN groups, not fresh inference."}


def summarize(records, protocol):
    valid = [r for r in records if r["all_candidates_full_horizon_valid"]]
    result = {"all_windows": len(records), "rankable_windows": len(valid),
              "excluded_incomplete_windows": len(records) - len(valid), "methods": {}, "by_variant": {}}
    for method in METHODS:
        rows = [r["metrics"][method] for r in valid]
        result["methods"][method] = {"choice_changes_vs_original_gru": sum(r["metrics"][method]["choice"] != r["metrics"]["original_gru"]["choice"] for r in valid),
                                      "tie_optimal_choices": sum(x["realized_tie_optimal"] for x in rows),
                                      "mean_regret_in_diagnostic_cost": float(np.mean([x["regret_in_diagnostic_cost"] for x in rows])) if rows else None}
    for variant in sorted({r["variant"] for r in records}):
        selected = [r for r in valid if r["variant"] == variant]
        groups = [r["group"] for r in selected]
        gain = [r["metrics"]["original_gru"]["regret_in_diagnostic_cost"] - r["metrics"]["gru_plus_oracle_response"]["regret_in_diagnostic_cost"] for r in selected]
        correction = [r["metrics"]["fixed_reference_truth"]["regret_in_diagnostic_cost"] - r["metrics"]["action_specific_truth"]["regret_in_diagnostic_cost"] for r in selected]
        result["by_variant"][variant] = {"rankable_windows": len(selected),
                                         "response_only_choice_changes": sum(r["metrics"]["gru_plus_oracle_response"]["choice"] != r["metrics"]["original_gru"]["choice"] for r in selected),
                                         "motion_truth_choice_changes": sum(r["metrics"]["fixed_reference_truth"]["choice"] != r["metrics"]["original_gru"]["choice"] for r in selected),
                                         "truth_response_choice_changes": sum(r["metrics"]["fixed_reference_truth"]["choice"] != r["metrics"]["action_specific_truth"]["choice"] for r in selected),
                                         "response_only_regret_reduction": group_bootstrap(gain, groups, protocol["group_bootstrap_draws"], protocol["group_bootstrap_seed"]),
                                         "response_value_with_correct_motion": group_bootstrap(correction, groups, protocol["group_bootstrap_draws"], protocol["group_bootstrap_seed"])}
    return result


class ScoreTap(BackboneTap):
    def __init__(self, *args):
        super().__init__(*args)
        parent = self.evaluator.DistributedMinimaxDNMPC
        adapter = self
        self.last_planner = None
        self.last_scenarios = None
        self.last_planner_observation = None
        self.last_step_index = None
        self.last_objective = None

        class TappedPlanner(parent):
            def plan(self, observation, scenarios, **kwargs):
                result = super().plan(observation, scenarios, **kwargs)
                adapter.last_planner = self
                adapter.last_scenarios = copy.deepcopy(scenarios.truncate(8))
                adapter.last_planner_observation = copy.deepcopy(observation)
                adapter.last_step_index = int(kwargs.get("step_index", 0))
                adapter.last_objective = float(result.diagnostics.objective_value)
                return result

        self.evaluator.DistributedMinimaxDNMPC = TappedPlanner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if sha(args.archive) != protocol["source_archive_sha256"]:
        raise ValueError("V4 archive differs from frozen protocol")
    with zipfile.ZipFile(args.archive) as archive:
        scenes = [json.loads(line) for line in archive.read("run/selected_scenes.jsonl").decode().splitlines()]
        windows = json.loads(archive.read("run/windows.json"))
        with np.load(io.BytesIO(archive.read("run/pairs.npz")), allow_pickle=False) as dataset:
            data = {k: dataset[k] for k in dataset.files}
        archived_trajectories = {r["episode_index"]: archive.read(f"run/observed/{r['episode_index']}.npz") for r in scenes}
    lookup = {(w["episode_index"], w["step"]): index for index, w in enumerate(windows)}
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = ScoreTap(args.capsule, args.output / "restored")
    from encirclement3d.minimax_mpc import ScenarioTrajectorySet, aggregate_scenario_costs
    records, episodes = [], []
    started = time.perf_counter()
    for scene in scenes:
        def observer(env, observation, actions, sequence):
            key = (scene["episode_index"], int(env.step_count))
            if key not in lookup:
                return
            index = lookup[key]
            valid = bool(data["valid"][index, :, :8].all())
            row = {"episode_index": scene["episode_index"], "group": str(scene["mirror_group_id"]), "variant": scene["variant"],
                   "step": int(env.step_count), "all_candidates_full_horizon_valid": valid}
            if sequence is None or sequence.tobytes() != data["proposed"][index, 0, :8].tobytes():
                raise AssertionError("Original plan differs from archived V4 sequence")
            if base.last_scenarios.trajectories[0].tobytes() != data["backbone"][index].tobytes():
                raise AssertionError("Original GRU differs from archived V4 backbone")
            if base.last_step_index != env.step_count:
                raise AssertionError("Planner timestamp differs from snapshot")
            before = fingerprint(env)
            planner_before = hashlib.sha256(__import__("pickle").dumps(base.last_planner.__dict__, protocol=5)).hexdigest()
            if valid:
                proposed = data["proposed"][index, :, :8]
                original = base.last_scenarios.trajectories[0]
                truth = data["target"][index, :, :8]
                responses = truth - truth[:1]
                paths = {"original_gru": np.repeat(original[None], 13, axis=0),
                         "gru_plus_oracle_response": original[None] + responses,
                         "fixed_reference_truth": np.repeat(truth[:1], 13, axis=0), "action_specific_truth": truth}
                costs = {}
                for method, candidates in paths.items():
                    values = []
                    for candidate, trajectory in zip(proposed, candidates):
                        planner = copy.deepcopy(base.last_planner)
                        # Diagnostic container only: no projection is asserted or called.
                        scenarios = ScenarioTrajectorySet(trajectory[None], np.ones(1), dynamics_status="raw")
                        scenario_costs = planner._team_scenario_costs(base.last_planner_observation, scenarios, candidate, base.last_step_index)
                        values.append(aggregate_scenario_costs(scenario_costs, scenarios.normalized_weights, planner.config.risk_mode, planner.config.cvar_alpha))
                    costs[method] = np.asarray(values)
                if not np.isclose(costs["original_gru"][0], base.last_objective, rtol=1e-10, atol=1e-8):
                    raise AssertionError("Copied full-team reference score differs from planner diagnostics")
                row["costs"] = {k: v.tolist() for k, v in costs.items()}
                row["metrics"] = {k: rank_metrics(v, costs["action_specific_truth"], protocol["tie_absolute_cost_tolerance"]) for k, v in costs.items()}
                row["reference_score_matches_original_diagnostics"] = True
            if fingerprint(env) != before or hashlib.sha256(__import__("pickle").dumps(base.last_planner.__dict__, protocol=5)).hexdigest() != planner_before:
                raise AssertionError("Offline scoring changed parent state")
            row["parent_integrity"] = True
            records.append(row)
            (args.output / "windows.json").write_text(json.dumps(records, indent=2))
            print(json.dumps({"episode": scene["episode_index"], "step": env.step_count, "rankable": valid}), flush=True)

        path = args.output / "trajectories" / f"{scene['episode_index']}.npz"
        result, _ = base.run(scene, path, observer=observer)
        with np.load(path, allow_pickle=False) as actual, np.load(io.BytesIO(archived_trajectories[scene["episode_index"]]), allow_pickle=False) as original:
            exact = all(actual[k].shape == original[k].shape and actual[k].dtype == original[k].dtype and actual[k].tobytes() == original[k].tobytes()
                        for k in ("target_positions", "defender_positions"))
        if not exact:
            raise AssertionError("Offline diagnostic changed original trajectory")
        episodes.append({"episode_index": scene["episode_index"], "trajectory_byte_equal": exact, "trajectory_sha256": sha(path),
                         **{k: result[k] for k in ("safe_capture_success", "collision", "boundary_violation", "timeout", "termination_reason")}})
    if {(r["episode_index"], r["step"]) for r in records} != set(lookup):
        raise AssertionError("Archived snapshot coverage mismatch")
    verify(base.root, base.capsule_manifest)
    summary = {"status": "offline_diagnostic_complete_not_promoted", "protocol": protocol,
               "enhanced_control_enabled": False, "source_archive_sha256": sha(args.archive), "capsule_sha256": sha(args.capsule),
               "source_hashes": {p.name: sha(p) for p in (Path(__file__), args.protocol)},
               "statistics": summarize(records, protocol), "episodes": episodes, "elapsed_seconds": time.perf_counter() - started,
               "limitations": ["Uses V4 exploratory TRAIN windows; no fresh independent confirmation, holdout, training or controller activation.",
                               "13 probes are not the complete DN-MPC local candidate library. Team diagnostic ranking does not equal distributed local decisions.",
                               "Actual action-specific truth with common noise is an upper-information diagnostic, not deployable or a formal upper bound.",
                               "All methods use unchanged proposed commands and original complete 8-step scoring; counterfactual CBF-dependent execution is not modeled by that score.",
                               "Windows censored before 8 steps are excluded jointly for all 13 probes; no post-terminal score padding.",
                               "No 24-step cost or terminal-value extension was introduced."]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": summary["status"], "statistics": summary["statistics"]}), flush=True)


if __name__ == "__main__":
    main()
