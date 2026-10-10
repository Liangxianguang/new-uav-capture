"""Actual delayed local-candidate shadow, never changing original control."""
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
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cwm_v8"), str(ROOT.parent / "cwm_v7"), str(ROOT.parent / "cwm_v2")]
from public_provider import PublicResponseProvider, delayed_joint_context
from s4_value import METHODS, summarize
from scout import BackboneTap
from baseline import Baseline
from diagnose import fingerprint
from freeze_baseline import sha, verify
from verify_release import check_archive
from response_model import ResponseModel
from geometry_release import arrays, compare_arrays, BASELINE_SHA
from collect_training import s4_rollout
from decision_value import rank_metrics
from s4_value import descriptive_bootstrap


def encode_public(value):
    if isinstance(value, np.ndarray):
        return {"array_dtype": value.dtype.str, "array_shape": list(value.shape), "array_values": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): encode_public(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode_public(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Unsupported public context type")


def decode_public(value):
    if isinstance(value, dict) and set(value) == {"array_dtype", "array_shape", "array_values"}:
        return np.asarray(value["array_values"], dtype=value["array_dtype"]).reshape(value["array_shape"])
    if isinstance(value, dict):
        return {k: decode_public(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_public(v) for v in value]
    return value


def public_call_snapshot(call):
    if call["planner"].config.fc_dbf_enabled or call["observation"].get("qdr_execution_aware", False):
        raise ValueError("Snapshot reconstruction is scoped to the frozen non-QDR/non-FC baseline")
    return encode_public({"observation": call["observation"], "known": {p: vars(m) for p, m in call["known"].items()},
                          "peer_sequences": call["peer_sequences"], "agent": call["agent"], "ordinal": call["ordinal"],
                          "local_candidates": call["local_candidates"], "selected": call["selected"], "selected_costs": call["selected_costs"],
                          "backbone": call["scenarios"].trajectories[0]})


def restore_public_call(snapshot, planner_config, distributed_config, backbone):
    from types import SimpleNamespace
    from encirclement3d.distributed_dn_mpc import DistributedMinimaxDNMPC
    from encirclement3d.minimax_mpc import ScenarioTrajectorySet
    call = decode_public(snapshot)
    call["known"] = {int(p): SimpleNamespace(**message) for p, message in call["known"].items()}
    call["peer_sequences"] = {int(p): actions for p, actions in call["peer_sequences"].items()}
    call["planner"] = DistributedMinimaxDNMPC(planner_config, distributed_config)
    call["scenarios"] = ScenarioTrajectorySet(backbone[None], np.ones(1), dynamics_status="raw")
    return call


def summarize_actual_choices(records, protocol):
    selected = [r for r in records if r["all_candidates_full_horizon_valid"]]
    groups = [r["group"] for r in selected]
    result = {}
    for method in METHODS:
        gains = [r["actual_selection_metrics"]["regret_in_diagnostic_cost"] - r["metrics"][method]["regret_in_diagnostic_cost"] for r in selected]
        result[method] = {"choice_changes_vs_actual_solver": sum(r["metrics"][method]["choice"] != r["actual_choice_pool_index"] for r in selected),
                          "gain_vs_actual_solver": descriptive_bootstrap(gains, groups, protocol["group_bootstrap_draws"], protocol["group_bootstrap_seed"])}
    return result


def model_loader(raw, kind, seed):
    def load():
        checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        if checkpoint["kind"] != kind or checkpoint["seed"] != seed or checkpoint["online_promoted"] or checkpoint["baseline_weights_included"]:
            raise ValueError("Frozen unpromoted model mismatch")
        # Constructing a module initializes tensors; preserve baseline Torch RNG.
        with torch.random.fork_rng(devices=[]):
            model = ResponseModel(kind, checkpoint["protocol"]["training"]["response_scale_m"])
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        mean = checkpoint["normalizer_mean"].numpy().reshape(8 if checkpoint["normalizer_mean"].numel() == 8 * 252 else 1, 252)
        scale = checkpoint["normalizer_scale"].numpy().reshape(mean.shape)
        def predict(history, relative, proposed, anchor, backbone):
            count = len(proposed)
            inputs = (torch.as_tensor((history - mean) / scale, dtype=torch.float32)[None].repeat(count, 1, 1),
                      torch.as_tensor(relative, dtype=torch.float32)[None].repeat(count, 1, 1),
                      torch.as_tensor(proposed, dtype=torch.float32), torch.as_tensor(anchor, dtype=torch.float32)[None].repeat(count, 1, 1, 1),
                      torch.as_tensor(backbone, dtype=torch.float64)[None].repeat(count, 1, 1))
            with torch.no_grad():
                prediction, response, _ = model(*inputs)
            return {"prediction": prediction.numpy(), "response": response.numpy()}
        return predict
    return load


class ActualLocalTap(BackboneTap):
    """Copy first actual invocation per agent, score only after original plan."""
    def __init__(self, capsule, restored, snapshot_steps):
        super().__init__(capsule, restored)
        adapter = self
        parent = self.evaluator.DistributedMinimaxDNMPC
        self.calls, self.current_step = [], None
        self.snapshot_steps = set(snapshot_steps)
        class TappedPlanner(parent):
            def plan(self, observation, scenarios, **kwargs):
                adapter.calls = []
                adapter.current_step = int(kwargs.get("step_index", 0))
                return super().plan(observation, scenarios, **kwargs)

            def _select_local_sequence(self, observation, scenarios, agent_id, local_candidates, known, peer_sequences):
                selected, costs = super()._select_local_sequence(observation, scenarios, agent_id, local_candidates, known, peer_sequences)
                if adapter.current_step in adapter.snapshot_steps and not any(r["agent"] == agent_id for r in adapter.calls):
                    adapter.calls.append({"agent": int(agent_id), "ordinal": len(adapter.calls), "planner": copy.deepcopy(self),
                                          "observation": copy.deepcopy(observation), "scenarios": copy.deepcopy(scenarios),
                                          "local_candidates": np.stack(local_candidates).copy(), "known": copy.deepcopy(known),
                                          "peer_sequences": copy.deepcopy(peer_sequences), "selected": selected.copy(), "selected_costs": costs.copy()})
                return selected, costs
        self.evaluator.DistributedMinimaxDNMPC = TappedPlanner


def stable_union(sequences):
    result = []
    for sequence in sequences:
        sequence = np.asarray(sequence, dtype=np.float64)
        if sequence.shape != (8, 3) or not np.isfinite(sequence).all() or np.linalg.norm(sequence, axis=-1).max() > 5. + 1e-8:
            raise ValueError("Candidate contract mismatch")
        if not any(np.array_equal(sequence, old) for old in result):
            result.append(sequence.copy())
    return np.stack(result)


def local_cost(call, actions, paths):
    from encirclement3d.minimax_mpc import ScenarioTrajectorySet, aggregate_scenario_costs
    values = []
    for action, path in zip(actions, paths):
        planner = copy.deepcopy(call["planner"])
        scenarios = ScenarioTrajectorySet(np.asarray(path)[None], np.ones(1), dynamics_status="raw")
        costs = planner._local_scenario_cost_matrix(call["observation"], scenarios, call["agent"], action[None], call["known"], call["peer_sequences"])[0]
        values.append(aggregate_scenario_costs(costs, scenarios.normalized_weights, planner.config.risk_mode, planner.config.cvar_alpha))
    return np.asarray(values)


def inspect_call(base, env, call, history, providers, protocol, noise_key):
    from encirclement3d.minimax_mpc import ScenarioTrajectorySet, aggregate_scenario_costs
    observation, planner, agent = call["observation"], call["planner"], call["agent"]
    reference, velocity = base.evaluator.belief_reference(observation, planner.config)
    backbone = call["scenarios"].trajectories[0]
    actual = call["local_candidates"]
    result = {"agent": agent, "ordinal": call["ordinal"], "actual_candidate_count": len(actual), "parent_integrity": True,
              "all_candidates_full_horizon_valid": False, "model_public_context_matches_current_replay": True}
    context = delayed_joint_context(observation, call["known"], call["peer_sequences"], agent, actual, call["selected"], reference, velocity)
    if context is None:
        return {**result, "status": "skipped_missing_delayed_peer_plan"}, None
    joint, anchor, relative = context
    selected_index = next(i for i, v in enumerate(actual) if np.array_equal(v, call["selected"]))
    copied_scores = local_cost(call, actual, np.repeat(backbone[None], len(actual), 0))
    original_score = aggregate_scenario_costs(call["selected_costs"], call["scenarios"].normalized_weights, planner.config.risk_mode, planner.config.cvar_alpha)
    if not np.isclose(copied_scores[selected_index], original_score, rtol=1e-10, atol=1e-8) or int(np.argmin(copied_scores)) != selected_index:
        raise AssertionError("Copied local scores do not reproduce actual local choice")
    initial_predictions = {kind: provider.forecast(history, relative, joint, anchor, backbone) for kind, provider in providers.items()}
    if any(value is None for value in initial_predictions.values()):
        return {**result, "status": "fallback_prediction_failed"}, None
    cv = reference[None] + .1 * np.arange(1, 9)[:, None] * velocity[None]
    pool = list(actual)
    def generate(path):
        cloned = copy.deepcopy(planner)
        forecast = ScenarioTrajectorySet(path[None], np.ones(1), dynamics_status="raw")
        # Original generator invoked on a copy; forecast is explicitly raw.
        return cloned._local_candidate_sequences(observation, forecast, agent, observation["defender_positions"][agent], call["known"])
    pool.extend(generate(cv))
    for kind in ("plain", "structured"):
        for path in initial_predictions[kind]["prediction"]:
            pool.extend(generate(path))
    pool = stable_union(pool)
    actual_pool_index = next(i for i, v in enumerate(pool) if np.array_equal(v, call["selected"]))
    context = delayed_joint_context(observation, call["known"], call["peer_sequences"], agent, pool, call["selected"], reference, velocity)
    joint, anchor, relative = context
    predictions = {kind: provider.forecast(history, relative, joint, anchor, backbone) for kind, provider in providers.items()}
    if any(value is None for value in predictions.values()):
        return {**result, "status": "fallback_prediction_failed"}, None
    # Branch labels collected only after public model inputs/union are frozen.
    branches = [s4_rollout(base, env, sequence, noise_key) for sequence in joint]
    anchor_branch = s4_rollout(base, env, anchor, noise_key)
    repeated = s4_rollout(base, env, anchor, noise_key)
    if not compare_arrays(anchor_branch, repeated):
        raise AssertionError("Repeated local anchor branch changed bytes")
    target = np.stack([b["target"] for b in branches])
    valid = np.stack([b["valid"] for b in branches])
    paths = {"original_gru": np.repeat(backbone[None], len(pool), 0), "constant_velocity": np.repeat(cv[None], len(pool), 0),
             "plain_learned": predictions["plain"]["prediction"], "structured_learned": predictions["structured"]["prediction"]}
    response = target - anchor_branch["target"][None]
    paths.update({"gru_plus_exact_response": backbone[None] + response,
                  "fixed_reference_truth": np.repeat(anchor_branch["target"][None], len(pool), 0), "action_specific_truth": target,
                  "reference_truth_plus_plain_response": anchor_branch["target"][None] + predictions["plain"]["response"],
                  "reference_truth_plus_structured_response": anchor_branch["target"][None] + predictions["structured"]["response"]})
    full = bool(valid.all() and anchor_branch["valid"].all())
    costs = {k: local_cost(call, pool, v) for k, v in paths.items() if full or k in ("original_gru", "constant_velocity", "plain_learned", "structured_learned")}
    public_choices = {k: int(np.argmin(costs[k])) for k in ("original_gru", "constant_velocity", "plain_learned", "structured_learned")}
    result.update(status="scored" if full else "excluded_incomplete_truth_support", candidate_count=len(pool), actual_choice=selected_index,
                  actual_choice_pool_index=actual_pool_index,
                  reference_score_matches_original_diagnostics=True, all_candidates_full_horizon_valid=full,
                  public_choice_indices=public_choices, public_choice_changes_vs_actual={k: v != actual_pool_index for k, v in public_choices.items()},
                  noise_key=noise_key, known_peer_sent_steps={str(p): int(m.sent_step) for p, m in call["known"].items()},
                  anchor_repeat_byte_equal=True,
                  branch_termination_counts={label: int(sum(b["termination"][np.flatnonzero(b["termination"] != "after_terminal")[-1]] == label for b in branches))
                                             for label in sorted({b["termination"][np.flatnonzero(b["termination"] != "after_terminal")[-1]] for b in branches})})
    result["costs"] = {k: v.tolist() for k, v in costs.items()}
    if full:
        result["metrics"] = {k: rank_metrics(v, costs["action_specific_truth"], protocol["tie_absolute_cost_tolerance"]) for k, v in costs.items()}
        regret = float(costs["action_specific_truth"][actual_pool_index] - costs["action_specific_truth"].min())
        result["actual_selection_metrics"] = {"choice": actual_pool_index, "regret_in_diagnostic_cost": regret,
                                               "realized_cost_at_choice": float(costs["action_specific_truth"][actual_pool_index]),
                                               "realized_tie_optimal": bool(regret <= protocol["tie_absolute_cost_tolerance"])}
    fields = {"history": history, "relative": relative, "backbone": backbone, "reference": reference, "velocity": velocity,
              "proposed": joint, "anchor": anchor, "target": target, "valid": valid,
              "anchor_target": anchor_branch["target"], "anchor_valid": anchor_branch["valid"],
              "termination": np.stack([b["termination"] for b in branches]),
              "anchor_termination": anchor_branch["termination"], "actual_candidates": actual, "original_local_costs": copied_scores,
              "commanded": np.stack([b["commanded"] for b in branches]), "executed": np.stack([b["executed"] for b in branches]),
              "defenders": np.stack([b["defenders"] for b in branches]),
              "anchor_commanded": anchor_branch["commanded"], "anchor_executed": anchor_branch["executed"]}
    for kind, prediction in predictions.items():
        fields[kind + "_prediction"] = prediction["prediction"]
        fields[kind + "_response"] = prediction["response"]
    return result, fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--mode", choices=("off", "shadow", "load_failed", "guarded_blocked"), default="off")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if (sha(args.capsule) != BASELINE_SHA or sha(args.archive) != protocol["source_archive_sha256"] or protocol["enhanced_control_enabled"]
            or protocol["holdout_used"] or protocol["new_model_training_enabled"] or protocol["prior_training_gate_override_allowed"]):
        raise ValueError("Frozen diagnostic contract mismatch")
    check_archive(args.archive, "ARTIFACT_MANIFEST.json", False)
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with zipfile.ZipFile(args.archive) as archive:
        scenes = [json.loads(line) for line in archive.read("data/selected_scenes.jsonl").decode().splitlines()]
        scenes = [r for r in scenes if r["model_split"] == protocol["split"]]
        reference_trajectories = {r["episode_index"]: archive.read(f"data/observed/{r['episode_index']}.npz") for r in scenes}
        training = json.loads(archive.read("primary/summary.json"))
        if training["chosen_median_seed_per_architecture"] != protocol["selected_median_seeds"]:
            raise ValueError("Frozen median models differ")
        providers = {}
        torch_rng_before = torch.get_rng_state().clone()
        for kind, seed in protocol["selected_median_seeds"].items():
            if args.mode == "shadow":
                raw = archive.read(f"primary/{kind}_seed{seed}.pt")
                provider = PublicResponseProvider("shadow", model_loader(raw, kind, seed))
            elif args.mode == "load_failed":
                def failed_loader():
                    raise OSError("Injected unavailable optional checkpoint")
                provider = PublicResponseProvider("shadow", failed_loader)
            elif args.mode == "guarded_blocked":
                provider = PublicResponseProvider("guarded", lambda: (_ for _ in ()).throw(AssertionError("Unqualified model must not load")),
                                                  qualified=training["development_gate_passed"])
            else:
                provider = PublicResponseProvider()
            provider.load()
            providers[kind] = provider
        if not torch.equal(torch.get_rng_state(), torch_rng_before):
            raise AssertionError("Optional provider loading changed baseline Torch RNG")
    shadow_enabled = args.mode == "shadow" and all(p.status == "shadow_ready" for p in providers.values())
    base = ActualLocalTap(args.capsule, args.output / "restored", protocol["snapshot_steps"]) if shadow_enabled else Baseline(args.capsule, args.output / "restored")
    sources = [Path(__file__), ROOT / "public_provider.py", args.protocol.resolve()] + [ROOT.parent / p for p in (
        "cwm_v1/baseline.py", "cwm_v1/freeze_baseline.py", "cwm_v1/verify_release.py", "cwm_v2/scout.py", "cwm_v4/diagnose.py",
        "cwm_v5/decision_value.py", "cwm_v7/collect_training.py", "cwm_v7/geometry.py", "cwm_v7/response_model.py", "cwm_v7/geometry_release.py", "cwm_v8/s4_value.py")]
    source_hashes = {p.relative_to(ROOT.parent).as_posix(): sha(p) for p in sources}
    for path in sources:
        destination = args.output / "source" / path.relative_to(ROOT.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    (args.output / "scenes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scenes))
    records, episodes = [], []
    started = time.perf_counter()
    for scene in scenes:
        history = []
        def observer(env, observation, actions, sequence):
            history.append(base.evaluator.policy_observations(env, observation).reshape(-1).copy())
            if env.step_count not in protocol["snapshot_steps"]:
                return
            public_history = np.stack([history[0]] * max(0, 8 - len(history)) + history[-8:])
            before = fingerprint(env)
            for call in base.calls:
                planner_before = hashlib.sha256(pickle.dumps(call["planner"].__dict__, protocol=5)).hexdigest()
                key = int.from_bytes(hashlib.sha256(f"{protocol['noise_seed']}/{scene['episode_index']}/{env.step_count}/{call['agent']}".encode()).digest()[:4], "little")
                row, fields = inspect_call(base, env, call, public_history, providers, protocol, key)
                row.update(episode_index=scene["episode_index"], group=scene["mirror_group_id"], variant=scene["variant"], step=int(env.step_count))
                if fields is not None:
                    path = args.output / "calls" / f"{scene['episode_index']}_{env.step_count}_{call['agent']}.npz"
                    path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(path, **fields)
                    row["arrays_path"] = path.relative_to(args.output).as_posix()
                    row["arrays_sha256"] = sha(path)
                context_path = args.output / "calls" / f"{scene['episode_index']}_{env.step_count}_{call['agent']}.json"
                context_path.parent.mkdir(exist_ok=True)
                context_path.write_text(json.dumps(public_call_snapshot(call), indent=2))
                row["context_path"] = context_path.relative_to(args.output).as_posix()
                row["context_sha256"] = sha(context_path)
                if planner_before != hashlib.sha256(pickle.dumps(call["planner"].__dict__, protocol=5)).hexdigest():
                    raise AssertionError("Shadow mutated copied source planner")
                records.append(row)
            if before != fingerprint(env):
                raise AssertionError("Shadow mutated actual parent environment")
            (args.output / "calls.json").write_text(json.dumps(records, indent=2))
        trajectory = args.output / "trajectories" / f"{scene['episode_index']}.npz"
        result, _ = base.run(scene, trajectory, observer if shadow_enabled else None)
        if not compare_arrays(arrays(trajectory.read_bytes()), arrays(reference_trajectories[scene["episode_index"]])):
            raise AssertionError("Optional mode changed original trajectory fields")
        episodes.append({"episode_index": scene["episode_index"], "trajectory_byte_equal": True, "trajectory_sha256": sha(trajectory),
                         **{k: result[k] for k in ("safe_capture_success", "collision", "boundary_violation", "timeout", "target_invalid_episode", "termination_reason")}})
        print(json.dumps({"mode": args.mode, "episodes": len(episodes), "calls": len(records), "rankable": sum(r["all_candidates_full_horizon_valid"] for r in records)}), flush=True)
    verify(base.root, base.capsule_manifest)
    if any(sha(ROOT.parent / name) != digest for name, digest in source_hashes.items()):
        raise AssertionError("Diagnostic sources changed during run")
    (args.output / "calls.json").write_text(json.dumps(records, indent=2))
    stats = summarize(records, protocol) if records else None
    summary = {"status": "actual_local_shadow_complete_not_promoted" if shadow_enabled else "original_control_replay_equal", "mode": args.mode,
               "protocol": protocol, "source_hashes": source_hashes, "source_archive_sha256": sha(args.archive), "capsule_sha256": sha(args.capsule),
               "statistics": stats, "actual_selection_comparisons": summarize_actual_choices(records, protocol) if records else None,
               "actual_local_calls": len(records), "missing_peer_calls": sum(r["status"] == "skipped_missing_delayed_peer_plan" for r in records),
               "incomplete_support_calls": sum(r["status"] == "excluded_incomplete_truth_support" for r in records),
               "providers": {k: {"status": p.status, "load_calls": p.load_calls, "prediction_calls": p.prediction_calls, "control_eligible": p.control_eligible} for k, p in providers.items()},
               "episodes": episodes, "prior_training_gate_passed": training["development_gate_passed"], "prior_training_gate_overridden": False,
               "enhanced_control_enabled": False, "holdout_used": False, "torch_rng_preserved_on_optional_load": True,
               "elapsed_seconds": time.perf_counter() - started, "limitations": protocol["limitations"]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": summary["status"], "statistics": stats}), flush=True)


if __name__ == "__main__":
    main()
