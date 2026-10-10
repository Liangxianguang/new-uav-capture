"""Reconstruct frozen local scores, re-infer public models and audit fallback."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cwm_v7")]
from local_shadow import (METHODS, model_loader, local_cost, restore_public_call, decode_public, delayed_joint_context,
                          PublicResponseProvider, stable_union, summarize, summarize_actual_choices, rank_metrics)
from training_release import audit_data, audit as audit_training
from geometry_release import arrays, compare_arrays, target_bad, BASELINE_SHA, sha
from freeze_baseline import restore, verify, REFERENCE
from verify_release import check_archive


def frozen_configs(capsule, destination):
    manifest = restore(capsule, destination)
    sys.path.insert(0, str(destination / "src"))
    from encirclement3d.minimax_mpc import MinimaxMPCConfig
    from encirclement3d.distributed_dn_mpc import DistributedDNMPCConfig
    provenance = json.loads((destination / REFERENCE / "manifest.json").read_text())
    verify(destination, manifest)
    return MinimaxMPCConfig.from_mapping(provenance["planner"]), DistributedDNMPCConfig.from_mapping(provenance["distributed"])


def valid_mask(target, termination, scenario):
    observed = termination != "after_terminal"
    count = int(observed.sum())
    if count <= 0 or not np.array_equal(observed, np.arange(8) < count):
        raise ValueError("Actual terminal support is not a contiguous prefix")
    result = np.zeros(8, dtype=bool)
    result[:count] = ~np.maximum.accumulate(target_bad(target[:count], scenario))
    if np.any(target[count:] != 0):
        raise ValueError("Post-terminal target labels are not zero padding")
    return result


def verify_call(row, values, snapshot, providers, configs, source_window, data, scene, protocol):
    from encirclement3d.minimax_mpc import ScenarioTrajectorySet
    call = restore_public_call(snapshot, *configs, decode_public(snapshot)["backbone"])
    agent = row["agent"]
    if call["agent"] != agent or call["ordinal"] != row["ordinal"] or call["observation"]["step"] != row["step"]:
        raise ValueError("Actual local context identity mismatch")
    if any(key in call["observation"] for key in ("target_position", "target_branch_sign", "target_maneuver_route")):
        raise ValueError("Private target state in public context")
    for peer, message in call["known"].items():
        if peer == agent or message.sender != peer or message.receiver != agent or message.delivery_step > row["step"] or row["step"] - message.sent_step > configs[1].max_message_age_steps:
            raise ValueError("Captured delayed-message provenance mismatch")
    context = delayed_joint_context(call["observation"], call["known"], call["peer_sequences"], agent,
                                    call["local_candidates"], call["selected"], data["reference"][source_window], data["reference_velocity"][source_window])
    if context is None:
        if row["status"] != "skipped_missing_delayed_peer_plan" or "arrays_path" in row or row["all_candidates_full_horizon_valid"]:
            raise ValueError("Missing peer was silently filled or scored")
        return
    if values is None:
        raise ValueError("Complete local call has no actual array evidence")
    if not row["anchor_repeat_byte_equal"] or not row["reference_score_matches_original_diagnostics"]:
        raise ValueError("Missing original local reproduction evidence")
    if (values["history"].tobytes() != data["history"][source_window].tobytes()
            or values["backbone"].tobytes() != data["backbone"][source_window].tobytes()
            or values["reference"].tobytes() != data["reference"][source_window].tobytes()
            or values["velocity"].tobytes() != data["reference_velocity"][source_window].tobytes()):
        raise ValueError("Actual common public history/backbone differs from original collection")
    observation, planner = call["observation"], call["planner"]
    actual = planner._local_candidate_sequences(observation, call["scenarios"], agent, observation["defender_positions"][agent], call["known"])
    if not np.array_equal(np.stack(actual), call["local_candidates"]) or not np.array_equal(values["actual_candidates"], call["local_candidates"]):
        raise ValueError("Actual frozen local candidate generation mismatch")
    copied_scores = local_cost(call, call["local_candidates"], np.repeat(values["backbone"][None], len(actual), 0))
    choice = int(np.argmin(copied_scores))
    if choice != row["actual_choice"] or not np.array_equal(call["selected"], actual[choice]) or not np.allclose(copied_scores, values["original_local_costs"], rtol=1e-10, atol=1e-8):
        raise ValueError("Reconstructed actual local choice/cost mismatch")
    if not np.isclose(copied_scores[choice], call["selected_costs"][0], rtol=1e-10, atol=1e-8):
        raise ValueError("Actual selected local cost mismatch")
    original_joint, anchor, relative = context
    initial = {kind: p.forecast(values["history"], relative, original_joint, anchor, values["backbone"]) for kind, p in providers.items()}
    pool = list(actual)
    cv = values["reference"][None] + .1 * np.arange(1, 9)[:, None] * values["velocity"][None]
    paths_to_generate = [cv] + [path for kind in ("plain", "structured") for path in initial[kind]["prediction"]]
    for path in paths_to_generate:
        scenario = ScenarioTrajectorySet(path[None], np.ones(1), dynamics_status="raw")
        pool.extend(planner._local_candidate_sequences(observation, scenario, agent, observation["defender_positions"][agent], call["known"]))
    pool = stable_union(pool)
    joint, anchor, relative = delayed_joint_context(observation, call["known"], call["peer_sequences"], agent, pool, call["selected"], values["reference"], values["velocity"])
    if not np.array_equal(joint, values["proposed"]) or not np.array_equal(anchor, values["anchor"]) or not np.array_equal(relative, values["relative"]) or len(pool) != row["candidate_count"]:
        raise ValueError("Actual candidate union or delayed public input mismatch")
    inferred = {kind: p.forecast(values["history"], relative, joint, anchor, values["backbone"]) for kind, p in providers.items()}
    for kind, prediction in inferred.items():
        for key, value in prediction.items():
            old = values[kind + "_" + key]
            if old.dtype != value.dtype or old.shape != value.shape or old.tobytes() != value.tobytes():
                raise ValueError("Reloaded public local forecast bytes differ")
    for target, mask, termination in zip(values["target"], values["valid"], values["termination"]):
        if not np.array_equal(valid_mask(target, termination, scene["scenario"]), mask):
            raise ValueError("Actual branch physics/support mismatch")
    if not np.array_equal(valid_mask(values["anchor_target"], values["anchor_termination"], scene["scenario"]), values["anchor_valid"]):
        raise ValueError("Actual anchor physics/support mismatch")
    actual_index = next(i for i, value in enumerate(pool) if np.array_equal(value, call["selected"]))
    if actual_index != row["actual_choice_pool_index"] or not np.array_equal(values["target"][actual_index], values["anchor_target"]) or not np.array_equal(values["valid"][actual_index], values["anchor_valid"]):
        raise ValueError("Actual local reference branch mismatch")
    complete = bool(values["valid"].all() and values["anchor_valid"].all())
    if complete != row["all_candidates_full_horizon_valid"]:
        raise ValueError("Actual joint support flag mismatch")
    response = values["target"] - values["anchor_target"][None]
    paths = {"original_gru": np.repeat(values["backbone"][None], len(pool), 0), "constant_velocity": np.repeat(cv[None], len(pool), 0),
             "plain_learned": inferred["plain"]["prediction"], "structured_learned": inferred["structured"]["prediction"]}
    if complete:
        paths.update(gru_plus_exact_response=values["backbone"][None] + response,
                     fixed_reference_truth=np.repeat(values["anchor_target"][None], len(pool), 0), action_specific_truth=values["target"],
                     reference_truth_plus_plain_response=values["anchor_target"][None] + inferred["plain"]["response"],
                     reference_truth_plus_structured_response=values["anchor_target"][None] + inferred["structured"]["response"])
    elif "metrics" in row or "actual_selection_metrics" in row:
        raise ValueError("Incomplete truth was ranked")
    if set(row["costs"]) != set(paths):
        raise ValueError("Incomplete information conditions")
    for method, path in paths.items():
        reconstructed = local_cost(call, pool, path)
        if not np.allclose(reconstructed, row["costs"][method], rtol=1e-10, atol=1e-8):
            raise ValueError("Independently reconstructed local cost differs")
        if method in row["public_choice_indices"] and int(np.argmin(reconstructed)) != row["public_choice_indices"][method]:
            raise ValueError("Actual public choice mismatch")
        if complete and rank_metrics(np.asarray(row["costs"][method]), row["costs"]["action_specific_truth"], protocol["tie_absolute_cost_tolerance"]) != row["metrics"][method]:
            raise ValueError("Actual local rank/regret mismatch")
    if complete:
        truth = np.asarray(row["costs"]["action_specific_truth"])
        selected_cost = np.ones(len(pool))
        selected_cost[actual_index] = 0.
        if rank_metrics(selected_cost, truth, protocol["tie_absolute_cost_tolerance"]) != row["actual_selection_metrics"]:
            raise ValueError("Actual solver comparator mismatch")


def audit(read, source_read, configs):
    data, indices, _, data_report = audit_data(source_read)
    windows = json.loads(source_read("data/windows.json"))
    lookup = {(windows[i]["episode_index"], windows[i]["step"]): int(i) for i in indices["development_validation"]}
    scenes = {r["episode_index"]: r for r in [json.loads(line) for line in source_read("data/selected_scenes.jsonl").decode().splitlines()] if r["model_split"] == "development_validation"}
    source_episodes = {r["episode_index"]: r for r in data_report["episodes_results"]}
    training = json.loads(source_read("primary/summary.json"))
    summaries = []
    providers = {kind: PublicResponseProvider("shadow", model_loader(source_read(f"primary/{kind}_seed{seed}.pt"), kind, seed))
                 for kind, seed in training["chosen_median_seed_per_architecture"].items()}
    for stage, mode in (("primary", "shadow"), ("repeated", "shadow"), ("off", "off"), ("failed", "load_failed"), ("blocked", "guarded_blocked")):
        summary = json.loads(read(stage + "/summary.json"))
        protocol = summary["protocol"]
        if (summary["mode"] != mode or summary["capsule_sha256"] != BASELINE_SHA or summary["source_archive_sha256"] != protocol["source_archive_sha256"]
                or summary["enhanced_control_enabled"] or summary["holdout_used"] or summary["prior_training_gate_overridden"]
                or summary["prior_training_gate_passed"] != training["development_gate_passed"] or not summary["torch_rng_preserved_on_optional_load"]
                or protocol["split"] != "development_validation" or protocol["enhanced_control_enabled"] or protocol["holdout_used"]
                or protocol["new_model_training_enabled"] or protocol["prior_training_gate_override_allowed"]
                or protocol["selected_median_seeds"] != training["chosen_median_seed_per_architecture"]):
            raise ValueError("Frozen shadow/qualification contract mismatch")
        for name, expected in summary["source_hashes"].items():
            if hashlib.sha256(read(f"{stage}/source/{name}")).hexdigest() != expected:
                raise ValueError("Actual source digest mismatch")
        if json.loads(read(stage + "/source/cwm_v9/protocol.json")) != protocol:
            raise ValueError("Actual protocol mismatch")
        actual_scenes = {r["episode_index"]: r for r in [json.loads(line) for line in read(stage + "/scenes.jsonl").decode().splitlines()]}
        if actual_scenes != scenes or len(summary["episodes"]) != 16 or {r["episode_index"] for r in summary["episodes"]} != set(scenes):
            raise ValueError("Actual development episode population mismatch")
        for row in summary["episodes"]:
            index = row["episode_index"]
            raw = read(f"{stage}/trajectories/{index}.npz")
            if hashlib.sha256(raw).hexdigest() != row["trajectory_sha256"] or not row["trajectory_byte_equal"]:
                raise ValueError("Actual replay digest mismatch")
            if not compare_arrays(arrays(raw), arrays(source_read(f"data/observed/{index}.npz"))):
                raise ValueError("Actual original replay arrays differ")
            if any(row[k] != source_episodes[index][k] for k in ("safe_capture_success", "collision", "boundary_violation", "timeout", "target_invalid_episode", "termination_reason")):
                raise ValueError("Actual original replay outcomes differ")
        records = json.loads(read(stage + "/calls.json"))
        if mode != "shadow":
            if records or summary["actual_local_calls"] or summary["statistics"] is not None:
                raise ValueError("Fallback performed optional shadow work")
            status = {"off": "off", "load_failed": "load_failed", "guarded_blocked": "qualification_failed"}[mode]
            for provider in summary["providers"].values():
                if provider["status"] != status or provider["control_eligible"] or provider["prediction_calls"] or provider["load_calls"] != int(mode == "load_failed"):
                    raise ValueError("Fallback loader/prediction contract mismatch")
        else:
            expected = {(episode, step, agent) for episode, step in lookup for agent in range(4)}
            if len(records) != len(expected) or {(r["episode_index"], r["step"], r["agent"]) for r in records} != expected:
                raise ValueError("Actual local invocation coverage mismatch")
            for row in records:
                if row["group"] != scenes[row["episode_index"]]["mirror_group_id"] or not row["parent_integrity"]:
                    raise ValueError("Actual local call provenance mismatch")
                raw = read(stage + "/" + row["context_path"])
                if hashlib.sha256(raw).hexdigest() != row["context_sha256"]:
                    raise ValueError("Actual delayed context digest mismatch")
                values = None
                if "arrays_path" in row:
                    raw_arrays = read(stage + "/" + row["arrays_path"])
                    if hashlib.sha256(raw_arrays).hexdigest() != row["arrays_sha256"]:
                        raise ValueError("Actual local arrays digest mismatch")
                    values = arrays(raw_arrays)
                verify_call(row, values, json.loads(raw), providers, configs, lookup[row["episode_index"], row["step"]], data, scenes[row["episode_index"]], protocol)
            if summary["statistics"] != summarize(records, protocol) or summary["actual_selection_comparisons"] != summarize_actual_choices(records, protocol):
                raise ValueError("Actual local group statistics mismatch")
            if summary["actual_local_calls"] != len(records) or summary["missing_peer_calls"] != sum(r["status"] == "skipped_missing_delayed_peer_plan" for r in records) or summary["incomplete_support_calls"] != sum(r["status"] == "excluded_incomplete_truth_support" for r in records):
                raise ValueError("Actual support population mismatch")
            for provider in summary["providers"].values():
                if provider["status"] != "shadow_ready" or provider["control_eligible"] or provider["load_calls"] != 1 or provider["prediction_calls"] != 2 * (len(records) - summary["missing_peer_calls"]):
                    raise ValueError("Actual shadow provider call count mismatch")
        summaries.append(summary)
    for summary in summaries[1:]:
        for key in ("protocol", "source_hashes", "capsule_sha256", "source_archive_sha256", "prior_training_gate_passed", "enhanced_control_enabled", "holdout_used"):
            if summary[key] != summaries[0][key]:
                raise ValueError("Independent mode sources/contracts differ")
    if read("primary/calls.json") != read("repeated/calls.json"):
        raise ValueError("Independent shadow call record bytes differ")
    for row in json.loads(read("primary/calls.json")):
        if read("primary/" + row["context_path"]) != read("repeated/" + row["context_path"]):
            raise ValueError("Independent public planning context differs")
        if "arrays_path" in row and not compare_arrays(arrays(read("primary/" + row["arrays_path"])), arrays(read("repeated/" + row["arrays_path"]))):
            raise ValueError("Independent branch/model arrays differ")
    return {"status": "passed", "modes": 5, "episodes_per_mode": 16, "shadow_calls": summaries[0]["actual_local_calls"],
            "rankable_calls": summaries[0]["statistics"]["rankable_windows"], "missing_peer_calls": summaries[0]["missing_peer_calls"],
            "incomplete_support_calls": summaries[0]["incomplete_support_calls"], "local_candidates_and_scores_reconstructed": True,
            "public_model_outputs_byte_equal": True, "independent_shadow_records_equal": True, "all_original_replay_arrays_equal": True,
            "default_off_no_optional_load_or_predict": True, "load_failure_falls_back": True, "failed_gate_blocks_load": True,
            "motion_research_signal": summaries[0]["statistics"]["motion_research_signal"],
            "response_research_signal_given_motion": summaries[0]["statistics"]["response_research_signal_given_motion"],
            "prior_gate_overridden": False, "enhanced_control_enabled": False, "holdout_used": False,
            "scope": "Development-only finite first-local-call shadow and cloned joint open-loop branches; not complete sequential enhanced solver, controller deployment, original-Level transfer or online latency qualification."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "cwm_v7/artifacts/paired_training_20261010.zip")
    parser.add_argument("--capsule", type=Path, default=ROOT.parent / "cwm_v1/baseline/capsule.zip")
    for name in ("primary", "repeated", "off", "failed", "blocked"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if sha(args.source) != protocol["source_archive_sha256"] or sha(args.capsule) != BASELINE_SHA:
        raise ValueError("Protected source/baseline digest mismatch")
    configs = frozen_configs(args.capsule, ROOT.parent.parent / "results/cwm_v9/audit_restored")
    check_archive(args.source, "ARTIFACT_MANIFEST.json", False)
    with zipfile.ZipFile(args.source) as source:
        training_result = audit_training(source.read)
        if args.verify:
            integrity = check_archive(args.verify, "ARTIFACT_MANIFEST.json", False)
            with zipfile.ZipFile(args.verify) as archive:
                result = audit(archive.read, source.read, configs)
            print(json.dumps({**result, "artifact_sha256": integrity["sha256"]}))
            return
        members = {}
        for stage in ("primary", "repeated", "off", "failed", "blocked"):
            run = getattr(args, stage)
            if run is None:
                parser.error("All five mode runs required")
            for file in run.rglob("*"):
                relative = file.relative_to(run)
                if file.is_file() and "restored" not in relative.parts:
                    members[f"{stage}/{relative.as_posix()}"] = file.read_bytes()
        for file in ROOT.glob("*.py"):
            members["verification_source/" + file.name] = file.read_bytes()
        result = audit(members.__getitem__, source.read, configs)
        path = ROOT / "artifacts/local_shadow_20261010.zip"
        path.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in members.items():
                archive.writestr(name, raw)
            archive.writestr("ARTIFACT_MANIFEST.json", json.dumps({n: hashlib.sha256(v).hexdigest() for n, v in members.items()}, indent=2))
        integrity = check_archive(path, "ARTIFACT_MANIFEST.json", False)
        with zipfile.ZipFile(path) as archive:
            audit(archive.read, source.read, configs)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    with (reports / "shadow_summary.json").open("xb") as file:
        file.write((args.primary / "summary.json").read_bytes())
    with (reports / "release_manifest.json").open("x") as file:
        json.dump({**result, "source_training_audit": training_result, "artifact_sha256": integrity["sha256"], "artifact_bytes": path.stat().st_size}, file, indent=2)
    print(json.dumps({**result, "artifact_sha256": integrity["sha256"], "artifact_bytes": path.stat().st_size}))


if __name__ == "__main__":
    main()
