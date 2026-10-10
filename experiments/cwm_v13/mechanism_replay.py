"""True TRAIN-only branch/CBF mechanism replay around frozen original control."""
import argparse
import copy
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cwm_v10"), str(ROOT.parent / "cwm_v9"), str(ROOT.parent / "cwm_v7")]
from local_shadow import ActualLocalTap, public_call_snapshot, decode_public
from replay_public import compare_public
from geometry_release import arrays, compare_arrays, sha
from diagnose import fingerprint
from freeze_baseline import verify
from verify_release import check_archive


def branch_trace(base, parent, sequence, noise_key):
    env = copy.deepcopy(parent)
    safety = base.filter_class(env)
    if safety.margin != .35 or bool(env.execution["enabled"]):
        raise ValueError("Original margin/disabled execution contract mismatch")
    observation = env.observe()
    rows = []
    for offset, proposed in enumerate(sequence):
        env.rng = np.random.default_rng(np.random.SeedSequence([noise_key, offset, 0]))
        env.execution_rng = np.random.default_rng(np.random.SeedSequence([noise_key, offset, 1]))
        command, _ = safety.filter(proposed, observation)
        target_before, velocity_before = env.target_position.copy(), env.target_velocity.copy()
        sign_before = 0 if env.target_branch_sign is None else int(env.target_branch_sign)
        observation, _, done, truncated, info = base.env_class.step(env, command)
        sign_after = 0 if env.target_branch_sign is None else int(env.target_branch_sign)
        # The original step applies defenders before evaluating target_action.
        distances = np.linalg.norm(env.defender_positions - target_before[None], axis=-1)
        exposure = np.maximum(float(env.pursuit["target_defender_avoidance_distance"]) - distances, 0.)
        scores_valid = bool(np.isfinite(env.target_branch_scores).all())
        rows.append({"target": env.target_position.copy(), "defenders": env.defender_positions.copy(),
                     "commanded": np.asarray(command).copy(), "executed": env.last_executed_actions.copy(),
                     "valid": not any(info.get(k, False) for k in ("target_invalid_episode", "target_boundary_violation", "target_obstacle_violation")),
                     "termination": str(info.get("termination_reason", "running")),
                     "pre_target_position_label_only": target_before, "pre_target_velocity_label_only": velocity_before,
                     "branch_before_label_only": sign_before, "branch_after_label_only": sign_after,
                     "branch_commit_label_only": sign_before == 0 and sign_after != 0,
                     "branch_scores_label_only": env.target_branch_scores.copy() if scores_valid else np.zeros(2),
                     "branch_scores_valid_label_only": scores_valid,
                     "avoidance_exposure_label_only": exposure})
        if done or truncated or not rows[-1]["valid"]:
            break
    fields = {}
    for key in rows[0]:
        sample = np.asarray(rows[0][key])
        if sample.dtype.kind in "US":
            fields[key] = np.full((len(sequence),), "after_terminal", dtype="U128")
        else:
            fields[key] = np.zeros((len(sequence),) + sample.shape, dtype=sample.dtype)
        fields[key][:len(rows)] = np.stack([r[key] for r in rows])
    return fields


def verify_original_fields(values, traces, anchor, atol):
    for name in ("target", "defenders", "commanded", "executed", "valid", "termination"):
        replayed = np.stack([t[name] for t in traces])
        original = values[name]
        equal = np.array_equal(replayed, original) if original.dtype.kind not in "fc" else np.allclose(replayed, original, rtol=0., atol=atol)
        if not equal:
            raise ValueError("Actual V12 candidate branch replay differs: " + name)
    for name in ("target", "commanded", "executed", "valid", "termination"):
        original = values["anchor_" + name]
        equal = np.array_equal(anchor[name], original) if original.dtype.kind not in "fc" else np.allclose(anchor[name], original, rtol=0., atol=atol)
        if not equal:
            raise ValueError("Actual V12 reference branch replay differs: " + name)


def call_metrics(values, mechanism, protocol):
    valid = values["valid"] & values["anchor_valid"][None]
    nonanchor = ~(values["proposed"] == values["anchor"][None]).all((1, 2, 3))
    common = valid & nonanchor[:, None]
    observed = values["termination"] != "after_terminal"
    filtered = np.linalg.norm(values["commanded"] - values["proposed"], axis=-1)
    execution = np.linalg.norm(values["executed"] - values["commanded"], axis=-1)
    response = np.linalg.norm(values["target"] - values["anchor_target"][None], axis=-1)
    sign, reference_sign = mechanism["branch_after_label_only"], mechanism["anchor_branch_after_label_only"]
    both = common & (sign != 0) & (reference_sign[None] != 0)
    flip = both & (sign != reference_sign[None])
    pending = common & ((sign == 0) | (reference_sign[None] == 0))
    same = both & ~flip
    sample = lambda data, mask: {"count": int(mask.sum()), "sum": float(data[mask].sum()), "maximum": float(data[mask].max()) if mask.any() else 0.}
    full_pairs = common.all(1)
    initial_branch = int(mechanism["initial_branch_label_only"])
    first_response = []
    for effect, support, selected in zip(response, common, nonanchor):
        indices = np.flatnonzero(support & (effect > protocol["timing_response_threshold_m"]))
        if selected and indices.size:
            first_response.append(int(indices[0]))
    if initial_branch != 0 and flip.any():
        raise ValueError("Committed original target branch changed after commitment")
    return {"candidate_count": len(nonanchor), "nonanchor_candidates": int(nonanchor.sum()),
        "nonanchor_common_points": int(common.sum()), "full_nonanchor_pairs": int(full_pairs.sum()),
        "response": sample(response, common), "same_branch_response": sample(response, same), "flipped_branch_response": sample(response, flip), "pending_branch_response": sample(response, pending),
        "branch_flip_points": int(flip.sum()), "branch_flip_pairs": int(flip.any(1).sum()), "committed_pairs": int(both.any(1).sum()),
        "cbf_changed_command_points": int((filtered[observed] > protocol["changed_command_threshold_mps"]).sum()),
        "observed_command_points": int(observed.sum() * 4), "cbf_correction_max_mps": float(filtered[observed].max()),
        "execution_error_max_mps": float(execution[observed].max()),
        "initial_committed": initial_branch != 0, "first_response_offsets": first_response,
        "strong_response_points": int((response[common] > protocol["response_threshold_m"]).sum()),
        "support_points_by_offset": common.sum(0).tolist(), "strong_response_points_by_offset": ((response > protocol["response_threshold_m"]) & common).sum(0).tolist(),
        "terminal_labels": {label: int((values["termination"][observed] == label).sum()) for label in sorted(set(values["termination"][observed].tolist()))}}


def summarize(records, protocol):
    groups = sorted({r["group"] for r in records})
    total = lambda key: sum(r["metrics"][key] for r in records)
    error_max = lambda key: max(r["metrics"][key] for r in records)
    response_summary = {}
    for name in ("response", "same_branch_response", "flipped_branch_response", "pending_branch_response"):
        by_group = {}
        for group in groups:
            chosen = [r["metrics"][name] for r in records if r["group"] == group]
            count = sum(r["count"] for r in chosen)
            if count:
                by_group[group] = sum(r["sum"] for r in chosen) / count
        response_summary[name] = {"groups_with_support": len(by_group), "points": sum(r["metrics"][name]["count"] for r in records),
                                  "group_equal_mean_m": float(np.mean(list(by_group.values()))) if by_group else None,
                                  "maximum_m": max(r["metrics"][name]["maximum"] for r in records)}
    by_step = {}
    for step in sorted({r["step"] for r in records}):
        subset = [r for r in records if r["step"] == step]
        by_step[str(step)] = {"calls": len(subset), "initial_committed_calls": sum(r["metrics"]["initial_committed"] for r in subset),
                             "branch_flip_pairs": sum(r["metrics"]["branch_flip_pairs"] for r in subset),
                             "common_points": sum(r["metrics"]["nonanchor_common_points"] for r in subset)}
    first = [offset for r in records for offset in r["metrics"]["first_response_offsets"]]
    return {"status": "train_mechanism_replay_complete_not_promoted", "calls": len(records), "groups": len(groups),
        "nonanchor_candidates": total("nonanchor_candidates"), "full_nonanchor_pairs": total("full_nonanchor_pairs"),
        "nonanchor_common_points": total("nonanchor_common_points"), "strong_response_points": total("strong_response_points"),
        "true_branch_flip_pairs": total("branch_flip_pairs"), "true_both_committed_pairs": total("committed_pairs"),
        "cbf_changed_command_points": total("cbf_changed_command_points"), "observed_command_points": total("observed_command_points"),
        "cbf_correction_max_mps": error_max("cbf_correction_max_mps"), "execution_error_max_mps": error_max("execution_error_max_mps"),
        "initial_committed_calls": total("initial_committed"), "response_by_true_branch": response_summary, "by_step": by_step,
        "first_response_offset_counts": {str(offset): first.count(offset) for offset in sorted(set(first))},
        "support_points_by_offset": np.sum([r["metrics"]["support_points_by_offset"] for r in records], axis=0).tolist(),
        "strong_response_points_by_offset": np.sum([r["metrics"]["strong_response_points_by_offset"] for r in records], axis=0).tolist(),
        "enhanced_control_enabled": False, "new_model_training_enabled": False, "holdout_used": False, "prior_gate_overridden": False,
        "private_mechanism_labels_network_inputs": False,
        "limitations": "TRAIN-only diagnosis, not new generalization/learning or control. True private branch/exposure labels are diagnostic/supervision-only, never public predictor inputs. Same and flipped cohorts are descriptive, not an intervention isolating branch from avoidance or CBF. Execution disabled in these scenes; no transfer claim to enabled execution. Prefix termination support retained, no future imputation. V12 failed gate unchanged."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "cwm_v12/artifacts/task_effect_training_20261010.zip")
    parser.add_argument("--capsule", type=Path, default=ROOT.parent / "cwm_v1/baseline/capsule.zip")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    if sha(args.source) != protocol["source_archive_sha256"] or sha(args.capsule) != protocol["baseline_capsule_sha256"]:
        raise ValueError("Protected source/baseline mismatch")
    if protocol["split"] != "train" or any(protocol[key] for key in ("new_model_training_enabled", "enhanced_control_enabled", "holdout_used", "prior_gate_override_allowed", "private_mechanism_labels_network_inputs")):
        raise ValueError("TRAIN-only disabled diagnostic contract mismatch")
    check_archive(args.source, "ARTIFACT_MANIFEST.json", False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.source) as source:
        data_report = json.loads(source.read("data/summary.json"))
        scenes = [json.loads(line) for line in source.read("data/scenes.jsonl").decode().splitlines()]
        scenes = [r for r in scenes if r["model_split"] == "train"]
        rows = [r for r in json.loads(source.read("data/calls.json")) if r["split"] == "train"]
        if len(scenes) != protocol["episodes"] or sum("arrays_path" in r for r in rows) != protocol["eligible_calls"]:
            raise ValueError("Actual TRAIN population mismatch")
        base = ActualLocalTap(args.capsule, args.output / "restored", data_report["protocol"]["snapshot_steps"])
        sources = sorted({Path(module.__file__).resolve() for module in list(sys.modules.values()) if getattr(module, "__file__", None)
                          and Path(module.__file__).resolve().is_relative_to(ROOT.parent)} | {ROOT / "protocol.json"})
        hashes = {p.relative_to(ROOT.parent).as_posix(): sha(p) for p in sources}
        for path in sources:
            target = args.output / "source" / path.relative_to(ROOT.parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        records, episodes = [], []
        started = time.perf_counter()
        for scene in scenes:
            selected = [r for r in rows if r["episode_index"] == scene["episode_index"]]
            def observer(env, observation, actions, sequence):
                before = fingerprint(env)
                for call in base.calls:
                    row = next(r for r in selected if r["step"] == env.step_count and r["agent"] == call["agent"])
                    if "arrays_path" not in row:
                        continue
                    if not compare_public(public_call_snapshot(call), json.loads(source.read("data/" + row["context_path"]))):
                        raise ValueError("Actual original local context replay differs")
                    values = arrays(source.read("data/" + row["arrays_path"]))
                    branches = [branch_trace(base, env, proposed, row["noise_key"]) for proposed in values["proposed"]]
                    anchor = branch_trace(base, env, values["anchor"], row["noise_key"])
                    repeated = branch_trace(base, env, values["anchor"], row["noise_key"])
                    if not compare_arrays(anchor, repeated):
                        raise ValueError("Repeated mechanism anchor differs")
                    verify_original_fields(values, branches, anchor, protocol["replay_array_atol"])
                    fields = {key: np.stack([branch[key] for branch in branches]) for key in branches[0] if key.endswith("label_only")}
                    fields.update({"anchor_" + key: value for key, value in anchor.items() if key.endswith("label_only")})
                    fields["initial_branch_label_only"] = np.asarray(0 if env.target_branch_sign is None else int(env.target_branch_sign))
                    relative = f"calls/{row['episode_index']}_{row['step']}_{row['agent']}.npz"
                    path = args.output / relative
                    path.parent.mkdir(exist_ok=True)
                    np.savez_compressed(path, **fields)
                    records.append({"episode_index": row["episode_index"], "step": row["step"], "agent": row["agent"], "group": row["group"], "split": "train",
                                    "mechanism_path": relative, "mechanism_sha256": sha(path), "original_branch_fields_replay_equal": True,
                                    "anchor_repeat_equal": True, "metrics": call_metrics(values, fields, protocol)})
                if fingerprint(env) != before:
                    raise ValueError("Mechanism diagnostic changed actual parent")
            trajectory = args.output / "trajectories" / f"{scene['episode_index']}.npz"
            outcome, _ = base.run(scene, trajectory, observer)
            if not compare_arrays(arrays(trajectory.read_bytes()), arrays(source.read(f"data/observed/{scene['episode_index']}.npz"))):
                raise ValueError("Mechanism replay changed original trajectory")
            episodes.append({"episode_index": scene["episode_index"], "original_arrays_equal": True, "trajectory_sha256": sha(trajectory), "safe_capture_success": outcome["safe_capture_success"]})
            print(json.dumps({"episodes": len(episodes), "calls": len(records)}), flush=True)
        if len(records) != protocol["eligible_calls"]:
            raise ValueError("Incomplete TRAIN mechanism coverage")
        report = summarize(records, protocol)
    verify(base.root, base.capsule_manifest)
    if any(sha(ROOT.parent / name) != digest for name, digest in hashes.items()):
        raise ValueError("Mechanism run-used sources changed")
    report.update(protocol=protocol, source_hashes=hashes, episodes=episodes, elapsed_seconds=time.perf_counter() - started)
    (args.output / "records.json").write_text(json.dumps(records, indent=2))
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
