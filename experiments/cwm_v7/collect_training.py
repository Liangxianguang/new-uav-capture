"""Collect preassigned training/validation groups on qualified translated S4 geometry."""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent / "cwm_v4"), str(ROOT.parent / "cwm_v2"), str(ROOT.parent / "cwm_v1")]
from scout import BackboneTap
from diagnose import fingerprint
from diagnose_interactions import summarized
from freeze_baseline import sha, verify
sys.path.insert(0, str(ROOT))
from geometry import translated_records


def s4_commands(sequence, observation, reference, protocol):
    sequence = np.asarray(sequence, dtype=np.float64)
    if sequence.shape != (8, 4, 3):
        raise ValueError("Frozen four-agent eight-step command contract required")
    original = np.concatenate([sequence, np.repeat(sequence[-1:], protocol["horizon_steps"] - 8, axis=0)])
    nearest = int(np.argmin(np.linalg.norm(np.asarray(observation["defender_positions"]) - reference, axis=-1)))
    values = [original.copy()]
    for magnitude in (3., 5.):
        for sign in (1., -1.):
            command = original.copy()
            command[:, :, 1] += sign * magnitude
            values.append(command)
    values.append(np.zeros_like(original))
    for sign in (1., -1.):
        command = original.copy()
        command[:, nearest, 1] += sign * 5.
        values.append(command)
    result = np.stack(values)
    # Never clip the reference branch; fail if the frozen contract is violated.
    if np.linalg.norm(result[0], axis=-1).max() > protocol["maximum_command_speed_mps"] + 1e-8:
        raise ValueError("Reference exceeds original speed limit")
    result[1:] *= np.minimum(1., protocol["maximum_command_speed_mps"] / np.maximum(np.linalg.norm(result[1:], axis=-1, keepdims=True), 1e-9))
    return result


def scene_records(base, protocol):
    from encirclement3d.showcase import s4_adaptive_branching_scenario, scenario_metadata, s4_branch_route_metrics
    conditions = list(itertools.product(protocol["target_speed_scales"], protocol["observation_conditions"]))
    scenes = []
    for group in range(protocol["groups"]):
        speed, condition = conditions[group % len(conditions)]
        for member, bias in enumerate(("upper", "lower")):
            record = {"episode_index": protocol["seed_start"] + group * 2 + member,
                      "episode_seed": protocol["seed_start"] + group * 2 + member,
                      "mirror_group_id": f"cwm-s4-{protocol['layout_seed_start'] + group}", "mirror_pair_member": bias,
                      "layout_seed": protocol["layout_seed_start"] + group, "target_motion_mode": "adaptive_branching",
                      "target_speed_scale": speed, "obstacle_count": 1, "variant": f"s4_{condition['name']}_{speed}",
                      "level": None, "model_split": protocol["stage"], "observation_condition": condition["name"],
                      "pursuit_overrides": copy.deepcopy(condition["overrides"]),
                      "execution": {"enabled": False, "action_delay_steps": 0, "command_noise_std": 0.}}
            if protocol["target_branch_rule_overrides"]:
                raise ValueError("Do not change existing S4 branch rules")
            env = base.env_class(base.configuration(record), obstacle_count=1, target_speed_scale=speed)
            scenario = s4_adaptive_branching_scenario(env, record["layout_seed"], bias, protocol["scene_variation"])
            record["scenario"] = scenario_metadata(scenario)
            record["route_validation"] = s4_branch_route_metrics(env, scenario)
            scenes.append(record)
    return scenes


def s4_rollout(base, parent, sequence, noise_key):
    env = copy.deepcopy(parent)
    safety = base.filter_class(env)
    observation = env.observe()
    rows = []
    for offset, proposed in enumerate(sequence):
        env.rng = np.random.default_rng(np.random.SeedSequence([noise_key, offset, 0]))
        env.execution_rng = np.random.default_rng(np.random.SeedSequence([noise_key, offset, 1]))
        command, _ = safety.filter(proposed, observation)
        observation, _, done, truncated, info = base.env_class.step(env, command)
        valid = not any(info.get(k, False) for k in ("target_invalid_episode", "target_boundary_violation", "target_obstacle_violation"))
        rows.append({"target": env.target_position.copy(), "defenders": env.defender_positions.copy(),
                     "commanded": np.asarray(command).copy(), "executed": env.last_executed_actions.copy(),
                     "branch_sign_label_only": 0 if env.target_branch_sign is None else int(env.target_branch_sign),
                     "valid": valid, "termination": str(info.get("termination_reason", "running"))})
        if done or truncated or not valid:
            break
    fields = {}
    for key in ("target", "defenders", "commanded", "executed"):
        value = np.zeros((len(sequence),) + rows[0][key].shape, dtype=np.float64)
        value[:len(rows)] = np.stack([r[key] for r in rows])
        fields[key] = value
    fields["valid"] = np.zeros(len(sequence), dtype=bool)
    fields["valid"][:len(rows)] = [r["valid"] for r in rows]
    fields["branch_sign_label_only"] = np.zeros(len(sequence), dtype=np.int8)
    fields["branch_sign_label_only"][:len(rows)] = [r["branch_sign_label_only"] for r in rows]
    fields["termination"] = np.full(len(sequence), "after_terminal", dtype="U128")
    fields["termination"][:len(rows)] = [r["termination"] for r in rows]
    return fields


def s4_statistics(data, protocol):
    result = {}
    for horizon in (8, 16):
        valid = data["valid"][:, 1:, :horizon] & data["valid"][:, :1, :horizon]
        effect = np.linalg.norm(data["target"][:, 1:, :horizon] - data["target"][:, :1, :horizon], axis=-1)
        signs = data["branch_sign_label_only"][:, :, :horizon]
        committed = valid & (signs[:, 1:] != 0) & (signs[:, :1] != 0)
        flipped = (signs[:, 1:] != signs[:, :1]) & committed
        support_pairs = committed.any(axis=-1)
        flip_pairs = flipped.any(axis=-1)
        full = valid.all(axis=-1)
        signal_groups = sorted({str(g) for g, e, m in zip(data["group"], effect, valid) if np.any(e[m] > protocol["effect_threshold_m"])})
        flip_groups = sorted({str(g) for g, f in zip(data["group"], flip_pairs) if f.any()})
        result[str(horizon)] = {"effect_m": summarized(effect[valid]),
                               "effect_over_0_05m_fraction": float(np.mean(effect[valid] > protocol["effect_threshold_m"])) if valid.any() else None,
                               "signal_groups": signal_groups, "branch_flip_groups": flip_groups,
                               "branch_flip_pairs": int(flip_pairs.sum()), "both_committed_support_pairs": int(support_pairs.sum()),
                               "branch_flip_pair_fraction": float(flip_pairs.sum() / support_pairs.sum()) if support_pairs.any() else None,
                               "full_horizon_pairs": int(full.sum()),
                               "full_horizon_endpoint_effect_m": summarized(effect[:, :, -1][full]),
                               "executed_action_difference_mps": summarized(np.linalg.norm(data["executed"][:, 1:, :horizon] - data["executed"][:, :1, :horizon], axis=-1).mean(-1)[valid])}
    h = result["8"]
    gate = protocol["data_gate"]
    adequate = ((h["effect_over_0_05m_fraction"] or 0.) >= gate["minimum_effect_point_fraction"]
                and (h["branch_flip_pair_fraction"] or 0.) >= gate["minimum_branch_flip_pair_fraction"]
                and len(h["signal_groups"]) >= gate["minimum_independent_signal_groups"]
                and len(h["branch_flip_groups"]) >= gate["minimum_independent_signal_groups"])
    return {"horizons": result, "short_horizon_data_gate_passed": adequate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "training_data_protocol.json")
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.protocol = args.protocol.resolve()
    protocol = json.loads(args.protocol.read_text())
    development = json.loads((args.development / "summary.json").read_text())
    qualification = json.loads((args.qualification / "summary.json").read_text())
    center = protocol["wall_center_x_m"]
    if development["selected_wall_center_x_m"] != center or qualification["selected_wall_center_x_m"] != center:
        raise ValueError("Geometry has not passed both development and fresh qualification")
    if protocol["enhanced_control_enabled"] or protocol["private_labels_model_inputs"]:
        raise ValueError("Training collection must not enable control or private inputs")
    for run in (args.development, args.qualification):
        for row in json.loads((run / "summary.json").read_text())["candidates"]:
            if row["wall_center_x_m"] == center:
                episodes = json.loads((run / f"wallx{center:g}/episodes.json").read_text())
                stress = json.loads((run / f"wallx{center:g}/stress.json").read_text())
                if any(r["target_invalid_episode"] for r in episodes) or any(r["target_invalid"] for r in stress):
                    raise ValueError("Actual geometry qualification contains target-invalid evidence")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = BackboneTap(args.capsule, args.output / "restored")
    scenes = translated_records(base, protocol, center)
    split_names = protocol["variant_cycles_split"]
    if len(scenes) != 8 * len(split_names):
        raise ValueError("One four-variant mirror cycle per preassigned split required")
    for group in range(protocol["groups"]):
        for record in scenes[group * 2:group * 2 + 2]:
            record["model_split"] = split_names[group // 4]
            record["variant"] += f"_wallx{center:g}"
    (args.output / "selected_scenes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scenes))
    sources = [Path(__file__), args.protocol, ROOT / "geometry.py", ROOT.parent / "cwm_v6/s4_collect.py", ROOT.parent / "cwm_v4/diagnose.py", ROOT.parent / "cwm_v2/scout.py",
               ROOT.parent / "cwm_v1/baseline.py", ROOT.parent / "cwm_v1/collect_pairs.py",
               ROOT.parent / "cwm_v1/diagnose_interactions.py", ROOT.parent / "cwm_v1/freeze_baseline.py"]
    source_hashes = {p.relative_to(ROOT.parent).as_posix(): sha(p) for p in sources}
    for path in sources:
        destination = args.output / "source" / path.relative_to(ROOT.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    fields, windows, episodes = defaultdict(list), [], []
    started = time.perf_counter()
    for record in scenes:
        history = []
        def observer(env, observation, actions, sequence):
            if any("branch" in k for k in observation):
                raise AssertionError("Private branch leaked into observation")
            history.append(base.evaluator.policy_observations(env, observation).reshape(-1).copy())
            if env.step_count not in protocol["snapshot_steps"] or sequence is None:
                return
            reference, velocity = base.evaluator.belief_reference(observation, base.planner_config)
            proposed = s4_commands(sequence, observation, reference, protocol)
            key = int.from_bytes(hashlib.sha256(f"{protocol['seed_start']}/{record['episode_index']}/{env.step_count}".encode()).digest()[:4], "little")
            before = fingerprint(env)
            branches = [s4_rollout(base, env, command, key) for command in proposed]
            repeated = s4_rollout(base, env, proposed[0], key)
            if any(repeated[k].tobytes() != branches[0][k].tobytes() for k in repeated):
                raise AssertionError("Cloned repeat mismatch")
            if fingerprint(env) != before:
                raise AssertionError("Clones changed parent")
            packed = {k: np.stack([b[k] for b in branches]) for k in branches[0]}
            padded = [history[0]] * max(0, 8 - len(history)) + history[-8:]
            packed.update(history=np.stack(padded), relative=np.concatenate([observation["defender_positions"] - reference,
                                                                             observation["defender_velocities"] - velocity], axis=-1),
                          proposed=proposed, backbone=base.last_backbone[0].copy(), reference=reference.copy(), reference_velocity=velocity.copy(),
                          group=str(record["mirror_group_id"]), variant=record["variant"], split=record["model_split"], step=int(env.step_count))
            for name, value in packed.items():
                fields[name].append(value)
            windows.append({"episode_index": record["episode_index"], "group": record["mirror_group_id"], "step": int(env.step_count),
                            "private_branch_at_snapshot_label_only": env.target_branch_sign, "parent_integrity": True, "repeat_exact": True})
            np.savez_compressed(args.output / "pairs_partial.npz", **{k: np.stack(v) for k, v in fields.items()})
            (args.output / "windows.json").write_text(json.dumps(windows, indent=2))
            print(json.dumps({"episode": record["episode_index"], "step": env.step_count, "windows": len(windows)}), flush=True)

        trajectory = args.output / "observed" / f"{record['episode_index']}.npz"
        row, _ = base.run(record, trajectory, observer=observer)
        plain_path = args.output / "unobserved" / trajectory.name
        plain, _ = base.run(record, plain_path)
        with np.load(trajectory, allow_pickle=False) as a, np.load(plain_path, allow_pickle=False) as b:
            exact = all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes() for k in ("target_positions", "defender_positions"))
        keys = ("safe_capture_success", "capture_event", "collision", "boundary_violation", "timeout", "target_invalid_episode", "termination_reason")
        equal = all(row[k] == plain[k] for k in keys)
        if not exact or not equal:
            raise AssertionError("S4 observer changed original controller")
        episodes.append({"episode_index": record["episode_index"], "split": record["model_split"], "trajectory_byte_equal": exact, "outcomes_equal": equal,
                         "observed_sha256": sha(trajectory), "unobserved_sha256": sha(plain_path), **{k: row[k] for k in keys}})
        (args.output / "episodes.json").write_text(json.dumps(episodes, indent=2))
    if not windows:
        raise ValueError("No windows")
    data = {k: np.stack(v) for k, v in fields.items()}
    np.savez_compressed(args.output / "pairs.npz", **data)
    statistics = s4_statistics(data, protocol)
    verify(base.root, base.capsule_manifest)
    if any(sha(ROOT.parent / name) != expected for name, expected in source_hashes.items()):
        raise AssertionError("Experiment source changed during collection")
    report = {"status": "signal_requires_independent_confirmation" if statistics["short_horizon_data_gate_passed"] else "s4_data_no_go_keep_baseline",
              "enhanced_control_enabled": False, "new_model_trained": False, "protocol": protocol,
              "states": len(windows), "episodes": len(episodes), "groups": len(set(data["group"])),
              "capsule_sha256": sha(args.capsule), "dataset_sha256": sha(args.output / "pairs.npz"), "source_hashes": source_hashes,
              "statistics": statistics, "episodes_results": episodes, "elapsed_seconds": time.perf_counter() - started,
              "development_summary_sha256": sha(args.development / "summary.json"), "qualification_summary_sha256": sha(args.qualification / "summary.json"),
              "target_valid_original_episodes": sum(not r["target_invalid_episode"] for r in episodes),
              "training_eligible": statistics["short_horizon_data_gate_passed"] and not any(r["target_invalid_episode"] for r in episodes),
              "group_split_preassigned": True, "holdout_collected": False,
              "limitations": ["Separate S4 mechanism scout under a different target strategy; not original Level1-8 improvement.",
                              "Mirrored scenes share a group; window/action points are not independent scenes.",
                              "Branch truth is labels only. New branch rule/code, weights, encoder or safety relaxation forbidden.",
                              "Original local CBF is empirical; observer isolation does not prove enhanced-controller safety.",
                              "16-step tail is a held original command, not a longer DN-MPC plan."]}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "states": len(windows), "statistics": statistics}), flush=True)


if __name__ == "__main__":
    main()
