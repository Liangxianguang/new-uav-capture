"""Public-geometry single-agent interventions on fresh original TRAIN groups."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT.parent / "cwm_v2"), str(ROOT.parent / "cwm_v1")]
from scout import BackboneTap, select_scenes
from freeze_baseline import sha, verify
from diagnose_interactions import summarized


def commands(sequence, observation, reference, protocol):
    original = np.asarray(sequence, dtype=np.float64)
    if original.shape != (8, 4, 3) or protocol["horizon_steps"] < 8:
        raise ValueError("Frozen 8-step four-agent contract required")
    maximum = protocol["maximum_command_speed_mps"]
    if np.linalg.norm(original, axis=-1).max() > maximum + 1e-8:
        raise ValueError("Baseline command exceeds frozen speed limit")
    original = np.concatenate([original, np.repeat(original[-1:], protocol["horizon_steps"] - 8, axis=0)])
    radial = np.asarray(reference) - np.asarray(observation["defender_positions"])
    radial /= np.maximum(np.linalg.norm(radial, axis=-1, keepdims=True), 1e-9)
    tangent = np.stack([-radial[:, 1], radial[:, 0], np.zeros(4)], axis=-1)
    values = [original.copy()]
    for agent in range(4):
        for sign in (-1, 1):
            changed = original.copy()
            changed[:, agent] += sign * protocol["single_agent_tangent_bias_mps"] * tangent[agent]
            changed[:, agent] *= np.minimum(1., maximum / np.maximum(np.linalg.norm(changed[:, agent], axis=-1, keepdims=True), 1e-9))
            values.append(changed)
        changed = original.copy()
        changed[:, agent] = 0.
        values.append(changed)
    return np.stack(values)


def fingerprint(env):
    # Trusted, in-memory serialization only. No pickle file is produced/loaded.
    return hashlib.sha256(pickle.dumps(env.__dict__, protocol=5)).hexdigest()


def rollout(base, parent, sequence, noise_key):
    env = copy.deepcopy(parent)
    safety = base.filter_class(env)
    observation = env.observe()
    rows = []
    for offset, proposed in enumerate(sequence):
        env.rng = np.random.default_rng(np.random.SeedSequence([noise_key, offset, 0]))
        env.execution_rng = np.random.default_rng(np.random.SeedSequence([noise_key, offset, 1]))
        command, _ = safety.filter(proposed, observation)
        last_replan = env.target_maneuver_last_replan_step
        observation, _, done, truncated, info = base.env_class.step(env, command)
        valid = not any(info.get(k, False) for k in ("target_invalid_episode", "target_boundary_violation", "target_obstacle_violation"))
        rows.append({"target": env.target_position.copy(), "defenders": env.defender_positions.copy(),
                     "commanded": np.asarray(command).copy(), "executed": env.last_executed_actions.copy(),
                     "sensor": env.target_maneuver_observed_positions.copy(),
                     "direction": env.target_maneuver_direction.copy(), "mode": str(env.target_maneuver_mode),
                     "route": str(env.target_maneuver_route), "replanned": env.target_maneuver_last_replan_step != last_replan,
                     "valid": valid, "termination": str(info.get("termination_reason", "running"))})
        if done or truncated or not valid:
            break
    return rows


def pack(branches, horizon):
    fields = {}
    for key in ("target", "defenders", "commanded", "executed", "sensor", "direction"):
        value = np.zeros((len(branches), horizon) + branches[0][0][key].shape, dtype=np.float64)
        for i, branch in enumerate(branches):
            value[i, :len(branch)] = np.stack([r[key] for r in branch])
        fields[key] = value
    for key in ("valid", "replanned"):
        value = np.zeros((len(branches), horizon), dtype=bool)
        for i, branch in enumerate(branches):
            value[i, :len(branch)] = [r[key] for r in branch]
        fields[key] = value
    for key in ("mode", "route", "termination"):
        value = np.full((len(branches), horizon), "after_terminal", dtype="U128")
        for i, branch in enumerate(branches):
            value[i, :len(branch)] = [r[key] for r in branch]
        fields[key] = value
    return fields


def summarize(data, protocol):
    report = {}
    for variant in protocol["variants"]:
        selected = data["variant"] == variant
        targets = data["target"][selected]
        masks = data["valid"][selected]
        report[variant] = {}
        for horizon in protocol["horizons"]:
            common = masks[:, 1:, :horizon] & masks[:, :1, :horizon]
            effect = np.linalg.norm(targets[:, 1:, :horizon] - targets[:, :1, :horizon], axis=-1)
            full = common.all(axis=-1)
            signal_groups = sorted({str(g) for g, e, m in zip(data["group"][selected], effect, common) if np.any(e[m] > .05)})
            full_groups = sorted({str(g) for g, e, m in zip(data["group"][selected], effect[:, :, -1], full) if np.any(e[m] > .05)})
            endpoint = effect[:, :, -1][full]
            metrics = {"states": int(selected.sum()), "common_prefix_effect_m": summarized(effect[common]),
                       "fraction_over_0_05m": float(np.mean(effect[common] > .05)) if common.any() else None,
                       "signal_groups": signal_groups, "full_horizon_pairs": int(full.sum()),
                       "full_horizon_endpoint_effect_m": summarized(endpoint),
                       "full_horizon_fraction_over_0_05m": float(np.mean(endpoint > .05)) if endpoint.size else None,
                       "full_horizon_signal_groups": full_groups,
                       "mode_differs_fraction": float(np.mean((data["mode"][selected, 1:, :horizon] != data["mode"][selected, :1, :horizon])[common])) if common.any() else None}
            for key, name in (("executed", "executed_difference_mps"), ("sensor", "target_sensor_difference_m")):
                value = data[key][selected]
                difference = np.linalg.norm(value[:, 1:, :horizon] - value[:, :1, :horizon], axis=-1).mean(axis=-1)
                metrics[name] = summarized(difference[common])
            # Report each predeclared action, not only the strongest selected one.
            metrics["branches"] = {name: {"effect_m": summarized(effect[:, i][common[:, i]]),
                                         "full_pairs": int(full[:, i].sum())}
                                   for i, name in enumerate(protocol["branches"][1:])}
            report[variant][str(horizon)] = metrics
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--training-scenes", type=Path, required=True)
    parser.add_argument("--exclude-scenes", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = BackboneTap(args.capsule, args.output / "restored")
    validation = base.root / "results/phase91_zero_cbf_curriculum_v5/validation/scenes.jsonl"
    exclusions = [validation, *args.exclude_scenes]
    forbidden = {str(json.loads(line)["mirror_group_id"]) for p in exclusions for line in p.read_text().splitlines() if line}
    scenes = select_scenes(args.training_scenes, protocol, forbidden)
    (args.output / "selected_scenes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scenes))
    source_paths = [Path(__file__), args.protocol, ROOT.parent / "cwm_v2/scout.py", ROOT.parent / "cwm_v1/baseline.py",
                    ROOT.parent / "cwm_v1/freeze_baseline.py", ROOT.parent / "cwm_v1/collect_pairs.py", ROOT.parent / "cwm_v1/diagnose_interactions.py"]
    fields, windows, episodes = defaultdict(list), [], []
    started = time.perf_counter()
    for record in scenes:
        history = []
        def observer(env, observation, actions, sequence):
            history.append(base.evaluator.policy_observations(env, observation).reshape(-1).copy())
            if env.step_count not in protocol["snapshot_steps"] or sequence is None:
                return
            reference, velocity = base.evaluator.belief_reference(observation, base.planner_config)
            proposed = commands(sequence, observation, reference, protocol)
            noise_key = int.from_bytes(hashlib.sha256(f"{protocol['selection_seed']}/{record['episode_index']}/{env.step_count}".encode()).digest()[:4], "little")
            before = fingerprint(env)
            branches = [rollout(base, env, s, noise_key) for s in proposed]
            packed = pack(branches, protocol["horizon_steps"])
            repeated = pack([rollout(base, env, proposed[0], noise_key)], protocol["horizon_steps"])
            if any(v[:1].tobytes() != repeated[k].tobytes() for k, v in packed.items()):
                raise AssertionError("Repeat differs")
            if fingerprint(env) != before:
                raise AssertionError("Clone mutated parent environment")
            padded = [history[0]] * max(0, 8 - len(history)) + history[-8:]
            packed.update(history=np.stack(padded), relative=np.concatenate([observation["defender_positions"] - reference,
                                                                            observation["defender_velocities"] - velocity], axis=-1),
                          proposed=proposed, backbone=base.last_backbone[0].copy(), reference=reference.copy(),
                          reference_velocity=velocity.copy(), group=str(record["mirror_group_id"]), variant=record["variant"], step=int(env.step_count))
            for key, value in packed.items():
                fields[key].append(value)
            windows.append({"episode_index": record["episode_index"], "group": str(record["mirror_group_id"]), "step": int(env.step_count),
                            "private_mode_diagnostic_only": str(env.target_maneuver_mode), "parent_integrity": True, "repeat_exact": True,
                            "public_nearest_distance_m": float(np.linalg.norm(packed["relative"][:, :3], axis=-1).min())})
            np.savez_compressed(args.output / "pairs_partial.npz", **{k: np.stack(v) for k, v in fields.items()})
            (args.output / "windows.json").write_text(json.dumps(windows, indent=2))
            print(json.dumps({"episode": record["episode_index"], "step": env.step_count, "states": len(windows)}), flush=True)

        path = args.output / "observed" / f"{record['episode_index']}.npz"
        row, _ = base.run(record, path, observer=observer)
        plain_path = args.output / "unobserved" / path.name
        plain_row, _ = base.run(record, plain_path)
        with np.load(path, allow_pickle=False) as a, np.load(plain_path, allow_pickle=False) as b:
            exact = all(a[k].shape == b[k].shape and a[k].dtype == b[k].dtype and a[k].tobytes() == b[k].tobytes()
                        for k in ("defender_positions", "target_positions"))
        keys = ("safe_capture_success", "capture_event", "collision", "boundary_violation", "timeout", "target_invalid_episode", "termination_reason")
        outcomes = all(row[k] == plain_row[k] for k in keys)
        if not exact or not outcomes:
            raise AssertionError("Observed TRAIN baseline differs")
        episodes.append({"episode_index": record["episode_index"], "variant": record["variant"], "trajectories_byte_equal": exact,
                         "outcomes_equal": outcomes, "trajectory_sha256": sha(path), "unobserved_sha256": sha(plain_path),
                         **{k: row[k] for k in keys}})
        (args.output / "episodes.json").write_text(json.dumps(episodes, indent=2))
    verify(base.root, base.capsule_manifest)
    if not windows:
        raise ValueError("No windows")
    data = {k: np.stack(v) for k, v in fields.items()}
    np.savez_compressed(args.output / "pairs.npz", **data)
    variants = summarize(data, protocol)
    candidates = []
    for variant, horizons in variants.items():
        for horizon, values in horizons.items():
            threshold = protocol["minimum_next_stage_signal"]
            if (values["fraction_over_0_05m"] or 0) >= threshold["effect_over_0_05m_valid_point_fraction"] and len(values["signal_groups"]) >= threshold["independent_scene_groups_with_effect_over_0_05m"]:
                candidates.append({"variant": variant, "horizon": int(horizon)})
    report = {"status": "signal_requires_independent_confirmation" if candidates else "data_adequacy_no_go_keep_baseline",
              "enhanced_control_enabled": False, "states": len(windows), "episodes": len(episodes), "groups": len(set(data["group"])),
              "protocol": protocol, "source_hashes": {str(p.relative_to(ROOT.parent)): sha(p) for p in source_paths},
              "training_scenes_sha256": sha(args.training_scenes), "capsule_sha256": sha(args.capsule),
              "exclusion_sha256": {str(p): sha(p) for p in exclusions}, "excluded_group_overlap": len(set(data["group"]) & forbidden),
              "dataset_sha256": sha(args.output / "pairs.npz"), "variants": variants, "signal_candidates": candidates,
              "episodes_results": episodes, "elapsed_seconds": time.perf_counter() - started,
              "limitations": ["Exploratory TRAIN-only action/horizon screen, not independent confirmation or closed-loop gains.",
                              "All actions chosen from public geometry at fixed snapshot times; private states are diagnostic labels only.",
                              "Common-prefix populations vary with termination. Full-horizon endpoint/support reported separately.",
                              "Original final command repeated after 8 steps; no planner horizon or target rules changed."]}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "states": len(windows), "signal_candidates": candidates}), flush=True)


if __name__ == "__main__":
    main()
