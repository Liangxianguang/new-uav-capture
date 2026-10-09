"""Collect same-state command interventions on TRAINING scenes only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from baseline import Baseline
from freeze_baseline import sha, verify

VARIANTS = {"mixed_three_evasive": 5, "dense_mixed": 6}
BRANCHES = ("baseline_plan", "left_pressure", "right_pressure", "half_speed")


def split_records(path: Path, per_variant: int, seed: int, forbidden_groups: set[str]):
    if per_variant < 6:
        raise ValueError("Need at least six independent groups per variant")
    by_variant = {variant: defaultdict(list) for variant in VARIANTS}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            variant = record["variant"]
            if variant in by_variant:
                group = str(record["mirror_group_id"])
                if group in forbidden_groups:
                    raise ValueError("Training scene overlaps frozen baseline validation")
                by_variant[variant][group].append(record)
    selected = []
    rng = np.random.default_rng(seed)
    for variant, groups in by_variant.items():
        keys = sorted(groups)
        if len(keys) < per_variant:
            raise ValueError(f"Insufficient groups in {variant}")
        keys = [keys[i] for i in rng.permutation(len(keys))[:per_variant]]
        test_count = max(1, per_variant // 6)
        calibration_count = max(1, per_variant // 6)
        train_count = per_variant - test_count - calibration_count
        for index, key in enumerate(keys):
            split = "train" if index < train_count else ("calibration" if index < train_count + calibration_count else "validation")
            for record in groups[key]:
                selected.append({**record, "level": VARIANTS[variant], "model_split": split})
    return selected


def interventions(sequence: np.ndarray, maximum_speed: float = 5.0) -> np.ndarray:
    original = np.asarray(sequence, dtype=np.float64)
    values = [original.copy()]
    for sign in (-1, 1):
        changed = original.copy()
        changed[:, :, 1] += sign * 1.5
        norm = np.linalg.norm(changed, axis=-1, keepdims=True)
        changed *= np.minimum(1.0, maximum_speed / np.maximum(norm, 1e-9))
        values.append(changed)
    values.append(original * 0.5)
    return np.stack(values)


def roll_branch(base: Baseline, env, sequence: np.ndarray, noise_key: int):
    clone = copy.deepcopy(env)
    safety = base.filter_class(clone)
    observations = clone.observe()
    targets, commanded, executed, masks, terminal = [], [], [], [], []
    ended = False
    for step, proposed in enumerate(sequence):
        if ended:
            targets.append(np.zeros(3))
            commanded.append(np.zeros_like(proposed))
            executed.append(np.zeros_like(proposed))
            masks.append(False)
            terminal.append("after_terminal")
            continue
        # Per-step independent streams prevent visibility-dependent consumption
        # at one step from shifting the target's next-step sensor noise.
        clone.rng = np.random.default_rng(np.random.SeedSequence([noise_key, step, 0]))
        clone.execution_rng = np.random.default_rng(np.random.SeedSequence([noise_key, step, 1]))
        command, _ = safety.filter(proposed, observations)
        observations, _, done, truncated, info = base.env_class.step(clone, command)
        target_valid = not any(info.get(key, False) for key in
                               ["target_invalid_episode", "target_boundary_violation", "target_obstacle_violation"])
        targets.append(clone.target_position.copy())  # Labels only.
        commanded.append(np.asarray(command).copy())
        executed.append(clone.last_executed_actions.copy())
        masks.append(target_valid)
        terminal.append(str(info.get("termination_reason", "running")))
        ended = bool(done or truncated or not target_valid)
    return {"target": np.stack(targets), "commanded": np.stack(commanded), "executed": np.stack(executed),
            "mask": np.asarray(masks), "termination": terminal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--training-scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--groups-per-variant", type=int, default=12)
    parser.add_argument("--seed", type=int, default=910910)
    args = parser.parse_args()
    if args.training_scenes.parent.name != "train":
        raise ValueError("Only the original train split may be collected")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = Baseline(args.capsule, args.output / "restored")
    forbidden = {str(r["mirror_group_id"]) for r in base.records()}
    records = split_records(args.training_scenes, args.groups_per_variant, args.seed, forbidden)
    (args.output / "selected_scenes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    buckets = defaultdict(list)
    state_metadata, episode_metadata = [], []
    started = time.perf_counter()
    for record in records:
        history = []

        def observer(env, observation, safe_action, sequence):
            history.append(base.evaluator.policy_observations(env, observation).reshape(-1).copy())
            if env.step_count not in (4, 12, 20) or sequence is None:
                return
            reference, reference_velocity = base.evaluator.belief_reference(observation, base.planner_config)
            relative = np.concatenate([np.asarray(observation["defender_positions"]) - reference,
                                       np.asarray(observation["defender_velocities"]) - reference_velocity], axis=-1)
            commands = interventions(sequence[:8])
            key_bytes = f"{args.seed}/{record['episode_index']}/{env.step_count}".encode()
            noise_key = int.from_bytes(hashlib.sha256(key_bytes).digest()[:4], "little")
            # Clone interventions must not consume parent RNG/state.
            before_rng = json.dumps(env.rng.bit_generator.state, sort_keys=True)
            before_execution_rng = json.dumps(env.execution_rng.bit_generator.state, sort_keys=True)
            before_positions = env.defender_positions.copy()
            before_target = env.target_position.copy()
            branches = [roll_branch(base, env, command, noise_key) for command in commands]
            repeated = roll_branch(base, env, commands[0], noise_key)
            if not np.array_equal(repeated["target"], branches[0]["target"]):
                raise AssertionError("Repeated intervention is not deterministic")
            if before_rng != json.dumps(env.rng.bit_generator.state, sort_keys=True) or before_execution_rng != json.dumps(env.execution_rng.bit_generator.state, sort_keys=True):
                raise AssertionError("Collection mutated parent RNG")
            if not np.array_equal(before_positions, env.defender_positions) or not np.array_equal(before_target, env.target_position):
                raise AssertionError("Collection mutated parent physical state")
            padded = [history[0]] * max(0, 8 - len(history)) + history[-8:]
            targets = np.stack([branch["target"] for branch in branches])
            valid = np.stack([branch["mask"] for branch in branches])
            displacement = targets - reference[None, None, :]
            displacement[~valid] = 0
            buckets["history"].append(np.stack(padded).astype(np.float32))
            buckets["relative"].append(relative.astype(np.float32))
            buckets["proposed"].append(commands.astype(np.float32))
            buckets["commanded"].append(np.stack([b["commanded"] for b in branches]).astype(np.float32))
            buckets["executed"].append(np.stack([b["executed"] for b in branches]).astype(np.float32))
            buckets["labels"].append(displacement.astype(np.float32))
            buckets["mask"].append(valid)
            buckets["reference_velocity"].append(reference_velocity.astype(np.float32))
            buckets["split"].append(record["model_split"])
            buckets["group"].append(str(record["mirror_group_id"]))
            state_metadata.append({"episode_index": record["episode_index"], "step": int(env.step_count),
                                   "model_split": record["model_split"], "noise_key": noise_key,
                                   "parent_state_unchanged": True, "repeated_branch_bitwise_equal": True,
                                   "termination": [b["termination"] for b in branches]})

        row, _ = base.run(record, observer=observer)
        episode_metadata.append({"episode_index": record["episode_index"], "model_split": record["model_split"],
                                 "safe_capture_success": row["safe_capture_success"]})
        progress = {"episodes_completed": len(episode_metadata), "episodes_total": len(records),
                    "states": len(state_metadata), "elapsed_seconds": time.perf_counter() - started}
        (args.output / "progress.json").write_text(json.dumps(progress, indent=2))
        print(json.dumps(progress), flush=True)
    if not state_metadata:
        raise RuntimeError("No intervention windows collected")
    arrays = {key: np.stack(value) for key, value in buckets.items()}
    for name, value in arrays.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite dataset field {name}")
    splits = {split: set(arrays["group"][arrays["split"] == split]) for split in ("train", "calibration", "validation")}
    if any(splits[a] & splits[b] for a, b in [("train", "validation"), ("train", "calibration"), ("calibration", "validation")]):
        raise AssertionError("Mirror-group split leakage")
    if any(not len(groups) for groups in splits.values()):
        raise RuntimeError("Empty model split")
    np.savez_compressed(args.output / "pairs.npz", **arrays)
    common = arrays["mask"][:, 1:] & arrays["mask"][:, :1]
    effects = np.linalg.norm(arrays["labels"][:, 1:] - arrays["labels"][:, :1], axis=-1)
    manifest = {"status": "pilot_collection_complete", "capsule_sha256": sha(args.capsule),
                "source_training_scenes_sha256": sha(args.training_scenes), "dataset_sha256": sha(args.output / "pairs.npz"),
                "groups_per_variant": args.groups_per_variant, "seed": args.seed, "branches": BRANCHES,
                "horizon_steps": 8, "history_steps": 8, "input_contract": "public 252-D history, public relative states, proposed command sequences; no future executed/truth inputs",
                "states": len(state_metadata), "episodes": len(episode_metadata),
                "split_groups": {key: sorted(value) for key, value in splits.items()},
                "split_states": {key: int(np.sum(arrays["split"] == key)) for key in splits},
                "paired_response_mean_m": float(np.mean(effects[common])),
                "paired_response_p95_m": float(np.quantile(effects[common], 0.95)),
                "effect_over_0_1m_fraction": float(np.mean(effects[common] > 0.1)),
                "state_metadata": state_metadata, "episode_metadata": episode_metadata,
                "limitations": ["All three model splits come only from the original TRAINING pool; not a formal generalization claim.",
                                 "Step-keyed shared streams are a coupling for paired command interventions; source-level exogenous-noise identification is not proved.",
                                 "Open-loop command suffixes do not equal receding-horizon feedback policies.",
                                 "Terminated branches are masked; no simulation beyond terminal states."]}
    verify(base.root, base.capsule_manifest)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
