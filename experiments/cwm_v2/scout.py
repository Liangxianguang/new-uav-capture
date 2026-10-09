"""Scout fresh TRAIN groups for learnable response without changing baseline control."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "cwm_v1"))
from baseline import Baseline
from collect_pairs import roll_branch
from freeze_baseline import sha, verify
from diagnose_interactions import summarized


def commands(sequence, observation, reference, maximum_speed=5.):
    sequence = np.asarray(sequence[:8], dtype=np.float64)
    if sequence.shape != (8, 4, 3):
        raise ValueError("Frozen horizon/agent contract mismatch")
    radial = reference - observation["defender_positions"]
    radial /= np.maximum(np.linalg.norm(radial, axis=-1, keepdims=True), 1e-9)
    tangent = np.stack([-radial[:, 1], radial[:, 0], np.zeros(len(radial))], axis=-1)
    result = np.stack([sequence, sequence - 2 * tangent[None], sequence + 2 * tangent[None], np.zeros_like(sequence)])
    result *= np.minimum(1., maximum_speed / np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-9))
    return result


class BackboneTap(Baseline):
    """Copy already-computed public GRU scenarios; no second predictor call."""
    def __init__(self, *args):
        super().__init__(*args)
        parent = self.evaluator.DistributedMinimaxDNMPC
        adapter = self
        self.last_backbone = None

        class TappedPlanner(parent):
            def plan(self, observation, scenarios, **kwargs):
                adapter.last_backbone = np.asarray(scenarios.trajectories, dtype=np.float64)[:, :8].copy()
                return super().plan(observation, scenarios, **kwargs)

        self.evaluator.DistributedMinimaxDNMPC = TappedPlanner


def select_scenes(path, protocol, forbidden):
    if path.parent.name != "train":
        raise ValueError("Original train partition required")
    pools = {key: defaultdict(list) for key in protocol["variants"]}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row["variant"] in pools:
            group = str(row["mirror_group_id"])
            if group not in forbidden:
                pools[row["variant"]][group].append(row)
    rng = np.random.default_rng(protocol["selection_seed"])
    selected = []
    for variant, groups in pools.items():
        keys = sorted(groups)
        count = protocol["groups_per_variant"]
        if len(keys) < count:
            raise ValueError("Insufficient unused training groups")
        for key in rng.permutation(keys)[:count]:
            row = sorted(groups[str(key)], key=lambda r: r["episode_index"])[0]
            selected.append({**row, "level": protocol["variants"][variant], "model_split": "scout_train_only"})
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--training-scenes", type=Path, required=True)
    parser.add_argument("--exclude-scenes", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=Path(__file__).with_name("scout_protocol.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    base = BackboneTap(args.capsule, args.output / "restored")
    validation = base.root / "results/phase91_zero_cbf_curriculum_v5/validation/scenes.jsonl"
    forbidden = {str(json.loads(line)["mirror_group_id"]) for path in (validation, args.exclude_scenes)
                 for line in path.read_text().splitlines() if line}
    scenes = select_scenes(args.training_scenes, protocol, forbidden)
    scene_text = "".join(json.dumps(r) + "\n" for r in scenes)
    (args.output / "selected_scenes.jsonl").write_text(scene_text)
    metadata, episode_rows = [], []
    fields = defaultdict(list)
    started = time.perf_counter()
    source_hashes = {p.name: sha(p) for p in (Path(__file__), args.protocol,
                     Path(__file__).parent.parent / "cwm_v1/baseline.py",
                     Path(__file__).parent.parent / "cwm_v1/collect_pairs.py")}
    for record in scenes:
        history = []
        base.last_backbone = None

        def observer(env, observation, actions, sequence):
            history.append(base.evaluator.policy_observations(env, observation).reshape(-1).copy())
            if env.step_count not in protocol["snapshot_steps"] or sequence is None:
                return
            if base.last_backbone is None or base.last_backbone.shape != (1, 8, 3):
                raise ValueError("Frozen K=1 backbone path missing")
            reference, velocity = base.evaluator.belief_reference(observation, base.planner_config)
            proposed = commands(sequence, observation, reference)
            key = int.from_bytes(hashlib.sha256(f"{protocol['selection_seed']}/{record['episode_index']}/{env.step_count}".encode()).digest()[:4], "little")
            before = (env.defender_positions.tobytes(), env.target_position.tobytes(),
                      json.dumps(env.rng.bit_generator.state, sort_keys=True),
                      json.dumps(env.execution_rng.bit_generator.state, sort_keys=True))
            branches = [roll_branch(base, env, value, key) for value in proposed]
            repeated = roll_branch(base, env, proposed[0], key)
            if any(repeated[k].tobytes() != branches[0][k].tobytes() for k in ("target", "commanded", "executed", "mask")):
                raise AssertionError("Branch determinism mismatch")
            after = (env.defender_positions.tobytes(), env.target_position.tobytes(),
                     json.dumps(env.rng.bit_generator.state, sort_keys=True),
                     json.dumps(env.execution_rng.bit_generator.state, sort_keys=True))
            if before != after:
                raise AssertionError("Parent state or RNG changed")
            padded = [history[0]] * max(0, 8 - len(history)) + history[-8:]
            target = np.stack([b["target"] for b in branches])
            mask = np.stack([b["mask"] for b in branches])
            labels = target - reference[None, None]
            labels[~mask] = 0
            relative = np.concatenate([observation["defender_positions"] - reference,
                                       observation["defender_velocities"] - velocity], axis=-1)
            for name, value in {"history": np.stack(padded), "relative": relative, "proposed": proposed,
                                "commanded": np.stack([b["commanded"] for b in branches]),
                                "executed": np.stack([b["executed"] for b in branches]),
                                "labels": labels, "backbone": base.last_backbone[0] - reference,
                                "reference_velocity": velocity}.items():
                fields[name].append(np.asarray(value, dtype=np.float32))
            fields["mask"].append(mask)
            fields["group"].append(str(record["mirror_group_id"]))
            fields["variant"].append(record["variant"])
            fields["step"].append(int(env.step_count))
            row = {"episode_index": record["episode_index"], "variant": record["variant"],
                   "group": str(record["mirror_group_id"]), "step": int(env.step_count),
                   "public_nearest_distance_m": float(np.linalg.norm(relative[:, :3], axis=-1).min()),
                   "private_target_mode_diagnostic_only": str(env.target_maneuver_mode),
                   "private_target_route_diagnostic_only": str(env.target_maneuver_route),
                   "steps_since_target_replan_diagnostic_only": int(env.step_count - env.target_maneuver_last_replan_step),
                   "parent_integrity_pass": True, "repeat_pass": True,
                   "termination": [b["termination"] for b in branches]}
            metadata.append(row)
            # Save collected evidence per window; a later failure must not erase it.
            np.savez_compressed(args.output / "pairs_partial.npz", **{k: np.stack(v) for k, v in fields.items()})
            (args.output / "windows.json").write_text(json.dumps(metadata, indent=2))
            print(json.dumps({"variant": record["variant"], "step": env.step_count,
                              "distance_m": row["public_nearest_distance_m"], "states": len(metadata)}), flush=True)

        row, _ = base.run(record, observer=observer)
        episode_rows.append({"variant": record["variant"], "episode_index": record["episode_index"],
                             **{k: row[k] for k in ("safe_capture_success", "collision", "boundary_violation", "timeout")}})
    verify(base.root, base.capsule_manifest)
    if not metadata:
        raise ValueError("No scout states")
    data = {key: np.stack(value) for key, value in fields.items()}
    np.savez_compressed(args.output / "pairs.npz", **data)
    report = {"status": "scout_complete_not_online_promoted", "states": len(metadata), "episodes": len(scenes),
              "dataset_sha256": sha(args.output / "pairs.npz"), "capsule_sha256": sha(args.capsule),
              "training_scenes_sha256": sha(args.training_scenes), "excluded_scenes_sha256": sha(args.exclude_scenes),
              "protocol": protocol, "source_hashes": source_hashes, "variants": {},
              "elapsed_seconds": time.perf_counter() - started, "episodes_results": episode_rows,
              "original_validation_and_v1_group_overlap": 0,
              "limitations": ["Training-pool diagnostic only; few groups, no generalization or causal-identification claim.",
                               "Terminated branches masked; report common-prefix and full-horizon support separately.",
                               "Backbone is the unmodified already-computed original GRU path; truth is labels only."]}
    for variant in protocol["variants"]:
        chosen = data["variant"] == variant
        common = data["mask"][chosen, 1:] & data["mask"][chosen, :1]
        effects = np.linalg.norm(data["labels"][chosen, 1:] - data["labels"][chosen, :1], axis=-1)
        valid = effects[common]
        full = common.all(axis=-1)
        signal_groups = set()
        for g, e, m in zip(data["group"][chosen], effects, common):
            if np.any(e[m] > .05):
                signal_groups.add(str(g))
        report["variants"][variant] = {"states": int(chosen.sum()), "effect_m": summarized(valid),
                                      "fraction_over_0_05m": float(np.mean(valid > .05)) if valid.size else None,
                                      "signal_groups": sorted(signal_groups),
                                      "full_horizon_pairs": int(full.sum()),
                                      "paired_effect_points": int(common.sum()),
                                      "backbone_ade_m": float(np.linalg.norm(data["backbone"][chosen, None] - data["labels"][chosen], axis=-1)[data["mask"][chosen]].mean())}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "variants": report["variants"]}))


if __name__ == "__main__":
    main()
