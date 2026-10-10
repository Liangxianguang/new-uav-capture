"""Fresh group-preassigned local-context data, leaving original control intact."""
import argparse
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
sys.path[:0] = [str(ROOT.parent / "cwm_v9"), str(ROOT.parent / "cwm_v7")]
from local_shadow import ActualLocalTap, PublicResponseProvider, model_loader, inspect_call, public_call_snapshot, fingerprint
from geometry import translated_records
from geometry_release import arrays, compare_arrays, target_bad, BASELINE_SHA, sha
from freeze_baseline import verify
from verify_release import check_archive


def statistics(records, scenes, read):
    result = {}
    for split in ("train", "development_validation"):
        selected = [r for r in records if r["split"] == split and "arrays_path" in r]
        groups = {r["group"] for r in selected}
        points, effect, anchor_points = 0, [], 0
        for row in selected:
            values = arrays(read(row["arrays_path"]))
            nonanchor = ~(values["proposed"] == values["anchor"][None]).all((1, 2, 3))
            common = values["valid"] & values["anchor_valid"][None] & nonanchor[:, None]
            points += int(common.sum())
            anchor_points += int(values["anchor_valid"].sum())
            effect.extend(np.linalg.norm(values["target"] - values["anchor_target"][None], axis=-1)[common].tolist())
        result[split] = {"calls": len(selected), "groups": len(groups), "common_valid_nonanchor_points": points,
                         "anchor_valid_points": anchor_points, "paired_effect_mean_m": float(np.mean(effect)) if effect else None,
                         "paired_effect_over_0_05m_fraction": float(np.mean(np.asarray(effect) > .05)) if effect else None}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--v9", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "data_protocol.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if sha(args.capsule) != BASELINE_SHA or any(sha(path) != protocol[key] for path, key in ((args.source, "source_archive_sha256"), (args.geometry, "geometry_archive_sha256"), (args.v9, "v9_archive_sha256"))):
        raise ValueError("Protected baseline/qualification source mismatch")
    if protocol["enhanced_control_enabled"] or protocol["private_labels_model_inputs"] or protocol["holdout_collected"]:
        raise ValueError("Disabled-control/public-input contract mismatch")
    for path in (args.source, args.geometry, args.v9):
        check_archive(path, "ARTIFACT_MANIFEST.json", False)
    with zipfile.ZipFile(args.geometry) as archive:
        qualification = json.loads(archive.read("qualification/summary.json"))
        if qualification["selected_wall_center_x_m"] != protocol["wall_center_x_m"]:
            raise ValueError("Previously qualified geometry differs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(args.source) as archive:
        old_scenes = [json.loads(line) for line in archive.read("data/selected_scenes.jsonl").decode().splitlines()]
        training = json.loads(archive.read("primary/summary.json"))
        if training["chosen_median_seed_per_architecture"] != protocol["selected_median_seeds"]:
            raise ValueError("Candidate-generation models changed")
        providers = {kind: PublicResponseProvider("shadow", model_loader(archive.read(f"primary/{kind}_seed{seed}.pt"), kind, seed)) for kind, seed in protocol["selected_median_seeds"].items()}
        rng_before = torch.get_rng_state().clone()
        if not all(provider.load() for provider in providers.values()) or not torch.equal(rng_before, torch.get_rng_state()):
            raise ValueError("Frozen candidate provider unavailable or changed RNG")
    base = ActualLocalTap(args.capsule, args.output / "restored", protocol["snapshot_steps"])
    scenes = translated_records(base, protocol, protocol["wall_center_x_m"])
    forbidden = {r["layout_seed"] for r in old_scenes} | set(range(1966010, 1966018)) | set(range(1961010, 1964018))
    if any(r["layout_seed"] in forbidden for r in scenes):
        raise ValueError("Fresh data reuses prior development or holdout")
    for index, scene in enumerate(scenes):
        scene["model_split"] = protocol["variant_cycles_split"][(index // 2) // 4]
        scene["variant"] += "_wallx1.5"
    (args.output / "scenes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in scenes))
    source_paths = [Path(__file__), args.protocol.resolve()] + [ROOT.parent / p for p in (
        "cwm_v9/local_shadow.py", "cwm_v9/public_provider.py", "cwm_v7/collect_training.py", "cwm_v7/geometry.py", "cwm_v7/geometry_release.py",
        "cwm_v7/response_model.py", "cwm_v6/s4_collect.py", "cwm_v2/scout.py", "cwm_v4/diagnose.py", "cwm_v5/decision_value.py",
        "cwm_v8/s4_value.py", "cwm_v1/baseline.py", "cwm_v1/freeze_baseline.py", "cwm_v1/verify_release.py")]
    source_hashes = {p.relative_to(ROOT.parent).as_posix(): sha(p) for p in source_paths}
    for path in source_paths:
        destination = args.output / "source" / path.relative_to(ROOT.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
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
                row, values = inspect_call(base, env, call, public_history, providers, protocol, key)
                row.update(episode_index=scene["episode_index"], group=scene["mirror_group_id"], variant=scene["variant"], split=scene["model_split"], step=int(env.step_count))
                path = args.output / "calls" / f"{scene['episode_index']}_{env.step_count}_{call['agent']}.npz"
                path.parent.mkdir(exist_ok=True)
                if values is not None:
                    np.savez_compressed(path, **values)
                    row.update(arrays_path=path.relative_to(args.output).as_posix(), arrays_sha256=sha(path))
                context = path.with_suffix(".json")
                context.write_text(json.dumps(public_call_snapshot(call), indent=2))
                row.update(context_path=context.relative_to(args.output).as_posix(), context_sha256=sha(context))
                if planner_before != hashlib.sha256(pickle.dumps(call["planner"].__dict__, protocol=5)).hexdigest():
                    raise AssertionError("Collection changed captured planner")
                records.append(row)
            if fingerprint(env) != before:
                raise AssertionError("Collection changed original parent")
            (args.output / "calls.json").write_text(json.dumps(records, indent=2))
        path = args.output / "observed" / f"{scene['episode_index']}.npz"
        original_path = args.output / "unobserved" / path.name
        row, _ = base.run(scene, path, observer)
        plain, _ = base.run(scene, original_path)
        if not compare_arrays(arrays(path.read_bytes()), arrays(original_path.read_bytes())):
            raise AssertionError("Local collection changed original trajectory fields")
        outcomes = ("safe_capture_success", "capture_event", "collision", "boundary_violation", "timeout", "target_invalid_episode", "termination_reason")
        if any(row[k] != plain[k] for k in outcomes):
            raise AssertionError("Local collection changed original outcomes")
        actual_bad = bool(target_bad(arrays(path.read_bytes())["target_positions"], scene["scenario"]).any())
        if actual_bad != row["target_invalid_episode"]:
            raise AssertionError("Target physics differs from episode flag")
        episodes.append({"episode_index": scene["episode_index"], "split": scene["model_split"], "trajectory_byte_equal": True, "outcomes_equal": True,
                         "observed_sha256": sha(path), "unobserved_sha256": sha(original_path), **{k: row[k] for k in outcomes}})
        (args.output / "episodes.json").write_text(json.dumps(episodes, indent=2))
        print(json.dumps({"episodes": len(episodes), "split": scene["model_split"], "calls": len(records)}), flush=True)
    stats = statistics(records, scenes, lambda name: (args.output / name).read_bytes())
    gate = protocol["data_gate"]
    eligible = (sum(r["target_invalid_episode"] for r in episodes) <= gate["maximum_original_target_invalid_episodes"]
                and stats["train"]["groups"] >= gate["minimum_train_groups"] and stats["development_validation"]["groups"] >= gate["minimum_development_groups"]
                and all(s["common_valid_nonanchor_points"] >= gate["minimum_common_valid_nonanchor_points_each_split"] for s in stats.values()))
    verify(base.root, base.capsule_manifest)
    if any(sha(ROOT.parent / name) != digest for name, digest in source_hashes.items()):
        raise AssertionError("Sources changed during collection")
    report = {"status": "collected_eligible_for_offline_training" if eligible else "collected_not_eligible_keep_baseline", "training_eligible": eligible,
              "protocol": protocol, "source_hashes": source_hashes, "capsule_sha256": sha(args.capsule), "source_archive_sha256": sha(args.source),
              "scenes_sha256": sha(args.output / "scenes.jsonl"), "calls_sha256": sha(args.output / "calls.json"), "statistics": stats,
              "episodes": episodes, "enhanced_control_enabled": False, "holdout_collected": False, "elapsed_seconds": time.perf_counter() - started}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["status"], "statistics": stats}), flush=True)


if __name__ == "__main__":
    main()
