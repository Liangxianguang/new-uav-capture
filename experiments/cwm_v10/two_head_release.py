"""Audit fresh geometry, original public replay, actual local data and retraining."""
import argparse
import copy
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cwm_v9"), str(ROOT.parent / "cwm_v7")]
from train_two_head import LocalTwoHead, normalization, evaluate, control_metrics, gate_and_selection
from collect_local import statistics
from replay_public import compare_public
from local_shadow import decode_public, PublicResponseProvider, model_loader
from local_shadow_release import frozen_configs, verify_call
from geometry_release import arrays, compare_arrays, target_bad, BASELINE_SHA, sha
from training_release import tensor_tree_equal
from freeze_baseline import REFERENCE
from verify_release import check_archive


def audit_sources(read, stage, report):
    for name, digest in report["source_hashes"].items():
        if hashlib.sha256(read(f"{stage}/source/{name}")).hexdigest() != digest:
            raise ValueError("Actual source digest mismatch")


def check_geometry(scene, protocol):
    group = scene["layout_seed"] - protocol["layout_seed_start"]
    member = 0 if scene["mirror_pair_member"] == "upper" else 1 if scene["mirror_pair_member"] == "lower" else -1
    if not 0 <= group < 32 or member < 0 or scene["episode_seed"] != protocol["seed_start"] + group * 2 + member or scene["episode_index"] != scene["episode_seed"]:
        raise ValueError("Fresh preassigned scene seed mismatch")
    if scene["model_split"] != protocol["variant_cycles_split"][group // 4] or scene["mirror_group_id"] != f"cwm-s4-{scene['layout_seed']}":
        raise ValueError("Actual scene split/group mismatch")
    conditions = [(s, c) for s in protocol["target_speed_scales"] for c in protocol["observation_conditions"]]
    speed, condition = conditions[group % 4]
    if scene["target_speed_scale"] != speed or scene["pursuit_overrides"] != condition["overrides"] or scene["target_motion_mode"] != "adaptive_branching":
        raise ValueError("Frozen target/sensor scenario contract mismatch")
    rng = np.random.default_rng(scene["layout_seed"])
    jitter = rng.uniform(-.16, .16, (4, 2))
    jitter[:, 0] = np.maximum(jitter[:, 0], -.08)
    names = ("wall_half_extent_x_m", "wall_half_extent_y_m", "wall_height_m", "target_initial_x_m", "target_initial_y_m", "target_altitude_m", "defender_x_offset_m", "defender_y_scale", "defender_z_offset_m")
    sampled = {key: float(rng.uniform(*protocol["scene_variation"][key])) for key in names}
    mirror = 1. if member == 0 else -1.
    defenders = np.array([[-8.20, 2.70, 3.50], [-8.55, 3.80, 6.50], [-8.60, -.80, 4.00], [-8.85, .50, 6.00]])
    defenders[:, 0] += sampled["defender_x_offset_m"] + jitter[:, 0]
    defenders[:, 1] = mirror * (defenders[:, 1] * sampled["defender_y_scale"] + jitter[:, 1])
    defenders[:, 2] += sampled["defender_z_offset_m"]
    target = [sampled["target_initial_x_m"], mirror * sampled["target_initial_y_m"], sampled["target_altitude_m"]]
    metadata = scene["scenario"]
    if not np.array_equal(defenders, metadata["defender_positions"]) or not np.array_equal(target, metadata["target_position"]) or len(metadata["obstacles"]) != 1:
        raise ValueError("Actual geometry differs from deterministic declared scene")
    wall = metadata["obstacles"][0]
    if wall["center_xy"] != [1.5, 0.] or wall["half_extents_xy"] != [sampled["wall_half_extent_x_m"], sampled["wall_half_extent_y_m"]] or wall["height"] != sampled["wall_height_m"] or wall["shape"] != "wall":
        raise ValueError("Actual wall differs from qualified distribution")


def encoder_environment(restored):
    from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv, pursuit_settings
    provenance = json.loads((restored / REFERENCE / "manifest.json").read_text())
    config = provenance["environment"]
    original = yaml.safe_load((restored / "configs/capture_radius_pursuit_central_v4_flee.yaml").read_text())
    env = CaptureRadiusPursuit3DEnv.__new__(CaptureRadiusPursuit3DEnv)
    env.world, env.agents, env.task = copy.deepcopy(config["world"]), copy.deepcopy(config["agents"]), copy.deepcopy(config["task"])
    env.pursuit = pursuit_settings(env.task)
    env.n_defenders = int(env.agents["defenders"])
    for key in ("max_observation_obstacles", "include_uncertainty_features", "include_prediction_features"):
        env.pursuit[key] = original["task"]["pursuit"][key]
    env.task.update(policy_obstacle_geometry=original["task"]["policy_obstacle_geometry"], policy_clearance_features=False, policy_role_slot_features=False, policy_route_intent_features=False)
    return env


def audit_data(read, source_read, configs, restored):
    from encirclement3d.minimax_mpc import belief_reference
    from encirclement3d.observation_encoding import policy_observations
    report = json.loads(read("data/summary.json"))
    protocol = report["protocol"]
    if report["capsule_sha256"] != BASELINE_SHA or report["enhanced_control_enabled"] or report["holdout_collected"] or protocol["target_branch_rule_overrides"]:
        raise ValueError("Frozen collection/holdout contract mismatch")
    for name in ("scenes", "calls"):
        extension = "jsonl" if name == "scenes" else "json"
        if hashlib.sha256(read(f"data/{name}.{extension}")).hexdigest() != report[name + "_sha256"]:
            raise ValueError("Actual dataset digest mismatch")
    audit_sources(read, "data", report)
    if json.loads(read("data/source/cwm_v10/data_protocol.json")) != protocol:
        raise ValueError("Actual collection protocol mismatch")
    scenes = [json.loads(line) for line in read("data/scenes.jsonl").decode().splitlines()]
    by_id = {r["episode_index"]: r for r in scenes}
    if len(scenes) != 64 or len(by_id) != 64 or len({r["mirror_group_id"] for r in scenes}) != 32:
        raise ValueError("Actual new-scene population mismatch")
    for scene in scenes:
        check_geometry(scene, protocol)
    public = json.loads(read("public/summary.json"))
    audit_sources(read, "public", public)
    if public["data_summary_sha256"] != hashlib.sha256(read("data/summary.json")).hexdigest() or public["capsule_sha256"] != BASELINE_SHA or public["enhanced_control_enabled"] or public["holdout_used"]:
        raise ValueError("Independent original replay contract mismatch")
    public_rows = {r["episode_index"]: r for r in public["episodes"]}
    if set(public_rows) != set(by_id) or len(public["episodes"]) != 64:
        raise ValueError("Independent original replay population mismatch")
    env = encoder_environment(restored)
    frames_by_id, expected_calls = {}, set()
    episodes = json.loads(read("data/episodes.json"))
    if episodes != report["episodes"] or len(episodes) != 64 or {r["episode_index"] for r in episodes} != set(by_id):
        raise ValueError("Actual original episode population mismatch")
    for row in episodes:
        index, scene = row["episode_index"], by_id[row["episode_index"]]
        a, b = arrays(read(f"data/observed/{index}.npz")), arrays(read(f"data/unobserved/{index}.npz"))
        if not compare_arrays(a, b) or not row["trajectory_byte_equal"] or not row["outcomes_equal"]:
            raise ValueError("Collection changed actual original arrays")
        for kind in ("observed", "unobserved"):
            if hashlib.sha256(read(f"data/{kind}/{index}.npz")).hexdigest() != row[kind + "_sha256"]:
                raise ValueError("Actual original trajectory digest mismatch")
        if bool(target_bad(a["target_positions"], scene["scenario"]).any()) != row["target_invalid_episode"]:
            raise ValueError("Actual original target physics mismatch")
        if not compare_arrays(a, arrays(read(f"public/trajectories/{index}.npz"))) or not public_rows[index]["original_arrays_byte_equal"]:
            raise ValueError("Independent original replay arrays differ")
        raw = read(f"public/frames/{index}.json")
        if hashlib.sha256(raw).hexdigest() != public_rows[index]["frames_sha256"]:
            raise ValueError("Actual public frame digest mismatch")
        frames = decode_public(json.loads(raw))
        if len(frames) != a["target_positions"].shape[0] - 1 or [r["step"] for r in frames] != list(range(len(frames))):
            raise ValueError("Incomplete independent public-frame trace")
        for frame in frames:
            feature = policy_observations(env, frame["observation"]).reshape(-1)
            if feature.size != 252 or feature.tobytes() != frame["feature"].tobytes():
                raise ValueError("Actual original 252 feature encoding mismatch")
            if not np.array_equal(frame["observation"]["defender_positions"], a["defender_positions"][frame["step"]]):
                raise ValueError("Public defender state differs from actual original trajectory")
        frames_by_id[index] = frames
        expected_calls.update((index, step, agent) for step in protocol["snapshot_steps"] if step < len(frames) for agent in range(4))
    records = json.loads(read("data/calls.json"))
    if len(records) != len(expected_calls) or {(r["episode_index"], r["step"], r["agent"]) for r in records} != expected_calls or public["checked_calls"] != len(records):
        raise ValueError("Actual local-call coverage mismatch")
    providers = {kind: PublicResponseProvider("shadow", model_loader(source_read(f"primary/{kind}_seed{seed}.pt"), kind, seed)) for kind, seed in protocol["selected_median_seeds"].items()}
    calls = []
    for row in records:
        scene = by_id[row["episode_index"]]
        if row["split"] != scene["model_split"] or row["group"] != scene["mirror_group_id"] or not row["parent_integrity"]:
            raise ValueError("Actual local-call split/provenance mismatch")
        raw_context = read("data/" + row["context_path"])
        if hashlib.sha256(raw_context).hexdigest() != row["context_sha256"]:
            raise ValueError("Actual local context digest mismatch")
        snapshot = json.loads(raw_context)
        context = decode_public(snapshot)
        frame = frames_by_id[row["episode_index"]][row["step"]]
        if any(not compare_public(v, context["observation"][k]) for k, v in frame["observation"].items()):
            raise ValueError("Actual local public context differs from original replay")
        if not np.array_equal(frame["backbone"], context["backbone"]):
            raise ValueError("Actual supplied GRU differs from original replay")
        prefix = frames_by_id[row["episode_index"]][:row["step"] + 1]
        history = np.stack([prefix[0]["feature"]] * max(0, 8 - len(prefix)) + [f["feature"] for f in prefix[-8:]])
        reference, velocity = belief_reference(context["observation"], configs[0])
        synthetic = {"history": history[None], "backbone": frame["backbone"][None], "reference": reference[None], "reference_velocity": velocity[None]}
        values = None
        if "arrays_path" in row:
            raw = read("data/" + row["arrays_path"])
            if hashlib.sha256(raw).hexdigest() != row["arrays_sha256"]:
                raise ValueError("Actual local training array digest mismatch")
            values = arrays(raw)
            calls.append({"record": row, "values": values})
        verify_call(row, values, snapshot, providers, configs, 0, synthetic, scene, protocol)
    stats = statistics(records, scenes, lambda name: read("data/" + name))
    gate = protocol["data_gate"]
    eligible = (sum(r["target_invalid_episode"] for r in episodes) <= gate["maximum_original_target_invalid_episodes"]
                and stats["train"]["groups"] >= gate["minimum_train_groups"] and stats["development_validation"]["groups"] >= gate["minimum_development_groups"]
                and all(r["common_valid_nonanchor_points"] >= gate["minimum_common_valid_nonanchor_points_each_split"] for r in stats.values()))
    if stats != report["statistics"] or report["training_eligible"] != eligible or not eligible:
        raise ValueError("Actual collection statistics/data gate mismatch")
    return calls, report


def audit(read, source_read, configs, restored):
    calls, data_report = audit_data(read, source_read, configs, restored)
    indices = {split: [i for i, c in enumerate(calls) if c["record"]["split"] == split] for split in ("train", "development_validation")}
    mean, scale = normalization(calls, indices["train"])
    controls = control_metrics(calls, indices["development_validation"])
    summaries, sets = [], []
    for stage in ("primary", "retrained"):
        report = json.loads(read(stage + "/summary.json"))
        protocol = report["protocol"]
        audit_sources(read, stage, report)
        if json.loads(read(stage + "/source/training_protocol.json")) != protocol or report["baseline_weights_included"] or report["enhanced_control_enabled"] or report["holdout_used"]:
            raise ValueError("Frozen training/public model contract mismatch")
        if report["data_summary_sha256"] != hashlib.sha256(read("data/summary.json")).hexdigest() or report["data_calls_sha256"] != data_report["calls_sha256"] or report["controls"] != controls:
            raise ValueError("Actual training provenance/control mismatch")
        if report["split_calls"] != {k: len(v) for k, v in indices.items()} or report["split_groups"] != {k: sorted({calls[i]["record"]["group"] for i in v}) for k, v in indices.items()}:
            raise ValueError("Actual trained splits differ")
        stored = arrays(read(stage + "/normalization.npz"))
        if not np.array_equal(stored["mean"], mean) or not np.array_equal(stored["scale"], scale):
            raise ValueError("Actual train-only normalization differs")
        if len(report["models"]) != 9 or {(r["kind"], r["seed"]) for r in report["models"]} != {(k, s) for k in protocol["models"] for s in protocol["training"]["seeds"]}:
            raise ValueError("Actual nine-model population mismatch")
        checkpoints = {}
        for row in report["models"]:
            name = f"{row['kind']}_seed{row['seed']}"
            raw = read(f"{stage}/{name}.pt")
            if hashlib.sha256(raw).hexdigest() != row["checkpoint_sha256"]:
                raise ValueError("Actual checkpoint digest mismatch")
            checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            if checkpoint["online_promoted"] or checkpoint["baseline_weights_included"] or checkpoint["protocol"] != protocol or checkpoint["data_summary_sha256"] != report["data_summary_sha256"] or checkpoint["source_hashes"] != report["source_hashes"]:
                raise ValueError("Actual checkpoint contract mismatch")
            if not np.array_equal(checkpoint["normalizer_mean"].numpy(), mean) or not np.array_equal(checkpoint["normalizer_scale"].numpy(), scale):
                raise ValueError("Checkpoint normalizer mismatch")
            model = LocalTwoHead(row["kind"], protocol["training"]["motion_scale_m"], protocol["training"]["response_scale_m"])
            model.load_state_dict(checkpoint["model_state"], strict=True)
            measured, predictions = evaluate(model, calls, indices["development_validation"], mean, scale)
            expected = {f"call{j}_{key}": v for j, output in enumerate(predictions) for key, v in output.items()}
            if measured != row["development"] or not compare_arrays(expected, arrays(read(f"{stage}/{name}_development_predictions.npz"))):
                raise ValueError("Reloaded two-head predictions/metrics differ")
            if row["parameters"] != sum(p.numel() for p in model.parameters()) or row["trainable_parameters"] != sum(p.numel() for p in model.parameters() if p.requires_grad):
                raise ValueError("Actual parameter budget mismatch")
            history = json.loads(read(f"{stage}/{name}_history.json"))
            if [r["epoch"] for r in history] != list(range(1, 81)) or not np.isfinite([r["mean_batch_loss_motion_response_energy"] for r in history]).all():
                raise ValueError("Actual fixed training history mismatch")
            checkpoints[name] = checkpoint
        gate, chosen = gate_and_selection(report["models"], controls, protocol)
        if report["development_gate_passed"] != gate or report["chosen_median_seed_per_architecture"] != chosen:
            raise ValueError("Actual development gate/median selection mismatch")
        summaries.append(report)
        sets.append(checkpoints)
    for key in ("protocol", "source_hashes", "data_summary_sha256", "data_calls_sha256", "split_calls", "split_groups", "controls", "development_gate_passed", "chosen_median_seed_per_architecture"):
        if summaries[0][key] != summaries[1][key]:
            raise ValueError("Independent retraining summaries differ")
    for name, checkpoint in sets[0].items():
        if not tensor_tree_equal(checkpoint, sets[1][name]) or read(f"primary/{name}_history.json") != read(f"retrained/{name}_history.json"):
            raise ValueError("Independent weights/optimizer/RNG/history differ")
    return {"status": "passed", "data_episodes": 64, "data_groups": 32, "train_calls": len(indices["train"]), "development_calls": len(indices["development_validation"]),
            "original_public_replay_calls": 1280, "models_per_run": 9, "independent_training_runs": 2, "weights_optimizer_rng_equal": True,
            "reload_prediction_bytes_equal": True, "train_only_normalization_verified": True,
            "development_gate_passed": summaries[0]["development_gate_passed"], "enhanced_control_enabled": False, "holdout_used": False,
            "scope": "Fresh-group offline two-head training and exact original/public replay; not decision-value, closed-loop success, safety/latency or deployment qualification."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "public", "primary", "retrained"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "cwm_v7/artifacts/paired_training_20261010.zip")
    parser.add_argument("--capsule", type=Path, default=ROOT.parent / "cwm_v1/baseline/capsule.zip")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    protocol = json.loads((ROOT / "data_protocol.json").read_text())
    if sha(args.source) != protocol["source_archive_sha256"] or sha(args.capsule) != BASELINE_SHA:
        raise ValueError("Protected baseline/source mismatch")
    restored = ROOT.parent.parent / "results/cwm_v10/audit_restored"
    configs = frozen_configs(args.capsule, restored)
    check_archive(args.source, "ARTIFACT_MANIFEST.json", False)
    with zipfile.ZipFile(args.source) as source:
        if args.verify:
            integrity = check_archive(args.verify, "ARTIFACT_MANIFEST.json", False)
            with zipfile.ZipFile(args.verify) as archive:
                result = audit(archive.read, source.read, configs, restored)
            print(json.dumps({**result, "artifact_sha256": integrity["sha256"]}))
            return
        members = {}
        for stage in ("data", "public", "primary", "retrained"):
            run = getattr(args, stage)
            if run is None:
                parser.error("Collection, independent public replay and two training runs required")
            for file in run.rglob("*"):
                relative = file.relative_to(run)
                if file.is_file() and "restored" not in relative.parts:
                    members[f"{stage}/{relative.as_posix()}"] = file.read_bytes()
        for file in ROOT.glob("*.py"):
            members["verification_source/" + file.name] = file.read_bytes()
        result = audit(members.__getitem__, source.read, configs, restored)
        path = ROOT / "artifacts/two_head_training_20261010.zip"
        path.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, raw in members.items():
                archive.writestr(name, raw)
            archive.writestr("ARTIFACT_MANIFEST.json", json.dumps({n: hashlib.sha256(v).hexdigest() for n, v in members.items()}, indent=2))
        integrity = check_archive(path, "ARTIFACT_MANIFEST.json", False)
        with zipfile.ZipFile(path) as archive:
            audit(archive.read, source.read, configs, restored)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    for stage, run in (("data", args.data), ("training", args.primary), ("public", args.public)):
        with (reports / (stage + "_summary.json")).open("xb") as file:
            file.write((run / "summary.json").read_bytes())
    with (reports / "release_manifest.json").open("x") as file:
        json.dump({**result, "artifact_sha256": integrity["sha256"], "artifact_bytes": path.stat().st_size}, file, indent=2)
    print(json.dumps({**result, "artifact_sha256": integrity["sha256"], "artifact_bytes": path.stat().st_size}))


if __name__ == "__main__":
    main()
