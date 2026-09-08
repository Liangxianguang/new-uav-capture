"""Evaluate centralized scenario MPC on the frozen S3 mixed-obstacle protocol.

All methods reuse the same independently generated S3 scene records.  The
planner receives only the environment's policy-safe observation and projected
prediction candidates; true target state is used only by the environment for
termination metrics.  Every method writes JSONL provenance and TensorBoard
scalars so hard-example analysis can be reproduced without rerunning scenes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.showcase import (  # noqa: E402
    random_central_mixed_obstacle_scenario,
    scenario_from_metadata,
    scenario_metadata,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from evaluate_minimax_mpc import (  # noqa: E402
    DEFAULT_ENVIRONMENT_CONFIG,
    DEFAULT_MPC_CONFIG,
    MinimaxMPCConfig,
    RobustCBFQPConfig,
    load_yaml,
    model_from_checkpoint,
    run_episode,
    select_device,
    source_hashes,
    summarize_rows,
)
from evaluate_minimax_mpc import add_safety_source_hashes, require_summary_writer  # noqa: E402
from encirclement3d.distributed_dn_mpc import DistributedDNMPCConfig  # noqa: E402


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "central_random_mixed_obstacle_s3_v5_protocol.yaml"
REQUIRED_SPLITS = ("train", "validation", "locked_test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--split", choices=REQUIRED_SPLITS, default="validation")
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_ENVIRONMENT_CONFIG)
    parser.add_argument("--mpc-config", type=Path, default=DEFAULT_MPC_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scene-records",
        type=Path,
        help="Reuse a previously generated scenes.jsonl after validating its locked specs.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--candidate-source", choices=("checkpoint", "belief"), default="checkpoint")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "dynamic_encirclement",
            "expected",
            "worst_case",
            "cvar",
            "distributed_ideal",
            "distributed_delayed",
            "distributed_dropout",
            "distributed_none",
        ),
    )
    parser.add_argument("--episodes", type=int, help="Smoke override; locked_test requires its configured count.")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--projection-iterations", type=int)
    parser.add_argument(
        "--prediction-refresh-interval-steps",
        type=int,
        help="Refresh learned prediction every N control steps; 1 preserves per-step sampling.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--safety-layer",
        choices=("local_cbf", "robust_cbf_qp"),
        default="local_cbf",
        help="Safety filter used after planning. robust_cbf_qp is velocity-level and conditional.",
    )
    parser.add_argument(
        "--safety-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "innovation_safety.yaml",
        help="Safety configuration used by --safety-layer robust_cbf_qp.",
    )
    parser.add_argument("--without-local-cbf", action="store_true")
    return parser.parse_args()


def load_protocol(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("S3 protocol YAML must be a mapping.")
    seed_blocks = document.get("seed_blocks")
    episodes_per_split = document.get("episodes_per_split")
    settings = document.get("s3")
    if not isinstance(seed_blocks, dict) or not isinstance(episodes_per_split, dict) or not isinstance(settings, dict):
        raise ValueError("S3 protocol requires seed_blocks, episodes_per_split, and s3 mappings.")
    missing = [name for name in REQUIRED_SPLITS if name not in seed_blocks or name not in episodes_per_split]
    if missing:
        raise ValueError(f"S3 protocol is missing split settings: {', '.join(missing)}")
    observations = settings.get("observation_conditions")
    if not isinstance(observations, list) or not observations:
        raise ValueError("S3 protocol must define observation_conditions.")
    return document


def episode_count(protocol: dict[str, Any], split: str, override: int | None) -> int:
    configured = int(protocol["episodes_per_split"][split])
    if split == "locked_test" and override is not None and int(override) != configured:
        raise ValueError(f"locked_test requires exactly {configured} episodes; got {override}")
    count = configured if override is None else int(override)
    if count <= 0:
        raise ValueError("episodes must be positive")
    return count


def episode_spec(protocol: dict[str, Any], split: str, episode_index: int) -> dict[str, Any]:
    settings = protocol["s3"]
    seed_block = int(protocol["seed_blocks"][split])
    episode_seed = seed_block + int(episode_index)
    layout_seed = seed_block + 1_000_000 + int(episode_index)
    minimum_count, maximum_count = (int(value) for value in settings["obstacle_count_range"])
    conditions = list(
        itertools.product(
            range(minimum_count, maximum_count + 1),
            settings["defender_sides"],
            settings["initial_side_distances"],
            settings["target_speed_scales"],
            settings["target_motion_modes"],
            settings["observation_conditions"],
        )
    )
    order = np.random.default_rng(seed_block + 2_000_000).permutation(len(conditions))
    obstacle_count, defender_side, initial_distance, speed_scale, motion_mode, observation = conditions[
        int(order[episode_index % len(order)])
    ]
    observation = dict(observation)
    return {
        "episode_seed": episode_seed,
        "layout_seed": layout_seed,
        "defender_side": str(defender_side),
        "initial_side_distance": float(initial_distance),
        "target_speed_scale": float(speed_scale),
        "target_motion_mode": str(motion_mode),
        "target_crossing_required": bool(settings.get("target_crossing_required", False)),
        "observation_condition": str(observation["name"]),
        "pursuit_overrides": copy.deepcopy(observation["pursuit_overrides"]),
        "obstacle_count": int(obstacle_count),
        "condition_index": int(order[episode_index % len(order)]),
        "condition_table_size": len(conditions),
    }


def config_for_spec(
    environment_config: Path,
    spec: dict[str, Any],
    max_steps: int | None,
) -> dict[str, Any]:
    config = load_yaml(environment_config)
    config = copy.deepcopy(config)
    pursuit = config.setdefault("task", {}).setdefault("pursuit", {})
    pursuit.update(copy.deepcopy(spec["pursuit_overrides"]))
    pursuit["target_motion_mode"] = str(spec["target_motion_mode"])
    config["experiments"] = [
        {
            "name": "phase3_s3_scenario_mpc",
            "episodes": 1,
            "obstacle_count": int(spec["obstacle_count"]),
            "target_speed_scale": float(spec["target_speed_scale"]),
        }
    ]
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max-steps must be positive")
        config["world"]["max_steps"] = int(max_steps)
    return config


def source_hashes_s3(protocol_path: Path, mpc_config_path: Path) -> dict[str, str]:
    hashes = source_hashes(mpc_config_path)
    tracked = (
        PROJECT_ROOT / "scripts" / "evaluate_minimax_mpc_s3.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "showcase.py",
        protocol_path.resolve(),
    )
    for path in tracked:
        key = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def grouped_summary(rows: list[dict[str, Any]], steps: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"overall": summarize_rows(rows, steps)}
    for field in ("observation_condition", "target_motion_mode", "defender_side", "obstacle_count"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        result[f"by_{field}"] = {}
        for key, subset in sorted(groups.items()):
            episode_ids = {int(row["episode_index"]) for row in subset}
            subset_steps = [step for step in steps if int(step["episode_index"]) in episode_ids]
            result[f"by_{field}"][key] = summarize_rows(subset, subset_steps)
    return result


def load_scene_records(
    path: Path,
    *,
    protocol: dict[str, Any],
    split: str,
    episodes: int,
) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != episodes:
        raise ValueError(f"Scene record count must be {episodes}; got {len(records)}")
    for episode_index, record in enumerate(records):
        if int(record.get("episode_index", -1)) != episode_index:
            raise ValueError("Scene records must have contiguous episode_index values.")
        expected = episode_spec(protocol, split, episode_index)
        if record.get("spec") != expected:
            raise ValueError(f"Scene record spec mismatch at episode {episode_index}.")
        if not isinstance(record.get("scenario"), dict):
            raise ValueError(f"Scene record scenario must be a mapping at episode {episode_index}.")
    return records


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    episodes = episode_count(protocol, args.split, args.episodes)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    mpc_document = load_yaml(args.mpc_config)
    planner_config = MinimaxMPCConfig.from_mapping(dict(mpc_document.get("planner", {})))
    distributed_mapping = dict(mpc_document.get("distributed", {}))
    prediction_config = dict(mpc_document.get("prediction", {}))
    methods = list(args.methods or mpc_document.get("evaluation", {}).get("methods", ["dynamic_encirclement", "expected", "worst_case", "cvar"]))
    if args.candidate_source == "checkpoint" and args.checkpoint is None:
        raise ValueError("--checkpoint is required when --candidate-source=checkpoint")
    device = select_device(args.device)
    checkpoint = args.checkpoint.resolve() if args.checkpoint is not None else None
    checkpoint_data = model_from_checkpoint(checkpoint, device) if checkpoint is not None else None
    if checkpoint_data is not None:
        horizon_count = int(checkpoint_data[3].get("model_config", {}).get("horizon_count", 0))
        if horizon_count < planner_config.horizon_steps:
            raise ValueError("Prediction checkpoint horizon is shorter than planner horizon")
    num_samples = int(args.num_samples if args.num_samples is not None else prediction_config.get("num_samples", 8))
    sampling_steps = int(args.sampling_steps if args.sampling_steps is not None else prediction_config.get("sampling_steps", 8))
    sampling_seed = int(args.sampling_seed if args.sampling_seed is not None else prediction_config.get("sampling_seed", 745102))
    projection_iterations = int(
        args.projection_iterations
        if args.projection_iterations is not None
        else prediction_config.get("projection_iterations", 4)
    )
    prediction_refresh_interval_steps = int(
        args.prediction_refresh_interval_steps
        if args.prediction_refresh_interval_steps is not None
        else prediction_config.get("refresh_interval_steps", 1)
    )
    if min(num_samples, sampling_steps, projection_iterations, prediction_refresh_interval_steps) <= 0:
        raise ValueError("Prediction sampling, projection and refresh settings must be positive")

    if args.safety_layer == "robust_cbf_qp" and args.without_local_cbf:
        raise ValueError("--without-local-cbf cannot be combined with --safety-layer robust_cbf_qp")
    safety_layer = args.safety_layer if not args.without_local_cbf else "none"
    robust_safety_config: RobustCBFQPConfig | None = None
    if safety_layer == "robust_cbf_qp":
        safety_document = load_yaml(args.safety_config)
        safety_mapping = dict(safety_document.get("safety", {}))
        safety_probe = CaptureRadiusPursuit3DEnv(
            copy.deepcopy(load_yaml(args.environment_config)),
            obstacle_count=0,
            target_speed_scale=float(protocol["s3"]["target_speed_scales"][0]),
        )
        safety_mapping.setdefault("max_speed_mps", float(safety_probe.agents["defender_max_speed"]))
        safety_mapping.setdefault("max_acceleration_mps2", float(safety_probe.agents["defender_max_acceleration"]))
        safety_mapping.setdefault("safety_margin_m", float(safety_probe.pursuit["safety_margin"]))
        robust_safety_config = RobustCBFQPConfig.from_mapping(safety_mapping)

    hashes = source_hashes_s3(protocol_path, args.mpc_config)
    if safety_layer == "robust_cbf_qp":
        add_safety_source_hashes(hashes, args.safety_config)
    if checkpoint is not None:
        hashes["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    run_config = {
        "protocol": str(protocol_path),
        "environment_config": str(args.environment_config.resolve()),
        "mpc_config": str(args.mpc_config.resolve()),
        "split": args.split,
        "episodes": episodes,
        "scene_records": None if args.scene_records is None else str(args.scene_records.resolve()),
        "methods": methods,
        "candidate_source": args.candidate_source,
        "checkpoint": None if checkpoint is None else str(checkpoint),
        "device": str(device),
        "planner": planner_config.__dict__,
        "distributed": distributed_mapping,
        "prediction": {
            "num_samples": num_samples,
            "sampling_steps": sampling_steps,
            "sampling_seed": sampling_seed,
            "projection_iterations": projection_iterations,
            "refresh_interval_steps": prediction_refresh_interval_steps,
        },
        "use_local_cbf": safety_layer == "local_cbf",
        "safety_layer": safety_layer,
        "safety_config": str(args.safety_config.resolve()) if safety_layer == "robust_cbf_qp" else None,
        "source_hashes": hashes,
    }
    output.joinpath("config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    output.joinpath("protocol.yaml").write_text(protocol_path.read_text(encoding="utf-8"), encoding="utf-8")

    if args.scene_records is not None:
        episode_records = load_scene_records(
            args.scene_records,
            protocol=protocol,
            split=args.split,
            episodes=episodes,
        )
    else:
        episode_records = []
        for episode_index in range(episodes):
            spec = episode_spec(protocol, args.split, episode_index)
            config = config_for_spec(args.environment_config, spec, args.max_steps)
            validation_env = CaptureRadiusPursuit3DEnv(
                config,
                obstacle_count=0,
                target_speed_scale=float(spec["target_speed_scale"]),
            )
            scenario = random_central_mixed_obstacle_scenario(
                validation_env,
                layout_seed=int(spec["layout_seed"]),
                initial_side_distance=float(spec["initial_side_distance"]),
                defender_side=str(spec["defender_side"]),
                target_crossing_required=bool(spec["target_crossing_required"]),
                obstacle_count_range=(int(spec["obstacle_count"]), int(spec["obstacle_count"])),
                max_attempts=int(protocol["s3"].get("max_sampling_attempts", 500)),
                required_defender_zone_entries=int(protocol["s3"].get("required_defender_zone_entries", 1)),
            )
            episode_records.append(
                {
                    "episode_index": episode_index,
                    "spec": spec,
                    "scenario": scenario_metadata(scenario),
                }
            )
    output.joinpath("scenes.jsonl").write_text(
        "".join(json.dumps(record, allow_nan=True) + "\n" for record in episode_records),
        encoding="utf-8",
    )

    all_summaries: dict[str, Any] = {}
    for method in methods:
        method_output = output / method
        method_output.mkdir(parents=True, exist_ok=True)
        method_run_config = {
            **run_config,
            "methods": [method],
            "method": method,
        }
        method_output.joinpath("config.yaml").write_text(
            yaml.safe_dump(method_run_config, sort_keys=False), encoding="utf-8"
        )
        rows: list[dict[str, Any]] = []
        all_steps: list[dict[str, Any]] = []
        with (
            method_output.joinpath("episodes.jsonl").open("w", encoding="utf-8") as episode_file,
            method_output.joinpath("steps.jsonl").open("w", encoding="utf-8") as step_file,
        ):
            for record in episode_records:
                episode_index = int(record["episode_index"])
                spec = record["spec"]
                config = config_for_spec(args.environment_config, spec, args.max_steps)
                scenario = scenario_from_metadata(record["scenario"])
                distributed_config = None
                distributed_modes = {
                    "distributed_ideal": "ideal",
                    "distributed_delayed": "delayed",
                    "distributed_dropout": "dropout",
                    "distributed_none": "none",
                }
                if method in distributed_modes:
                    distributed_config = DistributedDNMPCConfig.from_mapping(
                        {
                            **distributed_mapping,
                            "communication_mode": distributed_modes[method],
                        }
                    )
                row, steps = run_episode(
                    config,
                    seed=int(spec["episode_seed"]),
                    method=method,
                    planner_config=planner_config,
                    candidate_source=args.candidate_source,
                    checkpoint_data=checkpoint_data,
                    device=device,
                    num_samples=num_samples,
                    sampling_steps=sampling_steps,
                    sampling_seed=sampling_seed + episode_index * 1000,
                    projection_iterations=projection_iterations,
                    prediction_refresh_interval_steps=prediction_refresh_interval_steps,
                    use_local_cbf=safety_layer == "local_cbf",
                    safety_layer=safety_layer,
                    robust_safety_config=robust_safety_config,
                    distributed_config=distributed_config,
                    scenario=scenario,
                    validate_scenario=False,
                )
                row.update(
                    {
                        "episode_index": episode_index,
                        "episode_seed": int(spec["episode_seed"]),
                        "layout_seed": int(spec["layout_seed"]),
                        "observation_condition": str(spec["observation_condition"]),
                        "target_motion_mode": str(spec["target_motion_mode"]),
                        "defender_side": str(spec["defender_side"]),
                        "obstacle_count": int(spec["obstacle_count"]),
                        "target_speed_scale": float(spec["target_speed_scale"]),
                        "initial_side_distance_m": float(spec["initial_side_distance"]),
                        "layout_signature": "+".join(
                            f"{shape}{sum(item['shape'] == shape for item in record['scenario']['obstacles'])}"
                            for shape in ("cylinder", "box", "wall")
                            if any(item["shape"] == shape for item in record["scenario"]["obstacles"])
                        ),
                    }
                )
                rows.append(row)
                for step in steps:
                    enriched = {"episode_index": episode_index, "episode_seed": int(spec["episode_seed"]), **step}
                    all_steps.append(enriched)
                episode_file.write(json.dumps(row, allow_nan=True) + "\n")
                for step in steps:
                    step_file.write(
                        json.dumps(
                            {"episode_index": episode_index, "episode_seed": int(spec["episode_seed"]), **step},
                            allow_nan=True,
                        )
                        + "\n"
                    )
        summary = grouped_summary(rows, all_steps)
        method_output.joinpath("summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
        all_summaries[method] = summary
        with require_summary_writer()(log_dir=str(method_output / "tensorboard"), flush_secs=5) as writer:
            writer.add_text("Evaluation/config", yaml.safe_dump(run_config, sort_keys=False), 0)
            writer.add_text("Evaluation/source_hashes", json.dumps(hashes, indent=2), 0)
            writer.add_text("Evaluation/protocol", protocol_path.read_text(encoding="utf-8"), 0)
            for episode_index, row in enumerate(rows):
                for key in (
                    "safe_capture_success",
                    "capture_event",
                    "collision",
                    "timeout",
                    "min_clearance_m",
                    "mean_planner_latency_ms",
                    "mean_predictor_latency_ms",
                    "prediction_refresh_rate",
                    "mean_prediction_age_steps",
                    "max_prediction_age_steps",
                    "mean_total_control_latency_ms",
                    "safety_solver_success_rate",
                    "safety_certificate_valid_rate",
                    "safety_fallback_rate",
                    "safety_abort_required_rate",
                    "safety_precondition_valid_rate",
                    "safety_recovery_action_rate",
                    "safety_maximum_slack_m",
                    "safety_mean_active_constraint_count",
                    "safety_mean_constraint_count",
                    "minimum_safety_barrier_m",
                    "maximum_safety_constraint_violation_m",
                    "planner_fallback_count",
                    "planner_success_count",
                    "planner_valid_count",
                    "planner_converged_count",
                    "planner_local_solver_failures",
                    "messages_attempted",
                    "messages_sent",
                    "messages_received",
                    "messages_dropped",
                    "message_bytes_sent",
                    "max_message_age_steps",
                    "mean_distributed_iterations",
                    "mean_distributed_action_delta_mps",
                    "mean_candidate_worst_minimum_distance_m",
                    "mean_candidate_cvar_minimum_distance_m",
                ):
                    value = row.get(key)
                    if value is not None and np.isfinite(float(value)):
                        writer.add_scalar(f"Episode/{key}", float(value), episode_index)
            overall = summary["overall"]
            for key, value in overall.items():
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    writer.add_scalar(f"Summary/{key}", float(value), 0)
            writer.add_text(
                "Summary/SafetyFailureCategoryCounts",
                json.dumps(overall.get("safety_failure_category_counts", {}), sort_keys=True),
                0,
            )
            writer.add_text(
                "Summary/SafetyFallbackReasonCounts",
                json.dumps(overall.get("safety_fallback_reason_counts", {}), sort_keys=True),
                0,
            )
            for name, key in (
                ("PlannerLatency", "planner_latency_ms"),
                ("PredictorLatency", "predictor_latency_ms"),
                ("SafetyLatency", "safety_latency_ms"),
                ("TotalControlLatency", "total_control_latency_ms"),
            ):
                if key in overall:
                    for percentile in ("p50", "p95", "p99"):
                        writer.add_scalar(f"Summary/{name}/{percentile}_ms", overall[key][percentile], 0)
            writer.add_hparams(
                {
                     "method": method,
                     "distributed_method": int(method.startswith("distributed_")),
                    "split": args.split,
                    "horizon_steps": planner_config.horizon_steps,
                    "control_horizon_steps": planner_config.control_horizon_steps,
                    "num_samples": num_samples,
                    "sampling_steps": sampling_steps,
                    "use_local_cbf": int(safety_layer == "local_cbf"),
                    "safety_layer": safety_layer,
                },
                {
                    "hparam/safe_capture_rate": float(overall["safe_capture_rate"]),
                    "hparam/solver_success_rate": float(overall["solver_success_rate"]),
                    "hparam/worst_candidate_minimum_distance_m": float(overall["worst_candidate_minimum_distance_m"]),
                },
            )
    result = {
        "config": run_config,
        "methods": all_summaries,
        "decision": "diagnostic_only",
        "decision_reason": "P4 communication robustness is diagnostic until the formal S3 gate is reviewed.",
    }
    output.joinpath("summary.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
