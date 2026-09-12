"""Evaluate action-conditioned prediction checkpoints on frozen S4 scenes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.distributed_dn_mpc import DistributedDNMPCConfig  # noqa: E402
from encirclement3d.minimax_mpc import MinimaxMPCConfig  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.safety_qp import RobustCBFQPConfig  # noqa: E402
from encirclement3d.showcase import scenario_from_metadata  # noqa: E402
from evaluate_minimax_mpc import (  # noqa: E402
    DEFAULT_ENVIRONMENT_CONFIG,
    DEFAULT_MPC_CONFIG,
    add_safety_source_hashes,
    load_yaml,
    model_from_checkpoint,
    require_summary_writer,
    run_episode,
    select_device,
    source_hashes,
)
from evaluate_s4_branching import config_for_spec, grouped_summary, load_protocol  # noqa: E402


METHODS = (
    "dynamic_encirclement",
    "pure_pursuit",
    "expected",
    "worst_case",
    "cvar",
    "distributed_ideal",
    "distributed_delayed",
    "distributed_dropout",
    "distributed_none",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROJECT_ROOT / "configs" / "phase15_s4_branching_pilot.yaml")
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_ENVIRONMENT_CONFIG)
    parser.add_argument("--mpc-config", type=Path, default=DEFAULT_MPC_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--official-s4-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-source", choices=("checkpoint", "belief"), default="checkpoint")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["dynamic_encirclement", "worst_case"])
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--sampling-steps", type=int, default=4)
    parser.add_argument("--sampling-seed", type=int, default=745102)
    parser.add_argument("--projection-iterations", type=int, default=4)
    parser.add_argument("--prediction-refresh-interval-steps", type=int, default=20)
    queue_group = parser.add_mutually_exclusive_group()
    queue_group.add_argument("--queue-aware-rollout", dest="queue_aware_rollout", action="store_true")
    queue_group.add_argument("--no-queue-aware-rollout", dest="queue_aware_rollout", action="store_false")
    parser.set_defaults(queue_aware_rollout=None)
    adaptive_group = parser.add_mutually_exclusive_group()
    adaptive_group.add_argument("--adaptive-k", dest="adaptive_k", action="store_true")
    adaptive_group.add_argument("--no-adaptive-k", dest="adaptive_k", action="store_false")
    parser.set_defaults(adaptive_k=None)
    rnic_group = parser.add_mutually_exclusive_group()
    rnic_group.add_argument("--rnic", dest="rnic", action="store_true")
    rnic_group.add_argument("--no-rnic", dest="rnic", action="store_false")
    parser.set_defaults(rnic=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--safety-layer", choices=("none", "local_cbf", "robust_cbf_qp"), default="local_cbf")
    parser.add_argument("--safety-config", type=Path, default=PROJECT_ROOT / "configs" / "innovation_safety.yaml")
    parser.add_argument(
        "--decision",
        choices=("validation_selection", "locked_test_diagnostic", "ood_diagnostic"),
        default="locked_test_diagnostic",
        help="Explicit result-split label stored in the run metadata and summary.",
    )
    return parser.parse_args()


def read_scenes(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    records.sort(key=lambda value: int(value["episode_index"]))
    if limit is not None:
        if limit <= 0:
            raise ValueError("episodes must be positive when supplied.")
        records = records[:limit]
    if not records:
        raise ValueError("The frozen scene file contains no records.")
    required = {"episode_index", "episode_seed", "scenario", "target_speed_scale", "defender_bias", "pursuit_overrides"}
    missing = required.difference(records[0])
    if missing:
        raise ValueError(f"Frozen scene record is missing: {', '.join(sorted(missing))}")
    return records


def protocol_for_frozen_scenes(protocol_path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Adapt only non-geometric protocol metadata to the frozen scene contract."""

    protocol = load_protocol(protocol_path)
    settings = dict(protocol["s4"])
    settings["target_speed_scales"] = sorted({float(record["target_speed_scale"]) for record in records})
    settings["defender_biases"] = sorted({str(record["defender_bias"]) for record in records})
    settings["observation_conditions"] = []
    seen: set[str] = set()
    for record in records:
        name = str(record.get("observation_condition", "nominal"))
        if name in seen:
            continue
        seen.add(name)
        settings["observation_conditions"].append(
            {
                "name": name,
                "pursuit_overrides": copy.deepcopy(record["pursuit_overrides"]),
            }
        )
    protocol["s4"] = settings
    return protocol


def source_hashes_closed_loop(protocol: Path, scenes: Path, mpc: Path) -> dict[str, str]:
    hashes = source_hashes(mpc)
    for path in (
        PROJECT_ROOT / "scripts" / "evaluate_s4_closed_loop.py",
        PROJECT_ROOT / "scripts" / "evaluate_s4_branching.py",
        protocol.resolve(),
        scenes.resolve(),
    ):
        hashes[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def apply_phase17_execution_mapping(
    config: dict[str, Any],
    phase17_execution_mapping: dict[str, Any],
    *,
    frozen_scene_record: dict[str, Any],
) -> None:
    """Apply defaults without overwriting a frozen scene execution contract."""

    if not phase17_execution_mapping or "execution_overrides" in frozen_scene_record:
        return
    config.setdefault("dynamics", {}).setdefault("execution", {}).update(
        copy.deepcopy(phase17_execution_mapping)
    )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    if args.candidate_source == "checkpoint" and args.checkpoint is None:
        raise ValueError("--checkpoint is required when --candidate-source=checkpoint.")
    if args.prediction_refresh_interval_steps <= 0 or args.num_samples <= 0 or args.sampling_steps <= 0:
        raise ValueError("prediction sampling and refresh settings must be positive.")

    records = read_scenes(args.scenes, args.episodes)
    protocol = protocol_for_frozen_scenes(args.protocol.resolve(), records)
    mpc_document = load_yaml(args.mpc_config)
    phase17_mapping = dict(mpc_document.get("phase17", {}))
    rnic = bool(
        phase17_mapping.get("reachability_normalized_cost", False)
        if args.rnic is None
        else args.rnic
    )
    planner_mapping = dict(mpc_document.get("planner", {}))
    planner_mapping["reachability_normalized_cost_enabled"] = rnic
    planner_config = MinimaxMPCConfig.from_mapping(planner_mapping)
    queue_aware_rollout = bool(
        phase17_mapping.get("queue_aware_rollout", False)
        if args.queue_aware_rollout is None
        else args.queue_aware_rollout
    )
    adaptive_k = bool(
        phase17_mapping.get("adaptive_k", False)
        if args.adaptive_k is None
        else args.adaptive_k
    )
    adaptive_budget_mapping = dict(mpc_document.get("prediction", {}).get("adaptive_budget", {}))
    if adaptive_k and not adaptive_budget_mapping:
        raise ValueError("adaptive_k requires prediction.adaptive_budget configuration")
    phase17_execution_mapping = dict(phase17_mapping.get("execution", {}))
    distributed_mapping = dict(mpc_document.get("distributed", {}))
    device = select_device(args.device)
    checkpoint_data = (
        model_from_checkpoint(args.checkpoint, device, args.official_s4_root)
        if args.checkpoint is not None
        else None
    )
    if checkpoint_data is not None:
        checkpoint_config = checkpoint_data[3].get("model_config", {})
        if int(checkpoint_config.get("horizon_count", 0)) < planner_config.horizon_steps:
            raise ValueError("Prediction checkpoint horizon is shorter than planner horizon.")

    safety_config = None
    if args.safety_layer == "robust_cbf_qp":
        safety_document = load_yaml(args.safety_config)
        safety_mapping = dict(safety_document.get("safety", {}))
        probe_spec = records[0]
        probe_config = config_for_spec(args.environment_config, protocol, probe_spec, args.max_steps)
        probe_env = CaptureRadiusPursuit3DEnv(
            probe_config,
            obstacle_count=1,
            target_speed_scale=float(probe_spec["target_speed_scale"]),
        )
        safety_mapping.setdefault("max_speed_mps", float(probe_env.agents["defender_max_speed"]))
        safety_mapping.setdefault("max_acceleration_mps2", float(probe_env.agents["defender_max_acceleration"]))
        safety_mapping.setdefault("safety_margin_m", float(probe_env.pursuit["safety_margin"]))
        safety_config = RobustCBFQPConfig.from_mapping(safety_mapping)

    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes_closed_loop(args.protocol, args.scenes, args.mpc_config)
    if args.safety_layer == "robust_cbf_qp":
        add_safety_source_hashes(hashes, args.safety_config)
    run_config = {
        "scenes": str(args.scenes.resolve()),
        "protocol": str(args.protocol.resolve()),
        "environment_config": str(args.environment_config.resolve()),
        "mpc_config": str(args.mpc_config.resolve()),
        "checkpoint": None if args.checkpoint is None else str(args.checkpoint.resolve()),
        "official_s4_root": None if args.official_s4_root is None else str(args.official_s4_root.resolve()),
        "candidate_source": args.candidate_source,
        "methods": list(args.methods),
        "episodes": len(records),
        "max_steps": args.max_steps,
        "num_samples": args.num_samples,
        "sampling_steps": args.sampling_steps,
        "sampling_seed": args.sampling_seed,
        "projection_iterations": args.projection_iterations,
        "prediction_refresh_interval_steps": args.prediction_refresh_interval_steps,
        "queue_aware_rollout": queue_aware_rollout,
        "adaptive_k": adaptive_k,
        "adaptive_budget": adaptive_budget_mapping,
        "reachability_normalized_cost": rnic,
        "phase17": phase17_mapping,
        "device": str(device),
        "safety_layer": args.safety_layer,
        "decision": args.decision,
        "source_hashes": hashes,
    }
    output.joinpath("config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    output.joinpath("scenes.jsonl").write_text(
        "".join(json.dumps(record, allow_nan=True) + "\n" for record in records), encoding="utf-8"
    )

    all_summaries: dict[str, Any] = {}
    distributed_modes = {
        "distributed_ideal": "ideal",
        "distributed_delayed": "delayed",
        "distributed_dropout": "dropout",
        "distributed_none": "none",
    }
    for method in args.methods:
        method_output = output / method
        method_output.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        with method_output.joinpath("episodes.jsonl").open("w", encoding="utf-8") as episode_file, method_output.joinpath("steps.jsonl").open("w", encoding="utf-8") as step_file:
            for record in records:
                spec = dict(record)
                config = config_for_spec(args.environment_config, protocol, spec, args.max_steps)
                apply_phase17_execution_mapping(
                    config,
                    phase17_execution_mapping,
                    frozen_scene_record=spec,
                )
                distributed_config = None
                if method in distributed_modes:
                    distributed_config = DistributedDNMPCConfig.from_mapping(
                        {**distributed_mapping, "communication_mode": distributed_modes[method]}
                    )
                row, episode_steps = run_episode(
                    config,
                    seed=int(spec["episode_seed"]),
                    method=method,
                    planner_config=planner_config,
                    candidate_source=args.candidate_source,
                    checkpoint_data=checkpoint_data,
                    device=device,
                    num_samples=args.num_samples,
                    sampling_steps=args.sampling_steps,
                    sampling_seed=args.sampling_seed + int(spec["episode_index"]) * 1000,
                    projection_iterations=args.projection_iterations,
                    use_local_cbf=args.safety_layer == "local_cbf",
                    safety_layer=args.safety_layer,
                    robust_safety_config=safety_config,
                    prediction_refresh_interval_steps=args.prediction_refresh_interval_steps,
                    queue_aware_rollout=queue_aware_rollout,
                    adaptive_prediction_config=(adaptive_budget_mapping if adaptive_k else None),
                    distributed_config=distributed_config,
                    scenario=scenario_from_metadata(spec["scenario"]),
                    validate_scenario=False,
                )
                row.update(
                    {
                        "episode_index": int(spec["episode_index"]),
                        "target_speed_scale": float(spec["target_speed_scale"]),
                        "defender_bias": str(spec["defender_bias"]),
                        "observation_condition": str(spec.get("observation_condition", "unknown")),
                        "rollout_policy": str(spec.get("rollout_policy", "unknown")),
                        "target_branch_sign_label": spec.get("target_branch_sign"),
                    }
                )
                rows.append(row)
                episode_file.write(json.dumps(row, allow_nan=True) + "\n")
                for step in episode_steps:
                    step_record = {"episode_index": int(spec["episode_index"]), **step}
                    steps.append(step_record)
                    step_file.write(json.dumps(step_record, allow_nan=True) + "\n")
        summary = grouped_summary(rows, steps)
        method_output.joinpath("summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
        all_summaries[method] = summary
        overall = summary["overall"]
        with require_summary_writer()(log_dir=str(method_output / "tensorboard"), flush_secs=5) as writer:
            writer.add_text("Evaluation/config", yaml.safe_dump(run_config, sort_keys=False), 0)
            writer.add_text("Evaluation/source_hashes", json.dumps(run_config["source_hashes"], indent=2), 0)
            for episode_index, row in enumerate(rows):
                for key in (
                    "safe_capture_success",
                    "capture_event",
                    "collision",
                    "boundary_violation",
                    "timeout",
                    "mean_capture_time_seconds",
                    "mean_planner_latency_ms",
                    "mean_predictor_latency_ms",
                    "prediction_refresh_rate",
                    "mean_prediction_age_steps",
                    "mean_total_control_latency_ms",
                    "queue_aware_rollout_rate",
                    "mean_qdr_queue_length",
                    "mean_qdr_first_controllable_step",
                    "qdr_prefix_minimum_clearance_m",
                    "qdr_prefix_minimum_boundary_margin_m",
                    "qdr_prefix_minimum_inter_agent_distance_m",
                    "qdr_prefix_maximum_safety_margin_violation_m",
                    "qdr_endpoint_position_error_mean_m",
                    "qdr_endpoint_position_error_max_m",
                    "qdr_endpoint_velocity_error_mean_mps",
                    "qdr_endpoint_velocity_error_max_mps",
                    "qdr_endpoint_check_completed",
                    "adaptive_enabled_rate",
                    "mean_adaptive_uncertainty_score",
                    "mean_adaptive_k",
                    "mean_adaptive_refresh_interval_steps",
                    "adaptive_forced_refresh_rate",
                    "mean_adaptive_cache_age_steps",
                    "mean_adaptive_prediction_residual_m",
                    "rnic_enabled_rate",
                    "mean_rnic_latency_ms",
                    "rnic_minimum_best_slack_s",
                    "rnic_mean_best_slack_s",
                    "rnic_maximum_best_slack_s",
                    "rnic_unreachable_slot_ratio",
                    "rnic_margin_violation_ratio",
                    "rnic_earliest_feasible_intercept_step",
                    "rnic_mean_arrival_time_s",
                    "rnic_maximum_arrival_time_s",
                ):
                    value = row.get(key)
                    if value is not None and np.isfinite(float(value)):
                        writer.add_scalar(f"Episode/{key}", float(value), episode_index)
            for key, value in overall.items():
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    writer.add_scalar(f"Summary/{key}", float(value), 0)
            for prefix, key in (
                ("PlannerLatency", "planner_latency_ms"),
                ("PredictorLatency", "predictor_latency_ms"),
                ("SafetyLatency", "safety_latency_ms"),
                ("TotalControlLatency", "total_control_latency_ms"),
            ):
                latency = overall[key]
                writer.add_scalar(f"Summary/{prefix}/p50_ms", latency["p50"], 0)
                writer.add_scalar(f"Summary/{prefix}/p95_ms", latency["p95"], 0)
                writer.add_scalar(f"Summary/{prefix}/p99_ms", latency["p99"], 0)
            writer.add_scalar("Summary/QDR/latency_p50_ms", overall["qdr_latency_ms"]["p50"], 0)
            writer.add_scalar("Summary/QDR/latency_p95_ms", overall["qdr_latency_ms"]["p95"], 0)
            writer.add_scalar("Summary/QDR/latency_p99_ms", overall["qdr_latency_ms"]["p99"], 0)
            writer.add_scalar("Summary/QDR/prefix_minimum_clearance_m", overall["qdr_prefix_minimum_clearance_m"], 0)
            writer.add_scalar("Summary/QDR/prefix_minimum_boundary_margin_m", overall["qdr_prefix_minimum_boundary_margin_m"], 0)
            writer.add_scalar("Summary/QDR/prefix_violation_rate", overall["qdr_prefix_violation_rate"], 0)
            writer.add_scalar(
                "Summary/QDR/endpoint_position_error_mean_m",
                overall["qdr_endpoint_position_error_mean_m"],
                0,
            )
            writer.add_scalar(
                "Summary/QDR/endpoint_position_error_max_m",
                overall["qdr_endpoint_position_error_max_m"],
                0,
            )
            writer.add_scalar(
                "Summary/QDR/endpoint_velocity_error_mean_mps",
                overall["qdr_endpoint_velocity_error_mean_mps"],
                0,
            )
            writer.add_scalar(
                "Summary/QDR/endpoint_check_coverage",
                overall["qdr_endpoint_check_coverage"],
                0,
            )
            writer.add_text("Summary/UAKR/BucketCounts", json.dumps(overall.get("adaptive_bucket_counts", {})), 0)
            writer.add_scalar("Summary/RNIC/enabled_rate", overall["rnic_enabled_rate"], 0)
            writer.add_scalar("Summary/RNIC/latency_p50_ms", overall["rnic_latency_ms"]["p50"], 0)
            writer.add_scalar("Summary/RNIC/latency_p95_ms", overall["rnic_latency_ms"]["p95"], 0)
            writer.add_scalar("Summary/RNIC/latency_p99_ms", overall["rnic_latency_ms"]["p99"], 0)
            writer.add_scalar("Summary/RNIC/minimum_best_slack_s", overall["rnic_minimum_best_slack_s"], 0)
            writer.add_scalar("Summary/RNIC/mean_best_slack_s", overall["rnic_mean_best_slack_s"], 0)
            writer.add_scalar("Summary/RNIC/unreachable_slot_ratio", overall["rnic_unreachable_slot_ratio"], 0)
            writer.add_scalar("Summary/RNIC/earliest_feasible_intercept_step", overall["rnic_earliest_feasible_intercept_step"], 0)
            writer.add_hparams(
                {
                    "method": method,
                    "queue_aware_rollout": int(queue_aware_rollout),
                    "adaptive_k": int(adaptive_k),
                    "reachability_normalized_cost": int(rnic),
                    "num_samples": args.num_samples,
                    "prediction_refresh_interval_steps": args.prediction_refresh_interval_steps,
                },
                {
                    "hparam/safe_capture_rate": float(overall["safe_capture_rate"]),
                    "hparam/total_latency_p95_ms": float(overall["total_control_latency_ms"]["p95"]),
                },
            )

    result = {"protocol": run_config, "methods": all_summaries, "decision": args.decision}
    output.joinpath("summary.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
