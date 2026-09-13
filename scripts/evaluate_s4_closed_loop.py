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
from encirclement3d.delay_aware_conformal_tube import DelayAwareConformalReachableTube  # noqa: E402
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
    parser.add_argument(
        "--reachable-tube-calibration",
        type=Path,
        help="Optional frozen delay-aware conformal tube JSON artifact.",
    )
    parser.add_argument(
        "--tube-budget-weight",
        type=float,
        help="Optional UAKR weight for the frozen public tube-width feature.",
    )
    parser.add_argument(
        "--tube-radius-scale-m",
        type=float,
        help="Optional UAKR normalization scale for the tube-width feature in metres.",
    )
    parser.add_argument("--prediction-refresh-interval-steps", type=int, default=20)
    queue_group = parser.add_mutually_exclusive_group()
    queue_group.add_argument("--queue-aware-rollout", dest="queue_aware_rollout", action="store_true")
    queue_group.add_argument("--no-queue-aware-rollout", dest="queue_aware_rollout", action="store_false")
    parser.set_defaults(queue_aware_rollout=None)
    queue_safety_group = parser.add_mutually_exclusive_group()
    queue_safety_group.add_argument(
        "--queue-aware-safety-projection",
        dest="queue_aware_safety_projection",
        action="store_true",
        help="Apply local CBF at the first controllable delayed state when QDR is enabled.",
    )
    queue_safety_group.add_argument(
        "--no-queue-aware-safety-projection",
        dest="queue_aware_safety_projection",
        action="store_false",
        help="Keep local CBF anchored at the current state.",
    )
    parser.set_defaults(queue_aware_safety_projection=None)
    parser.add_argument(
        "--qdr-prefix-recovery-authority",
        choices=("immutable", "replace_nonexecuting", "flush_pending"),
        help="When a public-geometry QDR prefix is unsafe, request the configured emergency-brake authority.",
    )
    parser.add_argument(
        "--qdr-execution-tube-multiplier",
        type=float,
        help="Enable the validation-frozen empirical QDR execution tube with this multiplier.",
    )
    adaptive_group = parser.add_mutually_exclusive_group()
    adaptive_group.add_argument("--adaptive-k", dest="adaptive_k", action="store_true")
    adaptive_group.add_argument("--no-adaptive-k", dest="adaptive_k", action="store_false")
    parser.set_defaults(adaptive_k=None)
    parser.add_argument(
        "--adaptive-low-threshold",
        type=float,
        help="Optional validation-only override for the UAKR low/medium threshold.",
    )
    parser.add_argument(
        "--adaptive-high-threshold",
        type=float,
        help="Optional validation-only override for the UAKR medium/high threshold.",
    )
    parser.add_argument(
        "--adaptive-risk-calibration",
        type=Path,
        help="Frozen validation-only UAKR risk-calibration artifact supplying policy thresholds.",
    )
    parser.add_argument(
        "--adaptive-residual-high-trigger-m",
        type=float,
        help="Optional public prediction-residual threshold that forces high-K replanning.",
    )
    parser.add_argument(
        "--adaptive-residual-refresh-trigger-m",
        type=float,
        help="Optional public prediction-residual threshold that forces refresh without changing K.",
    )
    rnic_group = parser.add_mutually_exclusive_group()
    rnic_group.add_argument("--rnic", dest="rnic", action="store_true")
    rnic_group.add_argument("--no-rnic", dest="rnic", action="store_false")
    parser.set_defaults(rnic=None)
    parser.add_argument(
        "--rnic-activation-slack-s",
        type=float,
        help="Optional severe-unreachability threshold for gated RNIC; default preserves the legacy RNIC penalty.",
    )
    parser.add_argument(
        "--rnic-cost-mode",
        choices=("interceptor", "formation_slot"),
        help="Validation-only RNIC cost mode override.",
    )
    parser.add_argument(
        "--rnic-slot-radius-m",
        type=float,
        help="Validation-only cooperative formation-slot radius override in metres.",
    )
    escape_gap_group = parser.add_mutually_exclusive_group()
    escape_gap_group.add_argument(
        "--escape-gap",
        dest="escape_gap",
        action="store_true",
        help="Enable the deterministic escape-gap cooperative MPC term.",
    )
    escape_gap_group.add_argument(
        "--no-escape-gap",
        dest="escape_gap",
        action="store_false",
        help="Disable the escape-gap cooperative MPC term.",
    )
    parser.set_defaults(escape_gap=None)
    parser.add_argument(
        "--escape-gap-weight",
        type=float,
        help="Validation-only escape-gap cost weight override.",
    )
    fc_dbf_group = parser.add_mutually_exclusive_group()
    fc_dbf_group.add_argument(
        "--fc-dbf",
        dest="fc_dbf",
        action="store_true",
        help="Enable the finite feasible-consensus formation gate.",
    )
    fc_dbf_group.add_argument(
        "--no-fc-dbf",
        dest="fc_dbf",
        action="store_false",
        help="Disable the finite feasible-consensus formation gate.",
    )
    parser.set_defaults(fc_dbf=None)
    parser.add_argument(
        "--fc-dbf-cost-weight",
        type=float,
        help="Validation-only FC-DBF cost weight override.",
    )
    parser.add_argument(
        "--fc-dbf-slot-tolerance-m",
        type=float,
        help="Validation-only FC-DBF slot tracking tolerance override in metres.",
    )
    belief_fusion_group = parser.add_mutually_exclusive_group()
    belief_fusion_group.add_argument(
        "--belief-fusion",
        dest="belief_fusion",
        action="store_true",
        help="Enable freshness-covariance public-belief fusion.",
    )
    belief_fusion_group.add_argument(
        "--no-belief-fusion",
        dest="belief_fusion",
        action="store_false",
        help="Use the legacy confidence/age weighted public-belief reference.",
    )
    parser.set_defaults(belief_fusion=None)
    parser.add_argument("--belief-fusion-age-decay", type=float)
    parser.add_argument("--belief-fusion-age-inflation-m2", type=float)
    parser.add_argument("--belief-fusion-dropout-inflation-m2", type=float)
    parser.add_argument("--belief-fusion-covariance-floor-m2", type=float)
    parser.add_argument("--belief-fusion-confidence-power", type=float)
    parser.add_argument("--belief-fusion-min-effective-samples", type=float)
    parser.add_argument(
        "--execution-delay-steps",
        type=int,
        help="Validation-only override for dynamics.execution.action_delay_steps.",
    )
    parser.add_argument(
        "--execution-noise-std-mps",
        type=float,
        help="Validation-only override for dynamics.execution.command_noise_std.",
    )
    parser.add_argument(
        "--execution-noise-bound-sigma",
        type=float,
        help="Validation-only override for dynamics.execution.command_noise_bound_sigma.",
    )
    parser.add_argument(
        "--execution-tracking-time-constant-s",
        type=float,
        help="Validation-only override for dynamics.execution.velocity_time_constant_seconds.",
    )
    parser.add_argument(
        "--execution-drag-coefficient",
        type=float,
        help="Validation-only override for dynamics.execution.drag_coefficient.",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        help="Optional fixed Torch intra-op CPU thread count for reproducible runtime benchmarks.",
    )
    parser.add_argument(
        "--torch-num-interop-threads",
        type=int,
        help="Optional fixed Torch inter-op CPU thread count for reproducible runtime benchmarks.",
    )
    parser.add_argument("--safety-layer", choices=("none", "local_cbf", "robust_cbf_qp"), default="local_cbf")
    parser.add_argument("--safety-config", type=Path, default=PROJECT_ROOT / "configs" / "innovation_safety.yaml")
    parser.add_argument(
        "--decision",
        choices=("validation_selection", "validation_confirmation", "locked_test_diagnostic", "ood_diagnostic"),
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


def apply_execution_cli_overrides(
    execution_mapping: dict[str, Any],
    *,
    delay_steps: int | None = None,
    noise_std_mps: float | None = None,
    noise_bound_sigma: float | None = None,
    tracking_time_constant_s: float | None = None,
    drag_coefficient: float | None = None,
) -> dict[str, Any]:
    """Apply validated command-line execution overrides to an effective mapping."""

    overrides: dict[str, Any] = {
        "action_delay_steps": delay_steps,
        "command_noise_std": noise_std_mps,
        "command_noise_bound_sigma": noise_bound_sigma,
        "velocity_time_constant_seconds": tracking_time_constant_s,
        "drag_coefficient": drag_coefficient,
    }
    effective = dict(execution_mapping)
    for key, value in overrides.items():
        if value is None:
            continue
        numeric_value = float(value)
        if not np.isfinite(numeric_value) or numeric_value < 0.0:
            raise ValueError(f"execution override {key} must be finite and non-negative")
        if key == "action_delay_steps":
            if int(value) != numeric_value:
                raise ValueError("execution delay override must be an integer")
            effective[key] = int(value)
        else:
            effective[key] = numeric_value
    if any(value is not None for value in overrides.values()):
        effective["enabled"] = True
    return effective


def configure_torch_threads(
    num_threads: int | None,
    num_interop_threads: int | None,
) -> dict[str, int | None]:
    """Apply optional fixed Torch thread counts and return the observed settings."""

    for name, value in (
        ("torch-num-threads", num_threads),
        ("torch-num-interop-threads", num_interop_threads),
    ):
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if num_interop_threads is not None:
        torch.set_num_interop_threads(int(num_interop_threads))
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))
    return {
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
    }


def main() -> None:
    args = parse_args()
    torch_thread_settings = configure_torch_threads(
        args.torch_num_threads,
        args.torch_num_interop_threads,
    )
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
    if args.rnic_activation_slack_s is not None:
        if not np.isfinite(float(args.rnic_activation_slack_s)):
            raise ValueError("rnic-activation-slack-s must be finite")
        planner_mapping["reachability_activation_slack_s"] = float(args.rnic_activation_slack_s)
    if args.rnic_cost_mode is not None:
        planner_mapping["reachability_cost_mode"] = str(args.rnic_cost_mode)
    if args.rnic_slot_radius_m is not None:
        if not np.isfinite(float(args.rnic_slot_radius_m)) or float(args.rnic_slot_radius_m) <= 0.0:
            raise ValueError("rnic-slot-radius-m must be finite and positive")
        planner_mapping["reachability_slot_radius_m"] = float(args.rnic_slot_radius_m)
    if args.escape_gap is not None:
        planner_mapping["escape_gap_cost_enabled"] = bool(args.escape_gap)
    if args.escape_gap_weight is not None:
        if not np.isfinite(float(args.escape_gap_weight)) or float(args.escape_gap_weight) < 0.0:
            raise ValueError("escape-gap-weight must be finite and non-negative")
        planner_mapping["weight_escape_gap"] = float(args.escape_gap_weight)
    if args.fc_dbf is not None:
        planner_mapping["fc_dbf_enabled"] = bool(args.fc_dbf)
    if args.fc_dbf_cost_weight is not None:
        if not np.isfinite(float(args.fc_dbf_cost_weight)) or float(args.fc_dbf_cost_weight) < 0.0:
            raise ValueError("fc-dbf-cost-weight must be finite and non-negative")
        planner_mapping["fc_dbf_cost_weight"] = float(args.fc_dbf_cost_weight)
    if args.fc_dbf_slot_tolerance_m is not None:
        if not np.isfinite(float(args.fc_dbf_slot_tolerance_m)) or float(args.fc_dbf_slot_tolerance_m) <= 0.0:
            raise ValueError("fc-dbf-slot-tolerance-m must be finite and positive")
        planner_mapping["fc_dbf_slot_tolerance_m"] = float(args.fc_dbf_slot_tolerance_m)
    if args.belief_fusion is not None:
        planner_mapping["belief_fusion_mode"] = (
            "freshness_covariance" if args.belief_fusion else "legacy"
        )
    fusion_overrides = {
        "belief_fusion_age_decay": args.belief_fusion_age_decay,
        "belief_fusion_age_inflation_m2": args.belief_fusion_age_inflation_m2,
        "belief_fusion_dropout_inflation_m2": args.belief_fusion_dropout_inflation_m2,
        "belief_fusion_covariance_floor_m2": args.belief_fusion_covariance_floor_m2,
        "belief_fusion_confidence_power": args.belief_fusion_confidence_power,
        "belief_fusion_min_effective_samples": args.belief_fusion_min_effective_samples,
    }
    for key, value in fusion_overrides.items():
        if value is not None:
            if not np.isfinite(float(value)):
                raise ValueError(f"{key} must be finite")
            planner_mapping[key] = float(value)
    planner_config = MinimaxMPCConfig.from_mapping(planner_mapping)
    queue_aware_rollout = bool(
        phase17_mapping.get("queue_aware_rollout", False)
        if args.queue_aware_rollout is None
        else args.queue_aware_rollout
    )
    queue_aware_safety_projection = bool(
        phase17_mapping.get("queue_aware_safety_projection", False)
        if args.queue_aware_safety_projection is None
        else args.queue_aware_safety_projection
    )
    if queue_aware_safety_projection and not queue_aware_rollout:
        raise ValueError("queue-aware-safety-projection requires queue-aware-rollout")
    if queue_aware_safety_projection and args.safety_layer != "local_cbf":
        raise ValueError("queue-aware-safety-projection is only supported with --safety-layer local_cbf")
    qdr_prefix_recovery_authority = args.qdr_prefix_recovery_authority
    if qdr_prefix_recovery_authority is None:
        configured_authority = phase17_mapping.get("prefix_recovery_authority")
        qdr_prefix_recovery_authority = (
            None if configured_authority is None else str(configured_authority)
        )
    if qdr_prefix_recovery_authority is not None and not queue_aware_rollout:
        raise ValueError("qdr prefix recovery authority requires queue-aware-rollout")
    adaptive_k = bool(
        phase17_mapping.get("adaptive_k", False)
        if args.adaptive_k is None
        else args.adaptive_k
    )
    adaptive_budget_mapping = dict(mpc_document.get("prediction", {}).get("adaptive_budget", {}))
    if args.adaptive_risk_calibration is not None:
        if args.adaptive_low_threshold is not None or args.adaptive_high_threshold is not None:
            raise ValueError("adaptive-risk-calibration cannot be combined with explicit threshold overrides")
        calibration_path = args.adaptive_risk_calibration.resolve()
        calibration_document = json.loads(calibration_path.read_text(encoding="utf-8"))
        policy_thresholds = dict(calibration_document.get("policy_thresholds", {}))
        if not bool(policy_thresholds.get("valid_order", False)):
            raise ValueError("adaptive risk calibration artifact has invalid policy threshold order")
        adaptive_budget_mapping["low_threshold"] = float(policy_thresholds["low_threshold"])
        adaptive_budget_mapping["high_threshold"] = float(policy_thresholds["high_threshold"])
    if args.tube_budget_weight is not None:
        if not np.isfinite(float(args.tube_budget_weight)) or float(args.tube_budget_weight) < 0.0:
            raise ValueError("tube-budget-weight must be finite and non-negative")
        adaptive_budget_mapping["tube_weight"] = float(args.tube_budget_weight)
    if args.tube_radius_scale_m is not None:
        if not np.isfinite(float(args.tube_radius_scale_m)) or float(args.tube_radius_scale_m) <= 0.0:
            raise ValueError("tube-radius-scale-m must be finite and positive")
        adaptive_budget_mapping["tube_radius_scale_m"] = float(args.tube_radius_scale_m)
    if args.adaptive_low_threshold is not None:
        if not np.isfinite(float(args.adaptive_low_threshold)):
            raise ValueError("adaptive-low-threshold must be finite")
        adaptive_budget_mapping["low_threshold"] = float(args.adaptive_low_threshold)
    if args.adaptive_high_threshold is not None:
        if not np.isfinite(float(args.adaptive_high_threshold)):
            raise ValueError("adaptive-high-threshold must be finite")
        adaptive_budget_mapping["high_threshold"] = float(args.adaptive_high_threshold)
    if args.adaptive_residual_high_trigger_m is not None:
        if not np.isfinite(float(args.adaptive_residual_high_trigger_m)) or float(args.adaptive_residual_high_trigger_m) <= 0.0:
            raise ValueError("adaptive-residual-high-trigger-m must be finite and positive")
        adaptive_budget_mapping["residual_high_trigger_m"] = float(args.adaptive_residual_high_trigger_m)
    if args.adaptive_residual_refresh_trigger_m is not None:
        if not np.isfinite(float(args.adaptive_residual_refresh_trigger_m)) or float(args.adaptive_residual_refresh_trigger_m) <= 0.0:
            raise ValueError("adaptive-residual-refresh-trigger-m must be finite and positive")
        adaptive_budget_mapping["residual_refresh_trigger_m"] = float(args.adaptive_residual_refresh_trigger_m)
    if adaptive_k and not adaptive_budget_mapping:
        raise ValueError("adaptive_k requires prediction.adaptive_budget configuration")
    phase17_execution_mapping = apply_execution_cli_overrides(
        dict(phase17_mapping.get("execution", {})),
        delay_steps=args.execution_delay_steps,
        noise_std_mps=args.execution_noise_std_mps,
        noise_bound_sigma=args.execution_noise_bound_sigma,
        tracking_time_constant_s=args.execution_tracking_time_constant_s,
        drag_coefficient=args.execution_drag_coefficient,
    )
    distributed_mapping = dict(mpc_document.get("distributed", {}))
    if args.qdr_execution_tube_multiplier is not None:
        if (
            not np.isfinite(float(args.qdr_execution_tube_multiplier))
            or float(args.qdr_execution_tube_multiplier) < 1.0
        ):
            raise ValueError("qdr-execution-tube-multiplier must be finite and at least one")
        distributed_mapping["qdr_execution_tube_enabled"] = True
        distributed_mapping["qdr_execution_tube_multiplier"] = float(args.qdr_execution_tube_multiplier)
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

    reachable_tube = (
        None
        if args.reachable_tube_calibration is None
        else DelayAwareConformalReachableTube.from_json(args.reachable_tube_calibration)
    )

    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes_closed_loop(args.protocol, args.scenes, args.mpc_config)
    if args.reachable_tube_calibration is not None:
        tube_path = args.reachable_tube_calibration.resolve()
        hashes[str(tube_path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = hashlib.sha256(
            tube_path.read_bytes()
        ).hexdigest()
    if args.adaptive_risk_calibration is not None:
        calibration_path = args.adaptive_risk_calibration.resolve()
        hashes[str(calibration_path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = hashlib.sha256(
            calibration_path.read_bytes()
        ).hexdigest()
    if args.safety_layer == "robust_cbf_qp":
        add_safety_source_hashes(hashes, args.safety_config)
    run_config = {
        "scenes": str(args.scenes.resolve()),
        "protocol": str(args.protocol.resolve()),
        "environment_config": str(args.environment_config.resolve()),
        "mpc_config": str(args.mpc_config.resolve()),
        "planner": planner_config.__dict__,
        "distributed": distributed_mapping,
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
        "queue_aware_safety_projection": queue_aware_safety_projection,
        "qdr_prefix_recovery_authority": qdr_prefix_recovery_authority,
        "adaptive_k": adaptive_k,
        "adaptive_budget": adaptive_budget_mapping,
        "adaptive_risk_calibration": (
            None
            if args.adaptive_risk_calibration is None
            else str(args.adaptive_risk_calibration.resolve())
        ),
        "reachability_normalized_cost": rnic,
        "reachable_tube_calibration": (
            None
            if args.reachable_tube_calibration is None
            else str(args.reachable_tube_calibration.resolve())
        ),
        "phase17": phase17_mapping,
        "effective_execution": phase17_execution_mapping,
        "device": str(device),
        **torch_thread_settings,
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
                if qdr_prefix_recovery_authority is not None:
                    config.setdefault("dynamics", {}).setdefault("execution", {})[
                        "pending_command_authority"
                    ] = qdr_prefix_recovery_authority
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
                    queue_aware_safety_projection=queue_aware_safety_projection,
                    qdr_prefix_recovery_authority=qdr_prefix_recovery_authority,
                    adaptive_prediction_config=(adaptive_budget_mapping if adaptive_k else None),
                    reachable_tube=reachable_tube,
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
            writer.add_scalar("Evaluation/torch_num_threads", run_config["torch_num_threads"], 0)
            writer.add_scalar("Evaluation/torch_num_interop_threads", run_config["torch_num_interop_threads"], 0)
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
                    "qdr_prefix_minimum_obstacle_barrier_m",
                    "qdr_prefix_minimum_boundary_barrier_m",
                    "qdr_prefix_minimum_inter_agent_barrier_m",
                    "qdr_prefix_minimum_barrier_m",
                    "qdr_prefix_admissible_rate",
                    "qdr_prefix_first_violation_step",
                    "qdr_prefix_violation_step_count",
                    "qdr_prefix_violation_step_ratio",
                    "qdr_prefix_recovery_requested",
                    "qdr_prefix_recovery_applied",
                    "qdr_prefix_recovery_override_slots",
                    "qdr_suffix_minimum_clearance_m",
                    "qdr_suffix_minimum_barrier_m",
                    "qdr_suffix_admissible_rate",
                    "qdr_suffix_gate_exhausted_once",
                    "qdr_suffix_gate_first_exhaustion_step",
                    "qdr_suffix_gate_max_exhaustion_streak_steps",
                    "qdr_suffix_gate_recovery_count",
                    "qdr_execution_tube_enabled",
                    "qdr_execution_tube_multiplier",
                    "qdr_mean_execution_tube_radius_m",
                    "qdr_max_execution_tube_radius_m",
                    "qdr_precondition_recovery_recommended_rate",
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
                    "rnic_assignment_switch_rate",
                    "escape_gap_enabled_rate",
                    "mean_escape_gap_cost",
                    "mean_escape_gap_max_rad",
                    "mean_escape_gap_escape_rad",
                    "mean_escape_gap_coverage_ratio",
                    "mean_escape_gap_violation_rate",
                    "fc_dbf_enabled_rate",
                    "fc_dbf_feasible_rate",
                    "mean_fc_dbf_min_slot_slack_s",
                    "mean_fc_dbf_max_slot_error_m",
                    "mean_fc_dbf_slot_error_m",
                    "mean_fc_dbf_slot_progress_m",
                    "mean_fc_dbf_assignment_switch_rate",
                    "fc_dbf_gate_exhaustion_rate",
                    "mean_fc_dbf_cost",
                    "belief_fusion_enabled_rate",
                    "belief_fusion_effective_sample_size",
                    "belief_fusion_weight_entropy",
                    "belief_fusion_mean_age_steps",
                    "belief_fusion_max_age_steps",
                    "belief_fusion_mean_covariance_trace_m2",
                    "belief_fusion_fallback_rate",
                    "conformal_tube_enabled_rate",
                    "mean_conformal_tube_radius_m",
                    "maximum_conformal_tube_radius_m",
                    "mean_conformal_tube_budget_score",
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
            writer.add_scalar("Summary/QDR/prefix_admissible_rate", overall["qdr_prefix_admissible_rate"], 0)
            writer.add_scalar("Summary/QDR/prefix_minimum_barrier_m", overall["qdr_prefix_minimum_barrier_m"], 0)
            writer.add_scalar("Summary/QDR/suffix_minimum_clearance_m", overall["qdr_suffix_minimum_clearance_m"], 0)
            writer.add_scalar("Summary/QDR/suffix_minimum_barrier_m", overall["qdr_suffix_minimum_barrier_m"], 0)
            writer.add_scalar("Summary/QDR/suffix_admissible_rate", overall["qdr_suffix_admissible_rate"], 0)
            writer.add_scalar(
                "Summary/QDR/execution_tube_enabled_rate",
                overall.get("qdr_execution_tube_enabled_rate", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/execution_tube_multiplier",
                overall.get("mean_qdr_execution_tube_multiplier", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/mean_execution_tube_radius_m",
                overall.get("mean_qdr_execution_tube_radius_m", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/max_execution_tube_radius_m",
                overall.get("max_qdr_execution_tube_radius_m", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/suffix_gate_active_rate",
                overall.get("qdr_suffix_gate_active_rate", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/suffix_gate_exhaustion_rate",
                overall.get("qdr_suffix_gate_exhaustion_rate", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/suffix_gate_rejected_candidates",
                overall.get("qdr_suffix_gate_rejected_candidates", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/suffix_gate_exhausted_episode_rate",
                overall.get("qdr_suffix_gate_exhausted_episode_rate", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/suffix_gate_first_exhaustion_step",
                overall.get("qdr_suffix_gate_first_exhaustion_step", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/suffix_gate_max_exhaustion_streak_steps",
                overall.get("qdr_suffix_gate_max_exhaustion_streak_steps", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/suffix_gate_recovery_count",
                overall.get("qdr_suffix_gate_recovery_count", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/precondition_recovery_recommended_rate",
                overall["qdr_precondition_recovery_recommended_rate"],
                0,
            )
            writer.add_text(
                "Summary/QDR/prefix_violation_cause_counts",
                json.dumps(overall.get("qdr_prefix_violation_cause_counts", {}), sort_keys=True),
                0,
            )
            writer.add_text(
                "Summary/QDR/precondition_status_counts",
                json.dumps(overall.get("qdr_precondition_status_counts", {}), sort_keys=True),
                0,
            )
            writer.add_scalar("Summary/QDR/prefix_recovery_request_rate", overall["qdr_prefix_recovery_request_rate"], 0)
            writer.add_scalar("Summary/QDR/prefix_recovery_apply_rate", overall["qdr_prefix_recovery_apply_rate"], 0)
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
            writer.add_scalar("Summary/RNIC/assignment_switch_rate", overall["rnic_assignment_switch_rate"], 0)
            writer.add_text(
                "Summary/RNIC/cost_mode_counts",
                json.dumps(overall.get("rnic_cost_mode_counts", {}), sort_keys=True),
                0,
            )
            writer.add_scalar("Summary/EGC/enabled_rate", overall["escape_gap_enabled_rate"], 0)
            writer.add_scalar("Summary/EGC/mean_gap_cost", overall["mean_escape_gap_cost"], 0)
            writer.add_scalar("Summary/EGC/mean_max_gap_rad", overall["mean_escape_gap_max_rad"], 0)
            writer.add_scalar("Summary/EGC/mean_escape_gap_rad", overall["mean_escape_gap_escape_rad"], 0)
            writer.add_scalar("Summary/EGC/mean_coverage_ratio", overall["mean_escape_gap_coverage_ratio"], 0)
            writer.add_scalar("Summary/EGC/mean_violation_rate", overall["mean_escape_gap_violation_rate"], 0)
            writer.add_scalar("Summary/FCDBF/enabled_rate", overall["fc_dbf_enabled_rate"], 0)
            writer.add_scalar("Summary/FCDBF/feasible_rate", overall["fc_dbf_feasible_rate"], 0)
            writer.add_scalar(
                "Summary/FCDBF/mean_min_slot_slack_s",
                overall["mean_fc_dbf_min_slot_slack_s"],
                0,
            )
            writer.add_scalar(
                "Summary/FCDBF/mean_max_slot_error_m",
                overall["mean_fc_dbf_max_slot_error_m"],
                0,
            )
            writer.add_scalar(
                "Summary/FCDBF/mean_slot_error_m",
                overall["mean_fc_dbf_slot_error_m"],
                0,
            )
            writer.add_scalar(
                "Summary/FCDBF/mean_slot_progress_m",
                overall["mean_fc_dbf_slot_progress_m"],
                0,
            )
            writer.add_scalar(
                "Summary/FCDBF/assignment_switch_rate",
                overall["mean_fc_dbf_assignment_switch_rate"],
                0,
            )
            writer.add_scalar(
                "Summary/FCDBF/gate_exhaustion_rate",
                overall["fc_dbf_gate_exhaustion_rate"],
                0,
            )
            writer.add_scalar("Summary/FCDBF/mean_cost", overall["mean_fc_dbf_cost"], 0)
            writer.add_scalar(
                "Summary/BeliefFusion/enabled_rate",
                overall["belief_fusion_enabled_rate"],
                0,
            )
            writer.add_scalar(
                "Summary/BeliefFusion/effective_sample_size",
                overall["belief_fusion_effective_sample_size"],
                0,
            )
            writer.add_scalar(
                "Summary/BeliefFusion/weight_entropy",
                overall["belief_fusion_weight_entropy"],
                0,
            )
            writer.add_scalar(
                "Summary/BeliefFusion/mean_age_steps",
                overall["belief_fusion_mean_age_steps"],
                0,
            )
            writer.add_scalar(
                "Summary/BeliefFusion/max_age_steps",
                overall["belief_fusion_max_age_steps"],
                0,
            )
            writer.add_scalar(
                "Summary/BeliefFusion/mean_covariance_trace_m2",
                overall["belief_fusion_mean_covariance_trace_m2"],
                0,
            )
            writer.add_scalar(
                "Summary/BeliefFusion/fallback_rate",
                overall["belief_fusion_fallback_rate"],
                0,
            )
            writer.add_text(
                "Summary/BeliefFusion/fallback_reason_counts",
                json.dumps(overall.get("belief_fusion_fallback_reason_counts", {}), sort_keys=True),
                0,
            )
            writer.add_scalar("Summary/ConformalTube/enabled_rate", overall["conformal_tube_enabled_rate"], 0)
            writer.add_scalar("Summary/ConformalTube/mean_radius_m", overall["mean_conformal_tube_radius_m"], 0)
            writer.add_scalar("Summary/ConformalTube/maximum_radius_m", overall["maximum_conformal_tube_radius_m"], 0)
            writer.add_scalar("Summary/ConformalTube/mean_budget_score", overall["mean_conformal_tube_budget_score"], 0)
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
