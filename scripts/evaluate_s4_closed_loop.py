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
    "distributed_async",
    # Phase 56 baseline aliases.  Their effective contracts are resolved in
    # main() and recorded per method so the raw evaluator remains backwards
    # compatible with earlier result directories.
    "delayed_mpc",
    "fixed_tube_mpc",
    "tube_mpc",
    "queue_aware_tube_mpc",
    "qdr_mpc",
    "synchronous_distributed_mpc",
    "asynchronous_distributed_mpc",
    "qdr_asynchronous_mpc",
    "fixed_k8_qdr",
    "B0_current_state_delayed_mpc",
    "B1_qdr_mpc",
    "B2_fixed_tube_mpc",
    "B3_queue_aware_tube_mpc",
    "B4_synchronous_distributed_mpc",
    "B5_asynchronous_distributed_mpc",
    "B6_qdr_asynchronous_mpc",
    "B7_fixed_k8_qdr",
    # Phase 58 explicit baseline names.  The aliases keep result directories
    # aligned with the pre-registered method contract.
    "M0_current_state_delayed_mpc",
    "M1_known_delay_delayed_mpc",
    "M2_fixed_tube_mpc",
    "M3_queue_aware_tube_mpc",
    "M4_synchronous_distributed_mpc",
    "M5_asynchronous_distributed_mpc",
    "M6_qdr_synchronous_mpc",
    "M6_qdr_normalized_soft_progress",
    "M7_qdr_asynchronous_mpc",
    "M8_fixed_k8_qdr",
    "M9_qdr_bounded_recovery",
    # Phase 59 repair-calibration variants.  They all retain immutable queue
    # authority; the suffix indicates only the pre-registered intervention.
    "R0_phase59_qdr_baseline",
    "R1_phase59_queue_cbf_k1",
    "R2_phase59_queue_cbf_k4",
    "R3_phase59_queue_cbf_k8",
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
        "--fixed-tube-radius-m",
        type=float,
        help="Use a constant target tube radius for Phase 56 fixed-tube baselines.",
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
        "--qdr-exhaustion-policy",
        choices=("hard_min_violation", "normalized_soft_progress", "bounded_progress_recovery"),
        help=(
            "QDR candidate selection when every candidate fails the suffix gate; "
            "soft and bounded policies are development liveness diagnostics only."
        ),
    )
    queue_token_group = parser.add_mutually_exclusive_group()
    queue_token_group.add_argument(
        "--queue-token-contract",
        dest="queue_token_contract",
        action="store_true",
        help="Enable token-checked queue authority and ACK logging for a validation diagnostic.",
    )
    queue_token_group.add_argument(
        "--no-queue-token-contract",
        dest="queue_token_contract",
        action="store_false",
        help="Disable the opt-in queue-token contract.",
    )
    parser.set_defaults(queue_token_contract=None)
    parser.add_argument(
        "--queue-token-max-override-slots",
        type=int,
        help="Optional non-negative cap on queue slots changed by token-checked recovery.",
    )
    parser.add_argument(
        "--qdr-execution-tube-multiplier",
        type=float,
        help="Enable the validation-frozen empirical QDR execution tube with this multiplier.",
    )
    parser.add_argument(
        "--qdr-execution-tube-calibration",
        type=Path,
        help="Load a validation-frozen QDR execution-tube summary JSON and use its step radii.",
    )
    parser.add_argument(
        "--qdr-execution-tube-calibration-variant",
        type=str,
        help="Variant key in --qdr-execution-tube-calibration; required when it has multiple variants.",
    )
    parser.add_argument(
        "--qdr-execution-tube-active-steps",
        type=int,
        help="Apply the empirical QDR tube only to this many immediate suffix steps.",
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

    if not phase17_execution_mapping:
        return
    if "execution_overrides" in frozen_scene_record:
        # Frozen scene contracts own the physical delay/noise factors.  The
        # opt-in queue-token contract is a bookkeeping/authority layer and
        # may still be enabled without rewriting those physical factors.
        token_keys = {
            "queue_token_contract_enabled",
            "queue_token_max_override_slots",
        }
        token_mapping = {
            key: copy.deepcopy(value)
            for key, value in phase17_execution_mapping.items()
            if key in token_keys
        }
        if token_mapping:
            config.setdefault("dynamics", {}).setdefault("execution", {}).update(token_mapping)
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


def phase56_method_contract(
    method: str,
    *,
    queue_aware_rollout: bool,
    queue_aware_safety_projection: bool,
    adaptive_k: bool,
    num_samples: int,
    fixed_tube_radius_m: float | None,
) -> dict[str, Any]:
    """Resolve a named Phase 56 baseline into the shared evaluator contract.

    The aliases keep baseline definitions visible in result directories while
    reusing the same prediction, planner, safety, and latency accounting path.
    No alias changes the locked-test data or uses target truth online.
    """

    alias = {
        "B0_current_state_delayed_mpc": "delayed_mpc",
        "B1_qdr_mpc": "qdr_mpc",
        "B2_fixed_tube_mpc": "fixed_tube_mpc",
        "B3_queue_aware_tube_mpc": "queue_aware_tube_mpc",
        "B4_synchronous_distributed_mpc": "synchronous_distributed_mpc",
        "B5_asynchronous_distributed_mpc": "asynchronous_distributed_mpc",
        "B6_qdr_asynchronous_mpc": "qdr_asynchronous_mpc",
        "B7_fixed_k8_qdr": "fixed_k8_qdr",
        "M0_current_state_delayed_mpc": "delayed_mpc",
        "M1_known_delay_delayed_mpc": "known_delay_mpc",
        "M2_fixed_tube_mpc": "fixed_tube_mpc",
        "M3_queue_aware_tube_mpc": "queue_aware_tube_mpc",
        "M4_synchronous_distributed_mpc": "synchronous_distributed_mpc",
        "M5_asynchronous_distributed_mpc": "asynchronous_distributed_mpc",
        "M6_qdr_synchronous_mpc": "qdr_synchronous_mpc",
        "M6_qdr_normalized_soft_progress": "qdr_synchronous_mpc",
        "M7_qdr_asynchronous_mpc": "qdr_asynchronous_mpc",
        "M8_fixed_k8_qdr": "fixed_k8_qdr",
        "M9_qdr_bounded_recovery": "qdr_synchronous_mpc",
    }
    requested_method = method
    method = alias.get(method, method)
    contract = {
        "requested_method": requested_method,
        "canonical_method": method,
        "queue_aware_rollout": bool(queue_aware_rollout),
        "queue_aware_safety_projection": bool(queue_aware_safety_projection),
        "adaptive_k": bool(adaptive_k),
        "num_samples": int(num_samples),
        "candidate_budget_requested": int(num_samples),
        "fixed_tube_radius_m": fixed_tube_radius_m,
        "target_tube_cost_enabled": False,
        "known_delay_compensation": False,
        "distributed_mode": None,
        "qdr_exhaustion_policy": None,
    }
    if method == "delayed_mpc":
        contract.update(canonical_method="worst_case", queue_aware_rollout=False, queue_aware_safety_projection=False)
    elif method == "known_delay_mpc":
        contract.update(
            canonical_method="worst_case",
            queue_aware_rollout=False,
            queue_aware_safety_projection=False,
            known_delay_compensation=True,
        )
    elif method in {"fixed_tube_mpc", "tube_mpc"}:
        contract.update(
            canonical_method="worst_case",
            queue_aware_rollout=False,
            queue_aware_safety_projection=False,
            fixed_tube_radius_m=(0.35 if fixed_tube_radius_m is None else fixed_tube_radius_m),
            target_tube_cost_enabled=True,
        )
    elif method in {"queue_aware_tube_mpc", "qdr_mpc"}:
        contract.update(
            canonical_method="worst_case",
            queue_aware_rollout=True,
            queue_aware_safety_projection=bool(queue_aware_safety_projection),
            fixed_tube_radius_m=(0.35 if fixed_tube_radius_m is None else fixed_tube_radius_m),
            target_tube_cost_enabled=True,
        )
    elif method == "synchronous_distributed_mpc":
        contract.update(canonical_method="distributed_delayed", distributed_mode="delayed", queue_aware_rollout=False)
    elif method == "qdr_synchronous_mpc":
        contract.update(
            canonical_method="distributed_delayed",
            distributed_mode="delayed",
            queue_aware_rollout=True,
        )
    elif method == "asynchronous_distributed_mpc":
        contract.update(canonical_method="distributed_async", distributed_mode="asynchronous", queue_aware_rollout=False)
    elif method == "qdr_asynchronous_mpc":
        contract.update(canonical_method="distributed_async", distributed_mode="asynchronous", queue_aware_rollout=True)
    elif method == "distributed_async":
        contract.update(distributed_mode="asynchronous", queue_aware_rollout=False)
    elif method == "fixed_k8_qdr":
        contract.update(canonical_method="worst_case", queue_aware_rollout=True, adaptive_k=False, num_samples=max(8, num_samples))
    if requested_method == "R0_phase59_qdr_baseline":
        contract.update(
            canonical_method="distributed_delayed",
            distributed_mode="delayed",
            queue_aware_rollout=True,
            queue_aware_safety_projection=False,
            num_samples=int(num_samples),
        )
    elif requested_method == "R1_phase59_queue_cbf_k1":
        contract.update(
            canonical_method="distributed_delayed",
            distributed_mode="delayed",
            queue_aware_rollout=True,
            queue_aware_safety_projection=True,
            num_samples=1,
        )
    elif requested_method == "R2_phase59_queue_cbf_k4":
        contract.update(
            canonical_method="distributed_delayed",
            distributed_mode="delayed",
            queue_aware_rollout=True,
            queue_aware_safety_projection=True,
            num_samples=max(4, int(num_samples)),
        )
    elif requested_method == "R3_phase59_queue_cbf_k8":
        contract.update(
            canonical_method="distributed_delayed",
            distributed_mode="delayed",
            queue_aware_rollout=True,
            queue_aware_safety_projection=True,
            num_samples=max(8, int(num_samples)),
        )
    elif requested_method == "M6_qdr_normalized_soft_progress":
        contract["qdr_exhaustion_policy"] = "normalized_soft_progress"
    elif requested_method == "M9_qdr_bounded_recovery":
        contract["qdr_exhaustion_policy"] = "bounded_progress_recovery"
    if contract["queue_aware_safety_projection"] and not contract["queue_aware_rollout"]:
        contract["queue_aware_safety_projection"] = False
    # Keep the requested budget synchronized with alias-specific overrides.
    # The actual realized budget is measured from step logs after prediction;
    # this distinction is essential because a GRU checkpoint returns one mean
    # trajectory even when the CLI requests K>1.
    contract["candidate_budget_requested"] = int(contract["num_samples"])
    return contract


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
    qdr_segment_audit_mapping = dict(mpc_document.get("qdr_segment_audit", {}))
    qdr_segment_progress_tolerance_m = float(
        qdr_segment_audit_mapping.get("progress_tolerance_m", 1.0e-9)
    )
    if (
        not np.isfinite(qdr_segment_progress_tolerance_m)
        or qdr_segment_progress_tolerance_m < 0.0
    ):
        raise ValueError("qdr_segment_audit.progress_tolerance_m must be finite and non-negative")
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
    if args.queue_token_max_override_slots is not None:
        if int(args.queue_token_max_override_slots) < 0:
            raise ValueError("queue-token-max-override-slots must be non-negative")
        if int(args.queue_token_max_override_slots) != args.queue_token_max_override_slots:
            raise ValueError("queue-token-max-override-slots must be an integer")
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
    if args.qdr_exhaustion_policy is not None:
        distributed_mapping["qdr_exhaustion_policy"] = str(args.qdr_exhaustion_policy)
    if args.qdr_execution_tube_active_steps is not None:
        if int(args.qdr_execution_tube_active_steps) <= 0:
            raise ValueError("qdr-execution-tube-active-steps must be positive")
        distributed_mapping["qdr_execution_tube_active_steps"] = int(args.qdr_execution_tube_active_steps)
    qdr_execution_tube_calibration_path = None
    qdr_execution_tube_calibration_variant = None
    if args.qdr_execution_tube_calibration is not None:
        if args.qdr_execution_tube_multiplier is not None:
            raise ValueError(
                "qdr-execution-tube-calibration cannot be combined with an explicit multiplier"
            )
        qdr_execution_tube_calibration_path = args.qdr_execution_tube_calibration.resolve()
        calibration_document = json.loads(
            qdr_execution_tube_calibration_path.read_text(encoding="utf-8")
        )
        variants = dict(calibration_document.get("variants", {}))
        if not variants:
            raise ValueError("QDR execution-tube calibration has no variants")
        if args.qdr_execution_tube_calibration_variant is None:
            if len(variants) != 1:
                raise ValueError(
                    "--qdr-execution-tube-calibration-variant is required for multiple variants"
                )
            qdr_execution_tube_calibration_variant = next(iter(variants))
        else:
            qdr_execution_tube_calibration_variant = str(args.qdr_execution_tube_calibration_variant)
        if qdr_execution_tube_calibration_variant not in variants:
            raise ValueError(
                f"Unknown QDR execution-tube calibration variant: {qdr_execution_tube_calibration_variant}"
            )
        calibration_summary = dict(variants[qdr_execution_tube_calibration_variant])
        calibrated_radii = calibration_summary.get("simultaneous_calibrated_radius_m_by_step")
        if not isinstance(calibrated_radii, list):
            raise ValueError("QDR execution-tube calibration is missing step radii")
        if len(calibrated_radii) != planner_config.horizon_steps:
            raise ValueError("QDR execution-tube calibration horizon does not match the planner")
        selected_multiplier = max(1.0, float(calibration_summary.get("simultaneous_multiplier", 1.0)))
        if not np.isfinite(selected_multiplier) or selected_multiplier < 1.0:
            raise ValueError("QDR execution-tube calibration has an invalid multiplier")
        distributed_mapping["qdr_execution_tube_enabled"] = True
        distributed_mapping["qdr_execution_tube_multiplier"] = selected_multiplier
        distributed_mapping["qdr_execution_tube_radius_m_by_step"] = [
            float(value) for value in calibrated_radii
        ]
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
    checkpoint_model_kind = None if checkpoint_data is None else str(checkpoint_data[1])

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
    if args.fixed_tube_radius_m is not None:
        if not np.isfinite(float(args.fixed_tube_radius_m)) or float(args.fixed_tube_radius_m) < 0.0:
            raise ValueError("fixed-tube-radius-m must be finite and non-negative")

    output.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes_closed_loop(args.protocol, args.scenes, args.mpc_config)
    if args.reachable_tube_calibration is not None:
        tube_path = args.reachable_tube_calibration.resolve()
        hashes[str(tube_path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = hashlib.sha256(
            tube_path.read_bytes()
        ).hexdigest()
    if qdr_execution_tube_calibration_path is not None:
        hashes[
            str(qdr_execution_tube_calibration_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        ] = hashlib.sha256(qdr_execution_tube_calibration_path.read_bytes()).hexdigest()
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
        "qdr_execution_tube_calibration": (
            None
            if qdr_execution_tube_calibration_path is None
            else str(qdr_execution_tube_calibration_path)
        ),
        "qdr_execution_tube_calibration_variant": qdr_execution_tube_calibration_variant,
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
        "qdr_segment_audit": {
            **qdr_segment_audit_mapping,
            "progress_tolerance_m": qdr_segment_progress_tolerance_m,
        },
        "queue_token_contract": args.queue_token_contract,
        "queue_token_max_override_slots": args.queue_token_max_override_slots,
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
        "fixed_tube_radius_m": (
            None if args.fixed_tube_radius_m is None else float(args.fixed_tube_radius_m)
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
        "distributed_async": "asynchronous",
    }
    for method in args.methods:
        method_contract = phase56_method_contract(
            method,
            queue_aware_rollout=queue_aware_rollout,
            queue_aware_safety_projection=queue_aware_safety_projection,
            adaptive_k=adaptive_k,
            num_samples=args.num_samples,
            fixed_tube_radius_m=(
                None if args.fixed_tube_radius_m is None else float(args.fixed_tube_radius_m)
            ),
        )
        method_contract["checkpoint_model_kind"] = checkpoint_model_kind
        method_contract["candidate_budget_realization_model"] = (
            "single_mean_trajectory"
            if checkpoint_model_kind == "gru"
            else "sample_set"
            if checkpoint_model_kind is not None or args.candidate_source == "belief"
            else "not_applicable"
        )
        method_distributed_mapping = dict(distributed_mapping)
        if method_contract.get("qdr_exhaustion_policy") is not None:
            method_distributed_mapping["qdr_exhaustion_policy"] = str(
                method_contract["qdr_exhaustion_policy"]
            )
        method_output = output / method
        method_output.mkdir(parents=True, exist_ok=True)
        method_run_config = {
            **run_config,
            "distributed": method_distributed_mapping,
            "method_contract": method_contract,
        }
        method_output.joinpath("config.yaml").write_text(
            yaml.safe_dump(method_run_config, sort_keys=False), encoding="utf-8"
        )
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
                if (
                    qdr_prefix_recovery_authority is not None
                    and bool(method_contract["queue_aware_rollout"])
                ):
                    config.setdefault("dynamics", {}).setdefault("execution", {})[
                        "pending_command_authority"
                    ] = qdr_prefix_recovery_authority
                if (
                    args.queue_token_contract is not None
                    and bool(method_contract["queue_aware_rollout"])
                ):
                    config.setdefault("dynamics", {}).setdefault("execution", {})[
                        "queue_token_contract_enabled"
                    ] = bool(args.queue_token_contract)
                if (
                    args.queue_token_max_override_slots is not None
                    and bool(method_contract["queue_aware_rollout"])
                ):
                    config.setdefault("dynamics", {}).setdefault("execution", {})[
                        "queue_token_max_override_slots"
                    ] = int(args.queue_token_max_override_slots)
                distributed_config = None
                canonical_method = str(method_contract["canonical_method"])
                if canonical_method in distributed_modes:
                    effective_distributed_mapping = dict(distributed_mapping)
                    if method_contract.get("qdr_exhaustion_policy") is not None:
                        effective_distributed_mapping["qdr_exhaustion_policy"] = str(
                            method_contract["qdr_exhaustion_policy"]
                        )
                    if method_contract["distributed_mode"] == "asynchronous":
                        effective_distributed_mapping["communication_interval_steps"] = max(
                            2, int(effective_distributed_mapping.get("communication_interval_steps", 1))
                        )
                    distributed_config = DistributedDNMPCConfig.from_mapping(
                        {
                            **effective_distributed_mapping,
                            "communication_mode": distributed_modes[canonical_method],
                        }
                    )
                method_planner_config = MinimaxMPCConfig(
                    **{
                        **planner_config.__dict__,
                        "target_tube_cost_enabled": bool(
                            method_contract["target_tube_cost_enabled"]
                        ),
                    }
                )
                row, episode_steps = run_episode(
                    config,
                    seed=int(spec["episode_seed"]),
                    method=canonical_method,
                    planner_config=method_planner_config,
                    candidate_source=args.candidate_source,
                    checkpoint_data=checkpoint_data,
                    device=device,
                    num_samples=int(method_contract["num_samples"]),
                    sampling_steps=args.sampling_steps,
                    sampling_seed=args.sampling_seed + int(spec["episode_index"]) * 1000,
                    projection_iterations=args.projection_iterations,
                    use_local_cbf=args.safety_layer == "local_cbf",
                    safety_layer=args.safety_layer,
                    robust_safety_config=safety_config,
                    prediction_refresh_interval_steps=args.prediction_refresh_interval_steps,
                    queue_aware_rollout=bool(method_contract["queue_aware_rollout"]),
                    queue_aware_safety_projection=bool(method_contract["queue_aware_safety_projection"]),
                    known_delay_compensation=bool(method_contract["known_delay_compensation"]),
                    qdr_segment_progress_tolerance_m=qdr_segment_progress_tolerance_m,
                    qdr_prefix_recovery_authority=(
                        qdr_prefix_recovery_authority
                        if bool(method_contract["queue_aware_rollout"])
                        else None
                    ),
                    adaptive_prediction_config=(
                        adaptive_budget_mapping if bool(method_contract["adaptive_k"]) else None
                    ),
                    reachable_tube=reachable_tube,
                    fixed_tube_radius_m=method_contract["fixed_tube_radius_m"],
                    distributed_config=distributed_config,
                    scenario=scenario_from_metadata(spec["scenario"]),
                    validate_scenario=False,
                )
                row.update(
                    {
                        "method": method,
                        "phase56_baseline_id": method,
                        "phase56_canonical_method": canonical_method,
                        "phase58_known_delay_compensation": bool(
                            method_contract["known_delay_compensation"]
                        ),
                        "candidate_budget_requested": int(
                            method_contract["candidate_budget_requested"]
                        ),
                        "candidate_budget_realized_min": int(
                            row.get("candidate_budget_realized_min", 0)
                        ),
                        "candidate_budget_realized_max": int(
                            row.get("candidate_budget_realized_max", 0)
                        ),
                        "candidate_budget_realized_rate": float(
                            row.get("candidate_budget_realized_rate", float("nan"))
                        ),
                        "candidate_budget_mismatch_steps": int(
                            row.get("candidate_budget_mismatch_steps", 0)
                        ),
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
            writer.add_text("Evaluation/config", yaml.safe_dump(method_run_config, sort_keys=False), 0)
            writer.add_text(
                "Evaluation/source_hashes",
                json.dumps(method_run_config["source_hashes"], indent=2),
                0,
            )
            writer.add_text(
                "Evaluation/QDR/exhaustion_policy",
                str(
                    method_run_config.get("distributed", {}).get(
                        "qdr_exhaustion_policy", "hard_min_violation"
                    )
                ),
                0,
            )
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
                    "qdr_prefix_recovery_token_present_rate",
                    "qdr_prefix_recovery_ack_accept_rate",
                    "qdr_prefix_recovery_ack_apply_rate",
                    "qdr_suffix_minimum_clearance_m",
                    "qdr_suffix_minimum_barrier_m",
                    "qdr_suffix_admissible_rate",
                    "qdr_terminal_any_candidate_feasible_rate",
                    "qdr_terminal_all_candidate_feasible_rate",
                    "qdr_segment_earliest_any_capture_step",
                    "qdr_segment_earliest_all_capture_step",
                    "qdr_segment_best_terminal_distance_m",
                    "qdr_segment_worst_terminal_distance_m",
                    "qdr_segment_best_progress_m",
                    "qdr_segment_worst_progress_m",
                    "qdr_segment_finite_progress_rate",
                    "qdr_segment_horizon_steps",
                    "qdr_segment_candidate_count",
                    "qdr_suffix_gate_exhausted_once",
                    "qdr_suffix_gate_first_exhaustion_step",
                    "qdr_suffix_gate_max_exhaustion_streak_steps",
                    "qdr_suffix_gate_recovery_count",
                    "qdr_exhaustion_soft_fallback_count",
                    "qdr_exhaustion_recovery_active_rate",
                    "qdr_exhaustion_recovery_count",
                    "qdr_exhaustion_recovery_budget_steps",
                    "qdr_exhaustion_recovery_horizon_steps",
                    "qdr_exhaustion_recovery_steps",
                    "qdr_execution_tube_enabled",
                    "qdr_execution_tube_multiplier",
                    "qdr_execution_tube_active_steps",
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
                    "mean_adaptive_queue_prefix_risk",
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
                "Summary/QDR/execution_tube_active_steps",
                overall.get("mean_qdr_execution_tube_active_steps", float("nan")),
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
                "Summary/QDR/exhaustion_soft_fallback_count",
                overall.get("qdr_exhaustion_soft_fallback_count", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/exhaustion_recovery_active_rate",
                overall.get("qdr_exhaustion_recovery_active_rate", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/exhaustion_recovery_count",
                overall.get("qdr_exhaustion_recovery_count", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/exhaustion_recovery_budget_steps",
                overall.get("qdr_exhaustion_recovery_budget_steps", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/exhaustion_recovery_horizon_steps",
                overall.get("qdr_exhaustion_recovery_horizon_steps", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/exhaustion_recovery_steps",
                overall.get("qdr_exhaustion_recovery_steps", float("nan")),
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
            writer.add_text(
                "Summary/QDR/segment_status_counts",
                json.dumps(overall.get("qdr_segment_status_counts", {}), sort_keys=True),
                0,
            )
            writer.add_scalar("Summary/QDR/prefix_recovery_request_rate", overall["qdr_prefix_recovery_request_rate"], 0)
            writer.add_scalar("Summary/QDR/prefix_recovery_apply_rate", overall["qdr_prefix_recovery_apply_rate"], 0)
            writer.add_scalar(
                "Summary/QDR/prefix_recovery_token_present_rate",
                overall.get("qdr_prefix_recovery_token_present_rate", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/prefix_recovery_ack_accept_rate",
                overall.get("qdr_prefix_recovery_ack_accept_rate", float("nan")),
                0,
            )
            writer.add_scalar(
                "Summary/QDR/prefix_recovery_ack_apply_rate",
                overall.get("qdr_prefix_recovery_ack_apply_rate", float("nan")),
                0,
            )
            writer.add_text(
                "Summary/QDR/prefix_recovery_ack_reason_counts",
                json.dumps(overall.get("qdr_prefix_recovery_ack_reason_counts", {}), sort_keys=True),
                0,
            )
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
            writer.add_scalar(
                "Summary/UAKR/mean_queue_prefix_risk",
                overall.get("mean_adaptive_queue_prefix_risk", float("nan")),
                0,
            )
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
                    "canonical_method": str(method_contract["canonical_method"]),
                    "queue_aware_rollout": int(bool(method_contract["queue_aware_rollout"])),
                    "adaptive_k": int(bool(method_contract["adaptive_k"])),
                    "reachability_normalized_cost": int(rnic),
                    "num_samples": int(method_contract["num_samples"]),
                    "fixed_tube_radius_m": float(
                        method_contract["fixed_tube_radius_m"]
                        if method_contract["fixed_tube_radius_m"] is not None
                        else 0.0
                    ),
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
