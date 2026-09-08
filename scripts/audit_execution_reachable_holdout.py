"""Audit held-out execution reachable-set coverage and sensitivity.

The script freezes a margin on one sampled calibration split, evaluates it on
an independent in-domain split, and then reports one-factor sensitivity and
explicit out-of-calibration cases.  It is an empirical contract audit: it
does not prove a continuous-time reachable set or forward invariance.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (SOURCE_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    SummaryWriter = None  # type: ignore[assignment,misc]

from calibrate_execution_reachable_set import (  # noqa: E402
    _finite_range,
    _parameters,
    _sample_commands,
    upper_quantile,
)
from encirclement3d.execution_dynamics import (  # noqa: E402
    ExecutionParameters,
    position_uncertainty_radii,
    rollout_execution,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "innovation_execution_reachable_holdout_audit.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int)
    parser.add_argument("--holdout-samples", type=int)
    parser.add_argument("--sensitivity-samples", type=int)
    parser.add_argument("--quantile", type=float)
    parser.add_argument(
        "--parameter-contract",
        choices=("configured_nominal", "observed_execution_parameters"),
        help="Parameter contract used by the no-noise reference rollout.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def source_hashes(config_paths: tuple[Path, ...]) -> dict[str, str]:
    paths = config_paths + (
        PROJECT_ROOT / "scripts" / "audit_execution_reachable_holdout.py",
        PROJECT_ROOT / "scripts" / "calibrate_execution_reachable_set.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "execution_dynamics.py",
    )
    hashes: dict[str, str] = {}
    for path in paths:
        key = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING"
    return hashes


def _sample_rollout_error(
    base_environment: dict[str, Any],
    variant: dict[str, Any],
    *,
    rng: np.random.Generator,
    defenders: int,
    horizon_steps: int,
    parameter_contract: str,
) -> tuple[np.ndarray, np.ndarray, ExecutionParameters, dict[str, Any]]:
    actual_parameters = _parameters(base_environment, variant, rng, randomized=True)
    if parameter_contract == "observed_execution_parameters":
        nominal_parameters = actual_parameters
    else:
        nominal_parameters = _parameters(base_environment, variant, rng, randomized=False)
    action, action_queue = _sample_commands(
        rng,
        defenders=defenders,
        max_speed_mps=nominal_parameters.max_speed_mps,
        queue_length=actual_parameters.action_delay_steps,
    )
    initial_velocity = rng.normal(0.0, 0.5, size=(defenders, 3))
    positions = np.zeros((defenders, 3), dtype=np.float64)
    noises = [
        rng.uniform(
            -actual_parameters.command_noise_bound_mps,
            actual_parameters.command_noise_bound_mps,
            size=(defenders, 3),
        )
        for _ in range(horizon_steps)
    ]
    nominal_positions, _nominal_velocities, _nominal_steps = rollout_execution(
        positions,
        initial_velocity,
        action_queue,
        action,
        nominal_parameters,
        horizon_steps=horizon_steps,
    )
    actual_positions, _actual_velocities, _actual_steps = rollout_execution(
        positions,
        initial_velocity,
        action_queue,
        action,
        actual_parameters,
        horizon_steps=horizon_steps,
        noises=noises,
    )
    errors = np.max(np.linalg.norm(actual_positions - nominal_positions, axis=2), axis=1)
    base_radii = position_uncertainty_radii(actual_parameters, defenders, horizon_steps)[:, 0]
    parameters = {
        "action_delay_steps": int(actual_parameters.action_delay_steps),
        "command_noise_std_mps": float(actual_parameters.command_noise_std_mps),
        "command_noise_bound_mps": float(actual_parameters.command_noise_bound_mps),
        "velocity_time_constant_seconds": float(actual_parameters.velocity_time_constant_seconds),
        "drag_coefficient": float(actual_parameters.drag_coefficient),
        "max_speed_mps": float(actual_parameters.max_speed_mps),
        "max_acceleration_mps2": float(actual_parameters.max_acceleration_mps2),
        "mass_scale": float(actual_parameters.mass_scale),
    }
    return errors, base_radii, actual_parameters, parameters


def _domain_status(parameters: ExecutionParameters, variant: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    expected_delay = int(variant["action_delay_steps"])
    if int(parameters.action_delay_steps) != expected_delay:
        failures.append("action_delay_steps")
    expected_noise = float(variant["command_noise_std"])
    if not np.isclose(parameters.command_noise_std_mps, expected_noise, atol=1.0e-12):
        failures.append("command_noise_std")
    expected_tau = float(variant["velocity_time_constant_seconds"])
    if not np.isclose(parameters.velocity_time_constant_seconds, expected_tau, atol=1.0e-12):
        failures.append("velocity_time_constant_seconds")
    fields = (
        ("max_speed_scale_range", parameters.max_speed_mps, "max_speed_mps"),
        ("max_acceleration_scale_range", parameters.max_acceleration_mps2, "max_acceleration_mps2"),
        ("mass_scale_range", parameters.mass_scale, "mass_scale"),
        ("drag_coefficient_range", parameters.drag_coefficient, "drag_coefficient"),
    )
    for range_name, value, label in fields:
        low, high = _finite_range(variant[range_name], range_name)
        if range_name == "max_speed_scale_range":
            # The range is expressed relative to the configured defender speed.
            base_value = float(variant.get("_base_defender_max_speed_mps", 1.0))
            low_value, high_value = low * base_value, high * base_value
        elif range_name == "max_acceleration_scale_range":
            base_value = float(variant.get("_base_defender_max_acceleration_mps2", 1.0))
            low_value, high_value = low * base_value, high * base_value
        else:
            low_value, high_value = low, high
        if not low_value - 1.0e-12 <= value <= high_value + 1.0e-12:
            failures.append(label)
    return not failures, failures


def _variant_with_base_scales(base_environment: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(variant)
    agents = base_environment["agents"]
    updated["_base_defender_max_speed_mps"] = float(agents["defender_max_speed"])
    updated["_base_defender_max_acceleration_mps2"] = float(agents["defender_max_acceleration"])
    return updated


def _draw_samples(
    base_environment: dict[str, Any],
    variant: dict[str, Any],
    *,
    domain_variant: dict[str, Any] | None = None,
    samples: int,
    seed: int,
    defenders: int,
    horizon_steps: int,
    parameter_contract: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    errors = np.zeros((samples, horizon_steps), dtype=np.float64)
    base_radii = np.zeros_like(errors)
    records: list[dict[str, Any]] = []
    calibration_domain = variant if domain_variant is None else domain_variant
    for sample_index in range(samples):
        error, radius, parameters, parameter_record = _sample_rollout_error(
            base_environment,
            variant,
            rng=rng,
            defenders=defenders,
            horizon_steps=horizon_steps,
            parameter_contract=parameter_contract,
        )
        in_domain, out_of_domain_fields = _domain_status(parameters, calibration_domain)
        errors[sample_index] = error
        base_radii[sample_index] = radius
        records.append(
            {
                "sample_index": sample_index,
                "error_m_by_step": error.tolist(),
                "base_radius_m_by_step": radius.tolist(),
                "parameter_values": parameter_record,
                "in_calibration_domain": bool(in_domain),
                "out_of_calibration_fields": out_of_domain_fields,
                "parameter_contract": parameter_contract,
            }
        )
    return errors, base_radii, records


def fit_margin(
    errors: np.ndarray,
    base_radii: np.ndarray,
    *,
    quantile: float,
) -> dict[str, Any]:
    if errors.shape != base_radii.shape or errors.ndim != 2 or errors.shape[0] < 2:
        raise ValueError("errors and base_radii must be matching [samples, horizon] arrays")
    frozen_base = upper_quantile(base_radii, quantile, axis=0)
    simultaneous_ratios = np.max(errors / np.maximum(frozen_base[None, :], 1.0e-12), axis=1)
    multiplier = float(upper_quantile(simultaneous_ratios, quantile, axis=0))
    frozen_radius = frozen_base * multiplier
    runtime_ratios = np.max(errors / np.maximum(base_radii, 1.0e-12), axis=1)
    runtime_multiplier = float(upper_quantile(runtime_ratios, quantile, axis=0))
    runtime_radius = base_radii * runtime_multiplier
    return {
        "quantile": float(quantile),
        "base_radius_quantile_m_by_step": frozen_base.tolist(),
        "simultaneous_multiplier": multiplier,
        "frozen_radius_m_by_step": frozen_radius.tolist(),
        "calibration_coverage": float(np.mean(np.all(errors <= frozen_radius[None, :] + 1.0e-12, axis=1))),
        "calibration_horizon_coverage": np.mean(
            errors <= frozen_radius[None, :] + 1.0e-12, axis=0
        ).tolist(),
        "runtime_multiplier": runtime_multiplier,
        "runtime_calibration_coverage": float(
            np.mean(np.all(errors <= runtime_radius + 1.0e-12, axis=1))
        ),
    }


def evaluate_margin(
    errors: np.ndarray,
    frozen_radius: np.ndarray,
    *,
    records: list[dict[str, Any]],
    target_coverage: float,
    policy_decision: str,
    base_radii: np.ndarray | None = None,
    runtime_multiplier: float | None = None,
) -> dict[str, Any]:
    within = errors <= frozen_radius[None, :] + 1.0e-12
    simultaneous = np.all(within, axis=1)
    out_of_domain = np.asarray([not bool(record["in_calibration_domain"]) for record in records], dtype=bool)
    coverage = float(np.mean(simultaneous))
    summary = {
        "samples": int(errors.shape[0]),
        "simultaneous_coverage": coverage,
        "horizon_coverage": np.mean(within, axis=0).tolist(),
        "target_coverage": float(target_coverage),
        "coverage_pass": bool(coverage >= target_coverage),
        "in_calibration_domain_rate": float(np.mean(~out_of_domain)),
        "out_of_calibration_samples": int(np.sum(out_of_domain)),
        "policy_decision": policy_decision,
        "actionable_coverage": coverage if policy_decision == "allow" else None,
        "maximum_error_m": float(np.max(errors)),
        "maximum_normalized_error": float(np.max(errors / np.maximum(frozen_radius[None, :], 1.0e-12))),
    }
    if base_radii is not None and runtime_multiplier is not None:
        if base_radii.shape != errors.shape:
            raise ValueError("base_radii must match errors when runtime margin is evaluated")
        runtime_within = errors <= base_radii * float(runtime_multiplier) + 1.0e-12
        runtime_simultaneous = np.all(runtime_within, axis=1)
        summary.update(
            {
                "runtime_multiplier": float(runtime_multiplier),
                "runtime_simultaneous_coverage": float(np.mean(runtime_simultaneous)),
                "runtime_horizon_coverage": np.mean(runtime_within, axis=0).tolist(),
                "runtime_coverage_pass": bool(np.mean(runtime_simultaneous) >= target_coverage),
                "runtime_maximum_normalized_error": float(
                    np.max(errors / np.maximum(base_radii * float(runtime_multiplier), 1.0e-12))
                ),
            }
        )
    return summary


def _sensitivity_variant(variant: dict[str, Any], factor: str, level: float | int) -> dict[str, Any]:
    updated = copy.deepcopy(variant)
    if factor == "command_noise_std_multiplier":
        updated["command_noise_std"] = float(variant["command_noise_std"]) * float(level)
    elif factor == "velocity_time_constant_multiplier":
        updated["velocity_time_constant_seconds"] = float(variant["velocity_time_constant_seconds"]) * float(level)
    elif factor == "mass_scale_multiplier":
        updated["mass_scale"] = float(variant["mass_scale"]) * float(level)
        low, high = _finite_range(variant["mass_scale_range"], "mass_scale_range")
        updated["mass_scale_range"] = [low * float(level), high * float(level)]
    elif factor == "drag_coefficient_multiplier":
        updated["drag_coefficient"] = float(variant["drag_coefficient"]) * float(level)
        low, high = _finite_range(variant["drag_coefficient_range"], "drag_coefficient_range")
        updated["drag_coefficient_range"] = [low * float(level), high * float(level)]
    elif factor == "action_delay_steps":
        updated["action_delay_steps"] = int(level)
    else:
        raise ValueError(f"Unknown sensitivity factor: {factor}")
    return updated


def variant_matches_calibration_contract(candidate: dict[str, Any], calibration: dict[str, Any]) -> bool:
    """Require declared execution settings, not sampled values, to match exactly."""

    scalar_keys = (
        "action_delay_steps",
        "command_noise_std",
        "velocity_time_constant_seconds",
        "max_speed_scale",
        "max_acceleration_scale",
        "mass_scale",
        "drag_coefficient",
    )
    range_keys = (
        "max_speed_scale_range",
        "max_acceleration_scale_range",
        "mass_scale_range",
        "drag_coefficient_range",
    )
    for key in scalar_keys:
        if key == "action_delay_steps":
            if int(candidate[key]) != int(calibration[key]):
                return False
        elif not np.isclose(float(candidate[key]), float(calibration[key]), atol=1.0e-12):
            return False
    for key in range_keys:
        if not np.allclose(
            np.asarray(candidate[key], dtype=np.float64),
            np.asarray(calibration[key], dtype=np.float64),
            atol=1.0e-12,
        ):
            return False
    return True


def require_writer() -> Any:
    if SummaryWriter is None:
        raise RuntimeError("TensorBoard logging requires the 'tensorboard' package.")
    return SummaryWriter


def log_tensorboard(output: Path, config: dict[str, Any], summaries: dict[str, Any]) -> None:
    with require_writer()(log_dir=str(output / "tensorboard"), flush_secs=5) as writer:
        writer.add_text("Audit/config", yaml.safe_dump(config, sort_keys=False), 0)
        writer.add_text("Audit/summaries", json.dumps(summaries, indent=2), 0)
        for variant_name, summary in summaries["variants"].items():
            prefix = f"HeldOut/{variant_name}"
            writer.add_scalar(f"{prefix}/calibration_coverage", summary["calibration"]["calibration_coverage"], 0)
            writer.add_scalar(f"{prefix}/holdout_coverage", summary["holdout"]["simultaneous_coverage"], 0)
            writer.add_scalar(
                f"{prefix}/holdout_runtime_coverage",
                summary["holdout"].get("runtime_simultaneous_coverage", float("nan")),
                0,
            )
            writer.add_scalar(f"{prefix}/holdout_in_domain_rate", summary["holdout"]["in_calibration_domain_rate"], 0)
            writer.add_scalar(f"{prefix}/frozen_multiplier", summary["calibration"]["simultaneous_multiplier"], 0)
            writer.add_scalar(f"{prefix}/ood_coverage_observed", summary["out_of_calibration"]["simultaneous_coverage"], 0)
            writer.add_scalar(f"{prefix}/ood_samples_rejected", summary["out_of_calibration"]["out_of_calibration_samples"], 0)
            for factor, entries in summary["sensitivity"].items():
                for index, entry in enumerate(entries):
                    writer.add_scalar(
                        f"{prefix}/sensitivity/{factor}/coverage",
                        entry["simultaneous_coverage"],
                        index,
                    )
                    writer.add_scalar(
                        f"{prefix}/sensitivity/{factor}/in_domain",
                        entry["in_calibration_domain_rate"],
                        index,
                    )
        writer.add_hparams(
            {
                "calibration_samples_per_variant": int(config["calibration_samples_per_variant"]),
                "holdout_samples_per_variant": int(config["holdout_samples_per_variant"]),
                "sensitivity_samples_per_setting": int(config["sensitivity_samples_per_setting"]),
                "horizon_steps": int(config["horizon_steps"]),
                "target_coverage": float(config["target_coverage"]),
                "quantile": float(config["quantile"]),
                "parameter_contract": str(config.get("parameter_contract", "configured_nominal")),
            },
            {
                f"hparam/{name}/holdout_coverage": float(summary["holdout"]["simultaneous_coverage"])
                for name, summary in summaries["variants"].items()
            } | {
                f"hparam/{name}/holdout_runtime_coverage": float(
                    summary["holdout"].get("runtime_simultaneous_coverage", float("nan"))
                )
                for name, summary in summaries["variants"].items()
            },
        )


def main() -> None:
    args = parse_args()
    document = load_yaml(args.config)
    base_path = (args.config.parent / str(document["base_environment_config"])).resolve()
    execution_path = (args.config.parent / str(document["execution_audit_config"])).resolve()
    reference_path = (args.config.parent / str(document["reference_calibration_summary"])).resolve()
    base_environment = load_yaml(base_path)
    execution_document = load_yaml(execution_path)
    variants = dict(execution_document["variants"])
    selected = list(variants)
    calibration_samples = int(args.calibration_samples or document["calibration_samples_per_variant"])
    holdout_samples = int(args.holdout_samples or document["holdout_samples_per_variant"])
    sensitivity_samples = int(args.sensitivity_samples or document["sensitivity_samples_per_setting"])
    horizon_steps = int(document["horizon_steps"])
    defenders = int(document["defenders"])
    parameter_contract = str(
        args.parameter_contract or document.get("parameter_contract", "configured_nominal")
    )
    if parameter_contract not in {"configured_nominal", "observed_execution_parameters"}:
        raise ValueError("parameter_contract must be configured_nominal or observed_execution_parameters")
    quantile = float(args.quantile if args.quantile is not None else document["quantile"])
    target_coverage = float(document["target_coverage"])
    if min(calibration_samples, holdout_samples, sensitivity_samples, horizon_steps, defenders) <= 0:
        raise ValueError("sample counts, horizon_steps, and defenders must be positive")
    if not 0.0 < quantile < 1.0 or not 0.0 < target_coverage < 1.0:
        raise ValueError("quantile and target_coverage must lie in (0, 1)")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.exists() else None
    config_snapshot = {
        **document,
        "config": str(args.config.resolve()),
        "base_environment_config": str(base_path),
        "execution_audit_config": str(execution_path),
        "reference_calibration_summary": str(reference_path),
        "selected_variants": selected,
        "calibration_samples": calibration_samples,
        "holdout_samples": holdout_samples,
        "sensitivity_samples": sensitivity_samples,
        "quantile": quantile,
        "parameter_contract": parameter_contract,
        "source_hashes": source_hashes((args.config.resolve(), base_path, execution_path, reference_path)),
    }
    output.joinpath("config.yaml").write_text(yaml.safe_dump(config_snapshot, sort_keys=False), encoding="utf-8")
    calibration_records: list[dict[str, Any]] = []
    holdout_records: list[dict[str, Any]] = []
    sensitivity_records: list[dict[str, Any]] = []
    out_of_calibration_records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for variant_index, variant_name in enumerate(selected):
        variant = _variant_with_base_scales(base_environment, dict(variants[variant_name]))
        calibration_errors, calibration_base, records = _draw_samples(
            base_environment,
            variant,
            samples=calibration_samples,
            seed=int(document["seed"]) + variant_index * 100003,
            defenders=defenders,
            horizon_steps=horizon_steps,
            parameter_contract=parameter_contract,
        )
        margin = fit_margin(calibration_errors, calibration_base, quantile=quantile)
        frozen_radius = np.asarray(margin["frozen_radius_m_by_step"], dtype=np.float64)
        for record in records:
            calibration_records.append({"variant": variant_name, **record})
        holdout_errors, holdout_base, holdout = _draw_samples(
            base_environment,
            variant,
            samples=holdout_samples,
            seed=int(document["seed"]) + 500000 + variant_index * 100003,
            defenders=defenders,
            horizon_steps=horizon_steps,
            parameter_contract=parameter_contract,
        )
        for record in holdout:
            holdout_records.append({"variant": variant_name, **record})
        holdout_summary = evaluate_margin(
            holdout_errors,
            frozen_radius,
            records=holdout,
            target_coverage=target_coverage,
            policy_decision="allow",
            base_radii=holdout_base,
            runtime_multiplier=float(margin["runtime_multiplier"]),
        )
        sensitivity_summary: dict[str, list[dict[str, Any]]] = {}
        for factor_index, (factor, levels) in enumerate(dict(document["sensitivity"]).items()):
            entries: list[dict[str, Any]] = []
            for level_index, level in enumerate(levels):
                changed_variant = _sensitivity_variant(variant, factor, level)
                errors, base, setting_records = _draw_samples(
                    base_environment,
                    changed_variant,
                    domain_variant=variant,
                    samples=sensitivity_samples,
                    seed=int(document["seed"])
                    + 900000
                    + variant_index * 100003
                    + factor_index * 1009
                    + level_index * 17,
                    defenders=defenders,
                    horizon_steps=horizon_steps,
                    parameter_contract=parameter_contract,
                )
                in_domain = float(np.mean([bool(item["in_calibration_domain"]) for item in setting_records]))
                contract_in_domain = variant_matches_calibration_contract(changed_variant, variant)
                policy = "allow" if contract_in_domain else "reject_or_fallback"
                entry = {
                    "factor": factor,
                    "level": level,
                    "declared_contract_in_calibration_domain": bool(contract_in_domain),
                    **evaluate_margin(
                        errors,
                        frozen_radius,
                        records=setting_records,
                        target_coverage=target_coverage,
                        policy_decision=policy,
                        base_radii=base,
                        runtime_multiplier=float(margin["runtime_multiplier"]),
                    ),
                }
                entries.append(entry)
                sensitivity_records.append({"variant": variant_name, **entry})
            sensitivity_summary[factor] = entries
        ood_variant = _variant_with_base_scales(
            base_environment,
            {**variant, **dict(document["out_of_calibration"][variant_name])},
        )
        ood_errors, ood_base, ood = _draw_samples(
            base_environment,
            ood_variant,
            domain_variant=variant,
            samples=holdout_samples,
            seed=int(document["seed"]) + 1300000 + variant_index * 100003,
            defenders=defenders,
            horizon_steps=horizon_steps,
            parameter_contract=parameter_contract,
        )
        for record in ood:
            out_of_calibration_records.append({"variant": variant_name, **record})
        ood_summary = evaluate_margin(
            ood_errors,
            frozen_radius,
            records=ood,
            target_coverage=target_coverage,
            policy_decision="reject_or_fallback",
            base_radii=ood_base,
            runtime_multiplier=float(margin["runtime_multiplier"]),
        )
        ood_summary["declared_contract_in_calibration_domain"] = bool(
            variant_matches_calibration_contract(ood_variant, variant)
        )
        summaries[variant_name] = {
            "calibration": margin,
            "holdout": holdout_summary,
            "sensitivity": sensitivity_summary,
            "out_of_calibration": ood_summary,
            "reference_previous_calibration": None
            if reference is None
            else reference.get("variants", {}).get(variant_name),
        }
    output.joinpath("calibration_samples.jsonl").write_text(
        "".join(json.dumps(record, allow_nan=True) + "\n" for record in calibration_records), encoding="utf-8"
    )
    output.joinpath("holdout_samples.jsonl").write_text(
        "".join(json.dumps(record, allow_nan=True) + "\n" for record in holdout_records), encoding="utf-8"
    )
    output.joinpath("sensitivity.jsonl").write_text(
        "".join(json.dumps(record, allow_nan=True) + "\n" for record in sensitivity_records), encoding="utf-8"
    )
    output.joinpath("out_of_calibration.jsonl").write_text(
        "".join(json.dumps(record, allow_nan=True) + "\n" for record in out_of_calibration_records), encoding="utf-8"
    )
    result = {
        **config_snapshot,
        "decision": "empirical_held_out_margin_audit_only",
        "formal_forward_invariance": False,
        "continuous_time_proof": False,
        "out_of_calibration_policy": "reject_or_fallback",
        "variants": summaries,
    }
    output.joinpath("summary.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    log_tensorboard(output, config_snapshot, result)
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
