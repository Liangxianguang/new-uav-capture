"""Calibrate execution reachable-set position tubes on an independent sample set.

The calibration compares a nominal execution rollout with the same queued
commands under bounded noise and randomized execution parameters.  It is an
empirical coverage audit, not a formal reachable-set proof.
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
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    SummaryWriter = None  # type: ignore[assignment,misc]

from encirclement3d.execution_dynamics import (  # noqa: E402
    ExecutionParameters,
    position_uncertainty_radii,
    rollout_execution,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "innovation_execution_reachable_calibration.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def source_hashes(config_paths: tuple[Path, ...]) -> dict[str, str]:
    paths = config_paths + (
        PROJECT_ROOT / "scripts" / "calibrate_execution_reachable_set.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "execution_dynamics.py",
    )
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _finite_range(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must have two values")
    low, high = float(value[0]), float(value[1])
    if not np.isfinite([low, high]).all() or high < low:
        raise ValueError(f"{name} must satisfy finite low <= high")
    return low, high


def _parameters(
    base_environment: dict[str, Any],
    variant: dict[str, Any],
    rng: np.random.Generator,
    *,
    randomized: bool,
) -> ExecutionParameters:
    agents = base_environment["agents"]
    if randomized:
        speed_scale = float(rng.uniform(*_finite_range(variant["max_speed_scale_range"], "max_speed_scale_range")))
        acceleration_scale = float(
            rng.uniform(*_finite_range(variant["max_acceleration_scale_range"], "max_acceleration_scale_range"))
        )
        mass_scale = float(rng.uniform(*_finite_range(variant["mass_scale_range"], "mass_scale_range")))
        drag = float(rng.uniform(*_finite_range(variant["drag_coefficient_range"], "drag_coefficient_range")))
    else:
        speed_scale = float(variant["max_speed_scale"])
        acceleration_scale = float(variant["max_acceleration_scale"])
        mass_scale = float(variant["mass_scale"])
        drag = float(variant["drag_coefficient"])
    noise_std = float(variant["command_noise_std"]) if randomized else 0.0
    return ExecutionParameters(
        enabled=True,
        dt_seconds=float(base_environment["world"]["dt"]),
        action_delay_steps=int(variant["action_delay_steps"]),
        command_noise_std_mps=noise_std,
        command_noise_bound_mps=noise_std * float(variant["command_noise_bound_sigma"]),
        clip_command_noise=bool(variant["clip_command_noise"]),
        velocity_time_constant_seconds=float(variant["velocity_time_constant_seconds"]),
        drag_coefficient=drag,
        max_speed_mps=float(agents["defender_max_speed"]) * speed_scale,
        max_acceleration_mps2=float(agents["defender_max_acceleration"]) * acceleration_scale,
        mass_scale=max(mass_scale, 1.0e-9),
    )


def _sample_commands(
    rng: np.random.Generator,
    *,
    defenders: int,
    max_speed_mps: float,
    queue_length: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    def sample() -> np.ndarray:
        values = rng.normal(0.0, 1.0, size=(defenders, 3))
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        magnitudes = rng.uniform(0.0, max_speed_mps, size=(defenders, 1))
        return values / np.maximum(norms, 1.0e-12) * magnitudes

    action_queue = [sample() for _ in range(int(queue_length))]
    return sample(), action_queue


def calibrate_variant(
    base_environment: dict[str, Any],
    variant: dict[str, Any],
    *,
    variant_name: str,
    samples: int,
    horizon_steps: int,
    defenders: int,
    seed: int,
    quantile: float,
    target_coverage: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    errors = np.zeros((samples, horizon_steps), dtype=np.float64)
    base_radii = np.zeros_like(errors)
    records: list[dict[str, Any]] = []
    for sample_index in range(samples):
        actual_parameters = _parameters(base_environment, variant, rng, randomized=True)
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
        error_by_step = np.max(np.linalg.norm(actual_positions - nominal_positions, axis=2), axis=1)
        radius_by_step = position_uncertainty_radii(actual_parameters, defenders, horizon_steps)[:, 0]
        errors[sample_index] = error_by_step
        base_radii[sample_index] = radius_by_step
        records.append(
            {
                "variant": variant_name,
                "sample_index": sample_index,
                "error_m_by_step": error_by_step.tolist(),
                "base_radius_m_by_step": radius_by_step.tolist(),
                "ratio_by_step": (error_by_step / np.maximum(radius_by_step, 1.0e-12)).tolist(),
                "actual_mass_scale": float(actual_parameters.mass_scale),
                "actual_drag_coefficient": float(actual_parameters.drag_coefficient),
                "actual_max_speed_mps": float(actual_parameters.max_speed_mps),
                "actual_max_acceleration_mps2": float(actual_parameters.max_acceleration_mps2),
            }
        )
    quantile_errors = upper_quantile(errors, quantile, axis=0)
    quantile_base = upper_quantile(base_radii, quantile, axis=0)
    ratios = errors / np.maximum(base_radii, 1.0e-12)
    quantile_ratios = upper_quantile(ratios, quantile, axis=0)
    simultaneous_ratios = np.max(errors / np.maximum(quantile_base[None, :], 1.0e-12), axis=1)
    simultaneous_multiplier = float(upper_quantile(simultaneous_ratios, quantile, axis=0))
    simultaneous_radius = quantile_base * simultaneous_multiplier
    empirical_base_coverage = np.mean(np.all(errors <= base_radii + 1.0e-12, axis=1))
    calibrated_coverage = np.mean(np.all(errors <= simultaneous_radius[None, :] + 1.0e-12, axis=1))
    summary = {
        "variant": variant_name,
        "samples": samples,
        "horizon_steps": horizon_steps,
        "quantile": quantile,
        "target_coverage": target_coverage,
        "empirical_base_tube_coverage": float(empirical_base_coverage),
        "empirical_calibrated_tube_coverage": float(calibrated_coverage),
        "base_tube_passes_target": bool(empirical_base_coverage >= target_coverage),
        "calibrated_tube_passes_target": bool(calibrated_coverage >= target_coverage),
        "error_quantile_m_by_step": quantile_errors.tolist(),
        "base_radius_quantile_m_by_step": quantile_base.tolist(),
        "required_multiplier_quantile_by_step": quantile_ratios.tolist(),
        "frozen_base_radius_m_by_step": quantile_base.tolist(),
        "simultaneous_multiplier": simultaneous_multiplier,
        "simultaneous_calibrated_radius_m_by_step": simultaneous_radius.tolist(),
        "maximum_calibrated_radius_m": float(np.max(quantile_errors)),
        "maximum_required_multiplier": float(np.max(quantile_ratios)),
    }
    return summary, records


def require_writer() -> Any:
    if SummaryWriter is None:
        raise RuntimeError("TensorBoard logging requires the 'tensorboard' package.")
    return SummaryWriter


def upper_quantile(values: np.ndarray, quantile: float, *, axis: int = 0) -> np.ndarray:
    """Return an order statistic whose empirical coverage is at least quantile."""

    ordered = np.sort(np.asarray(values, dtype=np.float64), axis=axis)
    sample_count = ordered.shape[axis]
    index = min(sample_count - 1, max(0, int(np.ceil(float(quantile) * sample_count)) - 1))
    return np.take(ordered, index, axis=axis)


def log_tensorboard(output: Path, config: dict[str, Any], summaries: dict[str, Any]) -> None:
    with require_writer()(log_dir=str(output / "tensorboard"), flush_secs=5) as writer:
        writer.add_text("Calibration/config", yaml.safe_dump(config, sort_keys=False), 0)
        writer.add_text("Calibration/summaries", json.dumps(summaries, indent=2), 0)
        for variant_name, summary in summaries.items():
            prefix = f"Calibration/{variant_name}"
            writer.add_scalar(f"{prefix}/base_tube_coverage", summary["empirical_base_tube_coverage"], 0)
            writer.add_scalar(f"{prefix}/calibrated_tube_coverage", summary["empirical_calibrated_tube_coverage"], 0)
            writer.add_scalar(f"{prefix}/maximum_calibrated_radius_m", summary["maximum_calibrated_radius_m"], 0)
            writer.add_scalar(f"{prefix}/maximum_required_multiplier", summary["maximum_required_multiplier"], 0)
            writer.add_scalar(f"{prefix}/simultaneous_multiplier", summary["simultaneous_multiplier"], 0)
            for step, value in enumerate(summary["error_quantile_m_by_step"], start=1):
                writer.add_scalar(f"{prefix}/q{int(summary['quantile'] * 100)}_error_m/step_{step}", value, 0)
            for step, value in enumerate(summary["required_multiplier_quantile_by_step"], start=1):
                writer.add_scalar(f"{prefix}/q{int(summary['quantile'] * 100)}_multiplier/step_{step}", value, 0)
        writer.add_hparams(
            {
                "samples_per_variant": int(config["samples_per_variant"]),
                "horizon_steps": int(config["horizon_steps"]),
                "target_coverage": float(config["target_coverage"]),
                "quantile": float(config["quantile"]),
            },
            {
                f"hparam/{name}/calibrated_coverage": float(summary["empirical_calibrated_tube_coverage"])
                for name, summary in summaries.items()
            },
        )


def main() -> None:
    args = parse_args()
    document = load_yaml(args.config)
    base_path = (args.config.parent / str(document["base_environment_config"])).resolve()
    execution_path = (args.config.parent / str(document["execution_audit_config"])).resolve()
    base_environment = load_yaml(base_path)
    execution_document = load_yaml(execution_path)
    variants = dict(execution_document["variants"])
    selected = list(document["variants"])
    samples = int(args.samples or document["samples_per_variant"])
    horizon_steps = int(document["horizon_steps"])
    defenders = int(document["defenders"])
    quantile = float(document["quantile"])
    target_coverage = float(document["target_coverage"])
    if not 0.0 < quantile < 1.0 or not 0.0 < target_coverage < 1.0:
        raise ValueError("quantile and target_coverage must lie in (0, 1)")
    if samples <= 0 or horizon_steps <= 0 or defenders <= 0:
        raise ValueError("samples, horizon_steps, and defenders must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        **document,
        "config": str(args.config.resolve()),
        "base_environment_config": str(base_path),
        "execution_audit_config": str(execution_path),
        "selected_variants": selected,
        "samples": samples,
        "source_hashes": source_hashes((args.config.resolve(), base_path, execution_path)),
    }
    output.joinpath("config.yaml").write_text(yaml.safe_dump(config_snapshot, sort_keys=False), encoding="utf-8")
    summaries: dict[str, Any] = {}
    all_records: list[dict[str, Any]] = []
    for index, variant_name in enumerate(selected):
        if variant_name not in variants:
            raise ValueError(f"Unknown execution variant: {variant_name}")
        summary, records = calibrate_variant(
            base_environment,
            dict(variants[variant_name]),
            variant_name=variant_name,
            samples=samples,
            horizon_steps=horizon_steps,
            defenders=defenders,
            seed=int(document["seed"]) + index * 100003,
            quantile=quantile,
            target_coverage=target_coverage,
        )
        summaries[variant_name] = summary
        all_records.extend(records)
    output.joinpath("samples.jsonl").write_text(
        "".join(json.dumps(record, allow_nan=True) + "\n" for record in all_records),
        encoding="utf-8",
    )
    result = {
        **config_snapshot,
        "decision": "empirical_reachable_set_calibration_only",
        "formal_forward_invariance": False,
        "variants": summaries,
    }
    output.joinpath("summary.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    log_tensorboard(output, config_snapshot, summaries)
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
