"""Audit empirical execution-tube coverage across multiple horizons.

Each horizon has an independent calibration split and an independent holdout
split.  The output is an empirical coverage audit only; it is not a
continuous-time reachability or forward-invariance proof.
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
for path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_execution_reachable_holdout import (  # noqa: E402
    _draw_samples,
    _variant_with_base_scales,
    fit_margin,
    load_yaml,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phase10_reachable_tube_horizons.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int)
    parser.add_argument("--holdout-samples", type=int)
    parser.add_argument("--parameter-contract", choices=("configured_nominal", "observed_execution_parameters"))
    return parser.parse_args()


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson interval for a Bernoulli proportion."""

    if trials <= 0:
        raise ValueError("trials must be positive")
    count = max(0, min(int(successes), int(trials)))
    n = float(trials)
    p = float(count) / n
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    half_width = float(z) / denominator * np.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, center - half_width), min(1.0, center + half_width)


def source_hashes(paths: tuple[Path, ...]) -> dict[str, str]:
    tracked = paths + (
        PROJECT_ROOT / "scripts" / "audit_execution_reachable_horizons.py",
        PROJECT_ROOT / "scripts" / "audit_execution_reachable_holdout.py",
        PROJECT_ROOT / "scripts" / "calibrate_execution_reachable_set.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "execution_dynamics.py",
    )
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tracked
    }


def coverage_summary(
    errors: np.ndarray,
    frozen_radius: np.ndarray,
    *,
    base_radii: np.ndarray,
    runtime_multiplier: float,
    target_coverage: float,
) -> dict[str, Any]:
    frozen_within = np.all(errors <= frozen_radius[None, :] + 1.0e-12, axis=1)
    runtime_within = np.all(errors <= base_radii * float(runtime_multiplier) + 1.0e-12, axis=1)
    frozen_successes = int(np.sum(frozen_within))
    runtime_successes = int(np.sum(runtime_within))
    frozen_interval = wilson_interval(frozen_successes, len(frozen_within))
    runtime_interval = wilson_interval(runtime_successes, len(runtime_within))
    return {
        "samples": int(len(frozen_within)),
        "target_coverage": float(target_coverage),
        "frozen_coverage": float(np.mean(frozen_within)),
        "frozen_successes": frozen_successes,
        "frozen_wilson_95": list(frozen_interval),
        "frozen_coverage_pass": bool(np.mean(frozen_within) >= target_coverage),
        "runtime_multiplier": float(runtime_multiplier),
        "runtime_coverage": float(np.mean(runtime_within)),
        "runtime_successes": runtime_successes,
        "runtime_wilson_95": list(runtime_interval),
        "runtime_coverage_pass": bool(np.mean(runtime_within) >= target_coverage),
        "frozen_horizon_coverage": np.mean(
            errors <= frozen_radius[None, :] + 1.0e-12, axis=0
        ).tolist(),
        "runtime_horizon_coverage": np.mean(
            errors <= base_radii * float(runtime_multiplier) + 1.0e-12, axis=0
        ).tolist(),
        "maximum_normalized_error": float(
            np.max(errors / np.maximum(frozen_radius[None, :], 1.0e-12))
        ),
    }


def main() -> None:
    args = parse_args()
    document = load_yaml(args.config)
    base_path = (args.config.parent / str(document["base_environment_config"])).resolve()
    execution_path = (args.config.parent / str(document["execution_audit_config"])).resolve()
    base_environment = load_yaml(base_path)
    execution_document = load_yaml(execution_path)
    variants = dict(execution_document["variants"])
    selected_variants = list(document.get("variants", variants))
    horizons = [int(value) for value in document.get("horizon_steps", [1, 2, 3, 5])]
    if not horizons or any(value <= 0 for value in horizons):
        raise ValueError("horizon_steps must contain positive integers")
    calibration_samples = int(args.calibration_samples or document["calibration_samples_per_variant"])
    holdout_samples = int(args.holdout_samples or document["holdout_samples_per_variant"])
    defenders = int(document["defenders"])
    target_coverage = float(document["target_coverage"])
    quantile = float(document["quantile"])
    parameter_contract = str(
        args.parameter_contract or document.get("parameter_contract", "observed_execution_parameters")
    )
    if parameter_contract not in {"configured_nominal", "observed_execution_parameters"}:
        raise ValueError("unsupported parameter contract")
    if min(calibration_samples, holdout_samples, defenders) <= 0:
        raise ValueError("sample counts and defenders must be positive")
    if not 0.0 < target_coverage < 1.0 or not 0.0 < quantile < 1.0:
        raise ValueError("target_coverage and quantile must lie in (0, 1)")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    config_snapshot = {
        **document,
        "config": str(args.config.resolve()),
        "base_environment_config": str(base_path),
        "execution_audit_config": str(execution_path),
        "selected_variants": selected_variants,
        "horizon_steps": horizons,
        "calibration_samples": calibration_samples,
        "holdout_samples": holdout_samples,
        "parameter_contract": parameter_contract,
        "source_hashes": source_hashes((args.config.resolve(), base_path, execution_path)),
    }
    output.joinpath("config.yaml").write_text(yaml.safe_dump(config_snapshot, sort_keys=False), encoding="utf-8")

    summaries: dict[str, Any] = {}
    for variant_index, variant_name in enumerate(selected_variants):
        if variant_name not in variants:
            raise ValueError(f"Unknown execution variant: {variant_name}")
        variant = _variant_with_base_scales(base_environment, dict(variants[variant_name]))
        horizon_summaries: dict[str, Any] = {}
        for horizon_index, horizon in enumerate(horizons):
            seed_base = int(document["seed"]) + variant_index * 100003 + horizon_index * 700001
            calibration_errors, calibration_base, _calibration_records = _draw_samples(
                base_environment,
                variant,
                samples=calibration_samples,
                seed=seed_base,
                defenders=defenders,
                horizon_steps=horizon,
                parameter_contract=parameter_contract,
            )
            margin = fit_margin(calibration_errors, calibration_base, quantile=quantile)
            holdout_errors, holdout_base, _holdout_records = _draw_samples(
                base_environment,
                variant,
                samples=holdout_samples,
                seed=seed_base + 300000,
                defenders=defenders,
                horizon_steps=horizon,
                parameter_contract=parameter_contract,
            )
            frozen_radius = np.asarray(margin["frozen_radius_m_by_step"], dtype=np.float64)
            horizon_summaries[str(horizon)] = {
                "horizon_steps": horizon,
                "quantile": quantile,
                "calibration": margin,
                "holdout": coverage_summary(
                    holdout_errors,
                    frozen_radius,
                    base_radii=holdout_base,
                    runtime_multiplier=float(margin["runtime_multiplier"]),
                    target_coverage=target_coverage,
                ),
            }
        summaries[variant_name] = horizon_summaries

    result = {
        **config_snapshot,
        "decision": "empirical_multi_horizon_reachable_tube_audit_only",
        "formal_forward_invariance": False,
        "continuous_time_proof": False,
        "variants": summaries,
    }
    output.joinpath("summary.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
