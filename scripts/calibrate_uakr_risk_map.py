"""Fit a validation-only monotone risk map for UAKR.

The map is fitted on complete mirror groups assigned to a calibration subset.
The confirmation subset is reported only after fitting and is never used to
change the artifact.  This is a development calibration tool, not a safety
certificate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_uakr_reliability import (  # noqa: E402
    _auc,
    _assign_canonical_mirror_group_split,
    _episode_features,
)
from encirclement3d.uncertainty_calibration import MonotoneRiskCalibration  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - environments without TensorBoard
    SummaryWriter = None  # type: ignore[assignment,misc]


def _split_rows(
    runs: list[Path],
    scene_manifest: Path,
    *,
    split_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = [row for run in runs for row in _episode_features(run)]
    assigned, metadata = _assign_canonical_mirror_group_split(rows, scene_manifest, split_seed=split_seed)
    return (
        [row for row in assigned if row["split"] == "calibration"],
        [row for row in assigned if row["split"] == "confirmation"],
        metadata,
    )


def _summary(
    rows: list[dict[str, Any]],
    calibration: MonotoneRiskCalibration,
    *,
    score_key: str,
    label: str,
) -> dict[str, Any]:
    scores = np.asarray([float(row[score_key]) for row in rows], dtype=np.float64)
    labels = np.asarray([bool(row[label]) for row in rows], dtype=np.float64)
    predicted = calibration.predict(scores)
    return {
        "episode_count": len(rows),
        "positive_count": int(np.sum(labels)),
        "auc": _auc(scores, labels),
        "brier_score": float(np.mean((predicted - labels) ** 2)),
        "mean_predicted_risk": float(np.mean(predicted)),
        "observed_rate": float(np.mean(labels)),
        "predicted_risk_minimum": float(np.min(predicted)),
        "predicted_risk_maximum": float(np.max(predicted)),
    }


def calibrate(
    runs: list[Path],
    scene_manifest: Path,
    *,
    split_seed: int = 20260912,
    score_key: str = "first_uncertainty",
    label: str = "failure",
    low_target_risk: float = 0.02,
    high_target_risk: float = 0.10,
) -> dict[str, Any]:
    if not 0.0 <= float(low_target_risk) < float(high_target_risk) <= 1.0:
        raise ValueError("target risks must satisfy 0 <= low < high <= 1")
    calibration_rows, confirmation_rows, split_metadata = _split_rows(
        runs, scene_manifest, split_seed=split_seed
    )
    calibration_scores = np.asarray([float(row[score_key]) for row in calibration_rows], dtype=np.float64)
    calibration_labels = np.asarray([bool(row[label]) for row in calibration_rows], dtype=np.float64)
    risk_map = MonotoneRiskCalibration.fit(calibration_scores, calibration_labels, label=label)
    low_threshold = risk_map.threshold_for_risk(low_target_risk)
    high_threshold = risk_map.threshold_for_risk(high_target_risk)
    threshold_valid = bool(low_threshold < high_threshold)
    return {
        "split_strategy": "canonical_mirror_group_half",
        "split_metadata": split_metadata,
        "score_key": score_key,
        "label": label,
        "target_risk": {"low": float(low_target_risk), "high": float(high_target_risk)},
        "policy_thresholds": {
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
            "valid_order": threshold_valid,
        },
        "risk_map": risk_map.as_dict(),
        "calibration": _summary(calibration_rows, risk_map, score_key=score_key, label=label),
        "confirmation": _summary(confirmation_rows, risk_map, score_key=score_key, label=label),
        "runs": [str(run.resolve()) for run in runs],
        "scene_manifest": str(scene_manifest.resolve()),
        "locked_test_used": False,
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    calibration = result["calibration"]
    confirmation = result["confirmation"]
    thresholds = result["policy_thresholds"]
    lines = [
        "# UAKR Risk Calibration Artifact",
        "",
        "Validation-only monotone map; no locked-test data is used.",
        "",
        f"- Score: `{result['score_key']}`",
        f"- Label: `{result['label']}`",
        f"- Mirror-group split: `{result['split_metadata']['calibration_group_count']}` calibration / "
        f"`{result['split_metadata']['confirmation_group_count']}` confirmation groups",
        "",
        "| Split | Episodes | Positive | AUROC | Brier | Observed rate | Mean predicted risk |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, values in (("Calibration", calibration), ("Confirmation", confirmation)):
        auc = "NaN" if not math.isfinite(float(values["auc"])) else f"{values['auc']:.3f}"
        lines.append(
            f"| {name} | {values['episode_count']} | {values['positive_count']} | {auc} | "
            f"{values['brier_score']:.4f} | {values['observed_rate']:.2%} | "
            f"{values['mean_predicted_risk']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Derived UAKR thresholds",
            "",
            f"- low/medium threshold: `{thresholds['low_threshold']:.6f}`",
            f"- medium/high threshold: `{thresholds['high_threshold']:.6f}`",
            f"- valid order: `{thresholds['valid_order']}`",
            "",
            "The thresholds are proposed budget-policy values, not guarantees.",
            "They may be used online only after the confirmation gate is passed",
            "and the artifact is frozen.",
        ]
    )
    path.resolve().write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260912)
    parser.add_argument(
        "--score-key",
        choices=("first_uncertainty", "mean_uncertainty", "maximum_uncertainty"),
        default="first_uncertainty",
    )
    parser.add_argument(
        "--label",
        choices=("failure", "collision", "boundary_violation", "timeout"),
        default="failure",
    )
    parser.add_argument("--low-target-risk", type=float, default=0.02)
    parser.add_argument("--high-target-risk", type=float, default=0.10)
    args = parser.parse_args()
    result = calibrate(
        [path.resolve() for path in args.run],
        args.scene_manifest.resolve(),
        split_seed=args.split_seed,
        score_key=args.score_key,
        label=args.label,
        low_target_risk=args.low_target_risk,
        high_target_risk=args.high_target_risk,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    _write_markdown(result, output.with_suffix(".md"))
    if SummaryWriter is not None:
        with SummaryWriter(log_dir=str(output.parent / f"{output.stem}_tensorboard")) as writer:
            writer.add_scalar("Calibration/CalibrationBrier", result["calibration"]["brier_score"], 0)
            writer.add_scalar("Calibration/ConfirmationBrier", result["confirmation"]["brier_score"], 0)
            writer.add_scalar("Calibration/CalibrationAUROC", result["calibration"]["auc"], 0)
            writer.add_scalar("Calibration/ConfirmationAUROC", result["confirmation"]["auc"], 0)
            writer.add_scalar("Policy/LowThreshold", result["policy_thresholds"]["low_threshold"], 0)
            writer.add_scalar("Policy/HighThreshold", result["policy_thresholds"]["high_threshold"], 0)
            writer.add_scalar("Policy/ThresholdOrderValid", int(result["policy_thresholds"]["valid_order"]), 0)
            writer.flush()
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
