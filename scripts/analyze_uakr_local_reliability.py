"""Audit UAKR scores against local step-level safety diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from analyze_uakr_reliability import _auc, _read_jsonl, _scene_groups  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment,misc]


def _split_groups(groups: list[str], split_seed: int) -> tuple[set[str], set[str]]:
    if len(groups) < 4:
        raise ValueError("canonical local reliability split requires at least four groups")
    rng = np.random.default_rng(int(split_seed))
    permutation = [groups[index] for index in rng.permutation(len(groups)).tolist()]
    midpoint = max(1, len(permutation) // 2)
    calibration = set(permutation[:midpoint])
    confirmation = set(permutation[midpoint:])
    if not confirmation:
        moved = sorted(calibration)[-1]
        calibration.remove(moved)
        confirmation.add(moved)
    return calibration, confirmation


def _step_rows(
    runs: list[Path],
    scene_manifest: Path,
    *,
    include_queue_prefix_risk: bool = False,
) -> list[dict[str, Any]]:
    episode_to_group = _scene_groups(scene_manifest)
    output: list[dict[str, Any]] = []
    for run in runs:
        path = run / "distributed_delayed" / "steps.jsonl"
        for row in _read_jsonl(path):
            episode_index = int(row["episode_index"])
            if episode_index not in episode_to_group:
                raise ValueError(f"scene manifest is missing episode_index={episode_index}")
            if "adaptive_uncertainty_score" not in row:
                raise ValueError("step log is missing adaptive_uncertainty_score")
            item = {
                "run": str(run.resolve()),
                "episode_index": episode_index,
                "mirror_group_id": episode_to_group[episode_index],
                "uncertainty": float(row["adaptive_uncertainty_score"]),
                "prediction_residual": float(row.get("adaptive_prediction_residual_m", 0.0)),
                "next_state_violation": not bool(row.get("safety_independent_next_state_safe", True)),
                "current_state_violation": not bool(row.get("safety_independent_current_state_safe", True)),
            }
            if include_queue_prefix_risk:
                if "adaptive_queue_prefix_risk" not in row:
                    raise ValueError("step log is missing adaptive_queue_prefix_risk")
                item["queue_prefix_risk"] = float(row["adaptive_queue_prefix_risk"])
            output.append(item)
    if not output:
        raise ValueError("no step rows found")
    return output


def _summary(rows: list[dict[str, Any]], score_key: str, label_key: str) -> dict[str, Any]:
    if not rows:
        raise ValueError("reliability subset is empty")
    scores = np.asarray([float(row[score_key]) for row in rows], dtype=np.float64)
    labels = np.asarray([bool(row[label_key]) for row in rows], dtype=bool)
    return {
        "step_count": len(rows),
        "positive_count": int(np.sum(labels)),
        "positive_rate": float(np.mean(labels)),
        "auroc": _auc(scores, labels),
        "score_mean": float(np.mean(scores)),
        "score_minimum": float(np.min(scores)),
        "score_maximum": float(np.max(scores)),
    }


def analyze(
    runs: list[Path],
    scene_manifest: Path,
    *,
    split_seed: int = 20260912,
    include_queue_prefix_risk: bool = False,
) -> dict[str, Any]:
    rows = _step_rows(
        runs,
        scene_manifest,
        include_queue_prefix_risk=include_queue_prefix_risk,
    )
    groups = sorted({str(row["mirror_group_id"]) for row in rows})
    calibration_groups, confirmation_groups = _split_groups(groups, split_seed)
    result: dict[str, Any] = {
        "runs": [str(run.resolve()) for run in runs],
        "scene_manifest": str(scene_manifest.resolve()),
        "split_strategy": "canonical_mirror_group_half",
        "split_seed": int(split_seed),
        "group_count": len(groups),
        "calibration_group_count": len(calibration_groups),
        "confirmation_group_count": len(confirmation_groups),
        "score_keys": ["uncertainty", "prediction_residual"]
        + (["queue_prefix_risk"] if include_queue_prefix_risk else []),
        "labels": ["next_state_violation", "current_state_violation"],
        "splits": {},
    }
    for split, selected_groups in (
        ("calibration", calibration_groups),
        ("confirmation", confirmation_groups),
    ):
        selected = [row for row in rows if row["mirror_group_id"] in selected_groups]
        result["splits"][split] = {
            score: {
                label: _summary(selected, score, label)
                for label in result["labels"]
            }
            for score in result["score_keys"]
        }
    return result


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# UAKR Local Step-Level Reliability Audit",
        "",
        "This is an offline diagnostic. Realized safety outcomes are never fed",
        "back into the controller, and no locked-test data is used.",
        "",
        f"- Mirror groups: `{result['group_count']}`",
        f"- Calibration / confirmation groups: `{result['calibration_group_count']}` / `{result['confirmation_group_count']}`",
        "",
        "| Split | Score | Label | Steps | Positives | Rate | AUROC |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for split, scores in result["splits"].items():
        for score, labels in scores.items():
            for label, values in labels.items():
                auc = "NaN" if not math.isfinite(float(values["auroc"])) else f"{values['auroc']:.3f}"
                lines.append(
                    f"| {split} | `{score}` | `{label}` | {values['step_count']} | "
                    f"{values['positive_count']} | {values['positive_rate']:.2%} | {auc} |"
                )
    lines.extend(
        [
            "",
            "The AUROC is a ranking diagnostic only. In particular, a value near",
            "0.5 means the score cannot be used as a reliable local safety trigger.",
        ]
    )
    path.resolve().write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20260912)
    parser.add_argument(
        "--include-queue-prefix-risk",
        action="store_true",
        help="also audit the public queue-prefix risk score when present in step logs",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        [path.resolve() for path in args.run],
        args.scene_manifest.resolve(),
        split_seed=args.split_seed,
        include_queue_prefix_risk=args.include_queue_prefix_risk,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    _write_markdown(result, output.with_suffix(".md"))
    if SummaryWriter is not None:
        with SummaryWriter(log_dir=str(output.parent / f"{output.stem}_tensorboard")) as writer:
            writer.add_scalar("LocalReliability/GroupCount", result["group_count"], 0)
            for split, scores in result["splits"].items():
                for score, labels in scores.items():
                    for label, values in labels.items():
                        if math.isfinite(float(values["auroc"])):
                            writer.add_scalar(f"LocalReliability/AUROC/{split}/{score}/{label}", values["auroc"], 0)
                        writer.add_scalar(f"LocalReliability/PositiveRate/{split}/{label}", values["positive_rate"], 0)
            writer.flush()
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
