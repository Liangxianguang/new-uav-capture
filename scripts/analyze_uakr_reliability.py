"""Audit whether UAKR uncertainty ranks closed-loop failures.

The script consumes only retained episode/step logs.  It never feeds realized
outcomes back into the controller and is therefore an offline diagnostic, not
an online safety mechanism or a threshold-selection shortcut for locked-test.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - environments without TensorBoard
    SummaryWriter = None  # type: ignore[assignment,misc]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.resolve().read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object: {path}")
            rows.append(value)
    return rows


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = labels.astype(bool)
    negative_count = int(np.sum(~positives))
    positive_count = int(np.sum(positives))
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    positive_rank_sum = float(np.sum(ranks[positives]))
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (
        positive_count * negative_count
    )


def _episode_features(run: Path) -> list[dict[str, Any]]:
    episode_path = run / "distributed_delayed" / "episodes.jsonl"
    step_path = run / "distributed_delayed" / "steps.jsonl"
    episodes = _read_jsonl(episode_path)
    steps = _read_jsonl(step_path)
    by_episode: dict[int, list[dict[str, Any]]] = {}
    for row in steps:
        episode_index = int(row["episode_index"])
        by_episode.setdefault(episode_index, []).append(row)
    output: list[dict[str, Any]] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        rows = by_episode.get(episode_index, [])
        scores = np.asarray(
            [float(row["adaptive_uncertainty_score"]) for row in rows], dtype=np.float64
        )
        if scores.size == 0 or not np.isfinite(scores).all():
            raise ValueError(f"missing finite adaptive uncertainty for episode {episode_index} in {run}")
        output.append(
            {
                "run": str(run.resolve()),
                "episode_index": episode_index,
                "mean_uncertainty": float(np.mean(scores)),
                "maximum_uncertainty": float(np.max(scores)),
                "first_uncertainty": float(scores[0]),
                "failure": not bool(episode.get("safe_capture_success", False)),
                "collision": bool(episode.get("collision", False)),
                "boundary_violation": bool(episode.get("boundary_violation", False)),
                "timeout": bool(episode.get("timeout", False)),
            }
        )
    return output


def _scene_groups(path: Path) -> dict[int, str]:
    groups: dict[int, str] = {}
    for row in _read_jsonl(path):
        episode_index = int(row["episode_index"])
        group = row.get("mirror_group_id")
        groups[episode_index] = str(group if group is not None else f"episode:{episode_index}")
    if not groups:
        raise ValueError("scene manifest is empty")
    return groups


def _assign_mirror_group_split(
    rows: list[dict[str, Any]],
    scene_manifest: Path,
    *,
    split_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign complete per-run mirror groups to calibration/confirmation."""

    episode_to_group = _scene_groups(scene_manifest)
    group_to_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        episode_index = int(row["episode_index"])
        if episode_index not in episode_to_group:
            raise ValueError(f"scene manifest is missing episode_index={episode_index}")
        key = (str(row["run"]), episode_to_group[episode_index])
        group_to_rows.setdefault(key, []).append(row)
    if len(group_to_rows) < 4:
        raise ValueError("mirror-group split requires at least four run/group groups")
    keys = sorted(group_to_rows)
    rng = np.random.default_rng(int(split_seed))
    permutation = [keys[index] for index in rng.permutation(len(keys)).tolist()]
    midpoint = max(1, len(permutation) // 2)
    calibration_keys = set(permutation[:midpoint])
    confirmation_keys = set(permutation[midpoint:])
    if not confirmation_keys:
        moved = sorted(calibration_keys)[-1]
        calibration_keys.remove(moved)
        confirmation_keys.add(moved)
    assigned: list[dict[str, Any]] = []
    for key in keys:
        split = "calibration" if key in calibration_keys else "confirmation"
        for row in group_to_rows[key]:
            assigned.append({**row, "split": split, "mirror_group_key": f"{key[0]}::{key[1]}"})
    metadata = {
        "strategy": "mirror_group_half",
        "split_seed": int(split_seed),
        "group_count": len(keys),
        "calibration_group_count": len(calibration_keys),
        "confirmation_group_count": len(confirmation_keys),
        "calibration_episode_count": sum(len(group_to_rows[key]) for key in calibration_keys),
        "confirmation_episode_count": sum(len(group_to_rows[key]) for key in confirmation_keys),
    }
    return assigned, metadata


def _assign_canonical_mirror_group_split(
    rows: list[dict[str, Any]],
    scene_manifest: Path,
    *,
    split_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign one canonical scene mirror-group split shared by every run."""

    episode_to_group = _scene_groups(scene_manifest)
    group_to_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        episode_index = int(row["episode_index"])
        if episode_index not in episode_to_group:
            raise ValueError(f"scene manifest is missing episode_index={episode_index}")
        group_to_rows.setdefault(episode_to_group[episode_index], []).append(row)
    if len(group_to_rows) < 4:
        raise ValueError("canonical mirror-group split requires at least four groups")
    groups = sorted(group_to_rows)
    rng = np.random.default_rng(int(split_seed))
    permutation = [groups[index] for index in rng.permutation(len(groups)).tolist()]
    midpoint = max(1, len(permutation) // 2)
    calibration_groups = set(permutation[:midpoint])
    confirmation_groups = set(permutation[midpoint:])
    if not confirmation_groups:
        moved = sorted(calibration_groups)[-1]
        calibration_groups.remove(moved)
        confirmation_groups.add(moved)
    assigned: list[dict[str, Any]] = []
    for group in groups:
        split = "calibration" if group in calibration_groups else "confirmation"
        for row in group_to_rows[group]:
            assigned.append({**row, "split": split, "mirror_group_key": group})
    metadata = {
        "strategy": "canonical_mirror_group_half",
        "split_seed": int(split_seed),
        "group_count": len(groups),
        "calibration_group_count": len(calibration_groups),
        "confirmation_group_count": len(confirmation_groups),
        "calibration_episode_count": sum(
            len(group_to_rows[group]) for group in calibration_groups
        ),
        "confirmation_episode_count": sum(
            len(group_to_rows[group]) for group in confirmation_groups
        ),
        "calibration_groups": sorted(calibration_groups),
        "confirmation_groups": sorted(confirmation_groups),
    }
    return assigned, metadata


def _quartiles(rows: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
    order = np.argsort(np.asarray([float(row[score_key]) for row in rows]), kind="mergesort")
    groups = np.array_split(order, 4)
    output: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        selected = [rows[int(position)] for position in group]
        output.append(
            {
                "quartile": index,
                "count": len(selected),
                "mean_score": float(np.mean([float(row[score_key]) for row in selected])),
                "failure_rate": float(np.mean([bool(row["failure"]) for row in selected])),
                "collision_rate": float(np.mean([bool(row["collision"]) for row in selected])),
                "boundary_violation_rate": float(
                    np.mean([bool(row["boundary_violation"]) for row in selected])
                ),
                "timeout_rate": float(np.mean([bool(row["timeout"]) for row in selected])),
            }
        )
    return output


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("analysis split is empty")
    result: dict[str, Any] = {"episode_count": len(rows), "score_summary": {}, "auc": {}, "quartiles": {}}
    for score_key in ("first_uncertainty", "mean_uncertainty", "maximum_uncertainty"):
        scores = np.asarray([float(row[score_key]) for row in rows], dtype=np.float64)
        result["score_summary"][score_key] = {
            "mean": float(np.mean(scores)),
            "minimum": float(np.min(scores)),
            "maximum": float(np.max(scores)),
        }
        result["auc"][score_key] = {
            label: _auc(scores, np.asarray([bool(row[label]) for row in rows], dtype=bool))
            for label in ("failure", "collision", "boundary_violation", "timeout")
        }
        result["quartiles"][score_key] = _quartiles(rows, score_key)
    return result


def analyze(
    runs: list[Path],
    *,
    scene_manifest: Path | None = None,
    split_strategy: str = "all",
    split_seed: int = 20260912,
) -> dict[str, Any]:
    rows = [row for run in runs for row in _episode_features(run)]
    if not rows:
        raise ValueError("at least one non-empty run is required")
    if split_strategy not in {"all", "mirror_group_half", "canonical_mirror_group_half"}:
        raise ValueError("split_strategy must be all, mirror_group_half, or canonical_mirror_group_half")
    result: dict[str, Any] = {
        "runs": [str(run.resolve()) for run in runs],
        "split_strategy": split_strategy,
    }
    if split_strategy in {"mirror_group_half", "canonical_mirror_group_half"}:
        if scene_manifest is None:
            raise ValueError("scene_manifest is required for a mirror-group split")
        split_function = (
            _assign_mirror_group_split
            if split_strategy == "mirror_group_half"
            else _assign_canonical_mirror_group_split
        )
        rows, split_metadata = split_function(rows, scene_manifest, split_seed=split_seed)
        result["split_metadata"] = split_metadata
        result["splits"] = {
            split: _summarize([row for row in rows if row["split"] == split])
            for split in ("calibration", "confirmation")
        }
    else:
        result.update(_summarize(rows))
    return result


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    summary = result if "auc" in result else result["splits"]["confirmation"]
    lines = [
        "# UAKR Reliability Audit",
        "",
        "Offline diagnostic from retained validation logs; no realized outcome is",
        "fed back to the controller and no locked-test data is used.",
        "",
        f"- Split strategy: `{result['split_strategy']}`",
        f"- Runs: `{len(result['runs'])}`",
        "",
        "## AUROC",
        "",
        "| Score | Failure | Collision | Boundary | Timeout |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for score_key, values in summary["auc"].items():
        values = {key: ("NaN" if not math.isfinite(float(value)) else f"{float(value):.3f}") for key, value in values.items()}
        lines.append(
            f"| `{score_key}` | {values['failure']} | {values['collision']} | "
            f"{values['boundary_violation']} | {values['timeout']} |"
        )
    for score_key, groups in summary["quartiles"].items():
        lines.extend(["", f"## {score_key} quartiles", "", "| Q | Count | Mean score | Failure | Collision | Timeout |", "| ---: | ---: | ---: | ---: | ---: | ---: |"])
        for group in groups:
            lines.append(
                f"| {group['quartile']} | {group['count']} | {group['mean_score']:.4f} | "
                f"{group['failure_rate']:.2%} | {group['collision_rate']:.2%} | {group['timeout_rate']:.2%} |"
            )
    path.resolve().write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene-manifest",
        type=Path,
        help="Scene manifest required by --split-strategy mirror_group_half.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("all", "mirror_group_half", "canonical_mirror_group_half"),
        default="all",
    )
    parser.add_argument("--split-seed", type=int, default=20260912)
    args = parser.parse_args()
    result = analyze(
        [path.resolve() for path in args.run],
        scene_manifest=args.scene_manifest,
        split_strategy=args.split_strategy,
        split_seed=args.split_seed,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8")
    _write_markdown(result, output.with_suffix(".md"))
    if SummaryWriter is not None:
        with SummaryWriter(log_dir=str(output.parent / f"{output.stem}_tensorboard")) as writer:
            summary_results = result if "auc" in result else result["splits"]["confirmation"]
            writer.add_scalar("Reliability/Episodes", summary_results["episode_count"], 0)
            for score_key, values in summary_results["auc"].items():
                for label, value in values.items():
                    if math.isfinite(float(value)):
                        writer.add_scalar(f"Reliability/AUROC/{score_key}/{label}", value, 0)
            for score_key, groups in summary_results["quartiles"].items():
                for group in groups:
                    step = int(group["quartile"])
                    writer.add_scalar(f"Reliability/Quartile/{score_key}/failure_rate", group["failure_rate"], step)
                    writer.add_scalar(f"Reliability/Quartile/{score_key}/collision_rate", group["collision_rate"], step)
            writer.flush()
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
