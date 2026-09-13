"""Aggregate the independent Phase 59 repair calibration by scene block.

The script is intentionally separate from the Phase 58 gate audit.  Phase 59
is a diagnostic repair study: it estimates the queue-aware local-CBF effect
and audits whether a requested K=4/K=8 candidate budget was actually realized
by the frozen predictor checkpoint.  It does not open confirmation or locked
test data and it never promotes a repair automatically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.tensorboard import SummaryWriter

try:
    from scripts.aggregate_closed_loop_seed_results import (
        LATENCY_FIELDS,
        OUTCOME_METRICS,
        RunArtifact,
        latency_value,
        load_groups,
        metric_summary,
        metric_value,
    )
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from aggregate_closed_loop_seed_results import (
        LATENCY_FIELDS,
        OUTCOME_METRICS,
        RunArtifact,
        latency_value,
        load_groups,
        metric_summary,
        metric_value,
    )


METHODS = (
    "R0_phase59_qdr_baseline",
    "R1_phase59_queue_cbf_k1",
    "R2_phase59_queue_cbf_k4",
    "R3_phase59_queue_cbf_k8",
)
REPAIR_METRICS = (
    "safe_capture_success",
    "collision",
    "boundary_violation",
    "timeout",
    "capture_time_seconds",
    "min_clearance_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True, metavar="NAME=GLOB")
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scene_blocks(path: Path) -> dict[str, list[int]]:
    blocks: dict[str, list[int]] = {}
    for row in read_jsonl(path):
        blocks.setdefault(str(row["scene_block"]), []).append(int(row["episode_index"]))
    return {name: sorted(ids) for name, ids in blocks.items()}


def latency_summary(
    artifacts: list[RunArtifact], method: str, episode_ids: list[int]
) -> dict[str, dict[str, float]]:
    wanted = set(episode_ids)
    values: dict[str, list[float]] = {field: [] for field in LATENCY_FIELDS}
    for artifact in artifacts:
        for row in read_jsonl(artifact.path / method / "steps.jsonl"):
            if int(row["episode_index"]) not in wanted:
                continue
            for field in LATENCY_FIELDS:
                value = latency_value(row, field)
                if np.isfinite(value):
                    values[field].append(float(value))
    return {
        field: {
            "samples": len(items),
            "p50": float(np.quantile(items, 0.50)),
            "p95": float(np.quantile(items, 0.95)),
            "p99": float(np.quantile(items, 0.99)),
        }
        for field, items in values.items()
    }


def grouped_subset_metric_matrix(
    artifacts: list[RunArtifact], method: str, metric: str, episode_ids: list[int]
) -> tuple[list[int], np.ndarray]:
    """Return mirror-group means for only the requested scene-block episodes."""

    wanted = set(int(value) for value in episode_ids)
    matrices: list[np.ndarray] = []
    group_order: list[int] | None = None
    reference_groups_by_episode: dict[int, int] | None = None
    for artifact in artifacts:
        if method not in artifact.methods:
            raise ValueError(f"Method {method!r} is missing from {artifact.path}")
        rows = [
            row
            for row in artifact.methods[method]
            if int(row["episode_index"]) in wanted
        ]
        rows.sort(key=lambda row: int(row["episode_index"]))
        if [int(row["episode_index"]) for row in rows] != sorted(wanted):
            raise ValueError(f"{artifact.path} is missing requested block episodes")
        matrices.append(np.asarray([metric_value(row, metric) for row in rows], dtype=np.float64))
        scene_rows = read_jsonl(artifact.path / "scenes.jsonl")
        by_episode = {
            int(row["episode_index"]): int(row["mirror_group_id"])
            for row in scene_rows
        }
        current_groups = [by_episode[int(row["episode_index"])] for row in rows]
        if group_order is None:
            group_order = list(dict.fromkeys(current_groups))
            reference_groups_by_episode = {
                int(row["episode_index"]): int(group_id)
                for row, group_id in zip(rows, current_groups)
            }
        elif reference_groups_by_episode is None or current_groups != [
            reference_groups_by_episode[int(row["episode_index"])] for row in rows
        ]:
            raise ValueError("Mirror-group ordering differs across repair artifacts")
    assert group_order is not None
    assert reference_groups_by_episode is not None
    matrix = np.stack(matrices, axis=0)
    grouped = np.full((matrix.shape[0], len(group_order)), np.nan, dtype=np.float64)
    for group_index, group_id in enumerate(group_order):
        columns = [
            index
            for index, episode_id in enumerate(sorted(wanted))
            if reference_groups_by_episode[episode_id] == group_id
        ]
        values = matrix[:, columns]
        counts = np.isfinite(values).sum(axis=1)
        grouped[:, group_index] = np.divide(
            np.nansum(values, axis=1),
            counts,
            out=np.full(matrix.shape[0], np.nan, dtype=np.float64),
            where=counts > 0,
        )
    return group_order, grouped


def budget_summary(
    artifacts: list[RunArtifact], method: str, episode_ids: list[int]
) -> dict[str, Any]:
    wanted = set(episode_ids)
    requested: list[float] = []
    realized: list[float] = []
    mismatch_steps = 0
    for artifact in artifacts:
        for row in read_jsonl(artifact.path / method / "steps.jsonl"):
            if int(row["episode_index"]) not in wanted:
                continue
            requested_value = float(row.get("candidate_budget_requested", 0.0))
            realized_value = float(row.get("candidate_count", 0.0))
            if realized_value <= 0.0:
                continue
            requested.append(requested_value)
            realized.append(realized_value)
            if realized_value < requested_value:
                mismatch_steps += 1
    if not realized:
        return {
            "requested_min": None,
            "requested_max": None,
            "realized_min": None,
            "realized_max": None,
            "realized_rate": None,
            "mismatch_steps": 0,
            "samples": 0,
        }
    return {
        "requested_min": int(min(requested)),
        "requested_max": int(max(requested)),
        "realized_min": int(min(realized)),
        "realized_max": int(max(realized)),
        "realized_rate": float(np.mean(np.asarray(realized) >= np.asarray(requested))),
        "mismatch_steps": int(mismatch_steps),
        "samples": len(realized),
    }


def paired_delta(
    artifacts: list[RunArtifact],
    candidate: str,
    reference: str,
    metric: str,
    episode_ids: list[int],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    candidate_groups, candidate_matrix = grouped_subset_metric_matrix(
        artifacts, candidate, metric, episode_ids
    )
    reference_groups, reference_matrix = grouped_subset_metric_matrix(
        artifacts, reference, metric, episode_ids
    )
    if candidate_groups != reference_groups:
        raise ValueError("paired repair methods do not have identical mirror groups")
    result = metric_summary(candidate_matrix - reference_matrix, rng, bootstrap_samples)
    return {
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "mean_delta_candidate_minus_reference": result["mean"],
        "paired_bootstrap_95_ci": result["bootstrap_95_ci"],
    }


def summarize_block(
    artifacts: list[RunArtifact],
    method: str,
    episode_ids: list[int],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in REPAIR_METRICS:
        _, matrix = grouped_subset_metric_matrix(artifacts, method, metric, episode_ids)
        metrics[metric] = metric_summary(matrix, rng, bootstrap_samples)
    return {
        "training_seeds": [artifact.seed for artifact in artifacts],
        "episodes_per_seed": len(episode_ids),
        "mirror_groups": len(episode_ids) // 2,
        "episode_metrics": metrics,
        "candidate_budget_audit": budget_summary(artifacts, method, episode_ids),
        "pooled_step_latency_ms": latency_summary(artifacts, method, episode_ids),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def write_tensorboard(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(str(output_dir)) as writer:
        writer.add_text("Phase59/config", json.dumps(payload["config"], sort_keys=True), 0)
        writer.add_text("Phase59/scene_manifest_sha256", payload["scene_manifest_sha256"], 0)
        for block, block_payload in payload["blocks"].items():
            for method, method_payload in block_payload["methods"].items():
                tag = method.replace("/", "_")
                metrics = method_payload["episode_metrics"]
                for metric in ("safe_capture_success", "collision", "boundary_violation", "timeout"):
                    writer.add_scalar(f"Outcome/{tag}/{block}/{metric}", metrics[metric]["mean"], 0)
                audit = method_payload["candidate_budget_audit"]
                if audit["realized_rate"] is not None:
                    writer.add_scalar(
                        f"CandidateBudget/{tag}/{block}/realized_rate",
                        audit["realized_rate"],
                        0,
                    )
                for field, values in method_payload["pooled_step_latency_ms"].items():
                    component = field.removesuffix("_latency_ms")
                    for quantile in ("p50", "p95", "p99"):
                        writer.add_scalar(
                            f"Latency/{tag}/{block}/{component}/{quantile}_ms",
                            values[quantile],
                            0,
                        )
        writer.flush()


def main() -> None:
    args = parse_args()
    groups = load_groups([args.group])
    artifacts = next(iter(groups.values()))
    available = set.intersection(*(set(artifact.methods) for artifact in artifacts))
    methods = [method for method in METHODS if method in available]
    if len(methods) != len(METHODS):
        raise ValueError(f"missing Phase 59 methods; available={sorted(available)}")
    blocks = scene_blocks(args.scenes.resolve())
    rng = np.random.default_rng(args.bootstrap_seed)
    block_payload: dict[str, Any] = {}
    all_episode_ids = sorted(episode_id for ids in blocks.values() for episode_id in ids)

    def comparison_payload(episode_ids: list[int]) -> dict[str, Any]:
        comparisons: dict[str, Any] = {}
        for candidate, reference in (
            ("R1_phase59_queue_cbf_k1", "R0_phase59_qdr_baseline"),
            ("R2_phase59_queue_cbf_k4", "R1_phase59_queue_cbf_k1"),
            ("R3_phase59_queue_cbf_k8", "R1_phase59_queue_cbf_k1"),
        ):
            comparisons[f"{candidate}_vs_{reference}"] = {
                metric: paired_delta(
                    artifacts,
                    candidate,
                    reference,
                    metric,
                    episode_ids,
                    rng,
                    args.bootstrap_samples,
                )
                for metric in ("safe_capture_success", "collision", "timeout", "min_clearance_m")
            }
        return comparisons

    for block, episode_ids in blocks.items():
        block_methods = {
            method: summarize_block(artifacts, method, episode_ids, rng, args.bootstrap_samples)
            for method in methods
        }
        block_payload[block] = {
            "episodes": len(episode_ids),
            "mirror_groups": len(episode_ids) // 2,
            "methods": block_methods,
            "paired_comparisons": comparison_payload(episode_ids),
        }
    overall_methods = {
        method: summarize_block(artifacts, method, all_episode_ids, rng, args.bootstrap_samples)
        for method in methods
    }
    payload = {
        "schema_version": "phase59-repair-by-block-v1",
        "scene_manifest_sha256": artifacts[0].scene_hash,
        "config": {
            "scenes": str(args.scenes.resolve()),
            "group": args.group,
            "training_seeds": [artifact.seed for artifact in artifacts],
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_unit": "matched predictor seed and frozen-scene mirror_group",
            "candidate_budget_audit": "requested versus realized candidate_count from raw step logs",
            "local_cbf_claim": "empirical_filter_only",
            "robust_clbf_qp_proof": "not_claimed",
        },
        "overall": {
            "episodes": len(all_episode_ids),
            "mirror_groups": len(all_episode_ids) // 2,
            "methods": overall_methods,
            "paired_comparisons": comparison_payload(all_episode_ids),
        },
        "blocks": block_payload,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    write_tensorboard(payload, args.tensorboard_dir.resolve())
    print(json.dumps({"status": "complete", "output": str(output), "blocks": list(block_payload)}))


if __name__ == "__main__":
    main()
