"""Aggregate Phase 58 calibration results by frozen scene block.

The aggregate is descriptive and audit-oriented.  It keeps upper/lower mirror
episodes paired within each mirror group, then performs a hierarchical
bootstrap over predictor seeds and mirror groups.  No confirmation or locked
test data are opened by this script.
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
        load_groups,
        metric_summary,
        metric_value,
        latency_value,
    )
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from aggregate_closed_loop_seed_results import (
        LATENCY_FIELDS,
        OUTCOME_METRICS,
        RunArtifact,
        load_groups,
        metric_summary,
        metric_value,
        latency_value,
    )


METHODS = (
    "M0_current_state_delayed_mpc",
    "M1_known_delay_delayed_mpc",
    "M2_fixed_tube_mpc",
    "M3_queue_aware_tube_mpc",
    "M4_synchronous_distributed_mpc",
    "M5_asynchronous_distributed_mpc",
    "M6_qdr_synchronous_mpc",
    "M7_qdr_asynchronous_mpc",
    "M8_fixed_k8_qdr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        required=True,
        metavar="NAME=GLOB",
        help="Calibration run directories, for example phase58=results\\phase58_calibration_seed*_m0_m8.",
    )
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
    for scene in read_jsonl(path):
        blocks.setdefault(str(scene["scene_block"]), []).append(int(scene["episode_index"]))
    if not blocks:
        raise ValueError("scene manifest has no blocks")
    return {name: sorted(ids) for name, ids in blocks.items()}


def episode_matrix(
    artifacts: list[RunArtifact], method: str, metric: str, episode_ids: list[int]
) -> tuple[list[int], np.ndarray]:
    matrix = np.full((len(artifacts), len(episode_ids)), np.nan, dtype=np.float64)
    wanted = set(episode_ids)
    for seed_index, artifact in enumerate(artifacts):
        rows = {int(row["episode_index"]): row for row in artifact.methods[method]}
        if not wanted.issubset(rows):
            missing = sorted(wanted.difference(rows))
            raise ValueError(f"{artifact.path} is missing episodes {missing[:5]}")
        for episode_index, column in ((value, index) for index, value in enumerate(episode_ids)):
            matrix[seed_index, column] = metric_value(rows[episode_index], metric)
    return episode_ids, matrix


def grouped_matrix(
    artifacts: list[RunArtifact], method: str, metric: str, episode_ids: list[int]
) -> tuple[list[int], np.ndarray]:
    ids, matrix = episode_matrix(artifacts, method, metric, episode_ids)
    scenes = {int(row["episode_index"]): int(row["mirror_group_id"]) for row in read_jsonl(artifacts[0].path / "scenes.jsonl")}
    group_ids = list(dict.fromkeys(scenes[episode_id] for episode_id in ids))
    grouped = np.full((matrix.shape[0], len(group_ids)), np.nan, dtype=np.float64)
    for group_index, group_id in enumerate(group_ids):
        columns = [index for index, episode_id in enumerate(ids) if scenes[episode_id] == group_id]
        values = matrix[:, columns]
        counts = np.isfinite(values).sum(axis=1)
        grouped[:, group_index] = np.divide(
            np.nansum(values, axis=1),
            counts,
            out=np.full(matrix.shape[0], np.nan, dtype=np.float64),
            where=counts > 0,
        )
    return group_ids, grouped


def latency_summary(artifacts: list[RunArtifact], method: str, episode_ids: list[int]) -> dict[str, dict[str, float]]:
    wanted = set(episode_ids)
    values_by_field: dict[str, list[float]] = {field: [] for field in LATENCY_FIELDS}
    for artifact in artifacts:
        for row in artifact.steps[method]:
            if int(row["episode_index"]) not in wanted:
                continue
            for field in LATENCY_FIELDS:
                value = latency_value(row, field)
                if np.isfinite(value):
                    values_by_field[field].append(value)
    return {
        field: {
            "samples": len(values),
            "p50": float(np.quantile(values, 0.50)),
            "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
        }
        for field, values in values_by_field.items()
    }


def summarize_block(
    artifacts: list[RunArtifact],
    method: str,
    episode_ids: list[int],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> dict[str, Any]:
    group_ids, _ = grouped_matrix(artifacts, method, "safe_capture_success", episode_ids)
    metrics: dict[str, Any] = {}
    for metric in OUTCOME_METRICS:
        _, matrix = grouped_matrix(artifacts, method, metric, episode_ids)
        metrics[metric] = metric_summary(matrix, rng, bootstrap_samples)
    return {
        "training_seeds": [artifact.seed for artifact in artifacts],
        "episodes_per_seed": len(episode_ids),
        "mirror_groups": len(group_ids),
        "episode_metrics": metrics,
        "pooled_step_latency_ms": latency_summary(artifacts, method, episode_ids),
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
    candidate_groups, candidate_matrix = grouped_matrix(artifacts, candidate, metric, episode_ids)
    reference_groups, reference_matrix = grouped_matrix(artifacts, reference, metric, episode_ids)
    if candidate_groups != reference_groups:
        raise ValueError("paired methods do not have identical mirror groups")
    summary = metric_summary(candidate_matrix - reference_matrix, rng, bootstrap_samples)
    return {
        "candidate": candidate,
        "reference": reference,
        "metric": metric,
        "mean_delta_candidate_minus_reference": summary["mean"],
        "paired_bootstrap_95_ci": summary["bootstrap_95_ci"],
    }


def write_tensorboard(payload: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(str(output_dir)) as writer:
        writer.add_text("RuntimeContract/aggregate_config", json.dumps(payload["config"], sort_keys=True), 0)
        writer.add_text("RuntimeContract/scene_manifest_sha256", payload["scene_manifest_sha256"], 0)
        for block, block_payload in payload["blocks"].items():
            for method, method_payload in block_payload["methods"].items():
                tag = method.replace("/", "_")
                metrics = method_payload["episode_metrics"]
                for metric in ("safe_capture_success", "collision", "boundary_violation", "timeout"):
                    writer.add_scalar(f"Outcome/{tag}/{block}/{metric}", metrics[metric]["mean"], 0)
                for field, percentiles in method_payload["pooled_step_latency_ms"].items():
                    component = field.removesuffix("_latency_ms")
                    for quantile in ("p50", "p95", "p99"):
                        writer.add_scalar(
                            f"Latency/{tag}/{block}/{component}/{quantile}_ms",
                            percentiles[quantile],
                            0,
                        )
        writer.flush()


def json_safe(value: Any) -> Any:
    """Replace unavailable numeric summaries with JSON null."""

    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    groups = load_groups([args.group])
    artifacts = next(iter(groups.values()))
    available = set.intersection(*(set(artifact.methods) for artifact in artifacts))
    methods = [method for method in METHODS if method in available]
    if not methods:
        raise ValueError("no Phase 58 methods found in calibration artifacts")
    blocks = scene_blocks(args.scenes.resolve())
    rng = np.random.default_rng(args.bootstrap_seed)
    block_payload: dict[str, Any] = {}
    comparisons = {
        "M6_vs_M0": ("M6_qdr_synchronous_mpc", "M0_current_state_delayed_mpc"),
        "M7_vs_M5": ("M7_qdr_asynchronous_mpc", "M5_asynchronous_distributed_mpc"),
        "M7_vs_M6": ("M7_qdr_asynchronous_mpc", "M6_qdr_synchronous_mpc"),
        "M5_vs_M4": ("M5_asynchronous_distributed_mpc", "M4_synchronous_distributed_mpc"),
        "M3_vs_M2": ("M3_queue_aware_tube_mpc", "M2_fixed_tube_mpc"),
        "M8_vs_M6": ("M8_fixed_k8_qdr", "M6_qdr_synchronous_mpc"),
    }
    for block, episode_ids in blocks.items():
        block_methods = {
            method: summarize_block(artifacts, method, episode_ids, rng, args.bootstrap_samples)
            for method in methods
        }
        block_comparisons: dict[str, dict[str, Any]] = {}
        for name, (candidate, reference) in comparisons.items():
            if candidate not in methods or reference not in methods:
                continue
            block_comparisons[name] = {
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
        block_payload[block] = {
            "episodes": len(episode_ids),
            "mirror_groups": len({int(row["mirror_group_id"]) for row in read_jsonl(args.scenes) if int(row["episode_index"]) in set(episode_ids)}),
            "methods": block_methods,
            "paired_comparisons": block_comparisons,
        }
    payload = {
        "schema_version": "phase58-calibration-by-block-v1",
        "scene_manifest_sha256": artifacts[0].scene_hash,
        "config": {
            "scenes": str(args.scenes.resolve()),
            "group": args.group,
            "training_seeds": [artifact.seed for artifact in artifacts],
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_unit": "matched predictor seed and mirror group",
            "local_cbf_claim": "empirical_filter_only",
            "robust_clbf_qp_proof": "not_claimed",
        },
        "blocks": block_payload,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    write_tensorboard(payload, args.tensorboard_dir.resolve())
    print(json.dumps({"status": "complete", "output": str(output), "blocks": list(block_payload)}, indent=2))


if __name__ == "__main__":
    main()
