"""Run and aggregate a process-isolated Phase 57 runtime benchmark.

Each method and predictor checkpoint is evaluated in its own child process and
child processes are executed sequentially.  The benchmark reports both the
full closed-loop stream and a matched fixed-prefix stream, so runtime is not
confounded by either concurrent CPU load or different termination lengths.
The benchmark is diagnostic: it does not tune a method or open locked-test.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
from torch.utils.tensorboard import SummaryWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METHODS = (
    "B0_current_state_delayed_mpc",
    "B1_qdr_mpc",
    "B2_fixed_tube_mpc",
    "B3_queue_aware_tube_mpc",
    "B4_synchronous_distributed_mpc",
    "B5_asynchronous_distributed_mpc",
    "B6_qdr_asynchronous_mpc",
    "B7_fixed_k8_qdr",
)
LATENCY_KEYS = (
    "predictor_latency_ms",
    "planner_latency_ms",
    "qdr_latency_ms",
    "safety_latency_ms",
    "total_control_latency_ms",
)
OUTCOME_KEYS = (
    "safe_capture_success",
    "capture_event",
    "collision",
    "boundary_violation",
    "timeout",
    "capture_time_seconds",
    "min_clearance_m",
)
DEFAULT_SEEDS = (727201, 727202, 727203)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--environment-config", type=Path, required=True)
    parser.add_argument("--mpc-config", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--prefix-steps", type=int, default=18)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sampling-steps", type=int, default=8)
    parser.add_argument("--sampling-seed", type=int, default=745102)
    parser.add_argument("--projection-iterations", type=int, default=4)
    parser.add_argument("--prediction-refresh-interval-steps", type=int, default=1)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--safety-layer", choices=("none", "local_cbf"), default="local_cbf")
    parser.add_argument("--decision", default="validation_selection")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse a child result when its summary.json and steps.jsonl already exist.",
    )
    return parser.parse_args()


def load_steps(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                grouped[int(row["episode_index"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["step"]))
    return dict(grouped)


def load_episodes(path: Path) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                episode_index = int(row["episode_index"])
                if episode_index in result:
                    raise ValueError(f"duplicate episode index in {path}: {episode_index}")
                result[episode_index] = row
    return result


def percentile(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot compute percentile over an empty sequence")
    return float(np.percentile(array, quantile))


def summarize_latency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        key: {
            "p50": percentile((float(row[key]) for row in rows), 50),
            "p95": percentile((float(row[key]) for row in rows), 95),
            "p99": percentile((float(row[key]) for row in rows), 99),
        }
        for key in LATENCY_KEYS
    } | {"samples": len(rows)}


def outcome_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        return float("nan")
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    numeric = float(value)
    return numeric if np.isfinite(numeric) else float("nan")


def summarize_outcomes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in OUTCOME_KEYS:
        values = np.asarray([outcome_value(row, key) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result[key] = {
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "finite_samples": int(finite.size),
        }
    return result


def fixed_prefix(rows_by_episode: dict[int, list[dict[str, Any]]], prefix_steps: int) -> tuple[list[dict[str, Any]], list[int]]:
    if prefix_steps <= 0:
        raise ValueError("prefix_steps must be positive")
    selected: list[dict[str, Any]] = []
    retained: list[int] = []
    for episode_index in sorted(rows_by_episode):
        rows = rows_by_episode[episode_index]
        if len(rows) < prefix_steps:
            continue
        selected.extend(rows[:prefix_steps])
        retained.append(episode_index)
    if not retained:
        raise ValueError("no episode contains the requested fixed prefix")
    return selected, retained


def child_output_dir(output_root: Path, seed: int, method: str) -> Path:
    return output_root / f"seed{seed}" / method


def checkpoint_for(root: Path, seed: int) -> Path:
    candidates = (
        root / f"phase16_gru_both_seed{seed}" / "checkpoint.pt",
        root / f"seed{seed}" / "checkpoint.pt",
        root / f"{seed}" / "checkpoint.pt",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"could not find checkpoint for seed {seed}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def run_child(args: argparse.Namespace, seed: int, method: str, output: Path) -> list[str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_for(args.checkpoint_root.resolve(), seed)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "evaluate_s4_closed_loop.py"),
        "--scenes",
        str(args.scenes.resolve()),
        "--protocol",
        str(args.protocol.resolve()),
        "--environment-config",
        str(args.environment_config.resolve()),
        "--mpc-config",
        str(args.mpc_config.resolve()),
        "--checkpoint",
        str(checkpoint.resolve()),
        "--output-dir",
        str(output.resolve()),
        "--candidate-source",
        "checkpoint",
        "--methods",
        method,
        "--num-samples",
        str(args.num_samples),
        "--sampling-steps",
        str(args.sampling_steps),
        "--sampling-seed",
        str(args.sampling_seed),
        "--projection-iterations",
        str(args.projection_iterations),
        "--prediction-refresh-interval-steps",
        str(args.prediction_refresh_interval_steps),
        "--device",
        "cpu",
        "--torch-num-threads",
        str(args.torch_num_threads),
        "--torch-num-interop-threads",
        str(args.torch_num_interop_threads),
        "--safety-layer",
        args.safety_layer,
        "--decision",
        args.decision,
    ]
    if args.episodes is not None:
        command.extend(("--episodes", str(args.episodes)))
    result_dir = output / method
    if args.skip_existing and (result_dir / "summary.json").exists() and (result_dir / "steps.jsonl").exists():
        return command
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty child output: {output}")
    log_path = output.parent / f"{output.name}.isolated_child.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + json.dumps(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    if result.returncode != 0:
        raise RuntimeError(f"isolated child failed for {seed}/{method}; see {log_path}")
    return command


def aggregate_child(
    output_root: Path,
    seeds: list[int],
    method: str,
    prefix_steps: int,
) -> dict[str, Any]:
    full_rows: list[dict[str, Any]] = []
    prefix_rows: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        child = child_output_dir(output_root, seed, method) / method
        rows_by_episode = load_steps(child / "steps.jsonl")
        episodes = load_episodes(child / "episodes.jsonl")
        full = [row for rows in rows_by_episode.values() for row in rows]
        prefix, retained = fixed_prefix(rows_by_episode, prefix_steps)
        full_rows.extend(full)
        prefix_rows.extend(prefix)
        per_seed.append(
            {
                "seed": seed,
                "episodes": len(rows_by_episode),
                "prefix_retained_episodes": len(retained),
                "outcomes": summarize_outcomes(list(episodes.values())),
                "full": summarize_latency(full),
                "matched_prefix": summarize_latency(prefix),
            }
        )
    return {
        "method": method,
        "sequential_process_isolation": True,
        "prefix_steps": prefix_steps,
        "full": summarize_latency(full_rows),
        "matched_prefix": summarize_latency(prefix_rows),
        "outcomes": summarize_outcomes(
            [
                row
                for seed in seeds
                for row in load_episodes(
                    child_output_dir(output_root, seed, method) / method / "episodes.jsonl"
                ).values()
            ]
        ),
        "per_seed": per_seed,
    }


def log_tensorboard(result: dict[str, Any], tensorboard_dir: Path) -> None:
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(str(tensorboard_dir)) as writer:
        writer.add_text("Benchmark/config", json.dumps(result["config"], sort_keys=True), 0)
        writer.add_scalar("Benchmark/prefix_steps", result["config"]["prefix_steps"], 0)
        writer.add_scalar("Benchmark/seed_count", len(result["config"]["seeds"]), 0)
        for method, summary in result["methods"].items():
            method_tag = method.replace("/", "_")
            for view in ("full", "matched_prefix"):
                for key in LATENCY_KEYS:
                    component = key.removesuffix("_latency_ms")
                    for quantile in ("p50", "p95", "p99"):
                        writer.add_scalar(
                            f"RuntimeIsolated/{method_tag}/{view}/{component}/{quantile}_ms",
                            summary[view][key][quantile],
                            0,
                        )
        writer.flush()


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.prefix_steps <= 0:
        raise ValueError("prefix_steps must be positive")
    if not args.seeds:
        raise ValueError("at least one seed is required")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    commands: dict[str, dict[str, list[str]]] = {}
    for seed in args.seeds:
        for method in args.methods:
            output = child_output_dir(output_root, int(seed), method)
            command = run_child(args, int(seed), method, output)
            commands.setdefault(str(seed), {})[method] = command
            print(json.dumps({"status": "child_complete", "seed": seed, "method": method}), flush=True)
    methods = {
        method: aggregate_child(output_root, [int(seed) for seed in args.seeds], method, args.prefix_steps)
        for method in args.methods
    }
    return {
        "schema_version": "phase57-isolated-runtime-v1",
        "benchmark": "phase57_process_isolated_runtime",
        "config": {
            "scenes": str(args.scenes.resolve()),
            "protocol": str(args.protocol.resolve()),
            "environment_config": str(args.environment_config.resolve()),
            "mpc_config": str(args.mpc_config.resolve()),
            "checkpoint_root": str(args.checkpoint_root.resolve()),
            "output_root": str(output_root),
            "seeds": [int(seed) for seed in args.seeds],
            "methods": list(args.methods),
            "episodes": args.episodes,
            "prefix_steps": int(args.prefix_steps),
            "num_samples": int(args.num_samples),
            "sampling_steps": int(args.sampling_steps),
            "sampling_seed": int(args.sampling_seed),
            "projection_iterations": int(args.projection_iterations),
            "prediction_refresh_interval_steps": int(args.prediction_refresh_interval_steps),
            "torch_num_threads": int(args.torch_num_threads),
            "torch_num_interop_threads": int(args.torch_num_interop_threads),
            "safety_layer": args.safety_layer,
            "decision": args.decision,
        },
        "commands": commands,
        "methods": methods,
    }


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    output = args.output_root.resolve() / "aggregate.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log_tensorboard(result, args.tensorboard_dir.resolve())
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
