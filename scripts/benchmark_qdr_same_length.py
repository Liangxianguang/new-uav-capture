"""Compare QDR runtime on matched fixed-length control prefixes.

The benchmark deliberately ignores episode termination and selects the first
``steps_per_episode`` rows from both arms for the same episode index.  This
isolates per-control-step computation from the different capture/collision
length distributions of QDR-on and QDR-off.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from torch.utils.tensorboard import SummaryWriter


LATENCY_KEYS = (
    "predictor_latency_ms",
    "planner_latency_ms",
    "qdr_latency_ms",
    "safety_latency_ms",
    "total_control_latency_ms",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-dir", type=Path, action="append", required=True)
    parser.add_argument("--on-dir", type=Path, action="append", required=True)
    parser.add_argument("--steps-per-episode", type=int, default=18)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                grouped[int(row["episode_index"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["step"]))
    return dict(grouped)


def select_fixed_prefix_pairs(
    off_rows: dict[int, list[dict[str, Any]]],
    on_rows: dict[int, list[dict[str, Any]]],
    steps_per_episode: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    if steps_per_episode <= 0:
        raise ValueError("steps_per_episode must be positive.")
    common = sorted(set(off_rows).intersection(on_rows))
    selected_off: list[dict[str, Any]] = []
    selected_on: list[dict[str, Any]] = []
    retained: list[int] = []
    for episode_index in common:
        off_prefix = off_rows[episode_index][:steps_per_episode]
        on_prefix = on_rows[episode_index][:steps_per_episode]
        if len(off_prefix) < steps_per_episode or len(on_prefix) < steps_per_episode:
            continue
        selected_off.extend(off_prefix)
        selected_on.extend(on_prefix)
        retained.append(episode_index)
    if not retained:
        raise ValueError("No matched episode has the requested fixed prefix length.")
    return selected_off, selected_on, retained


def percentile(values: Iterable[float], quantile: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot compute a percentile over an empty sequence")
    return float(np.percentile(array, quantile))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        key: {
            "p50": percentile((float(row[key]) for row in rows), 50),
            "p95": percentile((float(row[key]) for row in rows), 95),
            "p99": percentile((float(row[key]) for row in rows), 99),
        }
        for key in LATENCY_KEYS
    } | {"samples": len(rows)}


def benchmark(
    off_dirs: list[Path],
    on_dirs: list[Path],
    steps_per_episode: int,
) -> dict[str, Any]:
    if len(off_dirs) != len(on_dirs):
        raise ValueError("off-dir and on-dir must have the same number of entries.")
    off_selected: list[dict[str, Any]] = []
    on_selected: list[dict[str, Any]] = []
    per_seed: list[dict[str, Any]] = []
    for off_dir, on_dir in zip(off_dirs, on_dirs):
        off_rows, on_rows = (
            load_rows(off_dir / "distributed_delayed" / "steps.jsonl"),
            load_rows(on_dir / "distributed_delayed" / "steps.jsonl"),
        )
        selected_off, selected_on, retained = select_fixed_prefix_pairs(
            off_rows, on_rows, steps_per_episode
        )
        off_selected.extend(selected_off)
        on_selected.extend(selected_on)
        per_seed.append(
            {
                "off_dir": str(off_dir.resolve()),
                "on_dir": str(on_dir.resolve()),
                "retained_episodes": len(retained),
                "off": summarize(selected_off),
                "on": summarize(selected_on),
            }
        )
    off_summary = summarize(off_selected)
    on_summary = summarize(on_selected)
    return {
        "benchmark": "qdr_same_length_prefix",
        "steps_per_episode": steps_per_episode,
        "seed_pair_count": len(per_seed),
        "total_retained_episodes": sum(item["retained_episodes"] for item in per_seed),
        "off": off_summary,
        "on": on_summary,
        "delta_on_minus_off": {
            key: {
                quantile: on_summary[key][quantile] - off_summary[key][quantile]
                for quantile in ("p50", "p95", "p99")
            }
            for key in LATENCY_KEYS
        },
        "per_seed": per_seed,
    }


def log_tensorboard(result: dict[str, Any], tensorboard_dir: Path) -> None:
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(str(tensorboard_dir)) as writer:
        writer.add_text(
            "Benchmark/config",
            json.dumps(
                {
                    "benchmark": result["benchmark"],
                    "steps_per_episode": result["steps_per_episode"],
                    "seed_pair_count": result["seed_pair_count"],
                    "total_retained_episodes": result["total_retained_episodes"],
                },
                sort_keys=True,
            ),
            0,
        )
        writer.add_scalar("Benchmark/steps_per_episode", result["steps_per_episode"], 0)
        writer.add_scalar("Benchmark/retained_episodes", result["total_retained_episodes"], 0)
        for mode in ("off", "on"):
            for key in LATENCY_KEYS:
                prefix = key.removesuffix("_latency_ms")
                for quantile in ("p50", "p95", "p99"):
                    writer.add_scalar(
                        f"{mode}/{prefix}/{quantile}_ms",
                        result[mode][key][quantile],
                        0,
                    )
        for key in LATENCY_KEYS:
            prefix = key.removesuffix("_latency_ms")
            for quantile in ("p50", "p95", "p99"):
                writer.add_scalar(
                    f"delta_on_minus_off/{prefix}/{quantile}_ms",
                    result["delta_on_minus_off"][key][quantile],
                    0,
                )
        writer.flush()


def main() -> None:
    args = parse_args()
    result = benchmark(args.off_dir, args.on_dir, args.steps_per_episode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log_tensorboard(result, args.tensorboard_dir)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
