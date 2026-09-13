"""Run the independent Phase 58 QDR time-index and double-delay checker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.qdr_formalization import audit_qdr_time_index  # noqa: E402

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment,misc]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--defenders", type=int, default=4)
    parser.add_argument("--horizon-steps", type=int, default=8)
    parser.add_argument("--dt-seconds", type=float, default=0.1)
    parser.add_argument("--queue-lengths", type=int, nargs="+", default=[0, 2, 4, 8, 10])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    if args.defenders <= 0 or args.horizon_steps <= 0 or any(value < 0 for value in args.queue_lengths):
        raise ValueError("defenders/horizon must be positive and queue lengths non-negative")
    rng = np.random.default_rng(int(args.seed))
    positions = rng.normal(size=(args.defenders, 3))
    velocities = rng.normal(size=(args.defenders, 3))
    audits = []
    for queue_length in args.queue_lengths:
        queue = rng.normal(size=(queue_length, args.defenders, 3))
        suffix = rng.normal(size=(args.horizon_steps, args.defenders, 3))
        normal = audit_qdr_time_index(
            positions,
            velocities,
            queue,
            suffix,
            dt_seconds=float(args.dt_seconds),
        ).as_dict()
        intentionally_double_delayed = audit_qdr_time_index(
            positions,
            velocities,
            queue,
            suffix,
            dt_seconds=float(args.dt_seconds),
            reported_terminal_index_steps=int(queue_length + 2 * args.horizon_steps),
        ).as_dict()
        audits.append(
            {
                "queue_length": int(queue_length),
                "normal": normal,
                "intentional_double_delay_probe": intentionally_double_delayed,
                "double_delay_probe_rejected": bool(
                    intentionally_double_delayed["double_delay_detected"]
                ),
            }
        )
    result = {
        "experiment_name": "phase58_qdr_time_index",
        "seed": int(args.seed),
        "defenders": int(args.defenders),
        "horizon_steps": int(args.horizon_steps),
        "dt_seconds": float(args.dt_seconds),
        "queue_lengths": [int(value) for value in args.queue_lengths],
        "audits": audits,
        "overall_pass": bool(
            all(bool(item["normal"]["passed"]) for item in audits)
            and all(bool(item["double_delay_probe_rejected"]) for item in audits)
        ),
        "claim_boundary": "time-index and queue-composition equivalence only; not a safety certificate",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (output / "effective_config.json").write_text(
        json.dumps({key: result[key] for key in ("experiment_name", "seed", "defenders", "horizon_steps", "dt_seconds", "queue_lengths")}, indent=2) + "\n",
        encoding="utf-8",
    )
    if SummaryWriter is not None:
        with SummaryWriter(log_dir=str(output / "tensorboard")) as writer:
            writer.add_text("Protocol/claim_boundary", result["claim_boundary"], 0)
            writer.add_scalar("Gate/time_index_equivalence", float(result["overall_pass"]), 0)
            for item in audits:
                step = int(item["queue_length"])
                normal = item["normal"]
                writer.add_scalar("QDR/time_index_error_steps", normal["terminal_time_index_error_steps"], step)
                writer.add_scalar("QDR/max_position_error_m", normal["max_position_error_m"], step)
                writer.add_scalar("QDR/max_velocity_error_mps", normal["max_velocity_error_mps"], step)
                writer.add_scalar("QDR/first_controllable_step", normal["first_controllable_step"], step)
                writer.add_scalar("QDR/double_delay_probe_rejected", float(item["double_delay_probe_rejected"]), step)
            writer.flush()
    print(json.dumps(result, indent=2))
    if not result["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
