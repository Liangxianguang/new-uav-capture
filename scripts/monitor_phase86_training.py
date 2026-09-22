"""Quiet 10-minute monitor for a long-running Phase86 PPO job.

The monitor owns checkpoint validation and emits no periodic status output.
It only appends an event when a scheduled checkpoint is independently
validated or when the training/validation process fails or stops unexpectedly.
No reward, optimizer, or hyperparameter is changed by this process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp_utc": utc_now(), **event}, allow_nan=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def training_running(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def run_validation(
    *,
    python_executable: Path,
    config: Path,
    checkpoint: Path,
    scenes: Path,
    output: Path,
    algorithm: str,
    recurrent: bool,
    device: str,
    use_cbf: bool,
) -> tuple[int, str, str]:
    command = [
        str(python_executable),
        str(ROOT / "scripts" / "evaluate_mappo_ippo_phase87.py"),
        "--config",
        str(config.resolve()),
        "--checkpoint",
        str(checkpoint.resolve()),
        "--scenes",
        str(scenes.resolve()),
        "--output",
        str(output.resolve()),
        "--algorithm",
        algorithm,
        "--episodes",
        "300",
        "--max-steps",
        "250",
        "--device",
        device,
    ]
    if recurrent:
        command.append("--recurrent")
    if use_cbf:
        command.append("--use-cbf")
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    return completed.returncode, completed.stdout[-4000:], completed.stderr[-4000:]


def scheduled_checkpoints(manifest_path: Path, completed_updates: int) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        return []
    manifest = read_json(manifest_path)
    results: list[dict[str, Any]] = []
    for item in manifest.get("checkpoints", []):
        if not isinstance(item, dict):
            continue
        update = int(item.get("update", 0))
        kind = str(item.get("kind", ""))
        if kind == "interval" and update > 0 and update % 500 == 0 and update <= completed_updates:
            results.append(item)
    return sorted(results, key=lambda item: int(item["update"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-pid", type=int)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-scenes", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--algorithm", choices=("mappo", "ippo"), required=True)
    parser.add_argument("--recurrent", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--period-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.period_seconds != 600:
        raise ValueError("The formal monitor period must be exactly 600 seconds.")

    training_output = args.training_output.resolve()
    manifest_path = training_output / "checkpoint_manifest.json"
    progress_path = training_output / "progress.json"
    validation_root = training_output / "independent_validation"
    event_path = training_output / "monitor_events.jsonl"
    handled_updates: set[int] = set()
    if event_path.is_file():
        for line in event_path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "checkpoint_validation" and event.get("status") == "passed":
                handled_updates.add(int(event["update"]))

    next_check = time.monotonic()
    while True:
        delay = max(0.0, next_check - time.monotonic())
        if delay:
            time.sleep(delay)
        # Schedule the next inspection only after this inspection and any
        # checkpoint validation finish.  This guarantees a full, silent
        # 600-second interval between scheduled checks.
        next_check = time.monotonic() + args.period_seconds

        progress: dict[str, Any] = {}
        if progress_path.is_file():
            try:
                progress = read_json(progress_path)
            except (OSError, json.JSONDecodeError):
                progress = {}
        completed_updates = int(progress.get("updates_completed", 0))
        targets = scheduled_checkpoints(manifest_path, completed_updates)
        for checkpoint_item in targets:
            update = int(checkpoint_item["update"])
            if update in handled_updates:
                continue
            checkpoint = training_output / str(checkpoint_item["path"])
            if not checkpoint.is_file():
                append_event(
                    event_path,
                    {
                        "event": "checkpoint_validation",
                        "status": "failed",
                        "update": update,
                        "reason": "checkpoint_missing",
                        "checkpoint": str(checkpoint),
                    },
                )
                handled_updates.add(update)
                continue
            update_root = validation_root / f"update_{update:06d}"
            raw_output = update_root / "raw"
            cbf_output = update_root / "eval_cbf"
            raw_code, raw_stdout, raw_stderr = run_validation(
                python_executable=args.python.resolve(),
                config=args.config,
                checkpoint=checkpoint,
                scenes=args.validation_scenes,
                output=raw_output,
                algorithm=args.algorithm,
                recurrent=args.recurrent,
                device=args.device,
                use_cbf=False,
            )
            cbf_code, cbf_stdout, cbf_stderr = run_validation(
                python_executable=args.python.resolve(),
                config=args.config,
                checkpoint=checkpoint,
                scenes=args.validation_scenes,
                output=cbf_output,
                algorithm=args.algorithm,
                recurrent=args.recurrent,
                device=args.device,
                use_cbf=True,
            )
            raw_result = raw_output / "evaluation.json"
            cbf_result = cbf_output / "evaluation.json"
            status = "passed" if raw_code == 0 and cbf_code == 0 and raw_result.is_file() and cbf_result.is_file() else "failed"
            event: dict[str, Any] = {
                "event": "checkpoint_validation",
                "status": status,
                "update": update,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": str(checkpoint_item.get("sha256", "")),
                "raw_artifact": str(raw_result),
                "eval_cbf_artifact": str(cbf_result),
                "raw_returncode": raw_code,
                "eval_cbf_returncode": cbf_code,
            }
            if status == "passed":
                raw = read_json(raw_result)
                cbf = read_json(cbf_result)
                event["raw_metrics"] = {
                    key: raw.get(key)
                    for key in ("safe_capture_rate", "collision_rate", "boundary_violation_rate", "target_invalid_episode_rate", "timeout_rate")
                }
                event["eval_cbf_metrics"] = {
                    key: cbf.get(key)
                    for key in ("safe_capture_rate", "collision_rate", "boundary_violation_rate", "target_invalid_episode_rate", "timeout_rate")
                }
                handled_updates.add(update)
            else:
                event["raw_stdout_tail"] = raw_stdout
                event["raw_stderr_tail"] = raw_stderr
                event["eval_cbf_stdout_tail"] = cbf_stdout
                event["eval_cbf_stderr_tail"] = cbf_stderr
            append_event(event_path, event)
            if status != "passed":
                return

        target_updates = int(progress.get("updates_target", 0))
        if target_updates > 0 and completed_updates >= target_updates:
            append_event(
                event_path,
                {
                    "event": "training_complete",
                    "status": "complete",
                    "updates": completed_updates,
                    "training_output": str(training_output),
                },
            )
            return
        if args.training_pid is not None and not training_running(args.training_pid):
            append_event(
                event_path,
                {
                    "event": "training_stopped",
                    "status": "failed",
                    "updates": completed_updates,
                    "updates_target": target_updates,
                    "training_pid": args.training_pid,
                },
            )
            return


if __name__ == "__main__":
    main()
