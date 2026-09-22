"""Run a tiny CUDA contract smoke before formal Phase86 training.

The smoke deliberately uses one calibration record and two updates.  It is a
hardware/serialization check only; its output must never be used as a policy
result.  When CUDA is unavailable the script writes a machine-readable blocked
artifact and exits with status 2 instead of silently falling back to CPU.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "current_rl_comparison_phase86_v1.yaml")
    parser.add_argument("--training-scenes", type=Path, default=ROOT / "results" / "phase86_nominal_repaired_development_calibration" / "scenes.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "current_rl_comparison_phase86_v1" / "cuda_smoke.json")
    parser.add_argument("--model-output", type=Path, default=ROOT / "models" / "_phase86_cuda_smoke_20260920")
    parser.add_argument("--updates", type=int, default=2)
    args = parser.parse_args()

    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    result = {
        "smoke": "phase86_cuda_contract",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()),
    }
    if torch.cuda.is_available():
        result["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
            }
            for index in range(torch.cuda.device_count())
        ]
    else:
        result.update(
            {
                "status": "blocked_cuda_unavailable",
                "hardware_gate_passed": False,
                "reason": "Installed PyTorch is CPU-only or no CUDA device is visible; refusing CPU fallback.",
            }
        )
        args.output.resolve().write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 2

    command = [
        sys.executable,
        str(ROOT / "scripts" / "train_mappo_ippo_baseline.py"),
        "--config",
        str(args.config.resolve()),
        "--output",
        str(args.model_output.resolve()),
        "--algorithm",
        "mappo",
        "--recurrent",
        "--seed",
        "101",
        "--device",
        "cuda",
        "--updates",
        str(args.updates),
        "--episodes-per-update",
        "1",
        "--training-scenes",
        str(args.training_scenes.resolve()),
        "--training-episodes",
        "1",
        "--max-steps",
        "20",
        "--checkpoint-interval-updates",
        "1",
        "--ppo-epochs",
        "1",
        "--minibatch-size",
        "32",
        "--torch-threads",
        "1",
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    result.update(
        {
            "status": "passed" if completed.returncode == 0 else "failed",
            "hardware_gate_passed": completed.returncode == 0,
            "command": command,
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
            "checkpoint_manifest": str((args.model_output / "checkpoint_manifest.json").resolve()),
        }
    )
    args.output.resolve().write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
