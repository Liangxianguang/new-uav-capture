"""Audit the opt-in Phase 62 queue-token/ACK transition contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.execution_dynamics import (  # noqa: E402
    QueueToken,
    apply_command_authority_with_token,
    commit_delayed_command_with_token,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment,misc]


MODES = ("immutable", "replace_nonexecuting", "flush_pending")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "phase62_queue_token_transition_audit.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _queue(queue_length: int, defenders: int) -> list[np.ndarray]:
    return [
        np.full((defenders, 3), float(index + 1), dtype=np.float64)
        for index in range(queue_length)
    ]


def _same_queue(left: list[np.ndarray], right: list[np.ndarray]) -> bool:
    return len(left) == len(right) and all(np.array_equal(a, b) for a, b in zip(left, right))


def _check(cases: list[dict[str, Any]], name: str, passed: bool, **details: Any) -> None:
    cases.append({"name": name, "passed": bool(passed), **details})


def run_audit(config: dict[str, Any]) -> dict[str, Any]:
    audit = config["audit"]
    defenders = int(audit["defenders"])
    queue_lengths = [int(value) for value in audit["queue_lengths"]]
    max_override_slots = int(audit["max_override_slots"])
    issued_step = int(audit["issued_step"])
    generation = int(audit["generation"])
    cases: list[dict[str, Any]] = []
    recovery_cases = 0
    recovery_rejections = 0
    applied_slots: dict[str, list[int]] = {mode: [] for mode in MODES}

    for queue_length in queue_lengths:
        original = _queue(queue_length, defenders)
        token = QueueToken(
            generation=generation,
            issued_step=issued_step,
            pending_length=queue_length,
        )
        new_action = np.full((defenders, 3), 99.0, dtype=np.float64)

        for mode in MODES:
            directive = {
                "mode": mode,
                "emergency_brake": True,
                "expected_queue_token": token.as_dict(),
                "max_override_slots": max_override_slots,
            }
            changed, after_token, ack = apply_command_authority_with_token(
                original,
                directive,
                allowed_mode=mode,
                current_token=token,
            )
            start = 0 if mode == "flush_pending" else 1
            expected_slots = (
                0
                if mode == "immutable"
                else min(max(queue_length - start, 0), max_override_slots)
            )
            expected = _queue(queue_length, defenders)
            for index in range(start, start + expected_slots):
                expected[index].fill(0.0)
            _check(
                cases,
                f"bounded_mutation/{mode}/q={queue_length}",
                ack.accepted
                and ack.applied == bool(expected_slots)
                and ack.overridden_slots == expected_slots
                and _same_queue(changed, expected)
                and after_token.generation == generation + (1 if expected_slots else 0),
                mode=mode,
                queue_length=queue_length,
                overridden_slots=int(ack.overridden_slots),
                ack_reason=ack.reason,
            )
            applied_slots[mode].append(int(ack.overridden_slots))

            stale = QueueToken(
                generation=max(generation - 1, 0),
                issued_step=issued_step,
                pending_length=queue_length,
            )
            stale_queue, stale_after, stale_ack = apply_command_authority_with_token(
                original,
                {
                    "mode": mode,
                    "emergency_brake": True,
                    "expected_queue_token": stale.as_dict(),
                    "max_override_slots": max_override_slots,
                },
                allowed_mode=mode,
                current_token=token,
            )
            recovery_cases += 1
            recovery_rejections += int(not stale_ack.accepted)
            _check(
                cases,
                f"stale_token_rejected/{mode}/q={queue_length}",
                not stale_ack.accepted
                and stale_ack.reason == "stale_queue_token"
                and _same_queue(stale_queue, original)
                and stale_after == token,
                mode=mode,
                queue_length=queue_length,
                ack_reason=stale_ack.reason,
            )

            missing_queue, _missing_after, missing_ack = apply_command_authority_with_token(
                original,
                {"mode": mode, "emergency_brake": True, "max_override_slots": max_override_slots},
                allowed_mode=mode,
                current_token=token,
            )
            if mode != "immutable":
                recovery_cases += 1
                recovery_rejections += int(not missing_ack.accepted)
            _check(
                cases,
                f"missing_token_rejected/{mode}/q={queue_length}",
                (mode == "immutable" and missing_ack.accepted and not missing_ack.applied)
                or (
                    mode != "immutable"
                    and not missing_ack.accepted
                    and missing_ack.reason == "missing_queue_token"
                    and _same_queue(missing_queue, original)
                ),
                mode=mode,
                queue_length=queue_length,
                ack_reason=missing_ack.reason,
            )

        mismatch_current = QueueToken(
            generation=generation,
            issued_step=issued_step,
            pending_length=queue_length + 1,
        )
        mismatch_queue, _mismatch_after, mismatch_ack = apply_command_authority_with_token(
            original,
            {
                "mode": "flush_pending",
                "emergency_brake": True,
                "expected_queue_token": mismatch_current.as_dict(),
            },
            allowed_mode="flush_pending",
            current_token=mismatch_current,
        )
        _check(
            cases,
            f"queue_length_mismatch_rejected/q={queue_length}",
            not mismatch_ack.accepted
            and mismatch_ack.reason == "queue_length_mismatch"
            and _same_queue(mismatch_queue, original),
            queue_length=queue_length,
            ack_reason=mismatch_ack.reason,
        )

        committed, due, next_token = commit_delayed_command_with_token(
            original,
            new_action,
            token,
            next_step=issued_step + 1,
        )
        expected_due = new_action if queue_length == 0 else original[0]
        expected_length = queue_length if queue_length > 0 else 0
        _check(
            cases,
            f"single_pop_commit/q={queue_length}",
            np.array_equal(due, expected_due)
            and len(committed) == expected_length
            and next_token.generation == generation + 1
            and next_token.issued_step == issued_step + 1
            and next_token.pending_length == expected_length,
            queue_length=queue_length,
            resulting_queue_length=len(committed),
        )

    passed = [case for case in cases if case["passed"]]
    failures = [case for case in cases if not case["passed"]]
    return {
        "experiment_name": "phase62_queue_token_transition_audit",
        "audit": {
            "defenders": defenders,
            "queue_lengths": queue_lengths,
            "max_override_slots": max_override_slots,
            "issued_step": issued_step,
            "generation": generation,
        },
        "authority_modes": list(MODES),
        "case_count": len(cases),
        "passed_case_count": len(passed),
        "failed_case_count": len(failures),
        "overall_pass": not failures,
        "recovery_negative_probe_count": recovery_cases,
        "recovery_negative_probe_rejection_rate": (
            float(recovery_rejections) / float(recovery_cases) if recovery_cases else 1.0
        ),
        "mean_applied_slots": {
            mode: float(np.mean(values)) if values else 0.0 for mode, values in applied_slots.items()
        },
        "failures": failures,
        "claim_boundary": (
            "queue-token freshness, bounded authority mutation, ACK semantics, and one-pop timing only; "
            "not a reachable-set, safety, or flight-controller certificate"
        ),
        "cases": cases,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result = run_audit(config)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_snapshot.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if SummaryWriter is not None:
        with SummaryWriter(log_dir=str(output / "tensorboard")) as writer:
            writer.add_text("Config/YAML", config_path.read_text(encoding="utf-8"), 0)
            writer.add_text("Audit/claim_boundary", result["claim_boundary"], 0)
            writer.add_scalar("Gate/overall_pass", float(result["overall_pass"]), 0)
            writer.add_scalar("Audit/case_count", float(result["case_count"]), 0)
            writer.add_scalar("Audit/failed_case_count", float(result["failed_case_count"]), 0)
            writer.add_scalar(
                "Audit/negative_probe_rejection_rate",
                result["recovery_negative_probe_rejection_rate"],
                0,
            )
            for mode in MODES:
                writer.add_scalar(
                    f"Authority/{mode}/mean_applied_slots",
                    result["mean_applied_slots"][mode],
                    0,
                )
            for case_index, case in enumerate(result["cases"]):
                writer.add_scalar(
                    "Audit/case_pass",
                    float(case["passed"]),
                    int(case_index),
                )
            writer.flush()

    print(json.dumps({key: result[key] for key in ("experiment_name", "case_count", "failed_case_count", "overall_pass")}, indent=2))


if __name__ == "__main__":
    main()
