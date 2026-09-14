"""Attribute Phase 60 closed-loop failures without using target truth online.

The analyzer joins evaluator episode summaries with the frozen scene manifest
and reports overlapping diagnostic flags.  It is a failure taxonomy, not a
causal proof and not a safety certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[assignment,misc]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object in {path}:{line_number}")
            rows.append(value)
    return rows


def _scene_blocks(path: Path) -> dict[int, str]:
    return {
        int(row["episode_index"]): str(row.get("scene_block", row.get("condition_id", "unknown")))
        for row in _load_jsonl(path)
    }


def _empty() -> dict[str, Any]:
    return {
        "episodes": 0,
        "safe_capture": 0,
        "collision": 0,
        "boundary_violation": 0,
        "timeout": 0,
        "timeout_with_qdr_exhaustion": 0,
        "collision_with_qdr_exhaustion": 0,
        "qdr_exhausted_episode": 0,
        "prefix_unrecoverable_episode": 0,
        "planner_fallback_episode": 0,
        "max_exhaustion_streak_steps": 0.0,
    }


def _update(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    bucket["episodes"] += 1
    termination = str(row.get("termination_reason", "unknown"))
    collision = bool(row.get("collision", False))
    boundary = bool(row.get("boundary_violation", False))
    timeout = bool(row.get("timeout", False)) or termination == "timeout"
    exhausted = float(row.get("qdr_suffix_gate_exhausted_once", 0.0) or 0.0) > 0.5
    statuses = row.get("qdr_precondition_status_counts", {})
    prefix_unrecoverable = isinstance(statuses, dict) and int(
        statuses.get("prefix_unsafe_unrecoverable", 0) or 0
    ) > 0
    planner_fallback = float(row.get("planner_fallback_count", 0.0) or 0.0) > 0.0
    bucket["safe_capture"] += int(bool(row.get("safe_capture_success", False)))
    bucket["collision"] += int(collision)
    bucket["boundary_violation"] += int(boundary)
    bucket["timeout"] += int(timeout)
    bucket["timeout_with_qdr_exhaustion"] += int(timeout and exhausted)
    bucket["collision_with_qdr_exhaustion"] += int(collision and exhausted)
    bucket["qdr_exhausted_episode"] += int(exhausted)
    bucket["prefix_unrecoverable_episode"] += int(prefix_unrecoverable)
    bucket["planner_fallback_episode"] += int(planner_fallback)
    bucket["max_exhaustion_streak_steps"] = max(
        float(bucket["max_exhaustion_streak_steps"]),
        float(row.get("qdr_suffix_gate_max_exhaustion_streak_steps", 0.0) or 0.0),
    )


def summarize_rows(
    rows: list[dict[str, Any]],
    scene_blocks: dict[int, str],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(_empty)
    for row in rows:
        method = str(row.get("method", "unknown"))
        episode_index = int(row.get("episode_index", -1))
        block = scene_blocks.get(episode_index, str(row.get("observation_condition", "unknown")))
        _update(grouped[(method, block)], row)
        _update(grouped[(method, "overall")], row)
    return {
        "groups": {
            f"{method}::{block}": {"method": method, "block": block, **metrics}
            for (method, block), metrics in sorted(grouped.items())
        }
    }


def _rate(metrics: dict[str, Any], key: str) -> str:
    episodes = max(int(metrics["episodes"]), 1)
    return f"{100.0 * float(metrics[key]) / episodes:.2f}%"


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Phase 60 Failure Taxonomy",
        "",
        "Flags are overlapping diagnostics joined to the frozen scene manifest; they do not establish causality or safety guarantees.",
        "",
        "| Method | Block | Episodes | Safe | Collision | Boundary | Timeout | Timeout+QDR exhaust | QDR-exhausted episodes | Prefix unrecoverable | Planner fallback | Max streak |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload["groups"].values():
        lines.append(
            "| {method} | {block} | {episodes} | {safe} | {collision} | {boundary} | {timeout} | {timeout_exhaust} | {exhausted} | {prefix} | {fallback} | {streak:.0f} |".format(
                method=item["method"],
                block=item["block"],
                episodes=item["episodes"],
                safe=_rate(item, "safe_capture"),
                collision=_rate(item, "collision"),
                boundary=_rate(item, "boundary_violation"),
                timeout=_rate(item, "timeout"),
                timeout_exhaust=_rate(item, "timeout_with_qdr_exhaustion"),
                exhausted=_rate(item, "qdr_exhausted_episode"),
                prefix=_rate(item, "prefix_unrecoverable_episode"),
                fallback=_rate(item, "planner_fallback_episode"),
                streak=float(item["max_exhaustion_streak_steps"]),
            )
        )
    lines.extend(["", "Claim boundary: empirical failure attribution only; local CBF is not an R-CLBF-QP proof.", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--tensorboard-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_blocks = _scene_blocks(args.scenes)
    rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for path in args.input:
        rows.extend(_load_jsonl(path))
        input_hashes[str(path)] = _sha256(path)
    summary = summarize_rows(rows, scene_blocks)
    payload = {
        "schema_version": "phase60-failure-taxonomy-v1",
        "scenes": str(args.scenes.resolve()),
        "scene_manifest_sha256": _sha256(args.scenes),
        "inputs": input_hashes,
        "claim_boundary": "empirical failure attribution only; no safety proof",
        **summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(_markdown(summary), encoding="utf-8")
    if SummaryWriter is not None:
        args.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        with SummaryWriter(log_dir=str(args.tensorboard_dir)) as writer:
            writer.add_text("Protocol/claim_boundary", payload["claim_boundary"], 0)
            for step, item in enumerate(payload["groups"].values()):
                prefix = f"FailureTaxonomy/{item['method']}/{item['block']}"
                episodes = max(int(item["episodes"]), 1)
                for key in (
                    "safe_capture",
                    "collision",
                    "boundary_violation",
                    "timeout",
                    "timeout_with_qdr_exhaustion",
                    "qdr_exhausted_episode",
                    "prefix_unrecoverable_episode",
                    "planner_fallback_episode",
                ):
                    writer.add_scalar(f"{prefix}/{key}_rate", float(item[key]) / episodes, step)
                writer.add_scalar(
                    f"{prefix}/max_exhaustion_streak_steps",
                    float(item["max_exhaustion_streak_steps"]),
                    step,
                )
            writer.flush()
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
