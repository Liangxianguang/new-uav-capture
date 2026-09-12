"""Summarize the public-geometry QDR prefix precondition audit.

The analyzer consumes evaluator ``steps.jsonl`` files and never uses target
truth or terminal outcomes to infer a prefix failure.  It is intentionally
compatible with legacy Phase 26 files: those files are reported as missing
the new classifier instead of being silently reinterpreted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


CLASSIFIER_KEYS = (
    "qdr_prefix_minimum_obstacle_barrier_m",
    "qdr_prefix_minimum_boundary_barrier_m",
    "qdr_prefix_minimum_inter_agent_barrier_m",
    "qdr_prefix_minimum_barrier_m",
    "qdr_prefix_admissible",
    "qdr_prefix_first_violation_step",
    "qdr_prefix_first_violation_cause",
    "qdr_prefix_violation_step_count",
    "qdr_prefix_violation_step_ratio",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(values: list[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            result.append(number)
    return result


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"expected an object in {path}:{line_number}")
            rows.append(value)
    return rows


def summarize_file(path: Path) -> dict[str, Any]:
    rows = _load_rows(path)
    qdr_rows = [row for row in rows if float(row.get("qdr_enabled", 0.0)) > 0.5]
    classified = [row for row in qdr_rows if all(key in row for key in CLASSIFIER_KEYS)]
    missing = len(qdr_rows) - len(classified)
    causes = Counter(
        str(row.get("qdr_prefix_first_violation_cause", "missing"))
        for row in classified
        if str(row.get("qdr_prefix_first_violation_cause", "none")) not in {"none", "None", ""}
    )

    def metric(key: str, reducer: str = "mean") -> float | None:
        values = _finite([row.get(key) for row in classified])
        if not values:
            return None
        if reducer == "min":
            return float(min(values))
        if reducer == "max":
            return float(max(values))
        return float(np.mean(values))

    return {
        "source": str(path),
        "sha256": _sha256(path),
        "rows": len(rows),
        "episodes": len({int(row["episode_index"]) for row in rows if "episode_index" in row}),
        "qdr_rows": len(qdr_rows),
        "classified_qdr_rows": len(classified),
        "missing_classifier_rows": missing,
        "classifier_complete": bool(qdr_rows) and missing == 0,
        "prefix_admissible_rate": metric("qdr_prefix_admissible"),
        "minimum_prefix_barrier_m": metric("qdr_prefix_minimum_barrier_m", "min"),
        "mean_prefix_violation_step_count": metric("qdr_prefix_violation_step_count"),
        "mean_prefix_violation_step_ratio": metric("qdr_prefix_violation_step_ratio"),
        "first_violation_step_min": metric("qdr_prefix_first_violation_step", "min"),
        "first_violation_cause_counts": dict(sorted(causes.items())),
    }


def _markdown(reports: list[dict[str, Any]]) -> str:
    lines = [
        "# QDR Prefix Failure Audit",
        "",
        "This report uses only public-geometry diagnostics recorded in `steps.jsonl`; it is not a safety proof and does not use target truth.",
        "",
        "| Run | QDR rows | Classified | Complete | Admissible rate | Min prefix barrier (m) | Mean violating-step ratio | First cause counts |",
        "|---|---:|---:|:---:|---:|---:|---:|---|",
    ]
    for report in reports:
        lines.append(
            "| {source} | {qdr_rows} | {classified_qdr_rows} | {classifier_complete} | {prefix_admissible_rate} | {minimum_prefix_barrier_m} | {mean_prefix_violation_step_ratio} | `{causes}` |".format(
                source=Path(report["source"]).parent.name,
                qdr_rows=report["qdr_rows"],
                classified_qdr_rows=report["classified_qdr_rows"],
                classifier_complete=report["classifier_complete"],
                prefix_admissible_rate=report["prefix_admissible_rate"],
                minimum_prefix_barrier_m=report["minimum_prefix_barrier_m"],
                mean_prefix_violation_step_ratio=report["mean_prefix_violation_step_ratio"],
                causes=json.dumps(report["first_violation_cause_counts"], ensure_ascii=False, sort_keys=True),
            )
        )
    lines.extend(
        [
            "",
            "Legacy files with `classifier_complete=false` are intentionally not assigned a cause; rerun the evaluator with the current schema before drawing a prefix-risk conclusion.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [summarize_file(path) for path in args.input]
    payload = {"reports": reports, "classifier_keys": list(CLASSIFIER_KEYS)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(_markdown(reports), encoding="utf-8")


if __name__ == "__main__":
    main()
