from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_uakr_reliability import analyze


def _write_run(root: Path, values: list[tuple[float, bool]]) -> None:
    method = root / "distributed_delayed"
    method.mkdir(parents=True)
    episodes = []
    steps = []
    for index, (score, success) in enumerate(values):
        episodes.append(
            {
                "episode_index": index,
                "safe_capture_success": success,
                "collision": not success,
                "boundary_violation": False,
                "timeout": False,
            }
        )
        steps.append({"episode_index": index, "adaptive_uncertainty_score": score})
    (method / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in episodes), encoding="utf-8")
    (method / "steps.jsonl").write_text("\n".join(json.dumps(row) for row in steps), encoding="utf-8")


def test_uakr_reliability_audit_reports_monotonic_failure_signal(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run, [(0.1, True), (0.2, True), (0.8, False), (0.9, False)])

    result = analyze([run])

    assert result["episode_count"] == 4
    assert result["auc"]["mean_uncertainty"]["failure"] == 1.0
    quartiles = result["quartiles"]["mean_uncertainty"]
    assert quartiles[0]["failure_rate"] == 0.0
    assert quartiles[-1]["failure_rate"] == 1.0
