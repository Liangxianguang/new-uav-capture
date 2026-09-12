from __future__ import annotations

import json

from scripts.analyze_qdr_prefix_failures import summarize_file


def _row(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "episode_index": 0,
        "qdr_enabled": 1.0,
        "qdr_prefix_minimum_obstacle_barrier_m": 0.2,
        "qdr_prefix_minimum_boundary_barrier_m": -0.1,
        "qdr_prefix_minimum_inter_agent_barrier_m": 0.4,
        "qdr_prefix_minimum_barrier_m": -0.1,
        "qdr_prefix_admissible": 0.0,
        "qdr_prefix_first_violation_step": 2.0,
        "qdr_prefix_first_violation_cause": "boundary",
        "qdr_prefix_violation_step_count": 1.0,
        "qdr_prefix_violation_step_ratio": 0.5,
    }
    value.update(updates)
    return value


def test_summarize_file_reports_cause_and_hash(tmp_path) -> None:
    path = tmp_path / "steps.jsonl"
    path.write_text(json.dumps(_row()) + "\n" + json.dumps(_row(episode_index=1, qdr_enabled=0.0)) + "\n")

    report = summarize_file(path)

    assert report["classifier_complete"] is True
    assert report["classified_qdr_rows"] == 1
    assert report["prefix_admissible_rate"] == 0.0
    assert report["first_violation_step_min"] == 2.0
    assert report["first_violation_cause_counts"] == {"boundary": 1}
    assert len(report["sha256"]) == 64


def test_summarize_file_does_not_infer_legacy_cause(tmp_path) -> None:
    path = tmp_path / "legacy_steps.jsonl"
    path.write_text(json.dumps({"episode_index": 0, "qdr_enabled": 1.0}) + "\n")

    report = summarize_file(path)

    assert report["classifier_complete"] is False
    assert report["missing_classifier_rows"] == 1
    assert report["first_violation_cause_counts"] == {}


def test_summarize_file_ignores_no_violation_sentinel_for_first_step(tmp_path) -> None:
    path = tmp_path / "mixed_steps.jsonl"
    path.write_text(
        json.dumps(_row(qdr_prefix_admissible=1.0, qdr_prefix_first_violation_step=-1.0))
        + "\n"
        + json.dumps(_row(episode_index=1, qdr_prefix_first_violation_step=1.0))
        + "\n"
    )

    report = summarize_file(path)

    assert report["first_violation_step_min"] == 1.0
