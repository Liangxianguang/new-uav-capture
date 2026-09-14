from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_phase60_failure_taxonomy import summarize_rows  # noqa: E402


def test_failure_taxonomy_keeps_overlapping_qdr_flags() -> None:
    rows = [
        {
            "method": "M6",
            "episode_index": 0,
            "safe_capture_success": False,
            "collision": False,
            "boundary_violation": False,
            "timeout": True,
            "termination_reason": "timeout",
            "qdr_suffix_gate_exhausted_once": 1.0,
            "qdr_suffix_gate_max_exhaustion_streak_steps": 31.0,
            "qdr_precondition_status_counts": {"prefix_unsafe_unrecoverable": 1},
            "planner_fallback_count": 0,
        },
        {
            "method": "M6",
            "episode_index": 1,
            "safe_capture_success": False,
            "collision": True,
            "boundary_violation": False,
            "timeout": False,
            "termination_reason": "collision",
            "qdr_suffix_gate_exhausted_once": 0.0,
            "qdr_suffix_gate_max_exhaustion_streak_steps": 0.0,
            "qdr_precondition_status_counts": {},
            "planner_fallback_count": 1,
        },
    ]

    result = summarize_rows(rows, {0: "joint_stress", 1: "joint_stress"})
    item = result["groups"]["M6::joint_stress"]
    assert item["episodes"] == 2
    assert item["timeout"] == 1
    assert item["timeout_with_qdr_exhaustion"] == 1
    assert item["qdr_exhausted_episode"] == 1
    assert item["prefix_unrecoverable_episode"] == 1
    assert item["planner_fallback_episode"] == 1
    assert item["max_exhaustion_streak_steps"] == 31.0
