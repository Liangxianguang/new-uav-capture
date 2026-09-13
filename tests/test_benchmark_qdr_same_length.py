from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_qdr_same_length import (  # noqa: E402
    select_fixed_prefix_pairs,
    summarize,
)


def _rows(episode_index: int, count: int, offset: float) -> dict[int, list[dict[str, float | int]]]:
    return {
        episode_index: [
            {
                "episode_index": episode_index,
                "step": step,
                "predictor_latency_ms": 1.0 + offset,
                "planner_latency_ms": 2.0 + offset,
                "qdr_latency_ms": 0.5 + offset,
                "safety_latency_ms": 0.25 + offset,
                "total_control_latency_ms": 3.75 + offset,
            }
            for step in range(count)
        ]
    }


def test_select_fixed_prefix_pairs_uses_only_common_complete_prefixes() -> None:
    off = {**_rows(0, 4, 0.0), **_rows(1, 2, 0.0)}
    on = {**_rows(0, 5, 1.0), **_rows(1, 3, 1.0)}
    selected_off, selected_on, retained = select_fixed_prefix_pairs(off, on, 3)
    assert retained == [0]
    assert [row["step"] for row in selected_off] == [0, 1, 2]
    assert [row["step"] for row in selected_on] == [0, 1, 2]


def test_summarize_reports_required_percentiles_and_sample_count() -> None:
    result = summarize(_rows(0, 3, 0.0)[0])
    assert result["samples"] == 3
    assert result["total_control_latency_ms"]["p50"] == 3.75
    assert set(result) == {
        "predictor_latency_ms",
        "planner_latency_ms",
        "qdr_latency_ms",
        "safety_latency_ms",
        "total_control_latency_ms",
        "samples",
    }
