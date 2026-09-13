from __future__ import annotations

from scripts.benchmark_phase57_isolated_runtime import fixed_prefix, summarize_latency


def _rows(count: int) -> dict[int, list[dict[str, float | int]]]:
    return {
        0: [
            {
                "episode_index": 0,
                "step": step,
                "predictor_latency_ms": 1.0 + step,
                "planner_latency_ms": 2.0 + step,
                "qdr_latency_ms": 0.5,
                "safety_latency_ms": 0.25,
                "total_control_latency_ms": 3.75 + step,
            }
            for step in range(count)
        ]
    }


def test_fixed_prefix_selects_exact_steps() -> None:
    selected, retained = fixed_prefix(_rows(4), 3)
    assert retained == [0]
    assert [row["step"] for row in selected] == [0, 1, 2]


def test_summary_contains_all_required_latency_percentiles() -> None:
    result = summarize_latency(_rows(3)[0])
    assert result["samples"] == 3
    assert result["predictor_latency_ms"]["p50"] == 2.0
    assert result["qdr_latency_ms"]["p99"] == 0.5
    assert set(result) == {
        "predictor_latency_ms",
        "planner_latency_ms",
        "qdr_latency_ms",
        "safety_latency_ms",
        "total_control_latency_ms",
        "samples",
    }
