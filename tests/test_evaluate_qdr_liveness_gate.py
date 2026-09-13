from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_qdr_liveness_gate import evaluate  # noqa: E402


def _aggregate(timeout: float, timeout_delta: float) -> dict[str, object]:
    return {
        "groups": {
            "on": {
                "distributed_delayed": {
                    "episode_metrics": {"timeout": {"mean": timeout}}
                }
            }
        },
        "paired_vs_reference": {
            "on": {
                "distributed_delayed": {
                    "episode_metrics": {
                        "timeout": {
                            "mean_delta_candidate_minus_reference": timeout_delta
                        }
                    }
                }
            }
        },
    }


def _summary(streak: float) -> dict[str, object]:
    return {"overall": {"qdr_suffix_gate_max_exhaustion_streak_steps": streak}}


def test_liveness_gate_passes_at_preregistered_limits() -> None:
    result = evaluate(
        _aggregate(0.05, 0.05),
        [_summary(24.0), _summary(20.0)],
        0.05,
        0.05,
        24.0,
    )
    assert result["status"] == "pass"
    assert all(check["pass"] for check in result["checks"].values())


def test_liveness_gate_rejects_excessive_streak() -> None:
    result = evaluate(
        _aggregate(0.04, 0.04),
        [_summary(25.0)],
        0.05,
        0.05,
        24.0,
    )
    assert result["status"] == "no_go"
    assert result["checks"]["max_exhaustion_streak_steps"]["pass"] is False
