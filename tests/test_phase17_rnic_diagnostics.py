from __future__ import annotations

import numpy as np

from scripts.analyze_phase17_rnic_diagnostics import auc, summarize


def test_tie_aware_auc_orders_high_risk_slack_first() -> None:
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    outcomes = np.array([0.0, 0.0, 1.0, 1.0])
    assert auc(scores, outcomes) == 1.0


def test_rnic_summary_reports_reliability_bins() -> None:
    rows = [
        {"rnic_minimum_best_slack_s": -0.8, "collision": True, "boundary_violation": False},
        {"rnic_minimum_best_slack_s": -0.4, "collision": True, "boundary_violation": False},
        {"rnic_minimum_best_slack_s": 0.1, "collision": False, "boundary_violation": False},
        {"rnic_minimum_best_slack_s": 0.3, "collision": False, "boundary_violation": False},
    ]
    result = summarize(rows)
    assert result["usable_episodes"] == 4
    assert len(result["risk_reliability_bins"]) == 4
    assert result["unsafe_auc"] == 1.0


def test_markdown_includes_pooled_group() -> None:
    from scripts.analyze_phase17_rnic_diagnostics import markdown

    payload = {
        "groups": {"seed1": summarize([{"rnic_minimum_best_slack_s": 0.1, "collision": False}])},
        "pooled": summarize([{"rnic_minimum_best_slack_s": 0.1, "collision": False}]),
    }
    assert "| pooled |" in markdown(payload)
