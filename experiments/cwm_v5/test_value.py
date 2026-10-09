import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from decision_value import METHODS, group_bootstrap, rank_metrics, summarize


def test_ranking_ties_and_truth_regret():
    value = rank_metrics(np.array([0., 0., 1.]), np.array([2., 0., 1.]))
    assert value["choice"] == 0
    assert value["regret_in_diagnostic_cost"] == 2.
    assert not value["realized_tie_optimal"]
    assert rank_metrics([3., 2.], [1e-10, 0.])["realized_tie_optimal"]
    with pytest.raises(ValueError):
        rank_metrics([np.nan], [0.])


def test_group_equal_bootstrap_not_window_pseudoreplication():
    result = group_bootstrap([0., 0., 0., 2.], ["a", "a", "a", "b"])
    assert result["groups"] == 2
    assert result["group_equal_mean"] == 1.
    assert result == group_bootstrap([0., 0., 0., 2.], ["a", "a", "a", "b"])
    assert group_bootstrap([], [])["group_equal_mean"] is None


def test_joint_censoring_excludes_all_methods():
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    rows = [{"variant": "fast_target", "group": "g", "all_candidates_full_horizon_valid": False}]
    result = summarize(rows, protocol)
    assert result["excluded_incomplete_windows"] == 1
    assert result["rankable_windows"] == 0
    assert all(result["methods"][m]["mean_regret_in_diagnostic_cost"] is None for m in METHODS)
