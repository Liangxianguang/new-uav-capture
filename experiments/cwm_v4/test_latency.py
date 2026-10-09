import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_latency import analyze


def test_latency_handles_censoring_and_does_not_call_pairs_independent():
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    protocol["variants"] = {"fast_target": 7}
    data = {"target": np.zeros((2, 13, 24, 3)), "valid": np.ones((2, 13, 24), dtype=bool),
            "variant": np.array(["fast_target"] * 2), "group": np.array(["same_group"] * 2)}
    data["target"][0, 1, 9:, 0] = .1
    data["target"][1, 1, 9:, 0] = .1
    data["target"][0, 2, 5:, 0] = .2
    data["valid"][0, 2, 4:] = False
    result = analyze(data, protocol)
    bucket = result["variants"]["fast_target"]
    assert result["status"] == "posthoc_diagnostic_not_a_new_gate"
    assert bucket["pairs_with_observed_effect_over_0_05m"] == 2
    assert bucket["first_observed_response_step"]["mean"] == 10
    assert bucket["response_first_after_original_8_step_horizon"] == 2
    assert bucket["response_first_within_original_8_step_horizon"] == 0
    assert bucket["full24_pair_count"] == 23
    assert bucket["branches"]["agent0_left"]["signal_groups"] == ["same_group"]
    assert bucket["branches"]["agent0_right"]["signal_pairs"] == 0
