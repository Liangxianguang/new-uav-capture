from __future__ import annotations

from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "phase61_qdr_recovery_authority_ablation.yaml"


def test_phase61_freezes_only_qdr_recovery_authority() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["method_contract"]["method"] == "M6_qdr_synchronous_mpc"
    assert config["method_contract"]["candidate_budget"] == 1
    assert config["method_contract"]["adaptive_k"] is False
    assert config["authority_modes"] == [
        "immutable",
        "replace_nonexecuting",
        "flush_pending",
    ]
    assert config["frozen_factors"]["action_delay_steps"] == 4
    assert config["frozen_factors"]["message_delay_steps"] == 2
    assert config["frozen_factors"]["command_noise_std_mps"] == 0.08
    assert config["gates"]["locked_test_access"] == "forbidden"
    assert config["reporting"]["latency_percentiles"] == [50, 95, 99]
    assert config["reporting"]["local_cbf_claim"] == "empirical_filter_only"
    assert config["reporting"]["robust_clbf_qp_proof"] == "not_claimed"
