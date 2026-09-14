from pathlib import Path

import yaml


CONFIG = Path(__file__).parents[1] / "configs" / "phase66_baseline_stress_calibration.yaml"


def test_phase66_contract_keeps_fair_baselines_and_claim_boundary() -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    methods = [item["id"] for item in document["methods"]]
    assert methods == [
        "M0_current_state_delayed_mpc",
        "M1_known_delay_delayed_mpc",
        "M2_fixed_tube_mpc",
        "M3_queue_aware_tube_mpc",
        "M5_asynchronous_distributed_mpc",
        "M6_qdr_synchronous_mpc",
        "M7_qdr_asynchronous_mpc",
    ]
    assert document["dataset"]["canonical_episodes"] == 360
    assert document["dataset"]["calibration_episodes"] == 120
    assert document["dataset"]["split_policy"] == "mirror_group_disjoint"
    assert document["qdr_segment_audit"]["progress_tolerance_m"] == 1.0e-9
    assert document["reporting"]["latency_components"] == [
        "predictor",
        "planner",
        "qdr_or_tube",
        "safety",
        "total_control",
    ]
    assert document["reporting"]["local_cbf_claim"] == "empirical_filter_only"
    assert document["reporting"]["robust_cbf_qp"] == "diagnostic_no_go"
