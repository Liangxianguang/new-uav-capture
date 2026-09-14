from pathlib import Path

import yaml


CONFIG = Path(__file__).parents[1] / "configs" / "phase67_bounded_progress_recovery.yaml"


def test_phase67_contract_freezes_bounded_recovery_and_claim_boundary() -> None:
    document = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    methods = [item["id"] for item in document["methods"]]
    assert methods == ["M6_qdr_synchronous_mpc", "M9_qdr_bounded_recovery"]
    assert document["dataset"]["canonical_episodes"] == 360
    assert document["dataset"]["calibration_episodes"] == 120
    assert document["dataset"]["calibration_mirror_groups"] == 60
    assert document["dataset"]["split_policy"] == "mirror_group_disjoint"
    assert document["distributed"]["qdr_exhaustion_recovery_budget_steps"] == 6
    assert document["distributed"]["qdr_exhaustion_recovery_horizon_steps"] == 3
    assert document["qdr_segment_audit"]["progress_tolerance_m"] == 1.0e-9
    assert document["evaluation"]["candidate_budget_requested"] == 8
    assert document["reporting"]["latency_components"] == [
        "predictor",
        "planner",
        "qdr_or_tube",
        "safety",
        "total_control",
    ]
    assert document["reporting"]["local_cbf_claim"] == "empirical_filter_only"
    assert document["reporting"]["robust_cbf_qp"] == "diagnostic_no_go"
    assert document["reporting"]["qdr_claim"] == "no_safety_proof"
    assert document["gates"]["no_locked_test_tuning"] is True
