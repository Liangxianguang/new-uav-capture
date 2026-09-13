from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_s4_closed_loop import phase56_method_contract  # noqa: E402


def _base() -> dict[str, object]:
    return {
        "queue_aware_rollout": True,
        "queue_aware_safety_projection": True,
        "adaptive_k": True,
        "num_samples": 4,
        "fixed_tube_radius_m": None,
    }


def test_phase56_qdr_async_contract_is_explicit() -> None:
    contract = phase56_method_contract("B6_qdr_asynchronous_mpc", **_base())
    assert contract["canonical_method"] == "distributed_async"
    assert contract["distributed_mode"] == "asynchronous"
    assert contract["queue_aware_rollout"] is True


def test_phase56_fixed_tube_has_validation_only_default() -> None:
    contract = phase56_method_contract(
        "B2_fixed_tube_mpc",
        **{**_base(), "queue_aware_rollout": False, "queue_aware_safety_projection": False},
    )
    assert contract["canonical_method"] == "worst_case"
    assert contract["queue_aware_rollout"] is False
    assert contract["fixed_tube_radius_m"] == 0.35
    assert contract["target_tube_cost_enabled"] is True


def test_phase56_non_tube_baseline_does_not_enable_tube_cost() -> None:
    contract = phase56_method_contract("B0_current_state_delayed_mpc", **_base())
    assert contract["target_tube_cost_enabled"] is False


def test_phase58_known_delay_baseline_does_not_read_queue() -> None:
    contract = phase56_method_contract("M1_known_delay_delayed_mpc", **_base())

    assert contract["canonical_method"] == "worst_case"
    assert contract["known_delay_compensation"] is True
    assert contract["queue_aware_rollout"] is False
    assert contract["target_tube_cost_enabled"] is False


def test_phase58_qdr_synchronous_composite_is_distributed_and_queue_aware() -> None:
    contract = phase56_method_contract("M6_qdr_synchronous_mpc", **_base())

    assert contract["canonical_method"] == "distributed_delayed"
    assert contract["distributed_mode"] == "delayed"
    assert contract["queue_aware_rollout"] is True
    assert contract["known_delay_compensation"] is False
