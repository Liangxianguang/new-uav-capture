from __future__ import annotations

from scripts.evaluate_minimax_mpc import _select_safety_observation


def test_qdr_safety_projection_uses_delayed_planning_state_only_when_enabled() -> None:
    current = {"state": "current"}
    delayed = {"state": "first_controllable"}

    assert (
        _select_safety_observation(
            current,
            delayed,
            queue_aware_rollout=True,
            queue_aware_safety_projection=True,
            local_cbf_enabled=True,
        )
        is delayed
    )
    assert (
        _select_safety_observation(
            current,
            delayed,
            queue_aware_rollout=True,
            queue_aware_safety_projection=False,
            local_cbf_enabled=True,
        )
        is current
    )


def test_qdr_safety_projection_never_reanchors_non_local_safety_filters() -> None:
    current = {"state": "current"}
    delayed = {"state": "first_controllable"}

    assert (
        _select_safety_observation(
            current,
            delayed,
            queue_aware_rollout=True,
            queue_aware_safety_projection=True,
            local_cbf_enabled=False,
        )
        is current
    )
