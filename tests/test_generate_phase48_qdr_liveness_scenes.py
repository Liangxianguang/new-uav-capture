from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_phase48_qdr_liveness_scenes import (  # noqa: E402
    LIVENESS_EXECUTION,
    NOMINAL_PURSUIT,
    validate_records,
)


def _records() -> list[dict[str, object]]:
    obstacle = {
        "shape": "wall",
        "center_xy": [0.0, 0.0],
        "radius": 0.55,
        "height": 9.45,
        "half_extents_xy": [0.55, 4.25],
    }
    return [
        {
            "episode_index": index,
            "mirror_group_id": 0,
            "defender_bias": bias,
            "layout_seed": 1,
            "target_speed_scale": 0.65,
            "pursuit_overrides": NOMINAL_PURSUIT.copy(),
            "execution_overrides": LIVENESS_EXECUTION.copy(),
            "scenario": {"obstacles": [obstacle]},
        }
        for index, bias in enumerate(("upper", "lower"))
    ]


def test_liveness_contract_is_explicit_and_bounded() -> None:
    assert LIVENESS_EXECUTION["action_delay_steps"] == 4
    assert LIVENESS_EXECUTION["pending_command_authority"] == "immutable"
    assert LIVENESS_EXECUTION["command_noise_std"] == 0.08
    assert NOMINAL_PURSUIT["message_delay_steps"] == 2


def test_validate_records_accepts_complete_mirror_group() -> None:
    validate_records(_records())


def test_validate_records_rejects_wrong_execution_contract() -> None:
    records = _records()
    records[0]["execution_overrides"] = {**LIVENESS_EXECUTION, "action_delay_steps": 2}
    try:
        validate_records(records)
    except ValueError as exc:
        assert "execution" in str(exc).lower()
    else:
        raise AssertionError("inconsistent execution contract was accepted")
