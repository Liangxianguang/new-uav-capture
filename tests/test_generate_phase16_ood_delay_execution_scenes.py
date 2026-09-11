from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from generate_phase16_ood_delay_execution_scenes import (  # noqa: E402
    OOD_EXECUTION,
    validate_records,
)


def test_execution_ood_contract_is_nontrivial() -> None:
    assert OOD_EXECUTION["enabled"] is True
    assert OOD_EXECUTION["action_delay_steps"] == 2
    assert OOD_EXECUTION["command_noise_std"] > 0.0
    assert OOD_EXECUTION["velocity_time_constant_seconds"] > 0.0


def test_validate_records_requires_execution_override_and_mirror_pair() -> None:
    obstacle = {
        "shape": "wall",
        "center_xy": [0.0, 0.0],
        "radius": 0.55,
        "height": 9.45,
        "half_extents_xy": [0.55, 4.25],
    }
    records = [
        {
            "mirror_group_id": 0,
            "defender_bias": bias,
            "layout_seed": 1,
            "target_speed_scale": 0.65,
            "execution_overrides": OOD_EXECUTION.copy(),
            "scenario": {"obstacles": [obstacle]},
        }
        for bias in ("upper", "lower")
    ]
    validate_records(records)
