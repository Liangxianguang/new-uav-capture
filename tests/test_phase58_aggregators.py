from __future__ import annotations

import math

from scripts.aggregate_phase58_calibration import json_safe, scene_blocks
from scripts.benchmark_phase57_isolated_runtime import PHASE58_METHODS


def test_phase58_runtime_benchmark_exposes_all_methods() -> None:
    assert PHASE58_METHODS[0] == "M0_current_state_delayed_mpc"
    assert PHASE58_METHODS[-1] == "M8_fixed_k8_qdr"
    assert len(PHASE58_METHODS) == 9


def test_json_safe_converts_unavailable_numeric_values() -> None:
    result = json_safe({"available": 1.0, "missing": float("nan"), "nested": [float("inf")]})
    assert result == {"available": 1.0, "missing": None, "nested": [None]}


def test_scene_blocks_preserves_episode_membership(tmp_path) -> None:
    manifest = tmp_path / "scenes.jsonl"
    manifest.write_text(
        '{"episode_index": 1, "scene_block": "id_reference"}\n'
        '{"episode_index": 0, "scene_block": "id_reference"}\n'
        '{"episode_index": 2, "scene_block": "joint_stress_transfer"}\n',
        encoding="utf-8",
    )
    assert scene_blocks(manifest) == {
        "id_reference": [0, 1],
        "joint_stress_transfer": [2],
    }


def test_json_safe_does_not_change_finite_values() -> None:
    assert math.isclose(json_safe(0.25), 0.25)
