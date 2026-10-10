import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from s4_collect import BackboneTap, scene_records, s4_commands, s4_statistics

PROTOCOL = json.loads(Path(__file__).with_name("s4_scout_protocol.json").read_text())


def test_s4_all_generated_scenes_pass_frozen_environment_contract(tmp_path):
    capsule = Path(__file__).parent.parent / "cwm_v1/baseline/capsule.zip"
    base = BackboneTap(capsule, tmp_path / "restored")
    records = scene_records(base, PROTOCOL)
    assert len(records) == 8
    assert len({r["mirror_group_id"] for r in records}) == 4
    for record in records:
        env = base.env_class(base.configuration(record), obstacle_count=1,
                             target_speed_scale=record["target_speed_scale"])
        assert env.pursuit["safety_margin"] == .75
        assert base.filter_class(env).margin == .35
        assert env.pursuit["target_branch_exit_offset_y"] == 5.30
        assert env.pursuit["target_branch_decision_x"] == -3.20
        assert record["route_validation"]["branch_route_feasible"]
    forbidden = json.loads(json.dumps(PROTOCOL))
    forbidden["target_branch_rule_overrides"] = {"target_branch_decision_x": -4.0}
    with pytest.raises(ValueError, match="branch rules"):
        scene_records(base, forbidden)


def test_s4_commands_only_public_geometry_and_preserves_anchor():
    plan = np.ones((8, 4, 3))
    original = plan.copy()
    positions = np.arange(12).reshape(4, 3)
    value = s4_commands(plan, {"defender_positions": positions}, np.zeros(3), PROTOCOL)
    assert value.shape == (8, 16, 4, 3)
    assert np.array_equal(plan, original)
    assert np.array_equal(value[0, :8], plan)
    assert np.linalg.norm(value, axis=-1).max() <= 5.00000001
    assert np.all(value[5] == 0)
    assert np.array_equal(value[6, :, 1:], value[0, :, 1:])
    with pytest.raises(ValueError):
        s4_commands(plan[:7], {"defender_positions": positions}, np.zeros(3), PROTOCOL)


def test_s4_gate_requires_independent_branch_flip_groups_and_short_signal():
    target = np.zeros((2, 8, 16, 3))
    target[:, 1:, :, 1] = .1
    signs = np.ones((2, 8, 16), dtype=np.int8)
    signs[:, 1:] = -1
    data = {"target": target, "branch_sign_label_only": signs, "valid": np.ones((2, 8, 16), dtype=bool),
            "group": np.array(["same", "same"]), "executed": np.zeros((2, 8, 16, 4, 3))}
    assert not s4_statistics(data, PROTOCOL)["short_horizon_data_gate_passed"]
    data["group"] = np.array(["a", "b"])
    assert s4_statistics(data, PROTOCOL)["short_horizon_data_gate_passed"]
    data["target"][:, 1:, :8] = 0
    data["branch_sign_label_only"][:, 1:, :8] = 1
    result = s4_statistics(data, PROTOCOL)
    assert not result["short_horizon_data_gate_passed"]
    assert result["horizons"]["16"]["branch_flip_pairs"] == 14


def test_s4_branch_unknown_and_post_terminal_are_not_flips():
    data = {"target": np.zeros((1, 8, 16, 3)), "branch_sign_label_only": np.zeros((1, 8, 16), dtype=np.int8),
            "valid": np.zeros((1, 8, 16), dtype=bool), "group": np.array(["a"]), "executed": np.zeros((1, 8, 16, 4, 3))}
    data["branch_sign_label_only"][:, 1:] = -1
    result = s4_statistics(data, PROTOCOL)["horizons"]["8"]
    assert result["branch_flip_pairs"] == 0
    assert result["both_committed_support_pairs"] == 0
    assert result["branch_flip_pair_fraction"] is None
