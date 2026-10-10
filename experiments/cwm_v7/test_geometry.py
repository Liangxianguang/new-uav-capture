import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from diagnose_geometry import select_geometry
from geometry import BackboneTap, translated_records


def test_geometry_selection_uses_validity_only():
    rows = [{"wall_center_x_m": 3., "original_target_invalid_episodes": 0, "stress_target_invalid_branches": 0, "original_safe_captures": 8},
            {"wall_center_x_m": 1.5, "original_target_invalid_episodes": 0, "stress_target_invalid_branches": 0, "original_safe_captures": 0},
            {"wall_center_x_m": 0., "original_target_invalid_episodes": 0, "stress_target_invalid_branches": 1, "original_safe_captures": 8}]
    assert select_geometry(rows) == 1.5
    rows[1]["original_target_invalid_episodes"] = 1
    assert select_geometry(rows) == 3.
    rows[0]["stress_target_invalid_branches"] = 1
    assert select_geometry(rows) is None


def test_geometry_moves_only_wall_and_declared_zone_not_agents_or_rules(tmp_path):
    protocol = json.loads(Path(__file__).with_name("geometry_protocol.json").read_text())
    protocol["groups"] = 1  # Unit invariant; complete scene populations are audited from raw runs.
    base = BackboneTap(Path(__file__).parent.parent / "cwm_v1/baseline/capsule.zip", tmp_path / "restored")
    original = translated_records(base, protocol, 0.)
    translated = translated_records(base, protocol, 3.)
    for a, b in zip(original, translated):
        assert a["pursuit_overrides"] == b["pursuit_overrides"]
        assert a["scenario"]["target_position"] == b["scenario"]["target_position"]
        assert a["scenario"]["defender_positions"] == b["scenario"]["defender_positions"]
        assert np.allclose(np.array(b["scenario"]["obstacle_zone_x"]) - a["scenario"]["obstacle_zone_x"], 3.)
        assert b["route_validation"]["branch_route_feasible"]
        env = base.env_class(base.configuration(b), obstacle_count=1, target_speed_scale=b["target_speed_scale"])
        assert env.pursuit["target_branch_waypoint_x"] == .85
        assert env.pursuit["target_branch_decision_x"] == -3.20
        assert env.pursuit["safety_margin"] == .75
        assert base.filter_class(env).margin == .35
    with pytest.raises(ValueError):
        translated_records(base, protocol, 4.)
