"""Geometry-only S4 development under frozen target/controller contracts."""
import copy
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "cwm_v6"))
from s4_collect import scene_records, s4_commands, s4_rollout, BackboneTap, fingerprint, sha, verify


def translated_records(base, protocol, wall_center_x):
    from encirclement3d.showcase import scenario_metadata, s4_branch_route_metrics, validate_s4_branching_scenario
    if protocol["target_branch_rule_overrides"]:
        raise ValueError("Target rules are frozen")
    if not np.isfinite(wall_center_x) or not 0 <= wall_center_x <= 3.0:
        raise ValueError("Geometry development translation outside predeclared range")
    records = scene_records(base, protocol)
    for record in records:
        env = base.env_class(base.configuration(record), obstacle_count=1,
                             target_speed_scale=record["target_speed_scale"])
        scenario = base.scenario_from_metadata(record["scenario"])
        wall = copy.deepcopy(scenario.obstacles[0])
        wall = replace(wall, center_xy=wall.center_xy + np.array([wall_center_x, 0.]))
        scenario = replace(scenario, obstacles=(wall,),
                           obstacle_zone_x=tuple(np.asarray(scenario.obstacle_zone_x) + wall_center_x),
                           name=scenario.name + f"_wallx{wall_center_x:g}")
        validate_s4_branching_scenario(env, scenario)
        record["scenario"] = scenario_metadata(scenario)
        record["route_validation"] = s4_branch_route_metrics(env, scenario)
        record["geometry_wall_center_x_m"] = wall_center_x
    return records
