"""Lightweight frozen-scene IO for the PPO baseline trainers.

The RL baseline should not import the full MPC evaluator merely to read scene
records.  Keeping this small adapter separate also allows the CUDA-enabled
Python 3.8 environment to run the baseline without parsing unrelated
Python-3.10-only evaluator code.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv
from encirclement3d.showcase import prepare_showcase_episode, scenario_from_metadata


def read_scenes(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in path.resolve().read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records.sort(key=lambda value: int(value["episode_index"]))
    if limit is not None:
        if limit <= 0:
            raise ValueError("episodes must be positive when supplied.")
        records = records[:limit]
    if not records:
        raise ValueError("The frozen scene file contains no records.")
    for record in records:
        nested_spec = record.get("spec")
        if isinstance(nested_spec, dict):
            for key in (
                "episode_seed",
                "layout_seed",
                "target_speed_scale",
                "defender_side",
                "initial_side_distance",
                "target_motion_mode",
                "target_crossing_required",
                "observation_condition",
                "pursuit_overrides",
                "obstacle_count",
                "mirror_group_id",
                "mirror_pair_member",
                "scene_block",
            ):
                if key not in record and key in nested_spec:
                    record[key] = copy.deepcopy(nested_spec[key])
    return records


def load_records(
    path: Path,
    limit: int | None,
    *,
    allow_target_crossing: bool = False,
) -> list[dict[str, Any]]:
    records = read_scenes(path.resolve(), limit)
    if not records:
        raise ValueError("The calibration scene file is empty.")
    if (not allow_target_crossing) and any(
        bool(item.get("target_crossing_required", False)) for item in records
    ):
        raise ValueError("RL baseline refuses target-crossing scenes.")
    if any("locked" in str(item.get("scene_block", "")).lower() for item in records):
        raise ValueError("RL baseline refuses locked-test records.")
    return records


def build_environment(
    environment_config: dict[str, Any],
    record: dict[str, Any],
    *,
    max_steps: int | None,
) -> tuple[CaptureRadiusPursuit3DEnv, Any, dict[str, Any]]:
    config = copy.deepcopy(environment_config)
    pursuit = config.setdefault("task", {}).setdefault("pursuit", {})
    pursuit.update(copy.deepcopy(record["pursuit_overrides"]))
    pursuit["target_motion_mode"] = str(record.get("target_motion_mode", "adaptive_maneuvering"))
    execution = config.setdefault("dynamics", {}).setdefault("execution", {})
    record_execution = copy.deepcopy(record.get("execution", {}))
    if "command_noise_std_mps" in record_execution and "command_noise_std" not in record_execution:
        record_execution["command_noise_std"] = record_execution.pop("command_noise_std_mps")
    execution.update(record_execution)
    config.setdefault(
        "experiments",
        [{"obstacle_count": int(record.get("obstacle_count", 3)), "target_speed_scale": float(record["target_speed_scale"])}],
    )
    config["world"]["max_steps"] = int(max_steps if max_steps is not None else config["world"].get("max_steps", 250))
    scenario = scenario_from_metadata(record["scenario"])
    env = CaptureRadiusPursuit3DEnv(
        config,
        obstacle_count=int(record.get("obstacle_count", len(scenario.obstacles))),
        target_speed_scale=float(record["target_speed_scale"]),
    )
    observation = prepare_showcase_episode(
        env,
        scenario,
        seed=int(record["episode_seed"]),
        record_history=True,
        validate_scenario=False,
    )
    return env, observation, scenario
