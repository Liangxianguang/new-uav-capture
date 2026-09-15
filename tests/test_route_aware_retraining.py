from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from encirclement3d.pursuit_controllers import (  # noqa: E402
    PublicBeliefRouteIntentController,
    SafetyFilteredPursuitController,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    central_mixed_obstacle_scenario,
    configure_target_crossing_episode,
    prepare_showcase_episode,
)
import train_capture_radius_recurrent_behavior_cloning as trainer  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def _environment() -> dict:
    return yaml.safe_load((ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml").read_text(encoding="utf-8"))


def test_public_belief_route_teacher_emits_finite_route_features_without_target_truth() -> None:
    config = _environment()
    config["task"]["pursuit"]["target_motion_mode"] = "flee_persistence"
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=0, target_speed_scale=0.55)
    scenario = central_mixed_obstacle_scenario(
        initial_side_distance=7.0,
        target_crossing_required=True,
        layout="mixed",
        defender_side="left",
    )
    configure_target_crossing_episode(env)
    observation = prepare_showcase_episode(env, scenario, seed=761201, record_history=False)
    teacher = PublicBeliefRouteIntentController(env)

    features = teacher.route_features(observation)
    assert features.shape == (env.n_defenders, 7)
    assert np.isfinite(features).all()
    assert np.allclose(features[:, 3:6].sum(axis=1), 1.0)
    action = SafetyFilteredPursuitController(teacher).act(observation)
    assert action.shape == (env.n_defenders, 3)
    assert np.isfinite(action).all()


def test_route_teacher_is_selected_explicitly() -> None:
    config = _environment()
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=3, target_speed_scale=0.55)
    controller = trainer.build_expert_controller(
        env,
        {
            "expert_controller": "public_belief_route_intent_v1",
            "teacher_horizon_seconds": 0.75,
            "teacher_replan_interval_steps": 8,
            "teacher_min_hold_steps": 6,
            "teacher_grid_step_m": 0.75,
            "teacher_route_margin_m": 0.85,
        },
    )
    assert isinstance(controller, SafetyFilteredPursuitController)
    assert isinstance(controller.controller, PublicBeliefRouteIntentController)
