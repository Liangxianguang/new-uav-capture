from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from encirclement3d.observation_encoding import policy_observations
from encirclement3d.pursuit_controllers import PublicBeliefRouteIntentController
from rl_scene_io import build_environment, read_scenes
from train_mappo_ippo_baseline import load_config, make_route_helper


ROOT = Path(__file__).resolve().parents[1]


def _route_environment_and_observation():
    config_path = ROOT / "configs" / "phase86_rl_warmup_route_intent_margin025.yaml"
    _document, environment = load_config(config_path)
    records = read_scenes(
        ROOT / "results" / "phase86_p1_speed005_warmup_development" / "scenes.jsonl",
        1,
    )
    return build_environment(environment, records[0], max_steps=120)


def test_route_intent_requires_a_persistent_helper() -> None:
    env, observation, _scenario = _route_environment_and_observation()
    with pytest.raises(ValueError, match="persistent public-belief route helper"):
        policy_observations(env, observation)


def test_route_intent_contract_is_seven_features_and_77_dimensions() -> None:
    env, observation, _scenario = _route_environment_and_observation()
    helper = make_route_helper(env)
    assert isinstance(helper, PublicBeliefRouteIntentController)
    encoded = policy_observations(env, observation, route_helper=helper)
    assert encoded.shape == (4, 77)
    assert encoded.dtype.name == "float32"
