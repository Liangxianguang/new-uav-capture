from __future__ import annotations

from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encirclement3d.residual_actor import RecurrentResidualActor  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def test_residual_actor_is_bounded_and_recurrent() -> None:
    actor = RecurrentResidualActor(local_observation_dim=11, hidden_dim=16, residual_scale=2.5)
    hidden = actor.initial_hidden(4)
    local = torch.randn(4, 11)
    base = torch.full((4, 3), 1.0)
    action, next_hidden, residual = actor.step(local, base, hidden, action_scale=5.0)
    assert action.shape == (4, 3)
    assert residual.shape == (4, 3)
    assert next_hidden.shape == (4, 16)
    assert torch.all(action <= 5.0)
    assert torch.all(action >= -5.0)
    assert not torch.equal(hidden, next_hidden)


def test_phase79_requires_192_demos_and_keeps_phase78_contract() -> None:
    payload = yaml.safe_load(
        (ROOT / "configs" / "phase79_dnmcp_dagger_residual.yaml").read_text(encoding="utf-8")
    )
    assert payload["experiment"]["demo_episodes"] >= 192
    assert payload["experiment"]["dagger_rounds"] >= 1
    assert payload["experiment"]["episodes_per_round"] > 0
    assert payload["environment_overrides"]["task"]["pursuit"]["target_maneuver_crossing_gain"] == 0.0
    assert payload["environment_overrides"]["task"]["pursuit"]["target_maneuver_enable_reverse_lane_change"] is False
    assert payload["experiment"]["filtered_route_base"] is True
