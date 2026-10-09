import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from diagnose import commands, pack, rollout, summarize

PROTOCOL = json.loads(Path(__file__).with_name("protocol.json").read_text())


def test_single_agent_public_commands():
    sequence = np.ones((8, 4, 3))
    positions = np.arange(12, dtype=float).reshape(4, 3)
    before = positions.copy()
    output = commands(sequence, {"defender_positions": positions}, np.zeros(3), PROTOCOL)
    assert output.shape == (13, 24, 4, 3)
    assert np.array_equal(output[0, :8], sequence)
    assert np.array_equal(output[0, 8:], np.repeat(sequence[-1:], 16, axis=0))
    assert np.array_equal(positions, before)
    assert np.linalg.norm(output, axis=-1).max() <= 5.00000001
    for agent in range(4):
        others = [i for i in range(4) if i != agent]
        for branch in range(1 + 3 * agent, 4 + 3 * agent):
            assert np.array_equal(output[branch][:, others], output[0][:, others])
        assert not output[3 + 3 * agent, :, agent].any()
    with pytest.raises(ValueError):
        commands(sequence[:7], {"defender_positions": positions}, np.zeros(3), PROTOCOL)


class FakeEnv:
    def __init__(self):
        self.target_maneuver_last_replan_step = 0
        self.target_position = np.zeros(3)
        self.defender_positions = np.zeros((4, 3))
        self.last_executed_actions = np.zeros((4, 3))
        self.target_maneuver_observed_positions = np.zeros((4, 3))
        self.target_maneuver_direction = np.zeros(3)
        self.target_maneuver_mode = "test"
        self.target_maneuver_route = "test"
        self.count = 0

    def observe(self):
        return {}

    def step(self, command):
        self.count += 1
        assert self.count <= 2, "Simulated after terminal"
        self.target_position += self.rng.normal(size=3)
        self.last_executed_actions = command.copy()
        return {}, 0., self.count == 2, False, {"termination_reason": "capture" if self.count == 2 else "running"}


class FakeFilter:
    def __init__(self, env):
        pass

    def filter(self, command, observation):
        return command, {}


def test_clone_terminal_masks_and_repeat():
    base = SimpleNamespace(env_class=FakeEnv, filter_class=FakeFilter)
    parent = FakeEnv()
    sequence = np.ones((24, 4, 3))
    a = pack([rollout(base, parent, sequence, 123)], 24)
    b = pack([rollout(base, parent, sequence, 123)], 24)
    assert parent.count == 0
    assert all(a[k].tobytes() == b[k].tobytes() for k in a)
    assert a["valid"].sum() == 2
    assert not a["valid"][0, 2:].any()
    assert np.all(a["termination"][0, 2:] == "after_terminal")


def test_summary_censoring_and_independent_group_counts():
    targets = np.zeros((2, 13, 24, 3))
    targets[:, 1:, :, 0] = .1
    masks = np.ones((2, 13, 24), dtype=bool)
    masks[0, :, 12:] = False
    data = {"target": targets, "valid": masks, "variant": np.array(["fast_target"] * 2), "group": np.array(["a", "b"]),
            "mode": np.full((2, 13, 24), "mode"), "executed": np.zeros((2, 13, 24, 4, 3)), "sensor": np.zeros((2, 13, 24, 4, 3))}
    protocol = {**PROTOCOL, "variants": {"fast_target": 7}}
    result = summarize(data, protocol)["fast_target"]
    assert result["8"]["full_horizon_pairs"] == 24
    assert result["24"]["full_horizon_pairs"] == 12
    assert result["24"]["signal_groups"] == ["a", "b"]
    assert result["24"]["full_horizon_signal_groups"] == ["b"]
    assert result["24"]["full_horizon_endpoint_effect_m"]["mean"] == pytest.approx(.1)
