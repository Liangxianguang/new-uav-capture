import copy
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT.parent / "cwm_v9")]
from task_score import FrozenLocalScore, original_scores, task_effect_loss
from local_shadow_release import frozen_configs
from local_shadow import restore_public_call
from geometry_release import arrays


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    torch.set_num_threads(1)
    configs = frozen_configs(ROOT.parent / "cwm_v1/baseline/capsule.zip", tmp_path_factory.mktemp("taskscore") / "restored")
    with zipfile.ZipFile(ROOT.parent / "cwm_v10/artifacts/two_head_training_20261010.zip") as archive:
        rows = json.loads(archive.read("data/calls.json"))
        row = next(r for r in rows if r.get("all_candidates_full_horizon_valid") and "arrays_path" in r)
        values = arrays(archive.read("data/" + row["arrays_path"]))
        call = restore_public_call(json.loads(archive.read("data/" + row["context_path"])), *configs, values["backbone"])
    actions = values["proposed"][:, :, row["agent"]]
    return call, actions, values


@pytest.mark.parametrize("mode", ["expected", "worst_case", "cvar"])
def test_full_score_matches_original_risk_and_role_switches(example, mode):
    from dataclasses import replace
    call, actions, values = copy.deepcopy(example)
    call["planner"].config = replace(call["planner"].config, risk_mode=mode)
    score = FrozenLocalScore(call, actions)
    for peer in score.peer_positions.numpy():
        paths = np.repeat((peer[None] + np.arange(8)[:, None] * .13)[None], len(actions), 0)
        assert np.allclose(score(torch.from_numpy(paths)).numpy(), original_scores(call, actions, paths), rtol=1e-10, atol=1e-8)


@pytest.mark.parametrize("flag", ["fc_dbf_enabled", "escape_gap_cost_enabled", "reachability_normalized_cost_enabled"])
def test_unsupported_objectives_rejected_not_omitted(example, flag):
    from dataclasses import replace
    call, actions, _ = copy.deepcopy(example)
    call["planner"].config = replace(call["planner"].config, **{flag: True})
    with pytest.raises(ValueError, match="Unsupported"):
        FrozenLocalScore(call, actions)


def test_execution_aware_objective_is_not_silently_replaced(example):
    call, actions, _ = copy.deepcopy(example)
    call["observation"]["qdr"] = {"execution_aware_action_rollout": True}
    with pytest.raises(ValueError, match="Unsupported"):
        FrozenLocalScore(call, actions)


def test_target_independent_safety_control_costs_retained(example):
    from dataclasses import replace
    call, actions, values = copy.deepcopy(example)
    score = FrozenLocalScore(call, actions)
    safety = copy.deepcopy(call)
    safety["planner"].config = replace(call["planner"].config, weight_distance=0., weight_capture_hinge=0., weight_formation=0., weight_terminal_distance=0.)
    zero = np.zeros((len(actions), 8, 3))
    assert np.allclose(score.offset.numpy(), original_scores(safety, actions, zero), rtol=1e-10, atol=1e-8)
    assert np.all(score.offset.numpy() >= 0.)


def test_exact_response_has_zero_task_effect_loss_and_motion_labels_detached(example):
    call, actions, values = example
    score = FrozenLocalScore(call, actions)
    anchor = torch.from_numpy(values["anchor_target"]).requires_grad_(True)
    target = torch.from_numpy(values["target"]).requires_grad_(True)
    response = (target.detach() - anchor.detach()[None]).requires_grad_(True)
    valid, anchor_valid = torch.from_numpy(values["valid"]), torch.from_numpy(values["anchor_valid"])
    loss = task_effect_loss(score, response, target, anchor, valid, anchor_valid)
    assert loss.item() == 0.
    loss.backward()
    assert anchor.grad is None and target.grad is None and response.grad is not None


def test_imperfect_response_effect_loss_backpropagates(example):
    call, actions, values = example
    score = FrozenLocalScore(call, actions)
    response = torch.full((len(actions), 8, 3), .04, dtype=torch.float64, requires_grad=True)
    loss = task_effect_loss(score, response, torch.from_numpy(values["target"]), torch.from_numpy(values["anchor_target"]),
                            torch.from_numpy(values["valid"]), torch.from_numpy(values["anchor_valid"]))
    loss.backward()
    assert torch.isfinite(response.grad).all() and response.grad.abs().sum() > 0 and loss > 0


def test_partial_terminal_support_cannot_be_used_as_task_label(example):
    call, actions, values = example
    valid = torch.from_numpy(values["valid"]).clone()
    valid[0, -1] = False
    with pytest.raises(ValueError, match="complete common terminal"):
        task_effect_loss(FrozenLocalScore(call, actions), torch.zeros_like(torch.from_numpy(values["target"])),
                         torch.from_numpy(values["target"]), torch.from_numpy(values["anchor_target"]), valid, torch.from_numpy(values["anchor_valid"]))


def test_speed_clipping_matches_original(example):
    call, actions, values = example
    enlarged = actions * 10.
    paths = np.repeat(values["backbone"][None], len(actions), 0)
    assert np.allclose(FrozenLocalScore(call, enlarged)(torch.from_numpy(paths)).numpy(), original_scores(call, enlarged, paths), rtol=1e-10, atol=1e-8)
