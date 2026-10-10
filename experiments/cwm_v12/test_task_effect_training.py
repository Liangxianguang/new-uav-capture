import copy
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_task_effect import LocalTwoHead, architecture, batch_loss, physical_loss, normalization, pack_inputs, state_digest, median_selection, qualification, PairedTaskLabel


def sample_calls():
    rng = np.random.default_rng(44)
    result = []
    for i in range(3):
        anchor = rng.normal(size=(8, 4, 3))
        values = {"history": rng.normal(size=(8, 252)), "relative": rng.normal(size=(4, 6)), "proposed": np.stack([anchor, anchor + .2]),
                  "anchor": anchor, "backbone": rng.normal(size=(8, 3)), "target": rng.normal(size=(2, 8, 3)), "anchor_target": rng.normal(size=(8, 3)),
                  "valid": np.ones((2, 8), bool), "anchor_valid": np.ones(8, bool)}
        result.append({"record": {"group": "a" if i < 2 else "b"}, "values": values})
    return result


@pytest.mark.parametrize("kind", ["plain", "structured"])
def test_same_architecture_pairs_have_identical_initializations(kind):
    torch.manual_seed(974101)
    mse = LocalTwoHead(architecture(kind + "_mse"))
    torch.manual_seed(974101)
    task = LocalTwoHead(architecture(kind + "_task"))
    assert state_digest(mse) == state_digest(task)


@pytest.mark.parametrize("name", ["motion_only", "plain_mse", "structured_mse"])
def test_mse_variant_preserves_original_physical_loss(name):
    torch.set_num_threads(1)
    calls = sample_calls()
    mean, scale = normalization(calls, [0, 1, 2])
    model = LocalTwoHead(architecture(name))
    config = json.loads((ROOT / "training_protocol.json").read_text())["training"]
    expected, original_terms = physical_loss(model, calls, [0, 1, 2], mean, scale, {"a": 2, "b": 1}, config)
    actual, terms = batch_loss(model, name, calls, [0, 1, 2], mean, scale, {"a": 2, "b": 1}, config, {})
    assert torch.equal(actual, expected) and torch.equal(terms[:3], original_terms) and terms[-1] == 0.


def test_no_complete_support_keeps_only_masked_physical_loss():
    calls = sample_calls()
    for call in calls:
        call["values"]["valid"][:, 5:] = False
        call["values"]["anchor_valid"][5:] = False
    mean, scale = normalization(calls, [0, 1, 2])
    model = LocalTwoHead("plain")
    config = json.loads((ROOT / "training_protocol.json").read_text())["training"]
    expected, _ = physical_loss(model, calls, [0, 1, 2], mean, scale, {"a": 2, "b": 1}, config)
    actual, terms = batch_loss(model, "plain_task", calls, [0, 1, 2], mean, scale, {"a": 2, "b": 1}, config, {})
    assert torch.equal(actual, expected) and terms[-1] == 0.
    altered = copy.deepcopy(calls)
    for call in altered:
        call["values"]["target"][:, 5:] += 1e6
        call["values"]["anchor_target"][5:] -= 1e6
    changed, _ = batch_loss(model, "plain_task", altered, [0, 1, 2], mean, scale, {"a": 2, "b": 1}, config, {})
    assert torch.equal(actual, changed)


def test_auxiliary_term_does_not_change_public_network_inputs():
    calls = sample_calls()
    mean, scale = normalization(calls, [0, 1])
    before, _ = pack_inputs(calls, [0, 1, 2], mean, scale)
    calls[2]["values"]["target"] += 1e3
    calls[2]["values"]["anchor_target"] -= 1e3
    after, _ = pack_inputs(calls, [0, 1, 2], mean, scale)
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_auxiliary_group_weighting_includes_partial_calls_as_zero():
    class Label:
        def loss(self, response):
            return response.sum() * 0. + 3.
    calls = sample_calls()
    mean, scale = normalization(calls, [0, 1, 2])
    model = LocalTwoHead("plain")
    config = json.loads((ROOT / "training_protocol.json").read_text())["training"]
    _, terms = batch_loss(model, "plain_task", calls, [0, 1, 2], mean, scale, {"a": 2, "b": 1}, config, {0: Label()})
    assert terms[-1] == .75  # (3/2) / (1/2 + 1/2 + 1)


def fake_runs(protocol):
    return [{"configuration": name, "seed": seed, "development": {"group_equal_ade_m": .3 + .01 * i, "group_equal_paired_response_error_m": .005},
             "task_effect": {"group_equal_normalized_task_effect_loss": .001 if name.endswith("_task") else .01}}
            for name in protocol["models"] for i, seed in enumerate(protocol["training"]["seeds"])]


def fake_decisions(protocol, gains):
    result = {}
    for name in protocol["models"]:
        for seed in protocol["training"]["seeds"]:
            result[f"{name}_seed{seed}"] = [{"episode_index": i, "step": 2, "agent": 0, "group": f"g{i}", "metrics": {
                "model": {"regret_in_diagnostic_cost": 1. if name == "motion_only" else 0.},
                "own_motion_component": {"regret_in_diagnostic_cost": float(gains[i])}, "constant_velocity": {"regret_in_diagnostic_cost": 1.}}} for i in range(8)]
    return result


def test_fixed_median_not_best_seed():
    protocol = json.loads((ROOT / "training_protocol.json").read_text())
    assert all(seed == 974102 for seed in median_selection(fake_runs(protocol), protocol).values())


def test_zero_or_unstable_response_gain_cannot_qualify():
    protocol = json.loads((ROOT / "training_protocol.json").read_text())
    controls = {"frozen_gru": {"group_equal_ade_m": .8, "group_equal_paired_response_error_m": .02}, "constant_velocity": {"group_equal_ade_m": .5}}
    _, gates = qualification(fake_runs(protocol), controls, fake_decisions(protocol, [0.] * 8), protocol)
    assert not any(g["research_eligible"] for g in gates.values())
    _, gates = qualification(fake_runs(protocol), controls, fake_decisions(protocol, [-1., 1.] * 4), protocol)
    assert not any(g["research_eligible"] for g in gates.values())


def test_prediction_gain_does_not_cover_failed_response_error_gate():
    protocol = json.loads((ROOT / "training_protocol.json").read_text())
    runs = fake_runs(protocol)
    for row in runs:
        row["development"]["group_equal_paired_response_error_m"] = .03
    controls = {"frozen_gru": {"group_equal_ade_m": .8, "group_equal_paired_response_error_m": .02}, "constant_velocity": {"group_equal_ade_m": .5}}
    _, gates = qualification(runs, controls, fake_decisions(protocol, [1.] * 8), protocol)
    assert not any(g["research_eligible"] for g in gates.values())
    assert all(not g["checks"]["median_response_error_vs_zero"] for g in gates.values())


def test_cached_task_supervision_exactly_matches_verified_v11_loss(tmp_path):
    from task_score import FrozenLocalScore, task_effect_loss
    from local_shadow_release import frozen_configs
    from local_shadow import restore_public_call
    from geometry_release import arrays
    configs = frozen_configs(ROOT.parent / "cwm_v1/baseline/capsule.zip", tmp_path / "restored")
    with zipfile.ZipFile(ROOT.parent / "cwm_v10/artifacts/two_head_training_20261010.zip") as source:
        row = next(r for r in json.loads(source.read("data/calls.json")) if "arrays_path" in r and r["all_candidates_full_horizon_valid"])
        values = arrays(source.read("data/" + row["arrays_path"]))
        call = restore_public_call(json.loads(source.read("data/" + row["context_path"])), *configs, values["backbone"])
    score = FrozenLocalScore(call, values["proposed"][:, :, row["agent"]])
    cache = PairedTaskLabel(score, values)
    a = torch.full(values["target"].shape, .025, dtype=torch.float64, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    actual = cache.loss(a)
    expected = task_effect_loss(score, b, torch.from_numpy(values["target"]), torch.from_numpy(values["anchor_target"]), torch.from_numpy(values["valid"]), torch.from_numpy(values["anchor_valid"]))
    assert torch.equal(actual, expected)
    actual.backward()
    expected.backward()
    assert torch.equal(a.grad, b.grad)
    assert not cache.reference.requires_grad and not cache.effect.requires_grad and not cache.scale.requires_grad


def test_cached_supervision_rejects_terminal_padding():
    values = sample_calls()[0]["values"]
    values["anchor_valid"][-1] = False
    with pytest.raises(ValueError, match="Full valid terminal support"):
        PairedTaskLabel(None, values)
