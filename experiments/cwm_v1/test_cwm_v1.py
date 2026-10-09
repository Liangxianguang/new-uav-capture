from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from collect_pairs import interventions, split_records
from freeze_baseline import restore
from model import ResponseModel, group_loss
from plugin import ResponsePlugin


def test_off_does_not_call_or_modify():
    def fail(_):
        raise AssertionError("Off mode called predictor")
    action = np.arange(12).reshape(4, 3)
    plugin = ResponsePlugin(predictor=fail)
    result = plugin.observe(action, {})
    assert np.array_equal(result, action)
    result[:] = -1
    assert action[0, 0] == 0
    assert plugin.last_diagnostics["predictor_called"] is False


@pytest.mark.parametrize("predictor", [None, lambda _: np.full((4, 8, 3), np.nan), lambda _: np.zeros(3)])
def test_shadow_invalid_returns_baseline(predictor):
    action = np.ones((4, 3))
    plugin = ResponsePlugin(mode="shadow", predictor=predictor)
    assert np.array_equal(plugin.observe(action, {}), action)
    assert plugin.last_diagnostics["status"] == "shadow_error"


def test_shadow_input_isolation_and_guard_closed():
    inputs = {"position": np.zeros(3)}
    def predictor(values):
        values["position"][:] = 100
        return np.zeros((4, 8, 3))
    plugin = ResponsePlugin(mode="shadow", predictor=predictor)
    action = np.ones((4, 3))
    assert np.array_equal(plugin.observe(action, inputs), action)
    assert np.array_equal(inputs["position"], np.zeros(3))
    with pytest.raises(ValueError):
        ResponsePlugin(mode="guarded")


def test_commands_are_bounded_and_original_unmodified():
    original = np.ones((8, 4, 3)) * 2
    result = interventions(original)
    assert result.shape == (4, 8, 4, 3)
    assert np.array_equal(result[0], original)
    assert np.max(np.linalg.norm(result, axis=-1)) <= 5 + 1e-12
    assert np.array_equal(original, np.ones_like(original) * 2)


def test_structural_pressure_permutation_invariant():
    torch.manual_seed(7)
    model = ResponseModel("structured")
    relative, actions = torch.randn(2, 4, 6), torch.randn(2, 8, 4, 3)
    order = [3, 1, 0, 2]
    assert torch.allclose(model.pressure(relative, actions), model.pressure(relative[:, order], actions[:, :, order]), atol=1e-7)
    assert torch.count_nonzero(model.pressure(relative, torch.zeros_like(actions))) == 0


def test_matched_capacity_and_finite_backward():
    plain, structured = ResponseModel("plain"), ResponseModel("structured")
    sizes = [sum(p.numel() for p in model.parameters()) for model in [plain, structured]]
    assert abs(sizes[0] - sizes[1]) / max(sizes) < .1
    for model in [plain, structured]:
        predicted = model(torch.randn(8, 8, 252), torch.randn(8, 4, 6), torch.randn(8, 8, 4, 3), torch.randn(8, 3))
        loss, _, _ = group_loss(predicted.reshape(2, 4, 8, 3), torch.zeros(2, 4, 8, 3), torch.ones(2, 4, 8, dtype=torch.bool))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_zip_traversal_rejected(tmp_path):
    capsule = tmp_path / "bad.zip"
    with zipfile.ZipFile(capsule, "w") as archive:
        archive.writestr("CAPSULE_MANIFEST.json", '{"files": {}}')
        archive.writestr("../outside", "bad")
    with pytest.raises(ValueError):
        restore(capsule, tmp_path / "restored")
    assert not (tmp_path / "outside").exists()
