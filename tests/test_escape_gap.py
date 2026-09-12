import numpy as np
import pytest

from encirclement3d.escape_gap import escape_gap_metrics
from encirclement3d.minimax_mpc import MinimaxMPCConfig


def _static_target_and_team(points: np.ndarray, horizon: int = 2) -> tuple[np.ndarray, np.ndarray]:
    team = np.repeat(np.asarray(points, dtype=np.float64)[None, None, :, :], horizon, axis=1)
    target = np.zeros((1, horizon, 3), dtype=np.float64)
    return team, target


def test_uniform_four_defender_ring_has_no_gap_penalty_above_safe_threshold():
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    team, target = _static_target_and_team(points)
    result = escape_gap_metrics(team, target, gap_safe_rad=1.80, escape_gap_safe_rad=1.80)
    assert result["cost"].shape == (1, 1)
    assert result["cost"][0, 0] == pytest.approx(0.0)
    assert result["max_gap_rad"][0, 0] == pytest.approx(np.pi / 2.0)
    assert result["escape_gap_rad"][0, 0] == pytest.approx(np.pi / 2.0)
    assert result["coverage_ratio"][0, 0] == pytest.approx(0.75)


def test_escape_heading_selects_a_nonmaximum_gap_when_available():
    angles = np.deg2rad([0.0, 60.0, 180.0, 240.0])
    points = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(4)])
    target_step = np.asarray([0.1, 0.1 * np.tan(np.deg2rad(30.0)), 0.0])
    team = np.stack([points, points + target_step], axis=0)[None, ...]
    # The target moves at 30 degrees, which lies in the 0--60 degree
    # gap rather than one of the 120 degree gaps.
    target = np.asarray([[[0.0, 0.0, 0.0], target_step]])
    result = escape_gap_metrics(team, target, gap_safe_rad=0.0, escape_gap_safe_rad=0.0)
    assert result["max_gap_rad"][0, 0] == pytest.approx(np.deg2rad(120.0))
    assert result["escape_gap_rad"][0, 0] == pytest.approx(np.deg2rad(60.0))


def test_gap_metric_vectorizes_sequences_and_candidates():
    points = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    team, target = _static_target_and_team(points, horizon=3)
    team = np.concatenate([team, team * np.asarray([1.0, 1.0, 1.0, 1.0])[None, None, :, None]], axis=0)
    targets = np.concatenate([target, target + np.asarray([0.2, 0.0, 0.0])[None, None, :]], axis=0)
    result = escape_gap_metrics(team, targets, horizon_discount=0.9)
    assert result["cost"].shape == (2, 2)
    assert all(np.isfinite(value).all() for value in result.values())


def test_mpc_config_accepts_egc_fields_and_rejects_invalid_angle():
    config = MinimaxMPCConfig.from_mapping(
        {
            "escape_gap_cost_enabled": True,
            "weight_escape_gap": 0.3,
            "escape_gap_safe_rad": 1.8,
            "escape_gap_escape_safe_rad": 1.8,
        }
    )
    assert config.escape_gap_cost_enabled is True
    assert config.weight_escape_gap == pytest.approx(0.3)
    with pytest.raises(ValueError, match=r"2\*pi"):
        MinimaxMPCConfig(escape_gap_safe_rad=2.0 * np.pi + 0.1)
