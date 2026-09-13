from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_s4_branching import load_protocol  # noqa: E402
from generate_phase27_rnic_confirmation_scenes import (  # noqa: E402
    DEFAULT_PROTOCOL,
    build_records,
    validate_records,
)


def test_phase27_generator_is_mirror_paired_and_training_support_bounded() -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    records = build_records(protocol, Path("configs/capture_radius_pursuit_central_v4_flee.yaml"), 8, 781101)

    validate_records(records)
    assert len(records) == 8
    assert len({record["layout_seed"] for record in records}) == 4
    assert {record["defender_bias"] for record in records} == {"upper", "lower"}
    assert all(record["scene_block"] == "phase27_rnic_fresh_validation" for record in records)


def test_phase27_generator_rejects_odd_episode_count() -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    with pytest.raises(ValueError, match="pair"):
        build_records(protocol, Path("configs/capture_radius_pursuit_central_v4_flee.yaml"), 3, 781101)


def test_generator_supports_named_validation_blocks() -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    records = build_records(
        protocol,
        Path("configs/capture_radius_pursuit_central_v4_flee.yaml"),
        2,
        991101,
        scene_block="phase34_qdr_fresh_validation",
        rollout_policy="qdr_precondition_evaluation_only",
    )
    assert {record["scene_block"] for record in records} == {"phase34_qdr_fresh_validation"}
    assert {record["rollout_policy"] for record in records} == {"qdr_precondition_evaluation_only"}
