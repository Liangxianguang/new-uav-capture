from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.aggregate_phase71_validation_seed_results import aggregate


def _write_result(directory: Path, seed_block: int, safe: list[bool]) -> None:
    directory.mkdir()
    (directory / "evaluation_metadata.json").write_text(
        json.dumps({"split": "validation", "not_a_locked_test": True, "locked_test": False, "seed_block": seed_block}),
        encoding="utf-8",
    )
    rows = [
        "safe_capture_success,capture_event,collision,world_violation_steps,termination_reason,min_clearance_m,capture_time_seconds,target_maneuver_switch_count",
    ]
    for value in safe:
        rows.append(f"{str(value).lower()},{str(value).lower()},false,0,{'safe_capture' if value else 'timeout'},0.4,{'4.0' if value else ''},2")
    (directory / "episodes.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_phase71_aggregation_reports_seed_variation_and_bootstrap(tmp_path: Path) -> None:
    first = tmp_path / "seed1"
    second = tmp_path / "seed2"
    _write_result(first, 711201, [True, False])
    _write_result(second, 711202, [True, True])

    report = aggregate([first, second], title="test", bootstrap_samples=100, bootstrap_seed=9)

    assert report["not_a_locked_test"] is True
    assert report["locked_test_included"] is False
    assert report["seed_blocks"] == [711201, 711202]
    assert report["total_episodes"] == 4
    assert report["pooled"]["safe_capture_rate"] == 0.75
    assert report["seed_variation"]["safe_capture_rate"]["min"] == 0.5
    assert len(report["pooled"]["bootstrap_ci95"]["safe_capture_rate"]) == 2


def test_phase71_aggregation_rejects_locked_test_metadata(tmp_path: Path) -> None:
    directory = tmp_path / "locked"
    directory.mkdir()
    (directory / "evaluation_metadata.json").write_text(
        json.dumps({"split": "locked_test", "not_a_locked_test": False, "locked_test": True, "seed_block": 1}),
        encoding="utf-8",
    )
    (directory / "episodes.csv").write_text("x\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Refusing non-validation input"):
        aggregate([directory, directory], title="test", bootstrap_samples=10, bootstrap_seed=1)
