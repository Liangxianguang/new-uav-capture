from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

from scripts.generate_phase73_crossing_scene_dataset import (
    generate,
    validate_dataset_contract,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs" / "phase73_crossing_scene_dataset.yaml"
ENVIRONMENT = ROOT / "configs" / "phase70_maneuvering_adversary_v2.yaml"


def test_phase73_contract_rejects_locked_test_and_requires_500_even_scenes() -> None:
    settings = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    assert validate_dataset_contract(settings)[0] == 600
    bad = copy.deepcopy(settings)
    bad["not_a_locked_test"] = False
    try:
        validate_dataset_contract(bad)
    except ValueError as error:
        assert "not_a_locked_test" in str(error)
    else:
        raise AssertionError("locked-test-like dataset contract was accepted")


def test_phase73_small_override_writes_mirrored_crossing_records(tmp_path: Path) -> None:
    manifest = generate(PROTOCOL, ENVIRONMENT, tmp_path / "dataset", episodes_override=6, allow_small=True)
    assert manifest["total_scenes"] == 6
    assert manifest["mirror_groups"] == 3
    assert manifest["all_mirror_groups_have_two_members"] is True
    assert manifest["target_crossing_required_rate"] == 1.0
    assert manifest["direct_path_blocked_rate"] == 1.0
    assert manifest["minimum_bypass_route_count"] >= 2

    records = [json.loads(line) for line in (tmp_path / "dataset" / "scenes.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 6
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records:
        grouped.setdefault(str(record["spec"]["mirror_group_id"]), []).append(record)
        assert record["spec"]["target_crossing_required"] is True
        assert record["route_certificate"]["direct_path_blocked"] is True
        assert int(record["route_certificate"]["bypass_route_count"]) >= 2
    assert all(len(items) == 2 for items in grouped.values())

    pair = next(iter(grouped.values()))
    left = next(item for item in pair if item["spec"]["mirror_pair_member"] == "left")
    right = next(item for item in pair if item["spec"]["mirror_pair_member"] == "right")
    left_scenario = left["scenario"]
    right_scenario = right["scenario"]
    assert np.allclose(
        np.asarray(left_scenario["target_position"], dtype=float) * np.array([-1.0, 1.0, 1.0]),
        np.asarray(right_scenario["target_position"], dtype=float),
    )
    left_centers = sorted(tuple(np.round(item["center_xy"], 6)) for item in left_scenario["obstacles"])
    right_centers = sorted(tuple(np.round(item["center_xy"], 6)) for item in right_scenario["obstacles"])
    mirrored_left_centers = sorted(
        tuple(np.round(np.asarray(center) * np.array([-1.0, 1.0]), 6)) for center in left_centers
    )
    assert right_centers == mirrored_left_centers
