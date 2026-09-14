from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_phase65_tail_stress_scenes.py"
SPEC = importlib.util.spec_from_file_location("generate_phase65_tail_stress_scenes", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_phase65_declares_strictly_extended_delay_noise_levels() -> None:
    assert max(item[0] for item in MODULE.EXECUTION_TAIL) == 12
    assert max(item[1] for item in MODULE.EXECUTION_TAIL) == 0.20
    assert max(item[0] for item in MODULE.COMMUNICATION_TAIL) == 8
    assert max(item[1] for item in MODULE.COMMUNICATION_TAIL) == 0.40
    assert max(item[2] for item in MODULE.EXECUTION_TAIL) == 0.30
    assert max(item[3] for item in MODULE.EXECUTION_TAIL) == 0.10


def test_phase65_small_matrix_preserves_mirror_and_split_contract() -> None:
    protocol = MODULE.load_protocol(MODULE.DEFAULT_PROTOCOL)
    records = MODULE.build_records(
        protocol,
        MODULE.DEFAULT_ENVIRONMENT_CONFIG,
        groups_per_block=3,
        seed_start=997101,
        rollout_policy="test_phase65",
    )
    MODULE.validate_records(records, groups_per_block=3)
    assert len(records) == 24
    assert {record["defender_bias"] for record in records} == {"upper", "lower"}
    assert {record["evaluation_split"] for record in records} == set(MODULE.SPLITS)
    _condition, _pursuit, execution, _factors = MODULE.condition_for(1, 44)
    assert execution["action_delay_steps"] == 12
    _condition, pursuit, _execution, _factors = MODULE.condition_for(2, 26)
    assert pursuit["message_delay_steps"] == 8
