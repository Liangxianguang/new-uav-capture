from __future__ import annotations

from pathlib import Path

from scripts.evaluate_minimax_mpc_s3 import episode_spec, load_protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_s3_mpc_episode_spec_is_reproducible_and_covers_protocol_fields() -> None:
    protocol_path = PROJECT_ROOT / "configs" / "central_random_mixed_obstacle_s3_v5_protocol.yaml"
    protocol = load_protocol(protocol_path)
    first = episode_spec(protocol, "validation", 0)
    second = episode_spec(protocol, "validation", 0)
    assert first == second
    assert first["episode_seed"] == 646101
    assert first["layout_seed"] == 1646101
    assert first["observation_condition"] in {"nominal", "delayed_noisy"}
    assert first["target_motion_mode"] in {"flee_persistence", "s_curve"}
    assert 3 <= first["obstacle_count"] <= 5


def test_s3_mpc_protocol_uses_distinct_split_seed_blocks() -> None:
    protocol = load_protocol(PROJECT_ROOT / "configs" / "central_random_mixed_obstacle_s3_v5_protocol.yaml")
    blocks = [int(protocol["seed_blocks"][split]) for split in ("train", "validation", "locked_test")]
    assert len(set(blocks)) == 3
