from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from merge_phase82_expert_calibration_parts import merge  # noqa: E402


def _row(index: int, accepted: bool) -> dict[str, object]:
    return {
        "episode_index": index,
        "mirror_group_id": f"mirror-{index}",
        "difficulty": "nominal",
        "expert_calibration_accepted": accepted,
        "expert_route_exercise_accepted": accepted,
        "safe_capture_in_pursuit": accepted,
        "capture_event": accepted,
        "physical_collision": False,
        "target_obstacle_collision": False,
        "target_boundary_violation": False,
        "target_invalid_episode": False,
        "defender_boundary_violation": False,
        "timeout": not accepted,
        "target_zone_entered": False,
        "target_crossed": False,
        "target_maneuver_fallback_count": 0,
        "min_clearance_m": 1.0,
        "capture_time_seconds": 1.0,
        "route_intent": "direct",
        "termination_reason": "safe_capture" if accepted else "timeout",
        "latency_samples_ms": [1.0, 2.0],
    }


def _write_part(path: Path, rows: list[dict[str, object]], source_hash: str) -> None:
    path.mkdir()
    (path / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    (path / "summary.json").write_text(
        json.dumps(
            {
                "locked_test_used": False,
                "source_scene_file_sha256": source_hash,
                "controller": "public_belief_route_intent_v1",
                "controller_mode": "public_belief_route",
                "expert_acceptance_contract": {},
            }
        ),
        encoding="utf-8",
    )


def test_merge_requires_exact_scene_coverage(tmp_path: Path) -> None:
    scenes_path = tmp_path / "scenes.jsonl"
    scenes = [
        {"episode_index": index, "variant": "command_noise"}
        for index in range(4)
    ]
    scenes_path.write_text("".join(json.dumps(row) + "\n" for row in scenes), encoding="utf-8")
    source_hash = hashlib.sha256(scenes_path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"not_a_locked_test": True, "locked_test": False}), encoding="utf-8"
    )
    first = tmp_path / "part1"
    second = tmp_path / "part2"
    _write_part(first, [_row(0, True), _row(1, False)], source_hash)
    _write_part(second, [_row(2, True), _row(3, True)], source_hash)

    summary = merge(scenes_path, [first, second], tmp_path / "merged")

    assert summary["raw_pool_episodes"] == 4
    assert summary["accepted_calibration_episodes"] == 3
    assert len((tmp_path / "merged" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()) == 4
