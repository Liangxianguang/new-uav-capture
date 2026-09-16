import json

from scripts.aggregate_phase82_dagger_results import aggregate


def _write_seed(root, seed, safe, collision=0.0, boundary=0.0):
    path = root / f"phase82_dnmcp_eligible_seed{seed}"
    path.mkdir()
    rows = [
        {
            "safe_capture_success": index < safe,
            "capture_event": index < safe,
            "collision": index < collision,
            "boundary_violation": index < boundary,
            "timeout": False,
        }
        for index in range(10)
    ]
    (path / "manifest.json").write_text(
        json.dumps({"locked_test_used": False, "source_scene_sha256": "scene-hash"}),
        encoding="utf-8",
    )
    (path / "evaluation.json").write_text(
        json.dumps(
            {
                "locked_test_used": False,
                "episodes": len(rows),
                "rows": rows,
                "latency_ms": {"total": {"p50": 1.0, "p95": 2.0, "p99": 3.0}},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_three_seed_gate_requires_every_seed(tmp_path):
    paths = [
        _write_seed(tmp_path, 824601, 8),
        _write_seed(tmp_path, 824602, 7),
        _write_seed(tmp_path, 824603, 9),
    ]
    output = tmp_path / "aggregate.json"
    payload = aggregate(paths, output)
    assert payload["three_seed_gate_passed"] is True
    assert payload["pooled_episode_summary"]["episodes"] == 30
    assert len(payload["seeds"]) == 3


def test_three_seed_gate_fails_on_collision(tmp_path):
    paths = [
        _write_seed(tmp_path, 824601, 8),
        _write_seed(tmp_path, 824602, 8, collision=1),
        _write_seed(tmp_path, 824603, 8),
    ]
    payload = aggregate(paths, tmp_path / "aggregate.json")
    assert payload["three_seed_gate_passed"] is False
    assert payload["seeds"][1]["gate"]["passed"] is False
