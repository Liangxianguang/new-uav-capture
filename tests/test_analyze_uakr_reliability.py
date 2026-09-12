from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_uakr_reliability import analyze


def _write_run(root: Path, values: list[tuple[float, bool]]) -> None:
    method = root / "distributed_delayed"
    method.mkdir(parents=True)
    episodes = []
    steps = []
    for index, (score, success) in enumerate(values):
        episodes.append(
            {
                "episode_index": index,
                "safe_capture_success": success,
                "collision": not success,
                "boundary_violation": False,
                "timeout": False,
            }
        )
        steps.append({"episode_index": index, "adaptive_uncertainty_score": score})
    (method / "episodes.jsonl").write_text("\n".join(json.dumps(row) for row in episodes), encoding="utf-8")
    (method / "steps.jsonl").write_text("\n".join(json.dumps(row) for row in steps), encoding="utf-8")


def test_uakr_reliability_audit_reports_monotonic_failure_signal(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run, [(0.1, True), (0.2, True), (0.8, False), (0.9, False)])

    result = analyze([run])

    assert result["episode_count"] == 4
    assert result["auc"]["mean_uncertainty"]["failure"] == 1.0
    quartiles = result["quartiles"]["mean_uncertainty"]
    assert quartiles[0]["failure_rate"] == 0.0
    assert quartiles[-1]["failure_rate"] == 1.0


def test_mirror_group_split_keeps_groups_intact(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_run(run, [(0.1, True), (0.2, True), (0.3, True), (0.4, True), (0.8, False), (0.9, False), (0.7, False), (0.6, False)])
    manifest = tmp_path / "scenes.jsonl"
    records = [
        {"episode_index": index, "mirror_group_id": f"group-{index // 2}"}
        for index in range(8)
    ]
    manifest.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    result = analyze(
        [run],
        scene_manifest=manifest,
        split_strategy="mirror_group_half",
        split_seed=17,
    )

    assert result["split_metadata"]["group_count"] == 4
    assert result["split_metadata"]["calibration_group_count"] == 2
    assert result["split_metadata"]["confirmation_group_count"] == 2
    assert result["split_metadata"]["calibration_episode_count"] == 4
    assert result["split_metadata"]["confirmation_episode_count"] == 4


def test_canonical_mirror_group_split_is_shared_across_runs(tmp_path: Path) -> None:
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    values = [(0.1, True), (0.2, True), (0.3, True), (0.4, True), (0.8, False), (0.9, False), (0.7, False), (0.6, False)]
    _write_run(run_a, values)
    _write_run(run_b, values)
    manifest = tmp_path / "scenes.jsonl"
    records = [
        {"episode_index": index, "mirror_group_id": f"group-{index // 2}"}
        for index in range(8)
    ]
    manifest.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    result = analyze(
        [run_a, run_b],
        scene_manifest=manifest,
        split_strategy="canonical_mirror_group_half",
        split_seed=17,
    )

    assert result["split_metadata"]["group_count"] == 4
    assert result["split_metadata"]["calibration_group_count"] == 2
    assert result["split_metadata"]["confirmation_group_count"] == 2
    assert result["split_metadata"]["calibration_episode_count"] == 8
    assert result["split_metadata"]["confirmation_episode_count"] == 8
