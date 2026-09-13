from __future__ import annotations

import json

from scripts.audit_phase58_scene_matrix import audit_matrix
from scripts.generate_phase58_delayed_planning_scenes import build_records, validate_records
from scripts.evaluate_s4_branching import load_protocol
from evaluate_minimax_mpc import DEFAULT_ENVIRONMENT_CONFIG


def test_phase58_small_matrix_audit_preserves_contract(tmp_path) -> None:
    protocol_path = tmp_path / "protocol.yaml"
    protocol_path.write_text(
        """seed_blocks:\n  pilot: 1\nepisodes_per_split:\n  pilot: 4\ns4:\n  target_motion_mode: adaptive_branching\n  target_speed_scales: [0.50, 0.65, 0.80]\n  defender_biases: [upper, lower]\n  observation_conditions:\n    - name: nominal\n      pursuit_overrides:\n        detection_range: 14.0\n        detection_dropout_probability: 0.15\n        observation_noise_std: 0.03\n        message_delay_steps: 2\n        message_dropout_probability: 0.05\n  branch_geometry:\n    target_branch_decision_x_m: -3.65\n    target_branch_exit_offset_y_m: 5.30\n    target_branch_waypoint_x_m: 0.85\n    target_branch_goal_x_m: 7.50\n  max_steps: 250\n""",
        encoding="utf-8",
    )
    protocol = load_protocol(protocol_path)
    records = build_records(protocol, DEFAULT_ENVIRONMENT_CONFIG, groups_per_block=3, seed_start=990101)
    validate_records(records, 3)
    scene_path = tmp_path / "scenes.jsonl"
    scene_path.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    import hashlib

    manifest_path.write_text(
        json.dumps({"episodes": len(records), "mirror_groups": len(records) // 2, "scene_manifest_sha256": hashlib.sha256(scene_path.read_bytes()).hexdigest()}),
        encoding="utf-8",
    )
    result = audit_matrix(scene_path, manifest_path)
    assert result["audit_pass"] is True
    assert result["episodes"] == 24
