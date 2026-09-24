"""Verify and materialize the frozen current Phase86 comparison contract.

The lock is intentionally separate from the scene files.  It records the
exact hashes and structural checks used by the formal comparison while
preserving the distinction between a frozen comparison contract and a locked
test set.  A dirty source worktree is recorded and prevents promotion of
results, but does not invalidate the scene-contract checks themselves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "current_rl_comparison_phase86_v1.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "current_rl_comparison_phase86_v1" / "protocol_lock.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest().upper()


def resolve_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def target_invalid_field_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        scenario = row.get("scenario", {})
        spec = row.get("spec", {})
        values = [
            row.get("target_invalid"),
            row.get("target_invalid_episode"),
            row.get("target_contract_invalid"),
            scenario.get("target_invalid") if isinstance(scenario, dict) else None,
            spec.get("target_invalid") if isinstance(spec, dict) else None,
        ]
        count += int(any(bool(value) for value in values))
    return count


def git_value(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify(config_path: Path) -> dict[str, Any]:
    document = yaml.safe_load(config_path.resolve().read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Protocol YAML must contain a mapping")
    contract = document.get("scene_contract")
    if not isinstance(contract, dict):
        raise ValueError("Protocol YAML must contain scene_contract")

    environment_path = resolve_path(config_path, str(document["environment_config"]))
    calibration_path = resolve_path(config_path, str(contract["calibration_scene_file"]))
    validation_path = resolve_path(config_path, str(contract["validation_scene_file"]))
    calibration_manifest = resolve_path(config_path, str(contract["calibration_manifest"]))
    validation_manifest = resolve_path(config_path, str(contract["validation_manifest"]))

    paths = {
        "protocol_config": config_path.resolve(),
        "environment_config": environment_path,
        "calibration_scenes": calibration_path,
        "calibration_manifest": calibration_manifest,
        "validation_scenes": validation_path,
        "validation_manifest": validation_manifest,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} does not exist: {path}")

    rows_by_split = {
        "calibration": read_jsonl(calibration_path),
        "validation": read_jsonl(validation_path),
    }
    expected_records = int(contract.get("records_per_split", 300))
    expected_groups = int(contract.get("mirror_groups_per_split", 150))
    checks: dict[str, Any] = {}
    checks["hashes_match_protocol"] = {
        "environment_config": sha256(environment_path) == str(contract["environment_config_sha256"]).upper(),
        "calibration_scenes": sha256(calibration_path) == str(contract["calibration_scene_sha256"]).upper(),
        "calibration_manifest": sha256(calibration_manifest) == str(contract["calibration_manifest_sha256"]).upper(),
        "validation_scenes": sha256(validation_path) == str(contract["validation_scene_sha256"]).upper(),
        "validation_manifest": sha256(validation_manifest) == str(contract["validation_manifest_sha256"]).upper(),
    }
    if not all(checks["hashes_match_protocol"].values()):
        raise ValueError(f"Protocol hash mismatch: {checks['hashes_match_protocol']}")

    split_summary: dict[str, Any] = {}
    all_layout_seeds: dict[str, set[int]] = {}
    all_episode_seeds: dict[str, set[int]] = {}
    for split, rows in rows_by_split.items():
        if len(rows) != expected_records:
            raise ValueError(f"{split} has {len(rows)} records, expected {expected_records}")
        groups = Counter(str(row.get("mirror_group_id")) for row in rows)
        if len(groups) != expected_groups or set(groups.values()) != {2}:
            raise ValueError(f"{split} mirror groups are not exactly two-member groups: {groups}")
        layout_seeds = {int(row["layout_seed"]) for row in rows}
        episode_seeds = {int(row["episode_seed"]) for row in rows}
        all_layout_seeds[split] = layout_seeds
        all_episode_seeds[split] = episode_seeds
        invalid_count = target_invalid_field_count(rows)
        if invalid_count != int(contract.get("target_invalid_scene_field_count", 0)):
            raise ValueError(f"{split} target-invalid scene field count is {invalid_count}")
        modes = sorted({str(row.get("target_motion_mode")) for row in rows})
        obstacle_counts = sorted({int(row.get("obstacle_count", -1)) for row in rows})
        delays = sorted({int(row.get("execution", {}).get("action_delay_steps", -1)) for row in rows})
        noises = sorted({float(row.get("execution", {}).get("command_noise_std_mps", -1.0)) for row in rows})
        split_summary[split] = {
            "records": len(rows),
            "mirror_groups": len(groups),
            "mirror_group_size_counts": dict(sorted(Counter(groups.values()).items())),
            "layout_seed_count": len(layout_seeds),
            "episode_seed_count": len(episode_seeds),
            "target_invalid_scene_field_count": invalid_count,
            "target_motion_modes": modes,
            "obstacle_counts": obstacle_counts,
            "execution_action_delay_steps": delays,
            "execution_command_noise_mps": noises,
        }

    overlap = {
        "layout_seed_overlap": len(all_layout_seeds["calibration"] & all_layout_seeds["validation"]),
        "episode_seed_overlap": len(all_episode_seeds["calibration"] & all_episode_seeds["validation"]),
    }
    if overlap["layout_seed_overlap"] != int(contract.get("layout_seed_overlap", 0)):
        raise ValueError(f"Unexpected layout-seed overlap: {overlap}")
    if overlap["episode_seed_overlap"] != int(contract.get("episode_seed_overlap", 0)):
        raise ValueError(f"Unexpected episode-seed overlap: {overlap}")

    dirty = bool(git_value("status", "--porcelain"))
    return {
        "protocol": str(document.get("experiment_name", config_path.stem)),
        "status": "frozen_comparison_contract",
        "formal_promotion_allowed": not dirty,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_head": git_value("rev-parse", "HEAD"),
        "source_worktree_dirty": dirty,
        "protocol_config_sha256": sha256(config_path),
        "files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "scene_summary": split_summary,
        "split_overlap": overlap,
        "contract": {
            "target_invalid_policy": document.get("experiment", {}).get("target_invalid_policy"),
            "actor_observation_contract": document.get("experiment", {}).get("actor_observation_contract"),
            "expert_action_or_route_labels": document.get("experiment", {}).get("expert_action_or_route_labels"),
            "target_future_truth_in_actor": document.get("experiment", {}).get("target_future_truth_in_actor"),
            "formal_route_intent_features": bool(document.get("environment_overrides", {}).get("task", {}).get("policy_route_intent_features", False)),
        },
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    lock = verify(args.config.resolve())
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.resolve().with_name(f"{args.output.name}.tmp")
    temporary.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    temporary.replace(args.output.resolve())
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()
