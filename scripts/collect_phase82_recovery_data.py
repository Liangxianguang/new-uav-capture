"""Collect targeted public-belief DN-MPC recovery/projection data.

The collector is development-only.  It runs the public-belief DN-MPC teacher
on calibration scenes and retains frames near an obstacle, with tight initial
formation spacing, or after a local-CBF projection.  Target truth is never
passed to the teacher; oracle truth is reserved for the separate calibration
screen.  The raw scene pool is not modified and locked-test data is rejected.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from train_phase79_dagger_residual import (  # noqa: E402
    DNMPCTeacher,
    build_environment,
    load_document,
    load_records,
    planner_config,
    resolve_config_path,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _target_obstacle_clearance(env: Any) -> float:
    if not env.obstacles:
        return float("inf")
    return float(
        min(env._obstacle_clearance(env.target_position, obstacle) for obstacle in env.obstacles)
    )


def _formation_spacing(env: Any) -> float:
    positions = np.asarray(env.defender_positions, dtype=np.float64)
    pairwise = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    pairwise += np.eye(len(positions), dtype=np.float64) * 1.0e6
    return float(np.min(pairwise))


def _boundary_margin(env: Any) -> float:
    positions = np.vstack([env.defender_positions, env.target_position[None, :]])
    return float(np.min(np.minimum(positions - env.lower[None, :], env.upper[None, :] - positions)))


def _reason_labels(
    teacher_recovery: bool,
    base_projection: float,
    teacher_projection: float,
    obstacle_clearance: float,
    formation_spacing: float,
    *,
    projection_threshold: float,
    obstacle_threshold: float,
    formation_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    if teacher_recovery:
        reasons.append("teacher_recovery")
    if max(base_projection, teacher_projection) >= projection_threshold:
        reasons.append("cbf_projection")
    if obstacle_clearance <= obstacle_threshold:
        reasons.append("obstacle_near")
    if formation_spacing <= formation_threshold:
        reasons.append("formation_spacing")
    return reasons


def collect(
    config_path: Path,
    scenes_path: Path,
    output_dir: Path,
    *,
    episodes: int | None = None,
    max_steps: int | None = None,
    projection_threshold: float = 0.05,
    obstacle_threshold: float = 2.0,
    formation_threshold: float = 1.30,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir.resolve()}")
    document, environment_config, settings = load_document(config_path)
    records = load_records(scenes_path, episodes)
    planner_path = resolve_config_path(str(document["mpc_config"]), config_path)
    planner = planner_config(planner_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_local: list[np.ndarray] = []
    selected_base: list[np.ndarray] = []
    selected_target: list[np.ndarray] = []
    selected_recovery: list[bool] = []
    selected_obstacle: list[float] = []
    selected_spacing: list[float] = []
    selected_projection: list[float] = []
    selected_base_projection: list[float] = []
    selected_teacher_projection: list[float] = []
    selected_teacher_base_gap: list[float] = []
    selected_boundary: list[float] = []
    frame_reasons: list[str] = []
    episode_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    feature_shape: tuple[int, int] | None = None
    start_time = time.perf_counter()

    for index, record in enumerate(records):
        env, observation, _scenario = build_environment(
            environment_config,
            record,
            max_steps=max_steps,
        )
        teacher = DNMPCTeacher(env, planner, settings)
        episode_selected = 0
        episode_reasons: Counter[str] = Counter()
        min_obstacle = float("inf")
        min_spacing = float("inf")
        max_projection = 0.0
        route_latencies: list[float] = []
        final_info: dict[str, Any] = {}
        while True:
            step_start = time.perf_counter()
            teacher_output = teacher.output(observation)
            route_latencies.append((time.perf_counter() - step_start) * 1000.0)
            local = np.concatenate(
                [policy_observations(env, observation), teacher_output.route_features], axis=1
            ).astype(np.float32)
            obstacle_clearance = _target_obstacle_clearance(env)
            spacing = _formation_spacing(env)
            boundary_margin = _boundary_margin(env)
            projection = max(
                float(teacher_output.base_cbf_correction_norm),
                float(teacher_output.teacher_cbf_correction_norm),
            )
            reasons = _reason_labels(
                bool(teacher_output.recovery),
                float(teacher_output.base_cbf_correction_norm),
                float(teacher_output.teacher_cbf_correction_norm),
                obstacle_clearance,
                spacing,
                projection_threshold=projection_threshold,
                obstacle_threshold=obstacle_threshold,
                formation_threshold=formation_threshold,
            )
            if reasons:
                selected_local.append(local.copy())
                selected_base.append(teacher_output.base_action.copy())
                selected_target.append(teacher_output.teacher_action.copy())
                selected_recovery.append(bool(teacher_output.recovery))
                selected_obstacle.append(obstacle_clearance)
                selected_spacing.append(spacing)
                selected_projection.append(projection)
                selected_base_projection.append(float(teacher_output.base_cbf_correction_norm))
                selected_teacher_projection.append(float(teacher_output.teacher_cbf_correction_norm))
                selected_teacher_base_gap.append(float(teacher_output.teacher_base_gap_norm))
                selected_boundary.append(boundary_margin)
                frame_reasons.append("+".join(reasons))
                episode_selected += 1
                for reason in reasons:
                    reason_counts[reason] += 1
                    episode_reasons[reason] += 1
            feature_shape = tuple(int(value) for value in local.shape)
            min_obstacle = min(min_obstacle, obstacle_clearance)
            min_spacing = min(min_spacing, spacing)
            max_projection = max(max_projection, projection)
            observation, _reward, terminated, truncated, final_info = env.step(
                teacher_output.teacher_action,
                record_history=True,
            )
            if terminated or truncated:
                break
        episode_rows.append(
            {
                "episode_index": int(record["episode_index"]),
                "episode_seed": int(record["episode_seed"]),
                "variant": str(record.get("variant", "unknown")),
                "single_factor": str(record.get("single_factor", "unknown")),
                "steps": int(env.step_count),
                "termination_reason": str(final_info.get("termination_reason", "unknown")),
                "safe_capture_success": bool(final_info.get("safe_capture_success", False)),
                "collision": bool(final_info.get("collision", False)),
                "boundary_violation": bool(final_info.get("world_violation_steps", 0) > 0),
                "timeout": bool(final_info.get("termination_reason") == "timeout"),
                "selected_frames": episode_selected,
                "selected_reason_counts": dict(episode_reasons),
                "min_target_obstacle_clearance_m": float(min_obstacle),
                "min_formation_spacing_m": float(min_spacing),
                "max_cbf_projection_norm_mps": float(max_projection),
                "route_latency_p50_ms": float(np.percentile(route_latencies, 50)) if route_latencies else 0.0,
                "route_latency_p95_ms": float(np.percentile(route_latencies, 95)) if route_latencies else 0.0,
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(
                f"collected {index + 1}/{len(records)} episodes, "
                f"selected_frames={len(selected_local)}",
                flush=True,
            )

    if feature_shape is None:
        raise ValueError("No frames were collected")
    if selected_local:
        local_array = np.stack(selected_local, axis=0).astype(np.float32)
        base_array = np.stack(selected_base, axis=0).astype(np.float32)
        target_array = np.stack(selected_target, axis=0).astype(np.float32)
    else:
        local_array = np.zeros((0, *feature_shape), dtype=np.float32)
        base_array = np.zeros((0, feature_shape[0], 3), dtype=np.float32)
        target_array = np.zeros((0, feature_shape[0], 3), dtype=np.float32)
    np.savez_compressed(
        output_dir / "recovery_projection_dataset.npz",
        local_observations=local_array,
        base_actions=base_array,
        target_actions=target_array,
        recovery=np.asarray(selected_recovery, dtype=np.bool_),
        target_obstacle_clearance_m=np.asarray(selected_obstacle, dtype=np.float32),
        formation_spacing_m=np.asarray(selected_spacing, dtype=np.float32),
        cbf_projection_norm_mps=np.asarray(selected_projection, dtype=np.float32),
        base_cbf_projection_norm_mps=np.asarray(selected_base_projection, dtype=np.float32),
        teacher_cbf_projection_norm_mps=np.asarray(selected_teacher_projection, dtype=np.float32),
        teacher_base_gap_norm_mps=np.asarray(selected_teacher_base_gap, dtype=np.float32),
        boundary_margin_m=np.asarray(selected_boundary, dtype=np.float32),
        frame_reasons=np.asarray(frame_reasons, dtype=str),
    )
    (output_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in episode_rows),
        encoding="utf-8",
    )
    (output_dir / "frame_reasons.json").write_text(
        json.dumps(dict(reason_counts), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "experiment_name": "phase82_targeted_recovery_projection_collection",
        "phase": "development_calibration_only",
        "not_a_locked_test": True,
        "locked_test_used": False,
        "target_truth_used_by_teacher": False,
        "controller": "public_belief_dnmpc_teacher",
        "config": str(config_path.resolve()),
        "scene_file": str(scenes_path.resolve()),
        "scene_file_sha256": _sha256(scenes_path),
        "episodes": len(episode_rows),
        "selected_frames": int(len(selected_local)),
        "feature_shape": list(feature_shape),
        "thresholds": {
            "cbf_projection_norm_mps": float(projection_threshold),
            "target_obstacle_clearance_m": float(obstacle_threshold),
            "formation_spacing_m": float(formation_threshold),
        },
        "continuous_diagnostics": [
            "base_cbf_projection_norm_mps",
            "teacher_cbf_projection_norm_mps",
            "teacher_base_gap_norm_mps",
        ],
        "reason_counts": dict(reason_counts),
        "safe_capture_episodes": int(sum(bool(row["safe_capture_success"]) for row in episode_rows)),
        "collision_episodes": int(sum(bool(row["collision"]) for row in episode_rows)),
        "boundary_episodes": int(sum(bool(row["boundary_violation"]) for row in episode_rows)),
        "timeout_episodes": int(sum(bool(row["timeout"]) for row in episode_rows)),
        "elapsed_seconds": float(time.perf_counter() - start_time),
        "local_cbf_is_empirical_filter_only": True,
        "formal_robust_cbf_qp_claim": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "phase81_dnmcp_dagger_residual_nominal_plus.yaml",
    )
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--projection-threshold", type=float, default=0.05)
    parser.add_argument("--obstacle-threshold", type=float, default=2.0)
    parser.add_argument("--formation-threshold", type=float, default=1.30)
    args = parser.parse_args()
    print(
        json.dumps(
            collect(
                args.config,
                args.scenes,
                args.output_dir,
                episodes=args.episodes,
                max_steps=args.max_steps,
                projection_threshold=args.projection_threshold,
                obstacle_threshold=args.obstacle_threshold,
                formation_threshold=args.formation_threshold,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
