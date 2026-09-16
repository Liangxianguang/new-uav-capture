"""Replay Phase 82 validation failures with separate boundary accounting.

The replay is development-only and reads the already completed validation
failure list.  It does not tune a checkpoint and refuses locked-test scenes.
Its purpose is to distinguish target and defender world-boundary violations
under the same actor-evaluation contract used by Phase 82.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.pursuit_controllers import (  # noqa: E402
    PublicBeliefRouteIntentController,
    PursuitCBFSafetyFilter,
)
from encirclement3d.residual_actor import RecurrentResidualActor  # noqa: E402
from train_phase79_dagger_residual import (  # noqa: E402
    actor_action,
    build_environment,
    load_document,
    load_records,
    resolve_config_path,
    _route_base_action,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.resolve().read_bytes()).hexdigest()


def _replay(
    environment: dict[str, Any],
    record: dict[str, Any],
    settings: dict[str, Any],
    actor: RecurrentResidualActor,
    device: torch.device,
    *,
    episode_seed: int,
    max_steps: int | None,
) -> dict[str, Any]:
    evaluation_record = dict(record)
    evaluation_record["episode_seed"] = int(episode_seed)
    env, observation, _scenario = build_environment(environment, evaluation_record, max_steps=max_steps)
    route = PublicBeliefRouteIntentController(
        env,
        horizon_seconds=float(settings.get("teacher_horizon_seconds", 0.75)),
        replan_interval_steps=int(settings.get("teacher_replan_interval_steps", 8)),
        min_hold_steps=int(settings.get("teacher_min_hold_steps", 6)),
        grid_step=float(settings.get("teacher_grid_step_m", 0.75)),
        route_margin=float(settings.get("teacher_route_margin_m", 0.85)),
    )
    safety = PursuitCBFSafetyFilter(env)
    hidden = actor.initial_hidden(env.n_defenders, device=device)
    while True:
        route_features = route.route_features(observation)
        base_action = _route_base_action(route, safety, observation, settings)
        local = np.concatenate([policy_observations(env, observation), route_features], axis=1).astype(np.float32)
        action, hidden = actor_action(
            actor,
            local,
            base_action,
            hidden,
            float(env.agents["defender_max_speed"]),
            device,
        )
        action, _diagnostics = safety.filter(action, observation)
        observation, _reward, terminated, truncated, info = env.step(action, record_history=True)
        if terminated or truncated:
            return {
                "safe_capture_success": bool(info["safe_capture_success"]),
                "termination_reason": str(info["termination_reason"]),
                "world_violation_steps": int(info["world_violation_steps"]),
                "target_world_violation_steps": int(info.get("target_world_violation_steps", 0)),
                "defender_world_violation_steps": int(info.get("defender_world_violation_steps", 0)),
                "first_target_boundary_violation_step": info.get("first_target_boundary_violation_step"),
                "first_defender_boundary_violation_step": info.get("first_defender_boundary_violation_step"),
                "collision": bool(info["collision"]),
                "boundary_violation": bool(info["world_violation_steps"] > 0),
                "min_clearance_m": float(info["min_clearance_so_far"]),
                "steps": int(env.step_count),
                "target_maneuver_fallback_count": int(info.get("target_maneuver_fallback_count", 0)),
                "target_maneuver_last_feasibility_failure": str(
                    info.get("target_maneuver_last_feasibility_failure", "none")
                ),
                "target_maneuver_feasible_candidate_count": int(
                    info.get("target_maneuver_feasible_candidate_count", 0)
                ),
                "target_maneuver_rejected_candidate_count": int(
                    info.get("target_maneuver_rejected_candidate_count", 0)
                ),
                "target_maneuver_route": str(info.get("target_maneuver_route", "unknown")),
            }


def diagnose(
    config_path: Path,
    checkpoint_path: Path,
    evaluation_path: Path,
    output_path: Path,
    *,
    max_failures: int | None = None,
    torch_threads: int = 1,
) -> dict[str, Any]:
    document, environment, settings = load_document(config_path)
    if not bool(document.get("not_a_locked_test", False)):
        raise ValueError("diagnostic replay requires an explicitly non-locked-test config")
    evaluation = _read_json(evaluation_path.resolve())
    if bool(evaluation.get("locked_test_used", False)):
        raise ValueError("locked-test evaluation is not accepted")
    failed_indices = [
        int(row["episode_index"])
        for row in evaluation.get("rows", [])
        if str(row.get("termination_reason")) != "safe_capture"
    ]
    if max_failures is not None:
        failed_indices = failed_indices[: int(max_failures)]
    scene_path = resolve_config_path(document["evaluation_scene_file"], config_path)
    records = load_records(scene_path, None)
    by_index = {int(record["episode_index"]): (position, record) for position, record in enumerate(records)}
    if any(index not in by_index for index in failed_indices):
        raise ValueError("failure list and validation scenes do not cover the same episode indices")
    device = torch.device("cpu")
    torch.set_num_threads(int(torch_threads))
    torch.set_num_interop_threads(int(torch_threads))
    checkpoint = torch.load(checkpoint_path.resolve(), map_location=device, weights_only=True)
    actor = RecurrentResidualActor(
        int(checkpoint["local_observation_dim"]),
        action_dim=int(checkpoint.get("action_dim", 3)),
        hidden_dim=int(checkpoint["recurrent_hidden_dim"]),
        residual_scale=float(checkpoint["residual_scale"]),
    ).to(device)
    actor.load_state_dict(checkpoint["state_dict"], strict=True)
    actor.eval()
    rows: list[dict[str, Any]] = []
    for index in failed_indices:
        position, record = by_index[index]
        replay = _replay(
            environment,
            record,
            settings,
            actor,
            device,
            episode_seed=int(record["episode_seed"]) + 910_000 + position,
            max_steps=None,
        )
        rows.append({"episode_index": index, "variant": record.get("variant"), **replay})
    payload = {
        "experiment_name": "phase82_boundary_failure_diagnostic_replay",
        "evaluation_split": "development_validation_only",
        "locked_test_used": False,
        "source_scene_file": str(scene_path.resolve()),
        "source_scene_sha256": _sha256(scene_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "source_failure_evaluation": str(evaluation_path.resolve()),
        "source_failure_count": len(
            [row for row in evaluation.get("rows", []) if str(row.get("termination_reason")) != "safe_capture"]
        ),
        "replayed_failure_count": len(rows),
        "rows": rows,
        "boundary_violation_cause_counts": {
            "target_only": sum(
                bool(row["target_world_violation_steps"] > 0 and row["defender_world_violation_steps"] == 0)
                for row in rows
            ),
            "defender_only": sum(
                bool(row["defender_world_violation_steps"] > 0 and row["target_world_violation_steps"] == 0)
                for row in rows
            ),
            "both": sum(
                bool(row["target_world_violation_steps"] > 0 and row["defender_world_violation_steps"] > 0)
                for row in rows
            ),
            "neither": sum(
                bool(row["target_world_violation_steps"] == 0 and row["defender_world_violation_steps"] == 0)
                for row in rows
            ),
        },
        "local_cbf_is_empirical_filter_only": True,
        "formal_robust_cbf_qp_claim": False,
    }
    output_path = output_path.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-failures", type=int)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    print(
        json.dumps(
            diagnose(
                args.config,
                args.checkpoint,
                args.evaluation,
                args.output,
                max_failures=args.max_failures,
                torch_threads=args.torch_threads,
            ),
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
