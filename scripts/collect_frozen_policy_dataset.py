"""Collect a policy-observation/action archive on a frozen scene manifest.

This utility is intentionally small and development-only.  It is used to
warm-start RL on the exact scene contract that will later be used for paired
evaluation; it never changes the environment's target-boundary contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.observation_encoding import policy_observations  # noqa: E402
from encirclement3d.pursuit_controllers import (  # noqa: E402
    DynamicEncirclementController,
    FixedRoleEncirclementController,
    PublicBeliefRouteIntentController,
    PursuitCBFSafetyFilter,
)
from rl_scene_io import build_environment, read_scenes  # noqa: E402
from train_mappo_ippo_baseline import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scenes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument(
        "--use-cbf",
        action="store_true",
        help="Apply the same local CBF layer used by the dynamic-encirclement baseline.",
    )
    parser.add_argument(
        "--controller",
        choices=("dynamic", "fixed_role", "route"),
        default="dynamic",
        help="Development-only teacher role assignment.",
    )
    parser.add_argument(
        "--interceptor-id",
        type=int,
        default=0,
        help="Fixed interceptor index when --controller=fixed_role.",
    )
    parser.add_argument(
        "--accepted-only",
        action="store_true",
        help="Keep only safe-capture, target-valid episodes in the development dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output}")
    if args.max_steps <= 0:
        raise ValueError("max-steps must be positive")

    document, environment = load_config(args.config.resolve())
    records = read_scenes(args.scenes.resolve(), args.episodes)
    if any("locked" in str(record.get("scene_block", "")).lower() for record in records):
        raise ValueError("Frozen training data must not contain locked-test records")

    local_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    expected_local_dim: int | None = None

    for record in records:
        env, observation, _scenario = build_environment(
            copy.deepcopy(environment), record, max_steps=args.max_steps
        )
        if args.controller == "fixed_role":
            controller = FixedRoleEncirclementController(env, interceptor_id=args.interceptor_id)
        elif args.controller == "route":
            controller = PublicBeliefRouteIntentController(
                env,
                interceptor_id=args.interceptor_id,
            )
        else:
            controller = DynamicEncirclementController(env)
        route_helper = (
            controller
            if args.controller == "route"
            else (
                PublicBeliefRouteIntentController(env, interceptor_id=args.interceptor_id)
                if bool(env.task.get("policy_route_intent_features", False))
                else None
            )
        )
        safety = PursuitCBFSafetyFilter(env) if args.use_cbf else None
        episode_local: list[np.ndarray] = []
        episode_actions: list[np.ndarray] = []
        final_info: dict[str, Any] = {}
        for _ in range(args.max_steps):
            local = np.asarray(
                policy_observations(env, observation, route_helper=route_helper),
                dtype=np.float32,
            )
            action = np.asarray(controller.act(observation), dtype=np.float64)
            if safety is not None:
                action, _diagnostics = safety.filter(action, observation)
            action = np.asarray(action, dtype=np.float32)
            if local.ndim != 2 or local.shape[0] != 4 or local.shape[1] <= 0:
                raise ValueError(f"Unexpected local observation shape: {local.shape}")
            if action.shape != (4, 3):
                raise ValueError(f"Unexpected teacher action shape: {action.shape}")
            expected_local_dim = local.shape[1] if expected_local_dim is None else expected_local_dim
            if local.shape[1] != expected_local_dim:
                raise ValueError("Frozen scene records produced inconsistent observation dimensions")
            episode_local.append(local)
            episode_actions.append(action)
            observation, _reward, terminated, truncated, final_info = env.step(action)
            if terminated or truncated:
                break

        accepted = bool(final_info.get("safe_capture_success", False)) and not bool(
            final_info.get("target_invalid_episode", False)
        )
        rows.append(
            {
                "episode_index": int(record["episode_index"]),
                "episode_seed": int(record["episode_seed"]),
                "steps": len(episode_local),
                "safe_capture_success": bool(final_info.get("safe_capture_success", False)),
                "target_invalid_episode": bool(final_info.get("target_invalid_episode", False)),
                "termination_reason": str(final_info.get("termination_reason", "unknown")),
                "accepted_for_dataset": accepted,
            }
        )
        if args.accepted_only and not accepted:
            continue
        local_parts.append(np.asarray(episode_local, dtype=np.float32))
        action_parts.append(np.asarray(episode_actions, dtype=np.float32))

    if not local_parts:
        raise RuntimeError("No episodes satisfied --accepted-only dataset selection.")

    max_length = max(len(part) for part in local_parts)
    local = np.zeros((len(local_parts), max_length, 4, int(expected_local_dim)), dtype=np.float32)
    actions = np.zeros((len(action_parts), max_length, 4, 3), dtype=np.float32)
    valid = np.zeros((len(local_parts), max_length), dtype=np.float32)
    for index, (local_part, action_part) in enumerate(zip(local_parts, action_parts)):
        length = len(local_part)
        local[index, :length] = local_part
        actions[index, :length] = action_part
        valid[index, :length] = 1.0
    args.output.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output / "expert_dataset.npz"
    np.savez_compressed(
        dataset_path,
        local_observations=local,
        actions=actions,
        valid_masks=valid,
    )
    metadata = {
        "dataset_type": "frozen_dynamic_encirclement_teacher_policy_actions",
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.resolve().read_bytes()).hexdigest(),
        "scenes": str(args.scenes.resolve()),
        "scenes_sha256": hashlib.sha256(args.scenes.resolve().read_bytes()).hexdigest(),
        "episodes": len(records),
        "selected_episodes": len(local_parts),
        "accepted_only": bool(args.accepted_only),
        "frames": int(valid.sum()),
        "sequence_length": int(max_length),
        "local_observation_dim": int(local.shape[-1]),
        "route_intent_features": bool(environment.get("task", {}).get("policy_route_intent_features", False)),
        "route_intent_feature_dim": 7 if bool(environment.get("task", {}).get("policy_route_intent_features", False)) else 0,
        "action_shape_per_frame": [4, 3],
        "controller": (
            "FixedRoleEncirclementController"
            if args.controller == "fixed_role"
            else (
                "PublicBeliefRouteIntentController"
                if args.controller == "route"
                else "DynamicEncirclementController"
            )
        ),
        "interceptor_id": int(args.interceptor_id) if args.controller in {"fixed_role", "route"} else None,
        "safety_layer": "local_cbf" if args.use_cbf else "none",
        "max_steps": args.max_steps,
        "selection_policy": "safe_capture_and_target_valid" if args.accepted_only else "all_rollouts",
        "rows": rows,
        "document_experiment_name": document.get("experiment_name"),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": str(dataset_path.resolve()),
                "metadata": str((args.output / "metadata.json").resolve()),
                "selected_episodes": int(len(local_parts)),
                "frames": int(valid.sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
