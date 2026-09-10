"""Collect S4 action-conditioned target-trajectory windows without input leakage."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import platform
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
    PredictionPursuitController,
    PurePursuitController,
    PursuitCBFSafetyFilter,
)
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    prepare_showcase_episode,
    s4_adaptive_branching_scenario,
    scenario_metadata,
)
from evaluate_minimax_mpc import DEFAULT_ENVIRONMENT_CONFIG, load_yaml  # noqa: E402
from evaluate_s4_branching import config_for_spec as s4_config_for_spec  # noqa: E402


DEFAULT_COLLECTION_CONFIG = PROJECT_ROOT / "configs" / "phase15_s4_branching_dataset.yaml"
DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "phase15_s4_branching_pilot.yaml"
POLICY_IDS = {
    "dynamic_encirclement": 0,
    "pure_pursuit": 1,
    "prediction_pursuit": 2,
    "safe_dynamic_encirclement": 3,
    "randomized_safe_mixture": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-config", type=Path, default=DEFAULT_COLLECTION_CONFIG)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_ENVIRONMENT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, help="Development-only count override.")
    return parser.parse_args()


def load_collection_config(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("S4 collection config must be a YAML mapping.")
    required_lists = ("target_speed_scales", "defender_biases", "rollout_policies", "observation_conditions")
    for key in required_lists:
        if not isinstance(document.get(key), list) or not document[key]:
            raise ValueError(f"S4 collection config requires non-empty {key}.")
    unknown_policies = sorted(set(document["rollout_policies"]).difference(POLICY_IDS))
    if unknown_policies:
        raise ValueError("Unsupported S4 rollout policies: " + ", ".join(unknown_policies))
    if int(document.get("episodes", 0)) <= 0:
        raise ValueError("S4 collection episodes must be positive.")
    if int(document.get("history_length", 0)) <= 0 or int(document.get("horizon_steps", 0)) <= 0:
        raise ValueError("S4 collection history_length and horizon_steps must be positive.")
    return document


def load_s4_protocol(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("s4"), dict):
        raise ValueError("S4 protocol must contain an s4 mapping.")
    if str(document["s4"].get("target_motion_mode")) != "adaptive_branching":
        raise ValueError("S4 dataset collection requires adaptive_branching target motion.")
    return document


def episode_spec(collection: dict[str, Any], episode_index: int) -> dict[str, Any]:
    conditions = list(
        itertools.product(
            collection["target_speed_scales"],
            collection["observation_conditions"],
            collection["defender_biases"],
            collection["rollout_policies"],
        )
    )
    seed_start = int(collection["seed_start"])
    order = np.random.default_rng(seed_start + 2_000_000).permutation(len(conditions))
    condition_index = int(order[episode_index % len(order)])
    target_speed_scale, observation, defender_bias, rollout_policy = conditions[condition_index]
    return {
        "episode_index": int(episode_index),
        "episode_seed": seed_start + int(episode_index),
        "layout_seed": seed_start + 1_000_000 + condition_index // len(collection["rollout_policies"]),
        "target_speed_scale": float(target_speed_scale),
        "observation_condition": str(observation["name"]),
        "pursuit_overrides": copy.deepcopy(observation["pursuit_overrides"]),
        "defender_bias": str(defender_bias),
        "rollout_policy": str(rollout_policy),
        "condition_index": condition_index,
        "condition_table_size": len(conditions),
    }


def controller_action(
    env: CaptureRadiusPursuit3DEnv,
    observation: dict[str, Any],
    policy_name: str,
    controller: Any,
    safety_filter: PursuitCBFSafetyFilter | None,
    rng: np.random.Generator,
) -> np.ndarray:
    desired = np.asarray(controller.act(observation), dtype=np.float64)
    if policy_name == "randomized_safe_mixture":
        noise = rng.normal(0.0, 1.0, size=desired.shape)
        desired = 0.70 * desired + 0.30 * env._clip_rows(noise, float(env.agents["defender_max_speed"]))
    if safety_filter is not None:
        desired, _diagnostics = safety_filter.filter(desired, observation)
    return env._clip_rows(desired, float(env.agents["defender_max_speed"]))


def make_controller(env: CaptureRadiusPursuit3DEnv, policy_name: str) -> tuple[Any, PursuitCBFSafetyFilter | None]:
    if policy_name == "pure_pursuit":
        return PurePursuitController(env), None
    if policy_name == "prediction_pursuit":
        return PredictionPursuitController(env), None
    controller = DynamicEncirclementController(env)
    if policy_name in {"safe_dynamic_encirclement", "randomized_safe_mixture"}:
        return controller, PursuitCBFSafetyFilter(env)
    if policy_name == "dynamic_encirclement":
        return controller, None
    raise ValueError(f"Unsupported rollout policy: {policy_name}")


def padded_history(values: list[np.ndarray], end_index: int, history_length: int) -> np.ndarray:
    selected = values[max(0, end_index - history_length + 1) : end_index + 1]
    return np.stack([selected[0]] * (history_length - len(selected)) + selected, axis=0)


def collect_episode(
    config: dict[str, Any],
    spec: dict[str, Any],
    history_length: int,
    horizon_steps: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    env = CaptureRadiusPursuit3DEnv(config, obstacle_count=1, target_speed_scale=float(spec["target_speed_scale"]))
    scenario = s4_adaptive_branching_scenario(
        env,
        layout_seed=int(spec["layout_seed"]),
        defender_bias=str(spec["defender_bias"]),
    )
    observation = prepare_showcase_episode(env, scenario, seed=int(spec["episode_seed"]), record_history=False)
    controller, safety_filter = make_controller(env, str(spec["rollout_policy"]))
    rng = np.random.default_rng(int(spec["episode_seed"]) + 4_000_000)

    local_frames: list[np.ndarray] = [policy_observations(env, observation).astype(np.float32)]
    visible_frames: list[np.ndarray] = [np.asarray(observation["target_visible"], dtype=bool)]
    message_age_frames: list[np.ndarray] = [np.asarray(observation["message_age_steps"], dtype=np.int16)]
    timestamp_frames: list[np.ndarray] = [
        np.asarray(observation["target_observation_timestamps"], dtype=np.int16)
    ]
    target_positions: list[np.ndarray] = [env.target_position.astype(np.float32).copy()]
    defender_positions: list[np.ndarray] = [env.defender_positions.astype(np.float32).copy()]
    defender_velocities: list[np.ndarray] = [env.defender_velocities.astype(np.float32).copy()]
    realized_actions: list[np.ndarray] = []
    final_info: dict[str, Any] = {}

    while True:
        action = controller_action(env, observation, str(spec["rollout_policy"]), controller, safety_filter, rng)
        observation, _reward, terminated, truncated, final_info = env.step(action)
        realized_actions.append(env.last_executed_actions.astype(np.float32).copy())
        local_frames.append(policy_observations(env, observation).astype(np.float32))
        visible_frames.append(np.asarray(observation["target_visible"], dtype=bool))
        message_age_frames.append(np.asarray(observation["message_age_steps"], dtype=np.int16))
        timestamp_frames.append(np.asarray(observation["target_observation_timestamps"], dtype=np.int16))
        target_positions.append(env.target_position.astype(np.float32).copy())
        defender_positions.append(env.defender_positions.astype(np.float32).copy())
        defender_velocities.append(env.defender_velocities.astype(np.float32).copy())
        if terminated or truncated:
            break

    action_count = len(realized_actions)
    samples: dict[str, list[np.ndarray]] = {
        "history_observations": [],
        "history_target_visible": [],
        "history_message_age_steps": [],
        "history_observation_timestamps": [],
        "future_target_positions": [],
        "future_defender_actions": [],
        "future_defender_positions": [],
        "future_defender_velocities": [],
        "sample_timesteps": [],
    }
    for timestep in range(0, max(action_count - horizon_steps + 1, 0)):
        samples["history_observations"].append(padded_history(local_frames, timestep, history_length))
        samples["history_target_visible"].append(padded_history(visible_frames, timestep, history_length))
        samples["history_message_age_steps"].append(padded_history(message_age_frames, timestep, history_length))
        samples["history_observation_timestamps"].append(padded_history(timestamp_frames, timestep, history_length))
        samples["future_target_positions"].append(
            np.stack(target_positions[timestep + 1 : timestep + horizon_steps + 1], axis=0)
        )
        samples["future_defender_actions"].append(
            np.stack(realized_actions[timestep : timestep + horizon_steps], axis=0)
        )
        samples["future_defender_positions"].append(
            np.stack(defender_positions[timestep : timestep + horizon_steps + 1], axis=0)
        )
        samples["future_defender_velocities"].append(
            np.stack(defender_velocities[timestep : timestep + horizon_steps + 1], axis=0)
        )
        samples["sample_timesteps"].append(np.asarray(timestep, dtype=np.int16))

    packed = {
        name: np.stack(values, axis=0) if values else np.empty((0,), dtype=np.float32)
        for name, values in samples.items()
    }
    episode_metadata = {
        **spec,
        "scenario": scenario_metadata(scenario),
        "frame_count": len(local_frames),
        "sample_count": int(packed["history_observations"].shape[0]),
        "target_branch_sign": final_info.get("target_branch_sign"),
        "target_branch_decision_step": final_info.get("target_branch_decision_step"),
        "termination_reason": final_info.get("termination_reason"),
    }
    return packed, episode_metadata


def validate_dataset(values: dict[str, np.ndarray], history_length: int, horizon_steps: int) -> None:
    count = int(values["history_observations"].shape[0])
    expected_shapes = {
        "history_target_visible": (count, history_length, 4),
        "history_message_age_steps": (count, history_length, 4),
        "history_observation_timestamps": (count, history_length, 4),
        "future_target_positions": (count, horizon_steps, 3),
        "future_defender_actions": (count, horizon_steps, 4, 3),
        "future_defender_positions": (count, horizon_steps + 1, 4, 3),
        "future_defender_velocities": (count, horizon_steps + 1, 4, 3),
    }
    if values["history_observations"].ndim != 4 or values["history_observations"].shape[1:3] != (history_length, 4):
        raise ValueError("history_observations has an invalid shape.")
    for name, expected in expected_shapes.items():
        if values[name].shape != expected:
            raise ValueError(f"{name} has shape {values[name].shape}, expected {expected}.")
    for name, value in values.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values.")
    if not np.isin(values["branch_sign"], (-1, 1)).all():
        raise ValueError("Every S4 sample must have a committed lower/upper branch label.")
    if values["sampling_weights"].shape != (count,) or np.any(values["sampling_weights"] <= 0.0):
        raise ValueError("sampling_weights must be positive with one value per S4 sample.")


def balanced_sampling_weights(policy_ids: np.ndarray, branch_sign: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Give every observed policy-by-branch stratum equal sampling mass."""

    policy_ids = np.asarray(policy_ids, dtype=np.int8)
    branch_sign = np.asarray(branch_sign, dtype=np.int8)
    if policy_ids.shape != branch_sign.shape or policy_ids.ndim != 1:
        raise ValueError("policy_ids and branch_sign must be one-dimensional and aligned.")
    strata = policy_ids.astype(np.int16) * 10 + (branch_sign > 0).astype(np.int16)
    unique, inverse, counts = np.unique(strata, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights /= float(np.mean(weights))
    summary = {str(int(key)): int(count) for key, count in zip(unique, counts, strict=True)}
    return weights.astype(np.float32), summary


def source_hashes(collection_config: Path, protocol_path: Path) -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "scripts" / "collect_s4_branching_dataset.py",
        PROJECT_ROOT / "scripts" / "evaluate_s4_branching.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "pursuit_env.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "showcase.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "observation_encoding.py",
        collection_config.resolve(),
        protocol_path.resolve(),
    )
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def main() -> None:
    args = parse_args()
    collection_path = args.collection_config.resolve()
    protocol_path = args.protocol.resolve()
    collection = load_collection_config(collection_path)
    protocol = load_s4_protocol(protocol_path)
    episodes = int(args.episodes if args.episodes is not None else collection["episodes"])
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    history_length = int(collection["history_length"])
    horizon_steps = int(collection["horizon_steps"])

    aggregates: dict[str, list[np.ndarray]] = {}
    sample_fields = (
        "history_observations",
        "history_target_visible",
        "history_message_age_steps",
        "history_observation_timestamps",
        "future_target_positions",
        "future_defender_actions",
        "future_defender_positions",
        "future_defender_velocities",
        "sample_timesteps",
        "episode_indices",
        "episode_seeds",
        "layout_seeds",
        "target_speed_scales",
        "branch_sign",
        "branch_decision_steps",
        "rollout_policy_ids",
        "defender_bias_sign",
    )
    for field in sample_fields:
        aggregates[field] = []
    episode_records: list[dict[str, Any]] = []
    for episode_index in range(episodes):
        spec = episode_spec(collection, episode_index)
        config_spec = {
            **spec,
            "pursuit_overrides": spec["pursuit_overrides"],
        }
        config = s4_config_for_spec(args.environment_config, protocol, config_spec, max_steps=None)
        packed, record = collect_episode(config, spec, history_length, horizon_steps)
        count = int(record["sample_count"])
        episode_records.append(record)
        if count == 0:
            continue
        for name in (
            "history_observations",
            "history_target_visible",
            "history_message_age_steps",
            "history_observation_timestamps",
            "future_target_positions",
            "future_defender_actions",
            "future_defender_positions",
            "future_defender_velocities",
            "sample_timesteps",
        ):
            aggregates[name].append(packed[name])
        branch_sign = int(record["target_branch_sign"])
        branch_decision_step = int(record["target_branch_decision_step"])
        aggregates["episode_indices"].append(np.full(count, episode_index, dtype=np.int32))
        aggregates["episode_seeds"].append(np.full(count, int(spec["episode_seed"]), dtype=np.int64))
        aggregates["layout_seeds"].append(np.full(count, int(spec["layout_seed"]), dtype=np.int64))
        aggregates["target_speed_scales"].append(np.full(count, float(spec["target_speed_scale"]), dtype=np.float32))
        aggregates["branch_sign"].append(np.full(count, branch_sign, dtype=np.int8))
        aggregates["branch_decision_steps"].append(np.full(count, branch_decision_step, dtype=np.int16))
        aggregates["rollout_policy_ids"].append(
            np.full(count, POLICY_IDS[str(spec["rollout_policy"])], dtype=np.int8)
        )
        aggregates["defender_bias_sign"].append(
            np.full(count, 1 if spec["defender_bias"] == "upper" else -1, dtype=np.int8)
        )
        if (episode_index + 1) % 10 == 0 or episode_index + 1 == episodes:
            print(
                json.dumps(
                    {
                        "status": "collecting",
                        "episodes_completed": episode_index + 1,
                        "episodes_requested": episodes,
                        "complete_window_count": int(sum(part.shape[0] for part in aggregates["episode_indices"])),
                    }
                ),
                flush=True,
            )

    if not aggregates["history_observations"]:
        raise RuntimeError("S4 collection produced no complete trajectory windows.")
    values = {name: np.concatenate(parts, axis=0) for name, parts in aggregates.items()}
    values["sampling_weights"], stratum_counts = balanced_sampling_weights(
        values["rollout_policy_ids"], values["branch_sign"]
    )
    validate_dataset(values, history_length, horizon_steps)
    np.savez_compressed(output / "dataset.npz", **values)
    output.joinpath("scenes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in episode_records), encoding="utf-8"
    )
    metadata = {
        "dataset_name": str(collection["dataset_name"]),
        "schema_version": int(collection["schema_version"]),
        "split": str(collection["split"]),
        "episodes_requested": episodes,
        "episodes_with_samples": int(sum(int(record["sample_count"]) > 0 for record in episode_records)),
        "sample_count": int(values["history_observations"].shape[0]),
        "history_length": history_length,
        "horizon_steps": horizon_steps,
        "feature_dim": int(values["history_observations"].shape[-1]),
        "rollout_policy_ids": POLICY_IDS,
        "sampling_contract": {
            "field": "sampling_weights",
            "strategy": "equal_total_mass_per_observed_rollout_policy_by_actual_branch_stratum",
            "stratum_encoding": "10 * rollout_policy_id + (branch_sign > 0)",
            "raw_stratum_counts": stratum_counts,
            "mean_weight": float(np.mean(values["sampling_weights"])),
        },
        "input_contract": {
            "history_observations": "policy-safe local observation encoder output",
            "history_target_visible": "published local detection mask",
            "history_message_age_steps": "published communication-age signal",
            "history_observation_timestamps": "published measurement timestamp",
            "contains_target_truth": False,
        },
        "label_contract": {
            "future_target_positions": "simulator truth; supervised label only",
            "future_defender_actions": "realized ideal velocity commands",
            "future_defender_positions_and_velocities": "future public defender state labels",
            "branch_sign_and_decision_step": "simulator-private target policy labels only",
        },
        "collection_config": collection,
        "protocol": str(protocol_path),
        "environment_config": str(args.environment_config.resolve()),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "source_hashes": source_hashes(collection_path, protocol_path),
    }
    output.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"dataset": str(output / "dataset.npz"), "metadata": metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
