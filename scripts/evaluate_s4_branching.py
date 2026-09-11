"""Run the frozen-scene S4 adaptive-branching pilot benchmark.

S4 is intentionally a calibration protocol, not a replacement for P4/P7.
The target commits to the less interceptable exit of a tall central wall from
the current defender geometry; the selected exit is never an actor input.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.distributed_dn_mpc import DistributedDNMPCConfig  # noqa: E402
from encirclement3d.pursuit_env import CaptureRadiusPursuit3DEnv  # noqa: E402
from encirclement3d.showcase import (  # noqa: E402
    s4_adaptive_branching_scenario,
    scenario_from_metadata,
    scenario_metadata,
)
from evaluate_minimax_mpc import (  # noqa: E402
    DEFAULT_ENVIRONMENT_CONFIG,
    DEFAULT_MPC_CONFIG,
    MinimaxMPCConfig,
    load_yaml,
    run_episode,
    select_device,
    source_hashes,
    summarize_rows,
)


DEFAULT_PROTOCOL = PROJECT_ROOT / "configs" / "phase15_s4_branching_pilot.yaml"
METHODS = (
    "dynamic_encirclement",
    "expected",
    "worst_case",
    "cvar",
    "distributed_ideal",
    "distributed_delayed",
    "distributed_dropout",
    "distributed_none",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--environment-config", type=Path, default=DEFAULT_ENVIRONMENT_CONFIG)
    parser.add_argument("--mpc-config", type=Path, default=DEFAULT_MPC_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, help="Pilot-only count override.")
    parser.add_argument("--max-steps", type=int, help="Pilot-only horizon override.")
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--candidate-source", choices=("belief",), default=None)
    parser.add_argument("--without-local-cbf", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def load_protocol(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("S4 protocol must be a YAML mapping.")
    if not isinstance(document.get("seed_blocks"), dict) or not isinstance(document.get("episodes_per_split"), dict):
        raise ValueError("S4 protocol requires seed_blocks and episodes_per_split mappings.")
    settings = document.get("s4")
    if not isinstance(settings, dict):
        raise ValueError("S4 protocol requires an s4 mapping.")
    if str(settings.get("target_motion_mode")) != "adaptive_branching":
        raise ValueError("S4 target_motion_mode must be adaptive_branching.")
    if not isinstance(settings.get("observation_conditions"), list) or not settings["observation_conditions"]:
        raise ValueError("S4 protocol requires observation_conditions.")
    return document


def episode_spec(protocol: dict[str, Any], episode_index: int) -> dict[str, Any]:
    settings = protocol["s4"]
    seed_block = int(protocol["seed_blocks"]["pilot"])
    conditions = list(
        itertools.product(
            settings["target_speed_scales"],
            settings["observation_conditions"],
            settings["defender_biases"],
        )
    )
    order = np.random.default_rng(seed_block + 2_000_000).permutation(len(conditions))
    condition_index = int(order[episode_index % len(order)])
    speed_scale, observation, defender_bias = conditions[condition_index]
    observation = dict(observation)
    # defender_bias is the final product axis, so upper/lower counterfactual
    # layouts share geometry noise while using independent motion seeds.
    return {
        "episode_seed": seed_block + int(episode_index),
        "layout_seed": seed_block + 1_000_000 + condition_index // 2,
        "target_speed_scale": float(speed_scale),
        "defender_bias": str(defender_bias),
        "observation_condition": str(observation["name"]),
        "pursuit_overrides": copy.deepcopy(observation["pursuit_overrides"]),
        "condition_index": condition_index,
        "condition_table_size": len(conditions),
    }


def config_for_spec(environment_config: Path, protocol: dict[str, Any], spec: dict[str, Any], max_steps: int | None) -> dict[str, Any]:
    config = copy.deepcopy(load_yaml(environment_config))
    pursuit = config.setdefault("task", {}).setdefault("pursuit", {})
    pursuit.update(copy.deepcopy(spec["pursuit_overrides"]))
    execution_overrides = spec.get("execution_overrides")
    if execution_overrides is not None:
        if not isinstance(execution_overrides, dict):
            raise ValueError("Frozen S4 execution_overrides must be a mapping.")
        config.setdefault("dynamics", {})["execution"] = copy.deepcopy(execution_overrides)
    geometry = dict(protocol["s4"]["branch_geometry"])
    pursuit.update(
        {
            "target_motion_mode": "adaptive_branching",
            "target_branch_decision_x": float(geometry["target_branch_decision_x_m"]),
            "target_branch_exit_offset_y": float(geometry["target_branch_exit_offset_y_m"]),
            "target_branch_waypoint_x": float(geometry["target_branch_waypoint_x_m"]),
            "target_branch_goal_x": float(geometry["target_branch_goal_x_m"]),
        }
    )
    config["experiments"] = [
        {
            "name": "phase15_s4_branching",
            "episodes": 1,
            "obstacle_count": 1,
            "target_speed_scale": float(spec["target_speed_scale"]),
        }
    ]
    configured_steps = int(protocol["s4"].get("max_steps", config["world"]["max_steps"]))
    config["world"]["max_steps"] = int(max_steps if max_steps is not None else configured_steps)
    if int(config["world"]["max_steps"]) <= 0:
        raise ValueError("max-steps must be positive.")
    return config


def source_hashes_s4(protocol_path: Path, mpc_config_path: Path) -> dict[str, str]:
    hashes = source_hashes(mpc_config_path)
    for path in (
        PROJECT_ROOT / "scripts" / "evaluate_s4_branching.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "showcase.py",
        protocol_path.resolve(),
    ):
        hashes[str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def grouped_summary(rows: list[dict[str, Any]], steps: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        episode_ids = {int(row["episode_index"]) for row in subset}
        subset_steps = [step for step in steps if int(step["episode_index"]) in episode_ids]
        summary = summarize_rows(subset, subset_steps)
        choices = Counter(
            "upper" if int(row["target_branch_sign"]) > 0 else "lower"
            for row in subset
            if row.get("target_branch_sign") in {-1, 1}
        )
        summary["target_branch_decision_rate"] = float(
            np.mean([row.get("target_branch_sign") in {-1, 1} for row in subset])
        )
        summary["target_branch_choice_counts"] = dict(sorted(choices.items()))
        return summary

    result: dict[str, Any] = {"overall": summarize(rows)}
    for field in ("observation_condition", "target_speed_scale", "defender_bias"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row[field])].append(row)
        result[f"by_{field}"] = {key: summarize(subset) for key, subset in sorted(groups.items())}
    return result


def main() -> None:
    args = parse_args()
    protocol_path = args.protocol.resolve()
    protocol = load_protocol(protocol_path)
    configured_episodes = int(protocol["episodes_per_split"]["pilot"])
    episodes = int(args.episodes if args.episodes is not None else configured_episodes)
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    mpc_document = load_yaml(args.mpc_config)
    planner_config = MinimaxMPCConfig.from_mapping(dict(mpc_document.get("planner", {})))
    distributed_mapping = dict(mpc_document.get("distributed", {}))
    defaults = dict(protocol.get("evaluation", {}))
    methods = list(args.methods or defaults.get("methods", ("dynamic_encirclement", "distributed_delayed")))
    candidate_source = str(args.candidate_source or defaults.get("candidate_source", "belief"))
    if candidate_source != "belief":
        raise ValueError("S4 pilot currently uses belief candidates; learned S4 data is collected separately.")
    use_local_cbf = not args.without_local_cbf and str(defaults.get("safety_layer", "local_cbf")) == "local_cbf"
    hashes = source_hashes_s4(protocol_path, args.mpc_config)
    run_config = {
        "protocol": str(protocol_path),
        "environment_config": str(args.environment_config.resolve()),
        "mpc_config": str(args.mpc_config.resolve()),
        "episodes": episodes,
        "methods": methods,
        "candidate_source": candidate_source,
        "use_local_cbf": use_local_cbf,
        "device": str(select_device(args.device)),
        "source_hashes": hashes,
    }
    output.joinpath("config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    output.joinpath("protocol.yaml").write_text(protocol_path.read_text(encoding="utf-8"), encoding="utf-8")

    records: list[dict[str, Any]] = []
    for episode_index in range(episodes):
        spec = episode_spec(protocol, episode_index)
        config = config_for_spec(args.environment_config, protocol, spec, args.max_steps)
        validation_env = CaptureRadiusPursuit3DEnv(config, obstacle_count=1, target_speed_scale=spec["target_speed_scale"])
        scenario = s4_adaptive_branching_scenario(
            validation_env,
            layout_seed=int(spec["layout_seed"]),
            defender_bias=str(spec["defender_bias"]),
        )
        records.append({"episode_index": episode_index, "spec": spec, "scenario": scenario_metadata(scenario)})
    output.joinpath("scenes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )

    all_summaries: dict[str, Any] = {}
    distributed_modes = {
        "distributed_ideal": "ideal",
        "distributed_delayed": "delayed",
        "distributed_dropout": "dropout",
        "distributed_none": "none",
    }
    for method in methods:
        method_output = output / method
        method_output.mkdir()
        rows: list[dict[str, Any]] = []
        steps: list[dict[str, Any]] = []
        with (
            method_output.joinpath("episodes.jsonl").open("w", encoding="utf-8") as episode_file,
            method_output.joinpath("steps.jsonl").open("w", encoding="utf-8") as step_file,
        ):
            for record in records:
                episode_index = int(record["episode_index"])
                spec = dict(record["spec"])
                config = config_for_spec(args.environment_config, protocol, spec, args.max_steps)
                distributed_config = None
                if method in distributed_modes:
                    distributed_config = DistributedDNMPCConfig.from_mapping(
                        {**distributed_mapping, "communication_mode": distributed_modes[method]}
                    )
                row, episode_steps = run_episode(
                    config,
                    seed=int(spec["episode_seed"]),
                    method=method,
                    planner_config=planner_config,
                    candidate_source="belief",
                    checkpoint_data=None,
                    device=select_device(args.device),
                    num_samples=8,
                    sampling_steps=8,
                    sampling_seed=745102 + episode_index * 1000,
                    projection_iterations=4,
                    prediction_refresh_interval_steps=1,
                    use_local_cbf=use_local_cbf,
                    distributed_config=distributed_config,
                    scenario=scenario_from_metadata(record["scenario"]),
                    validate_scenario=False,
                )
                row.update({"episode_index": episode_index, **spec})
                rows.append(row)
                for step in episode_steps:
                    steps.append({"episode_index": episode_index, **step})
                episode_file.write(json.dumps(row, allow_nan=True) + "\n")
                for step in episode_steps:
                    step_file.write(json.dumps({"episode_index": episode_index, **step}, allow_nan=True) + "\n")
        summary = grouped_summary(rows, steps)
        method_output.joinpath("summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
        all_summaries[method] = summary

    result = {
        "protocol": run_config,
        "methods": all_summaries,
        "decision": "pilot_only",
        "decision_rule": "Freeze formal S4 blocks only after this pilot shows non-saturated, non-degenerate capture rates.",
    }
    output.joinpath("summary.json").write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
