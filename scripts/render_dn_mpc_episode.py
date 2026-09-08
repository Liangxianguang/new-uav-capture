"""Replay locked DN-MPC episodes and render them as MP4/GIF media.

This command reuses the frozen scene record, prediction checkpoint, planner
configuration, and sampling settings from an existing S3 evaluation output.
It creates replay-only trajectory files and never edits the formal JSONL
statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from encirclement3d.distributed_dn_mpc import DistributedDNMPCConfig  # noqa: E402
from encirclement3d.showcase import scenario_from_metadata  # noqa: E402
from evaluate_minimax_mpc import (  # noqa: E402
    MinimaxMPCConfig,
    load_yaml,
    model_from_checkpoint,
    run_episode,
    select_device,
)
from evaluate_minimax_mpc_s3 import (  # noqa: E402
    config_for_spec,
    load_protocol,
    load_scene_records,
)
from render_3d_capture_animation import render_animation  # noqa: E402


METHOD_MODES = {
    "distributed_ideal": "ideal",
    "distributed_delayed": "delayed",
    "distributed_dropout": "dropout",
    "distributed_none": "none",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--method", choices=tuple(METHOD_MODES), default="distributed_delayed")
    parser.add_argument("--episode-indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--tail-length", type=int, default=0)
    parser.add_argument("--freeze-seconds", type=float, default=1.75)
    return parser.parse_args()


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def main() -> None:
    args = parse_args()
    if args.fps <= 0 or args.frame_stride <= 0 or args.tail_length < 0 or args.freeze_seconds < 0:
        raise ValueError("fps/frame-stride must be positive and tail/freeze values must be non-negative")
    evaluation_dir = _path(args.evaluation_dir)
    output_dir = _path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation = load_yaml(evaluation_dir / "config.yaml")
    protocol_path = _path(evaluation["protocol"])
    environment_path = _path(evaluation["environment_config"])
    mpc_path = _path(evaluation["mpc_config"])
    checkpoint_path = _path(evaluation["checkpoint"])
    scene_path = evaluation_dir / "scenes.jsonl"
    if not scene_path.is_file():
        scene_path = _path(evaluation["scene_records"])

    protocol = load_protocol(protocol_path)
    records = load_scene_records(
        scene_path,
        protocol=protocol,
        split=str(evaluation["split"]),
        episodes=int(evaluation["episodes"]),
    )
    indices = sorted(set(int(index) for index in args.episode_indices))
    if not indices or min(indices) < 0 or max(indices) >= len(records):
        raise ValueError(f"episode indices must be in [0, {len(records) - 1}]")

    mpc_document = load_yaml(mpc_path)
    planner_config = MinimaxMPCConfig.from_mapping(dict(mpc_document["planner"]))
    distributed_mapping = dict(evaluation.get("distributed", mpc_document.get("distributed", {})))
    prediction = dict(evaluation.get("prediction", mpc_document.get("prediction", {})))
    checkpoint_data = model_from_checkpoint(checkpoint_path, select_device(args.device))
    device = select_device(args.device)
    use_local_cbf = bool(evaluation.get("use_local_cbf", True))
    scenario_mode = METHOD_MODES[args.method]
    outputs: list[dict[str, Any]] = []

    for episode_index in indices:
        record = records[episode_index]
        spec = record["spec"]
        episode_dir = output_dir / f"episode_{episode_index:03d}"
        episode_dir.mkdir(parents=True, exist_ok=False)
        trajectory_path = episode_dir / "trajectory.npz"
        config = config_for_spec(environment_path, spec, max_steps=None)
        distributed_config = DistributedDNMPCConfig.from_mapping(
            {**distributed_mapping, "communication_mode": scenario_mode}
        )
        row, steps = run_episode(
            config,
            seed=int(spec["episode_seed"]),
            method=args.method,
            planner_config=planner_config,
            candidate_source=str(evaluation.get("candidate_source", "checkpoint")),
            checkpoint_data=checkpoint_data,
            device=device,
            num_samples=int(prediction.get("num_samples", 8)),
            sampling_steps=int(prediction.get("sampling_steps", 8)),
            sampling_seed=int(prediction.get("sampling_seed", 745102)) + episode_index * 1000,
            projection_iterations=int(prediction.get("projection_iterations", 4)),
            use_local_cbf=use_local_cbf,
            prediction_refresh_interval_steps=int(prediction.get("refresh_interval_steps", 1)),
            distributed_config=distributed_config,
            scenario=scenario_from_metadata(record["scenario"]),
            validate_scenario=False,
            record_history=True,
            trajectory_path=trajectory_path,
        )
        row.update(
            {
                "episode_index": episode_index,
                "episode_seed": int(spec["episode_seed"]),
                "layout_seed": int(spec["layout_seed"]),
                "observation_condition": str(spec["observation_condition"]),
                "target_motion_mode": str(spec["target_motion_mode"]),
                "defender_side": str(spec["defender_side"]),
                "obstacle_count": int(spec["obstacle_count"]),
                "render_source": str(evaluation_dir),
                "render_replay_only": True,
            }
        )
        result_path = episode_dir / "episode.json"
        result_path.write_text(json.dumps(row, indent=2, allow_nan=True), encoding="utf-8")
        (episode_dir / "steps.jsonl").write_text(
            "".join(json.dumps(step, allow_nan=True) + "\n" for step in steps),
            encoding="utf-8",
        )
        media = render_animation(
            trajectory_path,
            result_path,
            episode_dir,
            fps=args.fps,
            frame_stride=args.frame_stride,
            tail_length=args.tail_length,
            freeze_seconds=args.freeze_seconds,
        )
        row["media"] = media
        result_path.write_text(json.dumps(row, indent=2, allow_nan=True), encoding="utf-8")
        outputs.append({"episode_index": episode_index, "episode": row, "media": media})

    manifest = {
        "evaluation_dir": str(evaluation_dir),
        "method": args.method,
        "device": str(device),
        "episodes": outputs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
