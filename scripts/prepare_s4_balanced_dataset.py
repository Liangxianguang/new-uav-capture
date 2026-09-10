"""Create a weighted S4 training archive from a validated raw collection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from collect_s4_branching_dataset import balanced_sampling_weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw S4 dataset.npz archive.")
    parser.add_argument("--metadata", type=Path, required=True, help="Matching raw metadata.json.")
    parser.add_argument("--output", type=Path, required=True, help="Empty output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.resolve()
    metadata_path = args.metadata.resolve()
    if not source.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Input dataset and metadata must both exist.")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(source, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    required = {"history_observations", "rollout_policy_ids", "branch_sign"}
    missing = required.difference(values)
    if missing:
        raise ValueError("Raw S4 archive is missing fields: " + ", ".join(sorted(missing)))
    weights, stratum_counts = balanced_sampling_weights(values["rollout_policy_ids"], values["branch_sign"])
    values["sampling_weights"] = weights
    np.savez_compressed(output / "dataset.npz", **values)
    raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata = {
        **raw_metadata,
        "dataset_name": str(raw_metadata["dataset_name"]) + "_balanced_sampler",
        "derived_from": {
            "dataset": str(source),
            "metadata": str(metadata_path),
            "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "sampling_contract": {
            "field": "sampling_weights",
            "strategy": "equal_total_mass_per_observed_rollout_policy_by_actual_branch_stratum",
            "stratum_encoding": "10 * rollout_policy_id + (branch_sign > 0)",
            "raw_stratum_counts": stratum_counts,
            "mean_weight": float(np.mean(weights)),
            "weighting_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    output.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": str(output / "dataset.npz"),
                "sample_count": int(values["history_observations"].shape[0]),
                "sampling_strata": stratum_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
