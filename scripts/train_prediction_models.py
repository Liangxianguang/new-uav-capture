"""Train TensorBoard-tracked GRU and SSM-conditioned diffusion predictors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

# Windows environments that package both Intel and LLVM OpenMP runtimes can
# initialize incompatible duplicate runtimes before PyTorch is imported.
if os.name == "nt":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from encirclement3d.prediction import (  # noqa: E402
    ConditionalDiffusionTrajectoryPredictor,
    HistoryTargetPredictor,
    deterministic_mse,
    gaussian_nll,
    prediction_metrics,
)
from encirclement3d.trajectory_dataset import (  # noqa: E402
    PredictionDataset,
    load_prediction_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("gru", "diffusion"), required=True)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int, default=8)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--target-scale", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=745101)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def flatten_inputs(dataset: PredictionDataset) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.as_tensor(dataset.history_observations, dtype=torch.float32)
    inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
    targets = torch.as_tensor(dataset.future_target_displacements, dtype=torch.float32)
    return inputs, targets


def source_hashes() -> dict[str, str]:
    paths = (
        PROJECT_ROOT / "scripts" / "train_prediction_models.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "prediction.py",
        PROJECT_ROOT / "src" / "encirclement3d" / "trajectory_dataset.py",
    )
    return {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def build_model(args: argparse.Namespace, input_dim: int, horizon_count: int) -> torch.nn.Module:
    if args.model == "gru":
        return HistoryTargetPredictor(
            input_dim=input_dim,
            horizon_count=horizon_count,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        )
    return ConditionalDiffusionTrajectoryPredictor(
        input_dim=input_dim,
        horizon_count=horizon_count,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        diffusion_steps=args.diffusion_steps,
    )


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_kind: str,
    target_scale: float,
) -> float:
    model.train()
    losses: list[float] = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device) / target_scale
        if model_kind == "gru":
            mean, log_variance = model(inputs)
            loss = gaussian_nll(mean, log_variance, targets)
        else:
            loss = model.diffusion_loss(inputs, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {model_kind} training loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    dataset: PredictionDataset,
    device: torch.device,
    model_kind: str,
    target_scale: float,
    num_samples: int,
    sampling_steps: int,
) -> dict[str, float]:
    model.eval()
    inputs = inputs.to(device)
    targets = targets.to(device)
    normalized_targets = targets / target_scale
    if model_kind == "gru":
        mean, log_variance = model(inputs)
        loss = gaussian_nll(mean, log_variance, normalized_targets)
        candidates = mean[:, None] * target_scale
        deterministic = deterministic_mse(mean, normalized_targets)
    else:
        loss = model.diffusion_loss(inputs, normalized_targets)
        candidates = model.sample(
            inputs,
            num_samples=num_samples,
            sampling_steps=sampling_steps,
        ) * target_scale
        deterministic = torch.tensor(float("nan"), device=device)
    metrics = prediction_metrics(candidates, targets)
    constant_velocity = (
        torch.as_tensor(dataset.reference_velocities, device=device)[:, None, :]
        * float(dataset.dt_seconds)
        * torch.arange(1, targets.shape[1] + 1, device=device, dtype=targets.dtype)[None, :, None]
    )
    baseline = prediction_metrics(constant_velocity[:, None], targets)
    return {
        "loss": float(loss.detach().cpu()),
        "deterministic_mse": float(deterministic.detach().cpu()),
        **{f"model_{key}": value for key, value in metrics.items()},
        **{f"constant_velocity_{key}": value for key, value in baseline.items()},
    }


def write_metadata(
    output: Path,
    args: argparse.Namespace,
    train_dataset: PredictionDataset,
    validation_dataset: PredictionDataset,
    input_dim: int,
    device: torch.device,
) -> None:
    serializable_arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metadata = {
        "model": args.model,
        "model_backend": "portable_diagonal_ssm" if args.model == "diffusion" else "gru_gaussian",
        "arguments": serializable_arguments,
        "train_dataset": str(args.train_dataset.resolve()),
        "validation_dataset": str(args.validation_dataset.resolve()),
        "train_samples": train_dataset.sample_count,
        "validation_samples": validation_dataset.sample_count,
        "input_dim": input_dim,
        "horizon_count": train_dataset.horizon_steps,
        "dt_seconds": float(train_dataset.dt_seconds),
        "device": str(device),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": version("torch"),
        "tensorboard": version("tensorboard"),
        "runtime_workaround": {
            "windows_openmp_duplicate_runtime": os.name == "nt",
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "KMP_DUPLICATE_LIB_OK": os.environ.get("KMP_DUPLICATE_LIB_OK"),
        },
        "source_hashes": source_hashes(),
    }
    output.joinpath("metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output.joinpath("config.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.hidden_dim <= 0 or args.num_layers <= 0:
        raise ValueError("epochs, batch-size, hidden-dim, and num-layers must be positive.")
    if args.learning_rate <= 0.0 or args.target_scale <= 0.0:
        raise ValueError("learning-rate and target-scale must be positive.")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)
    train_dataset = load_prediction_dataset(str(args.train_dataset.resolve()))
    validation_dataset = load_prediction_dataset(str(args.validation_dataset.resolve()))
    if (
        train_dataset.history_length != validation_dataset.history_length
        or train_dataset.horizon_steps != validation_dataset.horizon_steps
        or train_dataset.defender_count != validation_dataset.defender_count
        or train_dataset.feature_dim != validation_dataset.feature_dim
        or not np.isclose(train_dataset.dt_seconds, validation_dataset.dt_seconds)
    ):
        raise ValueError("Train and validation prediction datasets have incompatible shapes.")
    train_inputs, train_targets = flatten_inputs(train_dataset)
    validation_inputs, validation_targets = flatten_inputs(validation_dataset)
    input_dim = int(train_inputs.shape[-1])
    model = build_model(args, input_dim, train_dataset.horizon_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    write_metadata(output, args, train_dataset, validation_dataset, input_dim, device)
    writer = SummaryWriter(log_dir=str(output / "tensorboard"), flush_secs=10)
    writer.add_text(
        "Config/arguments",
        yaml.safe_dump(
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            sort_keys=False,
        ),
        0,
    )
    writer.add_text(
        "Config/dataset_contract",
        yaml.safe_dump(
            {
                "inputs": "policy-safe observations",
                "labels": "future target truth",
                "dt_seconds": float(train_dataset.dt_seconds),
            },
            sort_keys=False,
        ),
        0,
    )
    writer.add_text("Runtime/openmp", json.dumps({
        "windows_duplicate_runtime_workaround": os.name == "nt",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "KMP_DUPLICATE_LIB_OK": os.environ.get("KMP_DUPLICATE_LIB_OK"),
    }), 0)
    writer.add_text("Model/backend", "portable_diagonal_ssm" if args.model == "diffusion" else "gru_gaussian", 0)
    history: list[dict[str, float | int]] = []
    try:
        for epoch in range(1, args.epochs + 1):
            started = time.perf_counter()
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                args.model,
                args.target_scale,
            )
            validation = evaluate(
                model,
                validation_inputs,
                validation_targets,
                validation_dataset,
                device,
                args.model,
                args.target_scale,
                args.num_samples,
                args.sampling_steps,
            )
            elapsed = time.perf_counter() - started
            record = {"epoch": epoch, "train_loss": train_loss, **validation, "epoch_seconds": elapsed}
            history.append(record)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/validation", validation["loss"], epoch)
            for key, value in validation.items():
                if key != "loss" and np.isfinite(value):
                    writer.add_scalar(f"Metrics/{key}", value, epoch)
            writer.add_scalar("Timing/epoch_seconds", elapsed, epoch)
            writer.add_scalar("Optimization/learning_rate", args.learning_rate, epoch)
            if epoch == 1 or epoch % 5 == 0:
                for name, parameter in model.named_parameters():
                    writer.add_histogram(f"Parameters/{name}", parameter.detach().cpu(), epoch)
            writer.flush()
    finally:
        writer.close()
    final = history[-1]
    with SummaryWriter(log_dir=str(output / "tensorboard_hparams")) as hparam_writer:
        hparam_writer.add_hparams(
            {
                "model": args.model,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "learning_rate": args.learning_rate,
                "target_scale": args.target_scale,
                "diffusion_steps": args.diffusion_steps,
                "sampling_steps": args.sampling_steps,
                "num_samples": args.num_samples,
            },
            {
                "hparam/final_loss": float(final["loss"]),
                "hparam/model_min_ade": float(final["model_min_ade"]),
                "hparam/model_min_fde": float(final["model_min_fde"]),
                "hparam/constant_velocity_min_ade": float(final["constant_velocity_min_ade"]),
            },
        )
    output.joinpath("training.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint = {
        "model_kind": args.model,
        "model_backend": "portable_diagonal_ssm" if args.model == "diffusion" else "gru_gaussian",
        "model_config": (
            model.model_config
            if args.model == "diffusion"
            else {
                "input_dim": input_dim,
                "horizon_count": train_dataset.horizon_steps,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
            }
        ),
        "target_scale": args.target_scale,
        "dt_seconds": float(train_dataset.dt_seconds),
        "seed": args.seed,
        "state_dict": model.state_dict(),
        "final_metrics": final,
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    print(json.dumps({"output": str(output), "device": str(device), "final": final}, indent=2))


if __name__ == "__main__":
    main()
