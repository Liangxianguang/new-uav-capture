# 3D Multi-UAV Cooperative Capture

This repository contains a reproducible kinematic benchmark for four UAV
pursuers cooperatively capturing one evasive target in a partially observable
three-dimensional obstacle field. Obstacles are central cylinders, boxes, and
walls; the target and pursuers start on opposite sides and may safely route
around or above obstacles.

The released workflow is intentionally narrow: train or evaluate a recurrent
behavior-cloning policy, execute it with the local CBF safety filter, and
replay the complete capture trajectory as PNG/GIF/H.264 MP4. Historical
branches that are not part of this workflow have been removed from the active
repository.

## Task And Scope

An episode is a **Cooperative Safe Capture** when all conditions hold:

1. At least one pursuer enters the target capture radius of `0.80 m`.
2. Capture occurs within the `250` control-step limit (`dt = 0.1 s`).
3. Before termination there is no obstacle collision, inter-UAV collision, or
   boundary violation.
4. At least two pursuers have entered the central obstacle zone. The target is
   not required to enter or cross that zone.

The world is `20 m x 20 m x 10 m`; the benchmark uses four pursuers with
maximum speed `5.0 m/s` and maximum acceleration `6.0 m/s^2`. Observations
include detection dropout/noise and delayed, lossy teammate messages. This is
a **kinematic simulation**. Capture-radius entry is not physical contact,
net capture, flight-control validation, real vision, SITL, or a flight test.

## Released Method And Evidence

The released checkpoint is **V5 exact-reactive recurrent behavior cloning +
local CBF**:

- a parameter-sharing recurrent actor consumes local target-belief, teammate,
  and shape-aware obstacle observations;
- it is trained from quality-gated rule-expert demonstrations;
- deployment resets the recurrent hidden state every control step, matching
  the V5 training contract (`sequence_length = 1`);
- the CBF filter modifies proposed velocity commands to respect obstacle,
  inter-UAV, and world-boundary constraints.

`Policy + CBF` is the deployed stack. It must never be reported as the raw
neural policy alone.

| Evidence | Cooperative Safe Capture | Status |
| --- | ---: | --- |
| V4 fixed cylinder / box / wall / mixed S2 | `100.0 / 100.0 / 98.7 / 100.0%` | Formal three-seed locked test |
| V4 random mixed S3, policy + CBF | `75.3% +/- 6.5%` | Formal three-seed locked test |
| V4 random mixed S3, raw policy | `2.3% +/- 1.2%` | Formal three-seed locked test |
| Released V5 random mixed S3, policy + CBF | `57/60 = 95.0%` | Development validation, one training seed |

The V5 checkpoint is the most useful runnable model, but its `95.0%` result is
**not** a multi-seed locked-test result. The formal V4 result remains the
defensible benchmark claim. Full evidence and reporting boundaries are in
[docs/evidence/README.md](docs/evidence/README.md).

## Repository Map

```text
configs/                         Versioned environment, S3 protocol, and retraining YAML
models/                          Released V5 development checkpoint and checksum
src/encirclement3d/              Pursuit environment, observations, actor, CBF, scenarios
scripts/                         Train, evaluate, replay, render, and PowerShell/BAT launchers
tests/                           Regression tests for task semantics and reproducible tools
docs/evidence/                   Formal result reports and structured summaries
docs/media/                      Reviewed successful capture PNG/GIF/MP4
results/                         Local run outputs only; ignored by Git
```

Important files:

| Purpose | File |
| --- | --- |
| Environment contract | `configs/capture_radius_pursuit_central_v4_flee.yaml` |
| Random mixed-obstacle protocol | `configs/central_random_mixed_obstacle_s3_v5_protocol.yaml` |
| Standalone retraining contract | `configs/capture_radius_recurrent_behavior_cloning_s3_retrain.yaml` |
| Released model | `models/v5_development_exact_reactive_seed661606.pt` |
| Training CLI | `scripts/train_capture_radius_recurrent_behavior_cloning.py` |
| Random S3 evaluator | `scripts/evaluate_random_central_mixed_obstacles.py` |
| Fixed S1/S2 evaluator | `scripts/evaluate_mixed_obstacle_showcase.py` |
| Scene replay and 3-D renderer | `scripts/render_random_capture_episode.py`, `scripts/render_3d_capture_animation.py` |

## Installation

The repository was validated on Windows, Python 3.11, PyTorch `2.7.1+cu126`,
and an RTX 4060. CUDA is recommended for training; CPU works for evaluation
and rendering.

```powershell
Set-Location F:\uav_capture\three_d_encirclement
conda env create -f environment.yml
conda activate uav-encirclement-gpu
$env:PYTHONPATH = "$PWD\src;$PWD\scripts"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

For an existing environment, run `conda env update -f environment.yml --prune`.

## Reproduce The Released V5 Result

The following command evaluates the published checkpoint on the frozen V5
**development validation** block. It creates the sampled maps, per-episode
CSV, JSON summary, and provenance under a new local directory.

```powershell
.\scripts\reproduce_released_v5.ps1 -Device cuda
```

The historical expected result is `57/60` Cooperative Safe Capture, `0%`
collision, `0%` boundary violation, and `100%` Transit. This is a verification
target for the released development model, not a new locked test.

Equivalent manual evaluation:

```powershell
python scripts/evaluate_random_central_mixed_obstacles.py `
  --checkpoint models/v5_development_exact_reactive_seed661606.pt `
  --environment-config configs/capture_radius_pursuit_central_v4_flee.yaml `
  --protocol configs/central_random_mixed_obstacle_s3_v5_protocol.yaml `
  --split validation `
  --output-dir results/reproduce_v5_s3_validation `
  --use-cbf --recurrent-reset-interval 1 --device cuda
```

To assess the safety layer, rerun the same frozen scenes without `--use-cbf`
in a different output directory. Do not tune the model or CBF from either
evaluation.

## Fixed-Scene Evaluation

This command checks a fixed mixed obstacle scene. `--protocol-config` is
required because it supplies the frozen central-zone task contract.

```powershell
python scripts/evaluate_mixed_obstacle_showcase.py `
  --checkpoint models/v5_development_exact_reactive_seed661606.pt `
  --method capture --scenario v4_s2 `
  --protocol-config configs/central_bidirectional_v4.yaml `
  --episodes 20 --seed 660501 `
  --output-dir results/reproduce_v5_fixed_s2 `
  --use-cbf --recurrent-reset-interval 1 --device cuda
```

## Replay And Render A Full Capture

The automatic reproduction script renders episode `0`. To render manually
after evaluation, use the generated `scenes.jsonl` file. The renderer writes a
top-down PNG/GIF/MP4 followed by a perspective 3-D PNG/GIF/MP4 with obstacle
volumes, altitude cues, trajectory tails, dynamic capture radius, and a frozen
capture frame.

```powershell
python scripts/render_random_capture_episode.py `
  --checkpoint models/v5_development_exact_reactive_seed661606.pt `
  --scenes results/reproduce_v5_s3_validation/scenes.jsonl `
  --episode-index 0 --use-cbf --recurrent-reset-interval 1 `
  --output-dir results/reproduce_v5_episode0 --device cuda

python scripts/render_3d_capture_animation.py `
  --trajectory results/reproduce_v5_episode0/trajectory.npz `
  --result results/reproduce_v5_episode0/episode.json `
  --output-dir results/reproduce_v5_episode0/three_d
```

One reviewed successful development replay is available in
[docs/media/README.md](docs/media/README.md):

![V5 capture frame](docs/media/v5_development_s3_episode0_capture_3d.png)

## Train A New Candidate From Scratch

The historical V4/V5 expert archives and three frozen V4 checkpoints are not
published. Therefore no clean clone can bitwise recreate those historical
models. The command below **does** reproduce the released standalone training
procedure: it collects new, quality-gated expert demonstrations locally,
trains a sequence-length-one recurrent BC actor, writes TensorBoard data, and
saves every effective configuration and source hash.

```powershell
python scripts/train_capture_radius_recurrent_behavior_cloning.py `
  --config configs/capture_radius_recurrent_behavior_cloning_s3_retrain.yaml `
  --output results/train_seed661606 `
  --seed 661606 --device cuda --sequence-length 1 --sequence-batch-size 16
```

Use a different output directory for each seed. The trainer refuses to
overwrite nonempty directories. Compare a newly trained model only on a fresh
development block; do not label it as the released V5 checkpoint or as a
formal V4 result.

PowerShell and BAT wrappers are included:

```powershell
.\scripts\run_capture_radius_recurrent_behavior_cloning.ps1 `
  -Output results\train_seed661606 -Seed 661606 -Device cuda

.\scripts\start_tensorboard.ps1 -LogDir results -Port 6006
```

## Verify The Repository

```powershell
python -m pytest -q
python scripts/evaluate_random_central_mixed_obstacles.py --help
python scripts/render_random_capture_episode.py --help
python scripts/render_3d_capture_animation.py --help
```

Generated checkpoints, expert archives, TensorBoard logs, CSV/NPZ trajectory
data, and newly rendered media are intentionally ignored. Each run stores its
effective YAML, metadata, and hashes inside its own `results/` directory.

## Reporting Rules

- Report raw policy and `Policy + CBF` separately.
- Report the V4 locked result as `75.3% +/- 6.5%` on random S3, not the V5
  single-seed `95.0%` development result.
- Transit is an independent route-feasibility diagnostic, not capture.
- A capture-radius event is not physical contact or entity capture.
- The active benchmark is kinematic. The local, uncommitted execution-dynamics
  draft in `src/encirclement3d/pursuit_env.py` is intentionally outside this
  release until it has a complete experimental protocol and tests.
