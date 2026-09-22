# Phase86 pure MAPPO/IPPO reproduction

This document records the reproducible training contract for the current
Phase86 pure-PPO experiment. The training rollout does **not** use CBF; CBF is
an optional evaluation-only safety layer.

## Current run

- Algorithm: MAPPO
- Seed: `101`
- Target: `5000` updates
- Episodes per update: `8`
- PPO epochs: `4`
- Minibatch size: `512`
- Hidden dimension: `128`
- Learning rate: `3e-4`
- Gamma / GAE lambda: `0.99 / 0.95`
- Max episode steps: `250`
- CBF training: disabled
- Checkpoint interval: **every 500 updates**
- Current output: `models/phase86_pure_mappo_route_nocbf_long_seed101_20260921_v2/`

At the documentation snapshot on 2026-09-22, the run had reached update
`1304/5000`. This is an ongoing development run, not a completed formal
comparison or a multi-seed result.

The launcher writes both `checkpoint_latest.pt` at the run root and archived
files such as `checkpoints/checkpoint_update_000500.pt` and
`checkpoints/checkpoint_update_001000.pt`. Generated checkpoints, progress
files, TensorBoard events, and process logs are intentionally excluded from
Git.

## Validation snapshot

The monitor evaluates each archived checkpoint on the frozen 300-episode
development validation split in two explicitly labelled modes. `raw` is the
pure PPO policy. `eval_cbf` applies an external local CBF safety layer and is
not a pure-PPO result.

| Checkpoint | Mode | Safe capture | Collision | Boundary violation | Target invalid | Timeout | Mean min clearance |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | raw | 24.67% | 75.33% | 0% | 0% | 0% | 0.008 m |
| 500 | eval-CBF | 86.33% | 0% | 0% | 0% | 13.67% | 1.007 m |
| 1000 | raw | 70.67% | 29.33% | 12.00% | 0% | 0% | 0.460 m |
| 1000 | eval-CBF | 98.67% | 0% | 0% | 0% | 1.33% | 1.012 m |

The latest online 8-episode rollout at that snapshot was `75%` safe capture,
`25%` defender physical collision, `0%` defender boundary violation, `0%`
target invalid, and `0%` timeout. It is included only as a training-progress
diagnostic; use the frozen 300-episode validation results for reporting.

## CUDA versus CPU

The formal training process is launched with `--device cuda`. The separate
independent-validation monitor is launched with `--device cpu` deliberately,
so validation does not compete with training for the GPU. TensorBoard also
runs on the CPU.

The small policy network and the current single-process environment rollout
can leave the GPU lightly utilized even when training is correctly on CUDA.
The training script uses `--torch-threads 1` to keep the formal run stable and
reproducible. On an RTX 5050 machine, use a CUDA-enabled PyTorch build and
verify `torch.cuda.is_available()` before starting; the portable launcher
refuses a silent CUDA-to-CPU fallback.

## Portable Windows launch

From the repository root on the second computer:

```powershell
.\scripts\run_phase86_pure_mappo.ps1 -Python python -Device cuda
```

If the input assets are stored elsewhere, pass relative or absolute paths:

```powershell
.\scripts\run_phase86_pure_mappo.ps1 `
  -Python C:\path\to\python.exe `
  -Device cuda `
  -TrainingScenes C:\data\phase86\scenes.jsonl `
  -ExpertDataset C:\data\phase86\expert_dataset.npz `
  -Output models\phase86_pure_mappo_seed101_repro
```

The same launcher supports IPPO for an apples-to-apples algorithm comparison:

```powershell
.\scripts\run_phase86_pure_mappo.ps1 -Algorithm ippo -Device cuda
```

Do not add `-UseCbfTrain` for the pure-PPO baseline. Use CBF only in a
separately labelled evaluation command.

## Required repository assets

The exact current training inputs are:

- `configs/phase86_pure_mappo_repro.yaml` (portable config used by the launcher)
- `configs/phase86_fix_route_mappo_development_v1.yaml` (original formal-run provenance)
- `configs/phase85_target_contract_repaired_environment.yaml`
- `results/phase86_nominal_repaired_development_calibration/scenes.jsonl`
- `models/phase86_fix_route_teacher_formal_contract_20260921/expert_dataset.npz`
- `scripts/train_mappo_ippo_baseline.py`
- `scripts/rl_scene_io.py`
- `src/encirclement3d/learning.py`
- `src/encirclement3d/observation_encoding.py`
- `src/encirclement3d/pursuit_controllers.py`
- `src/encirclement3d/pursuit_env.py`

The scene file and expert dataset are small reproducibility assets and are
versioned with this experiment. Large run outputs are not versioned.

## Resume

To resume, place the desired `checkpoint_latest.pt` inside the output
directory and pass it with `-Resume`. When resuming, the launcher does not
repeat behavior cloning. The checkpoint contract checks the algorithm,
observation dimensions, environment/config hashes, training-scene hash, and
restores optimizer and RNG state.

## Validation labels

- `independent_validation/update_N/raw/evaluation.json`: pure MAPPO/IPPO;
  `use_cbf=false`.
- `independent_validation/update_N/eval_cbf/evaluation.json`: the same policy
  with an external CBF evaluation layer; this is not a pure-PPO result.

Always report safe capture, defender physical collision, defender boundary
violation, target invalidity, and timeout separately.
