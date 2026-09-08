# Phase 5 Execution Reachable-Set Calibration

> Version: v1.0 (2026-09-08)
> Run: `results/phase5_execution_reachable_calibration_v2/`
> Decision: `empirical-margin-calibration-complete; formal-invariance-pending`

## 1. Purpose

The previous execution audit used a command-noise position tube, but the hard
randomized execution variant also changes speed, acceleration, mass, and drag.
This independent calibration measures how much that tube misses when the same
queued command is executed under the configured perturbation ranges.

For each of four execution variants, 2048 independent samples were generated.
Each sample uses the same initial state, queue, and selected command for a
nominal rollout and a perturbed rollout. The perturbed rollout includes
bounded command noise and, for hard variants, randomized execution parameters.
The error is the maximum defender position error at each preview step.

This is an empirical calibration experiment. It does not prove a reachable
set for an unseen distribution, a continuous-time system, or a real airframe.

## 2. Results

The base tube is the existing `position_uncertainty_radii` bound. Simultaneous
coverage requires the entire five-step error vector to remain inside one
frozen radius vector. The calibrated multiplier is computed from the upper
99th order statistic of the maximum normalized error over all five steps.

| Variant | Base tube coverage | Calibrated simultaneous coverage | Simultaneous multiplier | Frozen radius at step 5 (m) | Base tube decision |
| --- | ---: | ---: | ---: | ---: | --- |
| mild immutable | 100.00% | 99.023% | 0.864 | 0.197 | passes empirical target |
| hard randomized immutable | 45.557% | 99.023% | 2.004 | 0.435 | fails empirical target |
| mild flush pending | 100.00% | 99.023% | 0.862 | 0.197 | passes empirical target |
| hard flush pending | 46.680% | 99.023% | 1.988 | 0.432 | fails empirical target |

The hard variants demonstrate the main result: a noise-only tube is not an
adequate robust margin once execution parameters are randomized. A rounded
engineering multiplier near `2.1` would cover this locked calibration sample,
but it must not be inserted into the safety filter without a separate
calibration split, sensitivity analysis, and repeat on held-out parameters.

## 3. Protocol and artifacts

| Item | Value |
| --- | --- |
| Samples per variant | 2048 |
| Preview horizon | 5 steps |
| Defenders | 4 |
| Target simultaneous coverage | 99% |
| Quantile rule | upper order statistic, finite-sample conservative |
| Variants | mild/hard immutable and mild/hard flush pending |
| Per-sample record | `samples.jsonl` |
| TensorBoard | `tensorboard/` |

The run keeps `config.yaml`, `summary.json`, `samples.jsonl`, TensorBoard
events, and source hashes. Reproduction command:

```powershell
$env:PYTHONPATH = "src"
python scripts/calibrate_execution_reachable_set.py `
  --config configs/innovation_execution_reachable_calibration.yaml `
  --output-dir results/phase5_execution_reachable_calibration_repro
```

The output directory must be new; the evaluator refuses to overwrite an
existing directory.

## 4. Decision boundary

The empirical calibration task is complete, but the P5 safety gate remains
No-Go. The calibrated radius is evidence for a margin contract, not a safety
certificate. Before using it in the filter, the project still needs:

- a held-out execution-parameter calibration split;
- coverage sensitivity over delay, tracking, noise, mass, and drag ranges;
- an explicit policy for out-of-calibration execution states;
- continuous-time swept-volume and multi-step forward-invariance analysis;
- a solver that can enforce the resulting margins without excessive fallback.

TensorBoard retains the per-variant coverage, radius, and multiplier metrics;
the JSONL records permit recomputation without relying on the dashboard.
