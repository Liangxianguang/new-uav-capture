# Phase 17 UAKR/RNIC Pilot Report

Status: **pilot evidence only; no locked-test claim**.

These runs use the same eight seeds `648301..648308`, the same two-step
immutable execution queue, belief candidates, CPU execution, distributed
delayed DN-MPC and local CBF as the QDR smoke.  The pilot is intended to test
the implementation and logging contracts; it is not the planned validation
scale and it does not tune on locked-test data.

| Arm | QDR | UAKR | RNIC | Safe capture | Collision | Total p50/p95/p99 (ms) | Mean K | Refresh rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed-K reference | on | off | off | 87.5% | 12.5% | 73.83 / 97.09 / 102.78 | 8* | 100.0% |
| UAKR pilot | on | on | off | 87.5% | 12.5% | 61.47 / 94.89 / 101.47 | 1.50 | 69.1% |
| RNIC pilot | on | off | on | 87.5% | 12.5% | 108.26 / 146.77 / 163.60 | 8* | 100.0% |
| Full smoke | on | on | on | 87.5% | 12.5% | 89.23 / 148.05 / 231.65 | 1.50 | 69.1% |

\* The fixed-K rows use the belief source with eight candidates.  UAKR's
mean K is the actual selected budget, not a duplicated mean trajectory.

## UAKR diagnostics

The UAKR pilot selected low/medium/high buckets `104/18/1` times, with mean
uncertainty score `0.265`, mean cache age `0.579` steps, mean prediction
residual `0.378 m`, and forced-refresh rate `66.8%`.  The scheduler reduced
the refresh rate to `69.1%` and mean candidate budget to `1.50`, while this
small sample produced the same safe-capture/collision outcome as fixed-K.

The high forced-refresh rate is a warning, not a success claim: the current
residual trigger is too active for this smoke configuration and must be
calibrated on validation scenes.  The pilot used the belief candidate source,
so the UAKR result does not yet establish the main Diagonal-SSM diffusion
claim.

## RNIC diagnostics

RNIC was integrated into both the centralized and local distributed scenario
cost paths.  In this pilot it did not change the safe-capture or collision
outcome, while adding substantial planner cost: fixed-K planner p50/p95/p99
rose from `58.62/81.41/85.71 ms` to `92.62/129.19/145.12 ms`.  This is a
useful negative result: RNIC should not be retained in the main branch until
the planned speed-OOD validation shows a paired reduction in timeout or
unreachable interception decisions.

The RNIC implementation reports bounded acceleration/speed arrival-time
surrogates and normalized shortfall costs.  It remains a planner heuristic;
it is not a reachable-set proof, a CBF certificate, or a three-dimensional
minimum-time theorem.

## Reproducibility artifacts

Every run retains `config.yaml`, `episodes.jsonl`, `steps.jsonl`,
`summary.json`, and TensorBoard events under:

- `results/phase17_uakr_qdr_smoke_v2/distributed_delayed/tensorboard/`
- `results/phase17_rnic_qdr_fixedk_smoke/distributed_delayed/tensorboard/`
- `results/phase17_full_combination_smoke/distributed_delayed/tensorboard/`

The next required experiment is a validation-only paired matrix using the
Diagonal-SSM checkpoint and the frozen geometry, speed, and delay OOD blocks;
thresholds must be frozen before confirmation scenes are opened.
