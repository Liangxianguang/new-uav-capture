# Phase 17 Geometry-OOD Pilot

Status: **20-episode diagnostic pilot; not confirmation evidence**.

This pilot reuses the frozen Phase 16 geometry-shift manifest and the
validation-selected Diagonal-SSM checkpoint from training seed `727201`. It
uses 20 scenes, a two-step immutable execution queue, distributed delayed
DN-MPC, local CBF, CPU inference, and the same sampling seed across arms. The
pilot is deliberately labelled `ood_diagnostic`; no locked-test tuning was
performed.

| Arm | QDR | UAKR | RNIC | Safe capture | Capture event | Collision | Boundary | Total p50/p95/p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fixed K, no QDR | off | off | off | 100% | 100% | 0% | 0% | 112.32 / 141.90 / 148.15 |
| Fixed K + QDR | on | off | off | 85% | 90% | 15% | 0% | 114.93 / 139.76 / 151.60 |
| UAKR + QDR | on | on | off | 90% | 95% | 10% | 0% | 62.67 / 132.86 / 143.62 |

UAKR selected mean K `2.42`, refresh rate `34.3%`, mean cache age `1.16`
steps, and mean uncertainty score `0.331`. The UAKR arm's predictor latency
was `0/56.26/60.60 ms` at p50/p95/p99 because cache reuse produces zero
sampling time on non-refresh steps. QDR rollout latency was
`1.25/1.68/2.18 ms` at p50/p95/p99.

## Interpretation

The pilot suggests a real compute trade-off for UAKR on the Diagonal-SSM
checkpoint: total p50 decreased substantially while the 20-scene safe-capture
rate was higher than fixed-K+QDR. The result is too small for a statistical
claim, and the same pilot shows that QDR itself is currently harmful relative
to the no-QDR reference. Therefore the three mechanisms are not yet a
validated combined contribution.

The main failure mode to address is QDR's treatment of delayed execution: the
current adapter aligns the delayed state and target time index, but does not
yet make the immutable prefix a separately evaluated recovery/safety branch.
The next validation must add prefix clearance and endpoint-error diagnostics,
then test delay/noise axes with paired scenes. UAKR thresholds must be frozen
on validation data before any confirmation run. RNIC is not included here
because its smoke pilot added planner latency without changing behavior.

## Artifacts

Each run contains frozen-scene `config.yaml`, `scenes.jsonl`, episode/step
JSONL, summary JSON, source hashes, and TensorBoard events:

- `results/phase17_ood_geometry_diagonal_fixedk20_noqdr/distributed_delayed/tensorboard/`
- `results/phase17_ood_geometry_diagonal_fixedk20/distributed_delayed/tensorboard/`
- `results/phase17_ood_geometry_diagonal_uakr20/distributed_delayed/tensorboard/`

The Phase 16 source manifest is
`47c393d1d678f25aad14c978734aae5d89105d826ea3e7b32b16a46cc1654501`.
Generated results remain Git-ignored; the evaluator and configuration are
versioned.
