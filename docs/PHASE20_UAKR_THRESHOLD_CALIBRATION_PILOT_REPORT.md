# Phase 20 UAKR Threshold-Calibration Pilot

Status: **No-Go for promotion; retain as a negative ablation.**

## Question

The Phase 17 UAKR policy saved computation, but its non-inferiority gate was
not passed.  A reliability audit showed that episodes with larger mean
uncertainty were more likely to fail, so this pilot asks whether lowering the
bucket thresholds can allocate more high-budget predictions to difficult
episodes.

This is a development-only validation pilot.  The threshold was not selected
from locked-test data, and the result is not a formal three-seed confirmation.

## Protocol

- frozen validation scenes: first 20 records of the Phase 16 validation
  manifest;
- predictor: Diagonal-SSM checkpoint, seed `727201`;
- planner: distributed delayed DN-MPC;
- safety: local CBF;
- queue-aware rollout: disabled;
- RNIC: disabled;
- prediction sampling: `K_max=8`, 8 diffusion steps, fixed seed `745102`;
- baseline adaptive policy: `low=0.35`, `high=0.65`;
- pilot policy: `low=0.30`, `high=0.55`.

The new evaluator switches are `--adaptive-low-threshold` and
`--adaptive-high-threshold`.  They are explicit and serialized into the
effective configuration for reproducibility.

## Result

| Variant | Safe capture | Collision | Boundary | Timeout | Mean K | Refresh ratio | Total p50 / p95 / p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RNIC-off pilot reference | 95.0% | 5.0% | 0.0% | 0.0% | — | 100.0% | 36.58 / 51.43 / 59.87 |
| UAKR, `0.35/0.65` (historical 3-seed result) | 94.07% aggregate | — | — | — | 2.68 | 36.6% | 59.83 / 132.29 / 141.36* |
| UAKR, `0.30/0.55` pilot | 90.0% | 10.0% | 0.0% | 0.0% | 3.19 | 42.0% | 70.64 / 138.04 / 170.73 |

\* The historical row is the Phase 17 three-seed aggregate and is included
for context; it is not the same 20-episode reference protocol.

The lower-threshold policy changed one of the 20 paired outcomes from a
successful safe capture to a safety failure.  It increased average candidate
budget and did not improve the observed capture result.

## Decision

Lowering thresholds alone is rejected.  The current UAKR score has some
episode-level ranking signal, but it is not calibrated enough to justify a
larger budget on every moderately uncertain step.  Do not tune additional
thresholds on the locked-test.

The next valid UAKR experiment must use a fresh development calibration split
and an independently defined outcome label (for example, prediction tube
miss, timeout, or safety-margin violation).  It must evaluate reliability
before changing the online budget policy.  Until that succeeds, UAKR remains
a reproducible compute-efficiency ablation and Phase 16 remains the primary
model.

## Local artifact

```text
results/phase19_uakr_calibrated_pilot_seed727201_l030_h055/
```

The ignored artifact contains the effective configuration, source hashes,
episode/step JSONL, summary JSON, and TensorBoard events.
