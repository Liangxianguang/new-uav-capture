# Phase 12 Execution-Safety Progress Report

## Scope

This phase validates the continuous-segment geometry optimization and records
the next evidence needed for the queue-aware execution controller. It does not
claim closed-loop invariance, real-flight safety, or a completed high-capture
controller.

## Code changes

- `src/encirclement3d/safety_certificate.py` now evaluates each continuous
  action segment endpoint once and reuses shared endpoint geometry between
  adjacent subsegments.
- The independent certificate path skips position gradients when no Jacobian
  is requested.
- `scripts/evaluate_safety_execution.py` accepts `--seed-start` and
  `--seed-count` for reproducible consecutive unseen-seed blocks.
- Contract tests cover barrier values, robust margins, gradient ordering, and
  the holdout seed resolver.

The changes were committed as `de46be9` and `0dcf599` and pushed to
`origin/main`.

## Verification

The full repository test suite passed:

```text
145 passed in 645.32s
```

The isolated continuous geometry benchmark on one hard observation measured:

| Path | p50 (ms) | p95 (ms) | p99 (ms) |
| --- | ---: | ---: | ---: |
| continuous barrier values | 4.676 | 5.060 | 5.077 |
| analytic Jacobian | 10.051 | 10.374 | 10.489 |
| independent continuous certificate | 5.008 | 5.337 | 5.349 |

These are helper timings, not end-to-end filter latency.

## Short hard-execution smoke

The following runs used two locked seeds and a 20-step cap. They are
regression smoke tests only.

| Variant | Continuous certificate | Actual post-state safety | Safe capture | Collision | Boundary | p50 / p95 / p99 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hard replace, barrier recovery | 57.5% | 100% | 0/2 | 0% | 0% | 1,094 / 4,309 / 4,581 ms |
| hard flush, barrier recovery | 0% | 100% | 0/2 | 0% | 0% | 3,256 / 5,035 / 5,401 ms |
| hard flush, progress QP, horizon 3 | 92.5% | 100% | 0/2 | 0% | 0% | 213 / 1,228 / 1,346 ms |

The progress-QP row is promising as a recovery candidate, but it has no
episode-level capture evidence and remains below the required 95--99%
continuous-certificate target. The gap between actual post-state safety and
continuous safety remains unresolved.

## Reachable-tube holdout

The larger configured-nominal audit used 2,048 calibration and 4,096 holdout
samples per variant and horizon. For hard randomized execution, the runtime
coverage was 99.707%, 99.316%, 99.365%, and 99.512% for horizons 1, 2, 3,
and 5, with fitted runtime multipliers approximately 2.059, 2.058, 2.059,
and 2.061.

For hard flush pending, runtime coverage was 98.901% at horizon 1, 99.536%
at horizon 2, 99.438% at horizon 3, and 99.512% at horizon 5. The horizon-1
row fails the 99% point-estimate gate. The locked runtime multiplier 2.1 is
therefore retained; no queue-aware gain has been frozen for deployment.

Evidence:

- `results/phase12_reachable_tube_horizon_holdout_v1/summary.json`
- `results/phase12_continuous_geometry_cache_smoke_v1/summary.json`
- `results/phase12_progress_qp_geometry_cache_smoke_v1/summary.json`

## Next gates

1. Run the full eight-seed authority matrix with the optimized code and keep
   immutable, replace, and flush as separate executor assumptions.
2. Run the progress-QP variant on the full locked block before considering it
   a recovery candidate.
3. Use the new seed-block interface for at least 30--50 unseen closed-loop
   seeds after the recovery policy is frozen.
4. Report authority capability explicitly; flush pending is not a real-system
   conclusion unless the actuator queue supports cancellation.
5. Only after these gates pass should Mamba, diffusion prediction, or DN-MPC
   replace the nominal planner.

