# Phase 17 UAKR Validation Report

## Scope

This report evaluates Uncertainty-Triggered Adaptive K and Replanning (UAKR)
on the frozen 90-episode validation manifest used for configuration selection.
The comparison is paired by episode and predictor seed:

- predictor: Diagonal SSM conditional diffusion checkpoint;
- controller: distributed delayed DN-MPC + local CBF;
- execution delay: 2 steps, command noise: 0;
- fixed reference: K=8, refresh interval=1;
- adaptive policy: K in {1, 4, 8}, refresh interval in {4, 2, 1};
- three predictor seeds: 727201, 727202, 727203;
- sampling seed, scene manifest, projection iterations and horizon are fixed.

The validation manifest SHA-256 is
`5ba54afc62d2cf123014f1c5e598a0e3e914e9067870bc7a199ba3393d8f8d99`.
The raw JSONL and TensorBoard artifacts remain under the ignored `results/`
directory. The aggregate artifact is
`results/phase17_validation_uakr_aggregate.json`.

## Per-seed results

| Seed | Policy | Safe capture | Capture | Collision | Boundary | Mean K | Refresh ratio | Forced refresh | Total p50/p95/p99 (ms) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 727201 | fixed K=8 | 94.44% | 94.44% | 5.56% | 0.00% | 8.000 | 100.00% | 0.00% | 111.25/135.78/142.80 |
| 727201 | adaptive | 93.33% | 94.44% | 6.67% | 0.00% | 2.683 | 36.71% | 19.53% | 59.73/132.06/141.40 |
| 727202 | fixed K=8 | 93.33% | 93.33% | 6.67% | 0.00% | 8.000 | 100.00% | 0.00% | 111.53/135.67/143.68 |
| 727202 | adaptive | 94.44% | 95.56% | 5.56% | 1.11% | 2.688 | 36.54% | 19.44% | 60.16/133.39/142.73 |
| 727203 | fixed K=8 | 94.44% | 94.44% | 5.56% | 0.00% | 8.000 | 100.00% | 0.00% | 110.96/135.63/141.96 |
| 727203 | adaptive | 94.44% | 95.56% | 5.56% | 0.00% | 2.682 | 36.52% | 19.49% | 59.49/130.76/139.56 |

## Aggregate result and gate

Across the three matched seeds, safe capture is 94.07% for both policies.
The hierarchical paired bootstrap gives an adaptive-minus-fixed safe-capture
delta of `0.00 pp`, 95% CI `[-2.59, +2.96] pp`. The pre-registered
non-inferiority margin is `-2.0 pp`, so the lower confidence bound does not
pass the gate. Collision delta is `0.00 pp`, CI `[-2.59, +2.59] pp`; boundary
violation has a small positive point delta of `+0.37 pp`, CI `[0.00, +1.48] pp`.

The compute effect is real on this workload:

- mean K is approximately 2.68, a 66.5% reduction from K=8;
- refresh ratio is approximately 36.6%, a 63.4% reduction;
- predictor latency p50/p95/p99 is 0.00/55.85/58.89 ms because cached steps
  do not invoke the predictor;
- total-control latency p50/p95/p99 is 59.83/132.29/141.36 ms versus
  111.25/135.72/142.92 ms for fixed K=8.

**Decision: No-Go for promotion as a closed-loop performance contribution.**
UAKR is currently a reproducible compute-budget scheduler with no validated
ID safety/capture improvement. The next experiment must improve uncertainty
calibration or use the scheduler only as a secondary efficiency ablation;
thresholds must not be tuned on locked-test.

## Missing evidence before reconsideration

1. Reliability analysis: high-uncertainty bins must have a higher realized
   prediction/closed-loop failure rate than low-uncertainty bins.
2. A controlled single-process benchmark with fixed thread count must separate
   cache savings from predictor, planner, safety and total p50/p95/p99.
3. The policy must be evaluated on target-speed and delay/execution stress,
   with the same paired seeds, before any claim of robustness.
4. The adaptive schedule must log the exact feature components and replay them
   independently from step logs.
