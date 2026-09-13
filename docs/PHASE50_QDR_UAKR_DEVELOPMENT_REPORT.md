# Phase 50 QDR × UAKR Development Pilot

> Status: validation-development pilot. This result is not a locked-test
> result and was not used to tune thresholds or planner costs.

## 1. Objective and protocol

Phase 48 established a fresh 100-episode QDR liveness confirmation. Phase 50
tests whether the existing UAKR adaptive budget can reduce QDR computation on
the first 40 episodes of that fresh manifest without losing the fixed-K=8
reference behavior.

The comparison uses the same GRU `both` checkpoint seed `727201`, same 40
scenes, same sampling seed, same local CBF, same delay-4 bounded-noise
immutable execution contract, and QDR enabled in both arms. Only the prediction
budget policy changes:

| Arm | Candidate budget |
| --- | --- |
| Fixed-K=8 | eight candidates at every refresh, UAKR disabled |
| QDR+UAKR | existing frozen UAKR buckets `K={1,4,8}`, refresh `{4,2,1}` |

The evaluated 40-episode prefix has scene hash
`e772a306a3b4eea7c4ba09068e9fa3def3cc3bf6cec1222b7eb507132a758b53`; it is a
prefix of the parent Phase 48 manifest
`f69e612341aa094250ac9e4628dd103cde0b87255ccb0052a680743dceee8928`.
The parent manifest and all source/config hashes are retained in each run.

## 2. Closed-loop outcome

| Arm | Safe capture | Collision | Boundary | Timeout | Mean capture time (s) | Mean minimum clearance (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed-K=8 | 100.0% | 0.0% | 0.0% | 0.0% | 6.273 | 0.513 |
| QDR+UAKR | 97.5% | 0.0% | 0.0% | 2.5% | 7.392 | 0.495 |

The paired UAKR minus fixed-K=8 differences are:

```text
safe capture:       -2.50 pp [-7.50, 0.00]
collision:            0.00 pp [0.00, 0.00]
boundary:             0.00 pp [0.00, 0.00]
timeout:             +2.50 pp [0.00, 7.50]
capture time:        +1.074 s [-0.593, 2.675]
minimum clearance:   -0.018 m [-0.051, 0.015]
```

This is a one-seed, 40-episode development result; the bootstrap interval is
episode-paired and must not be interpreted as a three-seed confirmation
interval. Nevertheless, the point estimate crosses the pre-registered
`-2 pp` safe-capture non-inferiority boundary in the wrong direction, while
UAKR introduces a timeout that is absent from the fixed-K reference.

## 3. Budget and mechanism diagnostics

| Diagnostic | Fixed-K=8 | QDR+UAKR |
| --- | ---: | ---: |
| Mean K | 8 (configured) | 2.156 |
| Prediction refresh ratio | 100.0% | 30.54% |
| Mean refresh interval (steps) | 1.00 | 3.232 |
| Forced refresh rate | 0.0% | 21.19% |
| Mean uncertainty score | not applicable | 0.308 |
| Prefix admissibility | 94.66% | 94.28% |
| Suffix admissibility | 80.50% | 78.34% |
| Suffix-gate exhaustion | 23.49% | 19.02% |

UAKR reduces candidate budget and suffix-gate exhaustion, but its cached
prediction age and lower refresh frequency do not preserve the fixed-K capture
outcome in this pilot. The result therefore separates computational saving from
closed-loop non-inferiority.

## 4. Latency

All values are computed from the retained 40-episode summaries; predictor,
planner, QDR, safety, and total percentiles are reported as p50/p95/p99 in ms.

| Arm | Predictor | Planner | QDR | Safety | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed-K=8 | 10.74 / 21.83 / 23.42 | 24.35 / 39.70 / 47.14 | 1.11 / 2.18 / 2.49 | 1.17 / 2.58 / 2.92 | 43.61 / 64.28 / 72.99 |
| QDR+UAKR | 0.00 / 18.11 / 22.54 | 24.21 / 39.43 / 46.37 | 1.07 / 2.15 / 2.51 | 1.17 / 2.52 / 2.85 | 35.93 / 57.15 / 68.17 |

The zero predictor p50 for UAKR reflects cache hits, not zero-cost prediction
inference. UAKR reduces total p95 by about 11.1% in this pilot, but the
underlying evaluator runs are not a multi-seed or process-isolated deployment
benchmark. The 100 ms value remains an engineering reference only.

## 5. TensorBoard and artifacts

Both evaluator runs preserve effective configuration, source hashes, scene
hashes, episode/step JSONL, and TensorBoard events:

```text
results/phase50_qdr_fixedk8_seed727201/distributed_delayed/tensorboard
results/phase50_qdr_uakr_seed727201/distributed_delayed/tensorboard
results/phase50_qdr_uakr_aggregate.json
```

The evaluator records UAKR bucket counts, mean K, refresh interval, forced
refresh, cache age, QDR queue/gate diagnostics, and all five latency groups.

## 6. Decision

**QDR × UAKR promotion: No-Go for the current policy.** The pilot provides a
reproducible efficiency signal but fails the pre-registered safe-capture
non-inferiority direction and adds timeout. Freeze it as a negative/efficiency
ablation; do not scan thresholds on these scenes and do not reopen the full
QDR × UAKR × RNIC matrix.

QDR fixed-K=8 remains the main QDR branch under the Phase 48 contract. UAKR can
remain in the paper as an auditable budget diagnostic, not as a claimed
closed-loop performance improvement. A future UAKR study would need a new
intervention-effect calibration and an independent three-seed confirmation.

## 7. Next TodoList

- [x] Run fixed-K=8 and QDR+UAKR on the same 40-episode development prefix.
- [x] Record safe capture, timeout, K, refresh, QDR gate, and latency metrics.
- [x] Preserve both TensorBoard runs and the paired aggregate.
- [ ] Keep fixed-K=8 QDR as the main branch for the next independent transfer
  test.
- [ ] Only revisit UAKR after independently calibrating the intervention effect;
  do not tune the current thresholds on this pilot.
- [ ] Keep RNIC disabled and delay the Full combination until every selected
  module passes paired safety/liveness and runtime gates.
