# Phase 47 OOD Delay/Execution QDR Closed-Loop Report

> Status: frozen OOD diagnostic. This result is not used for model selection,
> hyper-parameter tuning, or locked-test promotion.

## 1. Question and scope

Phase 16 showed that the current distributed controller failed under combined
sensing, communication, and execution stress. Phase 47 asks one narrow causal
question: with the predictor, planner weights, local CBF, sampling seed, and
frozen scenes held fixed, does Queue-Aware Delayed-State Rollout (QDR) reduce
the failure rate on this stress axis?

The comparison is descriptive/paired at the episode level. It does not prove
robust safety, because the local CBF remains a nominal filter and the plant
contains delayed, noisy execution.

## 2. Reproducible protocol

The same frozen manifest is used for QDR-off and QDR-on:

```text
scenes: results/phase16_ood_delay_execution_shift/scenes.jsonl
episodes: 100 per predictor seed, 50 upper/lower mirror groups
training seeds: 727201, 727202, 727203
manifest SHA-256: 6478d0d51053b0eb0e9d5fbfbbce9c39e6be2c365c0ad28d2b5abce903f132aa
predictor: frozen GRU both checkpoint, checkpoint source
candidate samples / diffusion steps: 1 / 8
prediction refresh: every control step
safety layer: local CBF
evaluation split: ood_diagnostic
```

The manifest contains two sensing/communication conditions:

| Condition | Detection dropout | Observation noise | Message delay | Message dropout |
| --- | ---: | ---: | ---: | ---: |
| `delay6_dropout20` | 0.35 | 0.08 | 6 steps | 0.20 |
| `delay8_dropout25` | 0.40 | 0.10 | 8 steps | 0.25 |

Both conditions also activate the frozen execution contract: 2-step command
delay, command-noise standard deviation `0.08` clipped at `3 sigma`, first-order
velocity tracking with time constant `0.4 s`, drag range `[0.05, 0.20]`,
per-episode speed scale `[0.85, 0.98]`, acceleration scale `[0.70, 0.95]`, and
immutable pending-command authority. The scene-level execution overrides are
authoritative; the QDR queue therefore reports a mean and maximum queue length
of 2 and a first controllable step of 2 in this block.

QDR-on was executed with `--queue-aware-rollout
--queue-aware-safety-projection`; QDR-off was the previously frozen paired
run with both switches disabled. Neither arm uses target truth online.
Intervals use 10,000-sample hierarchical bootstrap over matched training seeds
and episode indices (seed `20260913`).

## 3. Closed-loop results

| Arm | Safe capture | Ordinary capture | Collision | Boundary | Timeout | Mean capture time (s) | Mean minimum clearance (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QDR-off | 47.00% [41.00%, 53.00%] | 47.00% [41.00%, 53.00%] | 53.00% [47.33%, 59.00%] | 19.00% [14.33%, 24.00%] | 0.00% | 3.874 [3.520, 4.265] | 0.136 [0.112, 0.159] |
| QDR-on | **93.67% [90.33%, 96.67%]** | 93.67% [90.33%, 96.67%] | **2.33% [0.00%, 5.33%]** | 2.33% [0.00%, 5.33%] | 4.00% [1.67%, 6.67%] | 5.681 [4.939, 6.450] | 0.522 [0.510, 0.535] |

The paired QDR-on minus QDR-off changes are:

```text
safe capture:       +46.67 pp [40.00, 53.00]
collision:          -50.67 pp [-57.00, -44.00]
boundary:           -16.67 pp (descriptive, paired CI not emitted by aggregator)
timeout:              +4.00 pp [1.67, 6.67]
capture time:        +1.562 s [0.476, 2.688]
minimum clearance:   +0.387 m [0.362, 0.411]
```

The safety improvement is large and consistent across the three frozen
predictor seeds:

| Seed | Safe capture | Collision | Boundary | Timeout | Mean minimum clearance (m) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 727201 | 95% | 0% | 0% | 5% | 0.522 |
| 727202 | 94% | 2% | 2% | 4% | 0.512 |
| 727203 | 92% | 5% | 5% | 3% | 0.533 |

## 4. Latency and mechanism diagnostics

The pooled step-level latency for QDR-on is:

| Component | p50 (ms) | p95 (ms) | p99 (ms) | Samples |
| --- | ---: | ---: | ---: | ---: |
| Predictor | 13.24 | 27.59 | 32.32 | 19,462 |
| Planner | 30.66 | 58.23 | 67.26 | 19,462 |
| QDR diagnostic | 1.03* | 1.68* | 2.18* | 3 summaries |
| Safety filter | 1.48 | 3.13 | 3.93 | 19,462 |
| Total control | 50.35 | 97.74 | 111.21 | 19,462 |

`*` QDR percentiles are the arithmetic mean of the three per-seed summary
percentiles; the aggregate script currently pools predictor/planner/safety/
total step samples but retains QDR component percentiles per run. They are
reported to make the QDR overhead visible, not as a pooled percentile.

For reference, the paired QDR-off total-control percentiles are
`86.09/122.82/178.46 ms`. This difference is not an efficiency claim: QDR-on
and QDR-off terminate with very different outcome mixtures and therefore have
different episode-length distributions. The 100 ms value remains an engineering
reference, not a hard acceptance gate.

QDR-on mechanism statistics, averaged over the three seed summaries, are:

| Diagnostic | Result |
| --- | ---: |
| Prefix admissibility | 95.84% |
| Suffix admissibility | 86.14% |
| Episode with at least one suffix-gate exhaustion | 96.67% |
| Step-level suffix-gate exhaustion | 19.26% |
| First exhaustion step | 15.95 |
| Maximum exhaustion streak | 20.67 steps |
| Recovery count per episode | 6.38 |
| Mean QDR queue length / max queue length | 2 / 2 |

The high episode-level “ever exhausted” rate together with repeated recovery
shows why exhaustion must not be used as a binary failure label. QDR is useful
here because it plans from the first controllable delayed state and evaluates
the immutable prefix, but the suffix gate still frequently becomes empty and
relies on conservative candidate behavior.

## 5. Interpretation

This is the strongest current evidence for QDR on the execution-stress axis:
it reduces collision and boundary failures by a large paired margin and raises
minimum clearance. The trade-off is clear: safe capture is not equal to
ordinary capture because QDR introduces 4% timeout, and successful captures
take longer on average. Therefore the result supports the claim
“queue-aware delayed-state planning substantially improves safety under this
frozen stress block, with a liveness cost,” not “QDR is universally better” or
“the closed loop is formally safe.”

The block remains conditional for three reasons:

1. It evaluates one OOD stress family and one controller/predictor family.
2. The physical safety contract is still not certified for arbitrary queue,
   tracking, and disturbance realizations.
3. QDR-on total p99 is `111.21 ms`, and the high gate-exhaustion rate indicates
   unresolved feasibility/liveness behavior.

## 6. Decision

**Conditional Go for QDR safety-axis evidence; No-Go for unconditional
promotion.** Keep QDR enabled for the documented stress-diagnostic branch and
retain the off/on paired artifacts. Do not tune on this OOD block, do not
change the locked-test result, and do not yet reopen the full QDR × UAKR × RNIC
matrix.

## 7. Next TodoList

- [x] Complete the three-seed QDR-on evaluation with 300 episodes and retain
  per-episode, per-step, summary, config, hash, and TensorBoard artifacts.
- [x] Perform matched QDR-off/on aggregation with hierarchical bootstrap.
- [x] Report predictor/planner/QDR/safety/total p50/p95/p99 and QDR gate
  persistence metrics.
- [ ] Run the remaining OOD axes separately: target-speed/behavior transfer
  and geometry transfer; no mixed-axis tuning.
- [ ] Add a pre-registered liveness gate: timeout non-inferiority and bounded
  exhaustion streak, evaluated on a new confirmation manifest.
- [ ] Benchmark QDR overhead in a controlled same-episode-length runtime
  harness before making any efficiency claim.
- [ ] Only if the new liveness confirmation passes, test QDR × UAKR on a fresh
  development block; RNIC remains disabled until its independent distributed
  latency issue is repaired.
- [ ] Keep robust CBF-QP as a diagnostic No-Go until a delayed execution-aware
  multi-step certificate is actually implemented and audited.
