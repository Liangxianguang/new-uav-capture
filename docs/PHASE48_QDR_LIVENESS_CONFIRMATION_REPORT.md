# Phase 48 QDR Liveness Confirmation Report

> Status: frozen validation-confirmation result. The manifest was created
> before evaluation, and no locked-test data or locked-test tuning was used.

## 1. Objective

Phase 47 showed that Queue-Aware Delayed-State Rollout (QDR) substantially
reduces safety failures under an OOD communication/execution block, but it
also exposed a potential liveness cost. Phase 48 tests that cost on a new
validation-confirmation manifest with nominal sensing/communication and a
fixed delayed execution contract.

The pre-registered liveness gate is:

```text
QDR-on timeout rate                    <= 5%
timeout delta versus paired QDR-off   <= 5 percentage points
maximum episode exhaustion streak      <= 24 control steps
```

The last limit equals three times the frozen eight-step prediction/planning
horizon and is checked on the maximum across the three predictor seeds, not on
an average alone.

## 2. Reproducible protocol

```text
manifest: results/phase48_qdr_liveness_confirmation_scenes/scenes.jsonl
episodes: 100 per arm and predictor seed; 50 mirror groups
training seeds: 727201, 727202, 727203
manifest SHA-256: f69e612341aa094250ac9e4628dd103cde0b87255ccb0052a680743dceee8928
predictor: frozen GRU both checkpoints, checkpoint source
candidate samples / sampling steps: 1 / 8
prediction refresh: every control step
safety layer: local CBF
split: validation_confirmation
bootstrap: 10,000 hierarchical samples, seed 20260913
```

Both arms use the same frozen scene indices, action conditions, checkpoint
seeds, sampling seed, and planner. The scene contract keeps the nominal Phase
15 sensing/communication support and explicitly sets:

- command delay: 4 steps;
- command noise: standard deviation `0.08`, clipped at `3 sigma`;
- pending-command authority: `immutable`;
- tracking time constant and drag: `0` in this confirmation block.

The candidate arm enables `queue_aware_rollout` and
`queue_aware_safety_projection`; the reference arm disables both. No target
truth is supplied to online prediction.

## 3. Closed-loop outcome

| Arm | Safe capture | Ordinary capture | Collision | Boundary | Timeout | Mean capture time (s) | Mean minimum clearance (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QDR-off | 83.33% [79.00%, 87.33%] | 83.33% [79.00%, 87.33%] | 16.67% [12.33%, 21.00%] | 2.67% [1.00%, 4.67%] | 0.00% | 2.594 [2.453, 2.756] | 0.266 [0.244, 0.289] |
| QDR-on | **98.33% [95.67%, 100.00%]** | 98.33% [95.67%, 100.00%] | **1.33% [0.00%, 3.33%]** | 1.33% [0.00%, 3.67%] | **0.33% [0.00%, 1.33%]** | 5.998 [5.304, 6.673] | 0.514 [0.502, 0.525] |

Paired QDR-on minus QDR-off differences are:

```text
safe capture:       +15.00 pp [10.33, 19.67]
collision:          -15.33 pp [-20.00, -11.00]
boundary:            -1.33 pp [-4.00, 1.33]
timeout:             +0.33 pp [0.00, 1.33]
capture time:        +3.262 s [2.562, 3.937]
minimum clearance:   +0.247 m [0.224, 0.270]
```

Per-seed safe capture for QDR-on is `100%`, `96%`, and `99%` for seeds
`727201`, `727202`, and `727203`; the corresponding QDR-off values are
`84%`, `83%`, and `83%`. Thus the safety effect is consistent across the
frozen predictor seeds, while the boundary interval alone does not establish
a statistically significant change.

## 4. Liveness gate

The gate was evaluated by
`scripts/evaluate_qdr_liveness_gate.py` using the matched aggregate and all
three QDR-on summaries:

| Check | Observed | Limit | Decision |
| --- | ---: | ---: | --- |
| QDR-on timeout rate | 0.33% | 5.00% | PASS |
| Timeout delta vs QDR-off | 0.33 pp | 5.00 pp | PASS |
| Maximum exhaustion streak | 23 steps | 24 steps | PASS |

The formal machine-readable gate result is
`results/phase48_qdr_liveness_gate.json`. The matched aggregate is
`results/phase48_qdr_liveness_aggregate.json`.

## 5. QDR mechanism and TensorBoard evidence

Across the three QDR-on seed summaries:

| Diagnostic | Result |
| --- | ---: |
| Mean/max QDR queue length | 4 / 4 |
| First controllable step | 4 |
| Prefix admissibility | 95.36% |
| Suffix admissibility | 84.44% |
| Step-level suffix-gate exhaustion | 20.89% |
| Episode with at least one exhaustion | 99.00% |
| Mean first exhaustion step | 13.52 |
| Maximum observed exhaustion streak | 23 steps |
| Mean recovery count | 5.74 |

“Ever exhausted” remains a process diagnostic rather than a failure label:
almost every episode encounters exhaustion, yet the maximum streak remains
bounded and the final safe-capture rate is high.

QDR-on pooled step-level latency is:

| Component | p50 (ms) | p95 (ms) | p99 (ms) | Samples |
| --- | ---: | ---: | ---: | ---: |
| Predictor | 24.30 | 30.06 | 36.37 | 18,415 |
| Planner | 46.66 | 61.74 | 71.25 | 18,415 |
| QDR diagnostic* | 2.11 | 2.94 | 3.75 | 3 summaries |
| Safety filter | 2.52 | 3.46 | 4.46 | 18,415 |
| Total control | 84.60 | 103.99 | 120.67 | 18,415 |

`*` QDR percentiles are the arithmetic mean of the three per-seed summary
percentiles; the aggregate artifact pools the other step-level components.
The QDR-off total is `30.81/59.88/69.16 ms`, but this is not interpreted as a
causal efficiency comparison because the two arms have different episode
outcomes and lengths. The 100 ms value remains a reference, not a hard gate.

Every evaluator run contains effective configuration, source hashes, scene
hash, episode/step JSONL, and TensorBoard events. The gate-specific TensorBoard
run is:

```text
results/phase48_qdr_liveness_gate_tb
```

It contains observed, limit, and pass scalars for all three checks plus the
overall gate result. Evaluator TensorBoard runs are under each
`distributed_delayed/tensorboard` directory in the six arm/seed result
directories.

## 6. Decision

**QDR liveness confirmation: Go for the frozen delay-4 confirmation contract.**
The result supports a reproducible QDR safety/liveness improvement on this
fresh ID confirmation block: safe capture rises by 15 percentage points while
the pre-registered timeout and exhaustion-streak limits pass.

This is still not a universal guarantee. Total p95/p99 are above the 100 ms
engineering reference, suffix admissibility is only 84.44%, and the local CBF
is not a delayed multi-step safety proof. QDR should therefore be promoted only
within the documented execution contract; robust CBF-QP remains diagnostic
No-Go and the full QDR × UAKR × RNIC combination remains unopened.

## 7. Next TodoList

- [x] Freeze a fresh 100-episode/50-mirror-group confirmation manifest.
- [x] Complete QDR-off/on evaluation for three predictor seeds.
- [x] Aggregate paired outcomes with hierarchical bootstrap.
- [x] Implement and execute the timeout/exhaustion-streak liveness gate.
- [x] Log gate, queue, exhaustion, and latency metrics to TensorBoard.
- [ ] Run a controlled same-episode-length runtime benchmark before claiming
  any QDR efficiency improvement.
- [ ] Test QDR × UAKR on a new development block, with UAKR thresholds frozen
  independently and RNIC disabled.
- [ ] Reopen the full combination only after QDR × UAKR passes paired safety,
  liveness, and latency gates.
- [ ] Keep robust CBF-QP separate until a delayed execution-aware multi-step
  certificate is implemented and independently audited.
