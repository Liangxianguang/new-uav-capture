# Phase 49 Matched-Length QDR Runtime Benchmark

> Status: frozen runtime diagnostic. This benchmark does not alter any model,
> planner weight, safety margin, or locked-test result.

## 1. Question

The Phase 48 QDR-on and QDR-off arms have different episode outcomes and
therefore different episode lengths. A raw total-latency comparison could
confound computation with termination distribution. Phase 49 asks whether the
QDR computation remains more expensive when both arms are restricted to the
same matched control prefix.

## 2. Controlled selection protocol

The benchmark reads the retained Phase 48 `steps.jsonl` artifacts. For each of
the three matched predictor seeds, it intersects episode indices and retains
the first 18 control steps only when both arms contain all 18 steps. All 100
episodes per seed satisfy this condition, giving 300 matched episodes and
5,400 steps per arm.

```text
off: results/phase48_qdr_liveness_off_seed{727201,727202,727203}
on:  results/phase48_qdr_liveness_on_seed{727201,727202,727203}
fixed prefix: 18 control steps / episode
latency percentiles: p50 / p95 / p99
```

The implementation is `scripts/benchmark_qdr_same_length.py`, with its
protocol recorded in `configs/phase49_qdr_runtime_benchmark.yaml`. It does not
rerun or modify the frozen episodes.

## 3. Results

| Arm | Predictor p50/p95/p99 (ms) | Planner p50/p95/p99 (ms) | QDR p50/p95/p99 (ms) | Safety p50/p95/p99 (ms) | Total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| QDR-off | 13.71 / 28.01 / 32.79 | 13.52 / 25.34 / 30.58 | 0 / 0 / 0 | 1.62 / 3.18 / 4.09 | 30.59 / 59.51 / 68.32 |
| QDR-on | 24.27 / 29.75 / 35.41 | 46.98 / 61.26 / 68.89 | 2.11 / 2.90 / 3.67 | 2.50 / 3.39 / 4.43 | 84.81 / 103.14 / 117.92 |
| On minus off | +10.56 / +1.75 / +2.62 | +33.46 / +35.92 / +38.32 | +2.11 / +2.90 / +3.67 | +0.88 / +0.21 / +0.35 | +54.21 / +43.63 / +49.60 |

Each arm contains exactly 5,400 selected step rows. The dominant additional
cost is the QDR planner path, not the QDR diagnostic itself: the QDR component
is about 2--4 ms at the reported percentiles, while planner p95 increases by
35.92 ms.

## 4. Interpretation and limitation

The matched-prefix result confirms that QDR's extra total latency is not solely
caused by QDR-on episodes running longer before capture. It is a genuine
per-control-step computation cost in this implementation, particularly in
candidate execution-dynamics rollout and suffix feasibility evaluation.

The benchmark is not a process-isolated deployment benchmark: Phase 48 logs
were produced by separate evaluator runs, and OS/BLAS scheduling can affect
absolute percentiles. Consequently, the result supports a negative runtime
ablation and identifies the planner path to optimize, but it does not claim a
hardware-independent overhead bound. A later single-process, fixed-thread
microbenchmark is still required for a deployment claim.

The 100 ms value remains an engineering reference only. QDR-on total p95 is
`103.14 ms` and p99 is `117.92 ms`; neither is an automatic rejection under the
project protocol.

## 5. TensorBoard artifact

The benchmark writes all arm/component percentiles and on-minus-off deltas to:

```text
results/phase49_qdr_same_length_runtime_benchmark_tb
```

The original evaluator runs retain their full configuration, source/manifest
hashes, per-episode and per-step JSONL, and TensorBoard events under each
`phase48_qdr_liveness_{off,on}_seed*/distributed_delayed/tensorboard` path.

## 6. Decision

**QDR efficiency promotion: No-Go.** Keep QDR as the safety/liveness method
under the Phase 48 execution contract, but do not claim that the current
implementation is computationally cheaper or deployment-ready. The next
optimization target is the candidate rollout/suffix gate, and any optimization
must preserve selected action semantics and pass the same safety/liveness
confirmation gates.

## 7. Next TodoList

- [x] Control episode termination by comparing matched 18-step prefixes.
- [x] Report predictor/planner/QDR/safety/total p50/p95/p99.
- [x] Log the benchmark and deltas to TensorBoard.
- [ ] Add a single-process fixed-thread microbenchmark for deployment-level
  latency claims.
- [ ] Test QDR × UAKR on a fresh development block with UAKR independently
  frozen; keep RNIC disabled.
- [ ] Reopen Full QDR × UAKR × RNIC only after paired safety, liveness, and
  latency gates pass.
