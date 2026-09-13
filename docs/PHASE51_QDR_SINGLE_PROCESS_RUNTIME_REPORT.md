# Phase 51 Single-Process Fixed-Thread QDR Runtime Benchmark

> Status: validation-development runtime diagnostic. This experiment uses a
> fresh Phase 48 validation prefix; it is not a locked-test result, does not
> tune any threshold or planner weight, and does not establish a safety proof.

## 1. Objective and protocol

Phase 49 showed that QDR-on was slower on matched episode prefixes, but its
source evaluator runs were produced under separate process/CPU scheduling
conditions. Phase 51 repeats the comparison sequentially, with one evaluator
process at a time and fixed Torch intra-op/inter-op thread counts of `1/1`.
The purpose is to remove concurrent CPU-load variation from the runtime
comparison; it is not a hardware-independent deployment benchmark.

Both arms use the same GRU `both` checkpoint seed `727201`, the same first 40
episodes of the Phase 48 confirmation manifest, the same local CBF, delay-4
immutable execution contract, bounded command noise (`0.08 m/s`, clipped at
`3 sigma`), sampling seed, `K=8`, and refresh interval of one control step.
Only QDR is changed:

| Arm | QDR configuration |
| --- | --- |
| QDR-off | queue-aware rollout and delayed-state safety projection disabled |
| QDR-on | queue-aware rollout and delayed-state safety projection enabled |

The run configuration is `configs/phase51_qdr_single_process_runtime.yaml`.
The evaluator records the effective Torch thread counts in `config.yaml` and
TensorBoard. Each arm was launched as a separate fresh process, but the arms
were run sequentially and never concurrently.

## 2. Closed-loop outcome

| Arm | Safe capture | Collision | Boundary | Timeout | Mean capture time (s) | Mean minimum clearance (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QDR-off | 85.0% (34/40) | 15.0% | 2.5% | 0.0% | 2.529 | 0.254 |
| QDR-on | 100.0% (40/40) | 0.0% | 0.0% | 0.0% | 6.273 | 0.513 |

The episode-paired QDR-on minus QDR-off bootstrap differences are:

```text
safe capture:       +15.00 pp [+5.00,+27.50]
collision:          -15.00 pp [-27.50,-5.00]
boundary:            -2.50 pp [-7.50, 0.00]
timeout:              0.00 pp [ 0.00, 0.00]
capture time:        +3.406 s [+1.961,+5.123]
minimum clearance:   +0.260 m [+0.198,+0.325]
```

This is a one-seed, 40-episode paired development result. It confirms the
same qualitative safety/liveness direction as Phase 48 on this prefix, but it
is not a multi-seed statistical confirmation and should not replace the
Phase 48 three-seed result.

## 3. Matched-length runtime

To control termination-length effects, the benchmark retains the first 18
control steps of every episode for both arms: 40 matched episodes and 720
steps per arm. Values are p50/p95/p99 in milliseconds.

| Arm | Predictor | Planner | QDR | Safety | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| QDR-off | 5.55 / 6.64 / 7.93 | 5.64 / 7.04 / 8.32 | 0 / 0 / 0 | 0.62 / 0.81 / 1.04 | 12.77 / 15.15 / 16.90 |
| QDR-on | 5.52 / 7.44 / 8.64 | 12.27 / 15.70 / 17.53 | 0.63 / 0.84 / 0.95 | 0.62 / 0.82 / 1.05 | 21.22 / 26.67 / 29.39 |
| On minus off | -0.03 / +0.80 / +0.71 | +6.63 / +8.66 / +9.21 | +0.63 / +0.84 / +0.95 | +0.00 / +0.01 / +0.01 | +8.45 / +11.53 / +12.48 |

With fixed threads, the QDR-on total p95 remains far below the earlier
concurrent-load comparison, while the planner p95 still increases by about
`8.66 ms`. The QDR diagnostic itself is below `1 ms` at p95; the dominant
implementation cost is therefore the QDR-enabled planner/candidate path,
not the queue diagnostic bookkeeping. The project does not treat `100 ms` as
a hard gate; all component and total percentiles are retained for engineering
comparison.

## 4. QDR mechanism diagnostics

For QDR-on, the mean/max queue length and mean first controllable delayed
step are both `4.0`. Prefix admissibility is `94.66%`, suffix admissibility is
`80.50%`, and the maximum continuous suffix-gate exhaustion streak is
`23` steps. The suffix gate was exhausted at least once in `97.5%` of these
episodes, reinforcing the earlier conclusion that “ever exhausted” is a
diagnostic event rather than a direct episode-failure label. Timeout remained
`0%` in this prefix.

## 5. Artifacts and reproducibility

The generated local artifacts are:

```text
results/phase51_qdr_single_fixed_threads_off_seed727201/
results/phase51_qdr_single_fixed_threads_on_seed727201/
results/phase51_qdr_single_process_runtime_aggregate.json
results/phase51_qdr_single_process_runtime_benchmark.json
results/phase51_qdr_single_process_runtime_benchmark_tb/
```

Each evaluator run preserves `config.yaml`, `scenes.jsonl`, source hashes,
episode/step JSONL, and TensorBoard events. TensorBoard contains the fixed
thread tags `Evaluation/torch_num_threads` and
`Evaluation/torch_num_interop_threads`, plus predictor/planner/QDR/safety/
total p50/p95/p99 tags. The matched-length benchmark writes its arm values
and on-minus-off deltas to the benchmark TensorBoard directory.

## 6. Decision

**QDR safety-axis: retained conditionally. QDR efficiency promotion: No-Go.**

The fixed-thread experiment removes the main process-concurrency caveat from
Phase 49 and still shows a real, reproducible planner overhead. It does not
justify claiming that QDR is computationally cheaper or deployment-ready.
Keep fixed-K=8 QDR as the main safety/liveness branch, optimize the candidate
rollout/suffix-gate path only under action-semantic regression tests, and do
not reopen the full QDR × UAKR × RNIC combination. UAKR remains frozen as a
negative/efficiency ablation after Phase 50.

## 7. Next TodoList

- [x] Add fixed intra-op/inter-op Torch thread controls to the evaluator.
- [x] Run sequential QDR-off/on paired evaluation on the same 40-episode
  development prefix.
- [x] Compute the matched 18-step runtime comparison.
- [x] Verify configuration and TensorBoard thread/latency logging.
- [ ] Optimize QDR candidate rollout and suffix feasibility without changing
  selected actions or outcome semantics.
- [ ] Re-run a fresh three-seed confirmation only after an optimization passes
  unit, schema, paired-outcome, and latency-regression checks.
- [ ] Keep RNIC disabled and do not reopen Full until UAKR independently
  satisfies the pre-registered safety/liveness gate.
