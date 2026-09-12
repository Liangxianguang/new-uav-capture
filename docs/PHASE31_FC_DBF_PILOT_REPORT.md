# Phase 31 FC-DBF Pilot Report

## 1. Scope and decision

Phase 31 implements **Feasible-Consensus Distributed Barrier Formation
(FC-DBF)** as a reproducible, finite candidate-set formation gate.  The
module is designed for the distributed delayed DN-MPC setting and does not
change the predictor backbone, access target truth, or replace the local CBF
safety layer.

The current decision is **engineering pass, promotion No-Go**.  The source,
configuration, unit tests, per-step diagnostics, source hashes and
TensorBoard scalars are complete.  In a controlled one-seed, 20-scene
development smoke, FC-DBF did not change safe-capture outcomes, while adding
substantial distributed planning latency.  This is not sufficient evidence
for a main-paper claim or for reopening the locked test.

## 2. Reproducible protocol

The paired comparison uses the same 20-scene validation-selection prefix,
checkpoint seed `727201`, predictor checkpoint, target-sampling seed
`745102`, one candidate sample, eight sampling steps, four projection
iterations, CPU execution, local CBF, and both `worst_case` and
`distributed_delayed` methods.  The only planned difference is
`fc_dbf_enabled`.

| Arm | Configuration | Artifact |
| --- | --- | --- |
| FC-DBF off | `configs/phase31_fc_dbf_distance_baseline.yaml` | `results/phase31_fc_dbf_baseline_smoke_seed727201_refresh4/` |
| FC-DBF on | `configs/phase31_fc_dbf_pilot.yaml` | `results/phase31_fc_dbf_pilot_smoke_seed727201_refresh4/` |

The artifacts are intentionally retained locally under the Git-ignored
`results/` directory.  The reports and source/configuration files are the
versioned reproducibility record.

## 3. Method

At each planning step, FC-DBF constructs a finite set of target-relative
formation slots.  A local defender uses its own rollout and the latest
available delayed peer messages.  When the peer view is incomplete, the
formation gate is omitted rather than filled with simulator state.

For a complete peer view, the gate evaluates:

1. slot tracking error over a short gate horizon;
2. bounded arrival-time slack under the existing speed/acceleration surrogate;
3. slot-error progress from the first to the terminal gate step; and
4. assignment switching relative to the previous consensus assignment.

The local implementation uses a **consensus token**: one nominal local
candidate undergoes the exact finite assignment search, and its delayed
assignment path is reused to evaluate all local candidate rollouts.  This
avoids repeating the permutation search for every candidate while preserving
the delayed information boundary.  If at least one local candidate satisfies
the declared gate, infeasible candidates are masked and the feasible ones
receive a soft FC-DBF cost.  If all candidates fail, the base objective is
used and `gate_exhausted` is recorded; the implementation never fabricates a
feasibility result.

The pilot contract is deliberately explicit: slot radius `1.40 m`, slot
tolerance `3.0 m`, minimum slot slack `-2.0 s`, minimum gate progress
`-0.10 m`, gate horizon `3` steps, maximum assignment switch rate `0.5`, and
FC-DBF cost weight `0.25`.  Negative slack/progress thresholds are development
contracts for this short finite-shooting horizon; they are not safety margins
and do not constitute a reachability proof.

## 4. Closed-loop results

The two arms produced identical episode-level outcomes in this smoke.

| Method | FC-DBF | Safe capture | Capture | Collision | Boundary | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Worst-case | off | 85.0% (17/20) | 90.0% | 15.0% | 0.0% | 0.0% |
| Worst-case | on | 85.0% (17/20) | 90.0% | 15.0% | 0.0% | 0.0% |
| Distributed delayed | off | 95.0% (19/20) | 95.0% | 5.0% | 0.0% | 0.0% |
| Distributed delayed | on | 95.0% (19/20) | 95.0% | 5.0% | 0.0% | 0.0% |

The observed paired safe-capture deltas are therefore `0.0 pp` for both
methods.  With only 20 scenes, no confidence interval or promotion claim is
attached to this smoke.

## 5. FC-DBF diagnostics

| Method | Enabled rate | Feasible rate | Gate exhaustion | Mean progress (m) | Mean min slack (s) | Mean max error (m) | Switch rate | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Worst-case | 100.0% | 100.0% | 0.0% | 0.584 | -0.807 | 4.507 | 0.0458 | 1.338 |
| Distributed delayed | 100.0% | 99.194% | 0.806% | 0.577 | -0.817 | 4.566 | 0.0694 | 1.386 |

The progress gate removes the earlier semantic saturation of the gate: it is
not exhausted in the centralized branch and is exhausted on only about 0.8%
of distributed planning steps.  This improves interpretability of the gate
diagnostics, but the smoke provides no evidence that the gate improves
capture or safety outcomes.

## 6. Latency

All latency values are milliseconds and are reported as p50/p95/p99.  The
100 ms value is not treated as a hard gate.

| Method | Arm | Predictor | Planner | Safety | Total control |
| --- | --- | ---: | ---: | ---: | ---: |
| Worst-case | off | 33.926 / 37.225 / 39.217 | 22.075 / 24.388 / 26.087 | 3.728 / 4.296 / 4.551 | 63.871 / 68.795 / 75.490 |
| Worst-case | on | 34.405 / 37.143 / 40.262 | 40.216 / 43.432 / 46.105 | 3.738 / 4.270 / 5.052 | 82.498 / 88.924 / 94.520 |
| Distributed delayed | off | 33.889 / 36.710 / 40.461 | 33.206 / 36.215 / 39.010 | 3.754 / 4.197 / 4.466 | 75.221 / 81.366 / 87.774 |
| Distributed delayed | on | 34.391 / 37.303 / 39.622 | 187.605 / 200.085 / 212.017 | 3.532 / 4.426 / 5.772 | 229.600 / 244.790 / 256.842 |

Relative to FC-DBF off, total p95 increases by about `20.13 ms` in the
centralized branch and `163.42 ms` in the distributed branch.  Thus the
consensus-token optimization is correct and auditable, but the current
implementation still has a major distributed compute bottleneck.  A future
optimization should profile and reduce the vectorized arrival/slack path;
more weight or more threshold scanning is not justified by this result.

## 7. TensorBoard and audit checks

Both arms contain effective planner configuration snapshots, source hashes,
episode/step JSONL and TensorBoard event files.  The following tags were
verified in both `worst_case` and `distributed_delayed` pilot/off logs:

```text
Summary/FCDBF/enabled_rate
Summary/FCDBF/feasible_rate
Summary/FCDBF/mean_min_slot_slack_s
Summary/FCDBF/mean_max_slot_error_m
Summary/FCDBF/mean_slot_error_m
Summary/FCDBF/mean_slot_progress_m
Summary/FCDBF/assignment_switch_rate
Summary/FCDBF/gate_exhaustion_rate
Summary/FCDBF/mean_cost
```

Focused tests pass (`22 passed`), Python compilation passes, and
`git diff --check` passes.  The run is validation-only; no locked-test result
was read or used for tuning.

## 8. Decision and next gate

FC-DBF is retained as a reproducible negative/engineering result.  It may
become a publishable contribution only if a pre-registered multi-seed
confirmation shows all of the following on a fresh validation block:

- paired safe-capture non-inferiority lower bound at least `-2 pp`;
- collision and boundary rates not worse than FC-DBF off;
- local-solver/fallback failure below `1%`;
- gate exhaustion below `1%` with a documented fallback policy; and
- a separately reported, acceptable planner and total-latency trade-off.

Only after that confirmation may one geometry or target-behavior OOD block be
opened.  The locked test remains closed.  FC-DBF is a formation-planning
gate, not an R-CLBF-QP safety certificate; local CBF and the existing safety
contract remain separate.
