# Phase 14 Continuous-QP Progress and Recovery Audit

> Version: v1.0 (2026-09-10)
> Scope: `progress_qp` fallback, continuous-segment constraints, delayed queue
> Contract: hard `flush_pending` simulation authority, reachable-tube multiplier `2.1`
> Decision: **P8 recovery gate remains No-Go**

## 1. Purpose

This audit evaluates whether the queue-aware robust CBF-QP can preserve task
progress after a full-horizon execution projection becomes infeasible. The
experiment keeps the independent execution, swept-volume, continuous-segment,
and actual post-state checks unchanged. `flush_pending` is a simulated queue
authority and is not evidence that a physical flight controller can cancel
already transmitted commands.

Two implementation issues were corrected before the final pilot:

1. Delayed fallback progress is now scored at the first point where the new
   candidate can execute (`queue length + 1`), instead of scoring only the
   already queued command.
2. An emergency brake may be repeated while the pending prefix remains
   inadmissible. The controller resumes normal queue filling only after the
   prefix is admissible again.

The evaluator also now defines `actual_post_robust_state_safe` using the
reachable-tube robust barrier, rather than the nominal current-state flag.

## 2. Short pilot comparison

Both pilots use two locked seeds and 60 steps. The only difference is the
number of internal continuous-segment samples.

| Configuration | Safe capture | Actual post robust safety | Continuous cert | Prefix admissible | Abort | Fallback cert | p95 / p99 ms | Mean fallback progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| h3, subdivisions 4 | 0% | 100% | 39.17% | 99.17% | 0.83% | 98.63% | 204.23 / 222.17 | 0.0052 m |
| h3, subdivisions 8 | 0% | 100% | 37.50% | 99.17% | 0.83% | 100% | 358.95 / 451.0 | 0.0546 m |

Increasing segment sampling from 4 to 8 does not improve the continuous
certificate and materially increases tail latency. The extra samples should
therefore remain a diagnostic option, not the default controller setting.
The delayed-progress correction improves the fallback ranking signal, but it
does not make the h3 full-horizon contract feasible.

Artifacts:

- `results/phase14_progress_continuous_qp_h3_pilot_v1/`
- `results/phase14_progress_continuous_qp_h3_s8_pilot_v4/`
- `results/phase14_progress_continuous_qp_h3_s8_pilot_v4/failure_analysis.json`

## 3. Long closed-loop audit

The final two-seed run uses h3, subdivisions 8, and 250 steps per episode.
It is an audit result, not a statistical locked test.

| Metric | Result |
| --- | ---: |
| Safe capture / ordinary capture | 0% / 0% |
| Collision / boundary violation | 0% / 0% |
| Timeout | 100% |
| Actual post-state safety | 72.2% |
| Actual post robust tube safety | 71.4% |
| Continuous-segment certificate | 12.4% |
| Prefix admissible | 65.2% |
| Abort required | 34.8% |
| Fallback certificate validity | 67.8% |
| Safety latency p50 / p95 / p99 | 194.38 / 356.57 / 369.41 ms |
| Mean fallback belief-goal progress | 0.0355 m |

The first episode leaves the robust execution contract around step 108--112.
Once the current state is too close to the robust boundary, braking is no
longer sufficient to restore the tube before the next execution update. The
repeated-brake change prevents the prior queue-resume behavior from creating
additional physical collisions in this two-seed run, but it does not create a
viability certificate. The correct interpretation is therefore “physical
collision avoided in this audit, robust contract still violated”, not “safe
closed-loop control passed”.

Artifact:

- `results/phase14_progress_continuous_qp_h3_s8_long_v2/`
- `results/phase14_progress_continuous_qp_h3_s8_long_v2/failure_analysis.json`

## 4. Current decision

The queue-aware robust CBF-QP remains useful as an auditable safety filter,
but the recovery path is not yet a closed-loop invariant controller. The
following gates remain open:

- a braking/viability-aware recovery condition that acts before the state
  reaches an unrecoverable robust margin;
- explicit separation of safe abort, robust-contract violation, and physical
  collision in the episode protocol;
- a long-run recovery audit with zero physical collision/boundary violation,
  at least 99% robust post-state coverage, and at least 95% continuous
  certificate coverage;
- only after those gates, a 30--50 unseen-seed holdout and renewed
  end-to-end prediction/planning evaluation.

The result does not justify lowering the reachable-tube multiplier, disabling
the independent certificate, or treating `flush_pending` as a real-flight
capability.
