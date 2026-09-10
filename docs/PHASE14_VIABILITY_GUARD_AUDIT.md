# Phase 14 Viability Guard Audit

> Version: v1.0 (2026-09-10)
> Scope: queue-aware execution recovery, viability guard, safety-abort accounting
> Contract: hard `flush_pending` simulation authority, reachable-tube multiplier `2.1`
> Decision: **P8 recovery gate remains No-Go**

## 1. Implementation status

This phase adds a short-horizon braking probe to the execution-aware robust
CBF-QP. The probe uses the independent execution certificate and exposes two
different decisions:

- `viability_guard_triggered`: the braking candidate is still usable but the
  nominal command should be replaced by braking/recovery;
- `safety_abort_requested`: the braking candidate is no longer certified above
  the configured abort margin, so the evaluator must stop before calling
  `env.step()`.

The evaluator now records `post_step_executed`. An unexecuted abort is retained
as a safety decision, but it is excluded from actual post-state safety and
physical execution-error statistics. The failure-analysis script applies the
same rule to `actual_post_robust_unsafe`.

The recovery configuration uses `execution_viability_guard_margin_m=0.03` and
`execution_viability_guard_abort_margin_m=0.012`. These values are calibration
parameters for this simulation audit, not a formal safety proof or a hardware
flight guarantee.

## 2. Results

### Guard pilot v2

Artifact: `results/phase14_progress_continuous_qp_h3_viability_guard_pilot_v2/`

Two seeds and 60 steps per episode produced:

| Metric | Result |
| --- | ---: |
| Safe capture | 0% |
| Collision / boundary violation | 0% / 0% |
| Safety abort | 0% |
| Actual post robust state safety | 100% |
| Prefix admissible | 100% |
| Guard brake decisions | 74 / 120 |
| Continuous certificate | 23.33% |
| Latency p50 / p95 / p99 | 77.64 / 203.23 / 217.79 ms |

The guard intervenes early and avoids physical safety loss in this short
window, but it does not preserve capture progress.

### Guard long audit v2

Artifact: `results/phase14_progress_continuous_qp_h3_viability_guard_long_v2/`

The original abort margin was `0.00 m`. One episode reached an abort request
after 79 executed transitions; the following action was not executed. The
other episode ran to timeout.

| Metric | Result |
| --- | ---: |
| Safe capture | 0% |
| Collision / boundary violation | 0% / 0% |
| Episode safety abort | 50% |
| Episode robust-contract violation | 50% |
| Executed transitions | 329 |
| Actual post robust unsafe transitions | 1 / 329 |
| Continuous-only gap | 276 / 329 executed transitions |
| Latency p50 / p95 / p99 | 102.28 / 219.16 / 235.44 ms |

This audit confirms that abort accounting is now execution-order correct, but
the zero abort margin is too late to prevent every robust-tube loss.

### Recovery calibration pilot v1

Artifact: `results/phase14_progress_continuous_qp_h3_viability_guard_recovery_v1_pilot/`

The calibrated configuration uses a 3 cm warning margin and a 1.2 cm abort
margin. It improves the step-level safety outcome, but remains unsuitable as a
finished capture controller.

| Metric | Result |
| --- | ---: |
| Safe capture | 0% |
| Collision / boundary violation | 0% / 0% |
| Episode safety abort | 50% |
| Executed transitions | 355 |
| Unexecuted abort decisions | 1 |
| Actual post robust unsafe transitions | 0 / 355 |
| Prefix admissible | 94.10% |
| Continuous certificate | 20.36% |
| Mean fallback belief-goal progress | 0.0492 m |
| Mean belief-goal progress | -0.0160 m |
| Latency p50 / p95 / p99 | 153.63 / 246.61 / 265.14 ms |

The result separates two facts: the execution abort path prevents a measured
post-state robust violation in this pilot, while the controller still spends
too much time in one-step recovery and does not close the target distance.

## 3. Failure attribution

Diagnostic artifacts:

- `results/phase14_progress_continuous_qp_h3_viability_guard_long_v2/failure_analysis.json`
- `results/phase14_progress_continuous_qp_h3_viability_guard_recovery_v1_pilot/failure_analysis.json`

The dominant failure is a conservative multi-step execution certificate, not a
physical collision. In the recovery pilot, 279 of 355 executed transitions
were continuous-only certificate gaps while the actual post robust state
remained safe. The fallback was certified in 279 of 355 decision rows, but
usually only for the one-step receding scope. This means the current guard is a
recovery mechanism, not a closed-loop invariant proof.

## 4. Gate decision and next work

P8 is not complete. The current evidence does not support an unseen-seed
holdout or renewed end-to-end Mamba/diffusion/DN-MPC combination yet.

The next experiments should:

1. calibrate the abort margin against the measured execution-error tail and
   queue depth, while keeping the reachable-tube multiplier at `2.1`;
2. replace repeated one-step braking with a queue-aware recovery tube that
   explicitly reserves progress and stopping distance;
3. improve continuous-segment feasibility without increasing tail latency by
   blindly increasing segment samples;
4. rerun the locked 8-seed audit only after prefix admissibility is at least
   95%, episode abort is at most 5%, actual robust post-state safety is 100% in
   the audit, and continuous certificate coverage reaches the project gate;
5. only then evaluate the prediction/planning stack in the full closed loop.

`flush_pending` remains a simulation authority for this audit. It must not be
presented as evidence that a physical flight controller can cancel already
transmitted commands.
