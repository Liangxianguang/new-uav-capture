# Phase 9 Queue-Aware Reachable-Tube and Progress-Aware Fallback Report

## 1. Scope

This phase evaluates the queue-aware execution contract for delayed multi-UAV
encirclement. The safety filter now distinguishes three command-authority
modes: `immutable`, `replace_nonexecuting`, and `flush_pending`. It also records
whether a fallback was accepted by the full preview certificate or by an
explicit one-step receding certificate.

The progress-aware fallback changes are deliberately belief-only. They use the
same weighted target prediction available to the pursuit controller and never
read the simulator target state.

## 2. Implementation Status

Completed:

- queue authority is checked before projection and recoverability is reported;
- full-horizon execution rollout and continuous-segment certificates share the
  same reachable-tube configuration;
- fallback candidates are independently rechecked by the execution certificate;
- queue clearing is treated as a one-shot emergency event, followed by an
  explicitly logged resume command;
- fallback diagnostics contain the reason, candidate type, certificate horizon,
  certificate scope, target-distance estimate, and belief-only progress;
- horizon 3, 4, and 5 locked-seed ablations were executed.
- a one-step progress-QP fallback was implemented and evaluated with and
  without continuous-segment constraints.
- continuous-segment constraints now use four explicit internal samples per
  action segment (five local subsegments) in both the QP Jacobian path and the
  independent certificate;
- step-level diagnostics now retain queue depth, pending-command norms, the
  first failed barrier, and the effective tube multiplier.

The one-step receding fallback is an operational recovery policy. It is not a
replacement for a full closed-loop multi-step proof.

The first unified-contract profiling smoke (hard + flush, horizon 1, two
locked seeds, 50 steps per episode) produced 100% QP continuous-segment
certificate validity, 100% independent continuous-segment validity, and 100%
actual post-state safety. It also produced 95% prefix admissibility, 5%
abort-required, and 762.5 ms p95 safety latency. This is a profiling result,
not a locked-matrix performance claim; it shows that the contracts now agree
and that the remaining immediate problem is projection cost and task progress.

## 3. Locked-Seed Results

All rows use the same eight seeds, three obstacles, target speed scale 0.45,
hard delayed execution, reachable-tube multiplier 2.1, and a 250-step limit.

| Preview horizon | Safe capture | Actual post-state safety | Full rollout cert. | Continuous cert. | Fallbacks | p95 latency |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 25.0% | 99.55% | 40.94% | 19.86% | 1075 | 222.8 ms |
| 4 | 50.0% | 99.75% | 58.48% | 12.17% | 749 | 358.8 ms |
| 5 | 25.0% | 99.75% | 44.99% | 5.00% | 1081 | 295.9 ms |

The final state-machine version keeps collision and boundary failure at 0% in
these runs. Horizon 4 reaches 50.0% on the eight locked seeds, but horizon 3
and horizon 5 reach only 25.0%; this variance is not evidence of a stable
improvement. The sample is too small to select a final horizon, and all three
settings remain below the working target of 80% safe capture.

### 3.1 Progress-QP fallback ablation

The progress-QP variant uses the belief target direction as the one-step
objective and independently rechecks the resulting command with the execution
certificate. The evaluation uses the same eight seeds and `hard_flush_pending`
variant as the horizon table above.

| Fallback variant | Continuous segment | Safe capture | Collision | Boundary | Actual post-state safety | Fallbacks | Mean fallback progress (m) | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| progress-QP, horizon 3 | off | 12.5% | 12.5% | 12.5% | 71.30% | 691 | 0.00255 | 682.5 ms |
| progress-QP, horizon 3 | on | 12.5% | 25.0% | 25.0% | 76.22% | 569 | 0.00339 | 453.0 ms |

The positive fallback-progress mean is not sufficient evidence of useful
closed-loop pursuit: both variants spend most steps in fallback, and neither
improves capture. The continuous-segment option improves p95 relative to this
particular run but still has poor episode-level safety and capture. These rows
are therefore negative ablations, not candidate final models.

### 3.2 Queue-authority ablation

The earlier locked matrix compares all three queue permissions on the same
eight seeds. The authority mode is an execution-system assumption: it cannot
be changed by a software safety filter unless the actuator queue actually
supports that operation.

| Variant | Safe capture | Abort required | Prefix admissible | Solver fallbacks | Actual post-state safety | Collision | p95 latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mild + immutable | 37.5% | 61.19% | 38.81% | 1007 | 55.50% | 0% | 107.5 ms |
| mild + replace nonexecuting | 100.0% | 0% | 100.0% | 46 | 100.0% | 0% | 201.4 ms |
| mild + flush pending | 75.0% | 0% | 100.0% | 22 | 100.0% | 0% | 160.7 ms |
| hard + immutable | 12.5% | 81.98% | 18.02% | 1598 | 27.40% | 0% | 110.6 ms |
| hard + replace nonexecuting | 12.5% | 8.55% | 91.45% | 1488 | 91.70% | 0% | 272.5 ms |
| hard + flush pending | 25.0% | 0.49% | 99.51% | 1291 | 99.55% | 0% | 247.6 ms |

This matrix does not meet the hard-scene capture target under any authority
mode. `replace_nonexecuting` and `flush_pending` meet the queue-feasibility
targets in the locked sample, but they are only valid conclusions for an
actuator that exposes the corresponding queue operation. `immutable` is the
only conservative software assumption until the real executor contract is
verified; under that assumption the hard scenario is a clear no-go.

## 4. Interpretation

The dominant bottleneck is still contract feasibility, not collision avoidance:

- the actual post-state remains safe in almost every step;
- the full reachable-tube rollout certificate rejects many commands that are
  safe after the simulated execution model runs;
- the continuous-segment contract is substantially more conservative as the
  horizon grows;
- the controller therefore spends many steps in barrier recovery or receding
  fallback, reducing pursuit progress;
- horizon 4 has the highest locked-seed capture rate, but it also has a
  358.8 ms p95 latency and needs unseen-seed confirmation;
- all three horizons exceed the intended 100 ms research control-cycle target
  at p95 in the final implementation;
- progress-QP does not solve the bottleneck: the hard flush progress-QP runs
  fall to 12.5% safe capture, with 71.30--76.22% actual post-state safety and
  453.0--682.5 ms p95 latency;
- the gap between command/certificate safety and actual post-state safety is
  now an explicit failure signal, rather than evidence that the fallback is
  working.
- queue permission is a first-order variable: the hard locked sample moves
  from 81.98% abort-required under `immutable` to 0.49% under `flush_pending`,
  while safe capture only moves from 12.5% to 25.0%; queue feasibility alone
  does not recover the task objective.

The result must not be described as a complete formal closed-loop safety proof.
In particular, the one-step receding fallback proves only the reported
one-step contract under the stated execution assumptions.

## 5. Next Work

### P9.1 Diagnose the queue and certificate failure before further fallback tuning

The progress-QP implementation is complete but failed the locked-seed
ablation. The next experiment should decompose each failure by queue prefix,
execution error, barrier type, and fallback reason. Compare, on exactly the
same seeds and variants:

- the original `barrier_recovery` fallback;
- `progress_qp`;
- a no-clear diagnostic mode that records infeasibility without claiming
  safety;
- a one-step-only certificate with the multi-step tube disabled for execution.

The report must include per-episode timelines, queue age/depth, candidate
action, executed action, certificate margins, and the first violated barrier.
Do not optimize capture rate until the cause of the post-state safety failures
is identified.

### P9.2 Recalibrate and validate the reachable tube

Use the existing calibration/holdout audit to tune the smallest multiplier that
preserves the required holdout coverage separately for mild and hard variants.
Any reduction of 2.1 must be justified by a new holdout set; it must not be
chosen only because it raises capture rate.

Phase 10 now provides an independent horizon 1/2/3/5 audit with Wilson
intervals and an opt-in `queue_aware` multiplier resolver. The resolver remains
disabled by default until its gains are fitted and frozen on a separate
calibration split.

### P9.3 Tighten the continuous-segment model

Separate endpoint, swept-volume, and continuous-segment contracts in the
report. The implementation now uses configured internal samples and a local
Lipschitz bound for each subsegment, rather than applying half of the whole
action path to every segment. Rerun the matched authority matrix and verify
that the independent certificate and QP use identical internal-sample
constraints; this remains an empirical continuous-segment contract, not a
real-airframe invariance proof.

### P9.4 Re-run unseen seeds only after the diagnostic gate

Do not spend the larger evaluation budget on the current progress-QP variant.
After P9.1-P9.3 produce a candidate that passes the diagnostic gate, run at
least 30--50 unseen seeds per candidate and repeat the selected configuration
on the same unseen set.
The acceptance gate should be:

- hard safe capture at least 80%;
- collision and boundary failure 0% in the evaluation set;
- actual post-state safety at least 99%;
- holdout tube coverage at the declared confidence level;
- p95 latency reported, with 100 ms treated as a target rather than a hard
  blocker;
- no hidden use of target ground truth by the controller or safety filter.

Only after this gate should the Mamba/diffusion predictor and DN-MPC be
connected to the end-to-end pipeline again.
