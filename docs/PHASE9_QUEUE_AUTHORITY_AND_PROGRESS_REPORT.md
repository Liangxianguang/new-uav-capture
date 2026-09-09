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

The one-step receding fallback is an operational recovery policy. It is not a
replacement for a full closed-loop multi-step proof.

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

### P9.3 Tighten the continuous-segment model

Separate endpoint, swept-volume, and continuous-segment contracts in the
report. Replace the current half-path-length bound with an exact or tighter
segment clearance bound for cylinders, boxes, boundaries, and inter-agent
separation, then rerun the Jacobian and invariance tests.

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
