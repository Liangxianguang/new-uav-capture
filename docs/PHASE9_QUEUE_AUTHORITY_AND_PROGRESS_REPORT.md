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

The result must not be described as a complete formal closed-loop safety proof.
In particular, the one-step receding fallback proves only the reported
one-step contract under the stated execution assumptions.

## 5. Next Work

### P9.1 Make the fallback genuinely progress-aware

Replace the finite candidate list with a one-step QP objective that maximizes
belief-target distance reduction subject to the one-step robust certificate,
then use the multi-step tube only as a feasibility and audit constraint. Report
the fraction of fallback steps with positive progress and compare it with the
current barrier-recovery policy. Keep the clear/resume state machine, but
validate it against the old semantics on the same seeds before claiming a
capture improvement.

### P9.2 Recalibrate the reachable tube

Use the existing calibration/holdout audit to tune the smallest multiplier that
preserves the required holdout coverage separately for mild and hard variants.
Any reduction of 2.1 must be justified by a new holdout set; it must not be
chosen only because it raises capture rate.

### P9.3 Tighten the continuous-segment model

Separate endpoint, swept-volume, and continuous-segment contracts in the
report. Replace the current half-path-length bound with an exact or tighter
segment clearance bound for cylinders, boxes, boundaries, and inter-agent
separation, then rerun the Jacobian and invariance tests.

### P9.4 Re-run unseen seeds

Before selecting horizon 3 or 4, run at least 30--50 unseen seeds per
candidate. After P9.1-P9.3, repeat the selected configuration on the same
unseen set.
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
