# Phase 17 RNIC Target-Speed OOD Report

## Protocol

This is a frozen OOD diagnostic block with 100 episodes and 50 mirror groups.
The scene manifest is `results/phase16_ood_target_speed_shift/scenes.jsonl`.
It changes target speed to the unseen `[0.82, 0.90]` range while keeping the
geometry, target branching, observation and communication settings fixed.
Both arms use the same Diagonal SSM checkpoint family, distributed delayed
DN-MPC, local CBF, K=8, refresh interval 1, delay 2, and sampling seed. Only
the RNIC cost switch changes.

Three matched predictor seeds were evaluated: 727201, 727202 and 727203.
The raw runs and TensorBoard event files remain under ignored `results/`.

## Per-seed outcomes

| Seed | Cost | Safe capture | Capture | Collision | Boundary | Timeout | Total p50/p95/p99 (ms) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 727201 | fixed distance | 95.00% | 95.00% | 5.00% | 0.00% | 0.00% | 39.66/133.11/144.89 |
| 727201 | RNIC | 94.00% | 95.00% | 6.00% | 0.00% | 0.00% | 141.86/177.78/187.38 |
| 727202 | fixed distance | 96.00% | 97.00% | 4.00% | 0.00% | 0.00% | 114.63/141.85/153.84 |
| 727202 | RNIC | 96.00% | 97.00% | 4.00% | 0.00% | 0.00% | 142.22/177.72/186.71 |
| 727203 | fixed distance | 94.00% | 96.00% | 6.00% | 0.00% | 0.00% | 112.60/137.87/147.98 |
| 727203 | RNIC | 93.00% | 95.00% | 7.00% | 0.00% | 0.00% | 141.18/176.39/185.50 |

## Aggregate decision

The hierarchical paired bootstrap gives RNIC-minus-fixed safe-capture delta
`-0.67 pp` with 95% CI `[-2.00, 0.00] pp`. Collision delta is `+0.67 pp`
with CI `[0.00, +2.00] pp`; timeout is `0%` in both arms. Aggregate total
latency is `138.44 ms` p95 for fixed distance and `177.33 ms` p95 for RNIC.

The newly logged nominal RNIC diagnostics are consistent across seeds:

- minimum best slack is approximately `-0.871 s`;
- mean best slack is `0.007--0.013 s`;
- unreachable-slot ratio is `44.37--44.91%`;
- earliest all-scenario feasible intercept step is `5.96--6.00`;
- RNIC diagnostic computation itself is approximately `2.68--3.80 ms`
  p50/p95/p99 per logged control step.

**Decision: No-Go for the RNIC main contribution.** On this frozen speed-OOD
block RNIC does not reduce timeout or collision and increases latency. The
reachability diagnostic is useful for explaining why many candidate-time
points are infeasible, but the current cost does not turn that information
into a better closed-loop action.

## Remaining evidence

The predicted-versus-realized arrival-time error audit has not yet been added;
the current slack values are nominal planning diagnostics only. Before any
future RNIC variant is considered, add that post-hoc audit and tune only on a
development stress split. Otherwise retain RNIC as a transparent negative
ablation and focus the main paper on a delay-aware, conformally calibrated
prediction tube that can drive both uncertainty budgeting and reachability
normalization.
