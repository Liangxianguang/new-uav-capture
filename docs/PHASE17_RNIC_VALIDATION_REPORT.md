# Phase 17 RNIC Validation Report

## Scope

This report evaluates Reachability-Normalized Interception Cost (RNIC) on the
same frozen 90-episode validation manifest used for UAKR selection. The only
changed factor is the RNIC cost switch:

- predictor: Diagonal SSM conditional diffusion;
- controller: distributed delayed DN-MPC + local CBF;
- execution delay: 2 steps, command noise: 0;
- fixed reference: Euclidean distance cost, K=8, refresh interval=1;
- candidate: the same fixed K=8 configuration with RNIC enabled;
- three predictor seeds: 727201, 727202, 727203.

The validation manifest SHA-256 is
`5ba54afc62d2cf123014f1c5e598a0e3e914e9067870bc7a199ba3393d8f8d99`.
The aggregate artifact is `results/phase17_validation_rnic_aggregate.json`.

## Result

All three seeds produce the same episode-level outcomes for the reference and
RNIC candidate:

| Policy | Safe capture | Capture | Collision | Boundary | Timeout | Mean capture time (s) | Total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed distance cost | 94.07% [91.11%, 96.67%] | 94.07% | 5.93% | 0.00% | 0.00% | 1.922 | 111.25/135.72/142.92 |
| Fixed K=8 + RNIC | 94.07% [91.11%, 96.67%] | 94.07% | 5.93% | 0.00% | 0.00% | 1.922 | 140.03/174.19/183.55 |

The paired outcome deltas are exactly zero for safe capture, collision,
boundary and timeout on this validation block. Planner latency increases from
49.36/72.21/80.35 ms to 77.77/111.06/118.63 ms at p50/p95/p99.

**Decision: No-Go on the current ID validation.** RNIC is active in the
planner call path, but it has not changed the selected closed-loop action on
this scene block and has introduced a substantial planner/total latency cost.
This is a negative result, not evidence that reachability-normalized costs are
generally ineffective.

## Diagnostic limitation

The current closed-loop episode schema does not yet persist the RNIC-specific
diagnostics required by the protocol: selected intercept step, minimum and
mean reachability slack, unreachable-slot ratio, and predicted-versus-realized
arrival-time error. Therefore the RNIC speed-stress gate is not fully
observable yet. The implementation must expose these diagnostics before the
target-speed OOD experiment can be interpreted as a complete RNIC test.

## Required follow-up

1. Add step/episode logging for the RNIC diagnostics without feeding simulator
   truth into control.
2. Run the frozen target-speed OOD block with RNIC off/on and three matched
   seeds; keep geometry, target behavior and delay fixed.
3. Compare formation-slot and interceptor-only RNIC, then tune only on the
   development stress split.
4. If speed-OOD timeout does not improve by the pre-registered margin, remove
   RNIC from the main method and retain it as a negative ablation.
