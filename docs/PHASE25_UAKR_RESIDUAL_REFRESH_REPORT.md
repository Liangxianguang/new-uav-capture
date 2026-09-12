# Phase 25 UAKR Residual-Only Refresh Pilot

Status: **No-Go for promotion; residual intervention is not reliable in the
current closed loop.**

## Motivation

Phase 24 escalated both candidate count and replanning when the public
prediction-to-belief residual exceeded `0.40 m`, and degraded performance.
This pilot isolates the two effects: it keeps the original adaptive K bucket
and only forces a prediction refresh at the same residual threshold.

## Protocol and result

The pilot uses the first 20 Phase 16 validation scenes, Diagonal-SSM seed
`727201`, distributed delayed DN-MPC, local CBF, no QDR/RNIC, and fixed
sampling seed `745102`.

| Policy | Safe capture | Collision | Boundary | Timeout | Mean K | Refresh ratio | Total p50 / p95 / p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original UAKR | 90.0% | 10.0% | 0.0% | 0.0% | 2.67 | 37.65% | 60.15 / 133.44 / 139.84 |
| Residual-only refresh, `0.40 m` | 85.0% | 15.0% | 0.0% | 0.0% | 2.67 | 36.96% | 67.25 / 142.34 / 154.52 |

The refresh-only intervention changes two paired outcomes in the wrong
direction and increases the upper latency quantiles. The pilot is small and
is not a significance claim, but it fails the predeclared direction-of-benefit
gate and does not justify further residual-threshold scanning.

## Decision

Do not promote residual-only refresh or residual-triggered high-K escalation.
The result separates a useful offline residual correlation from a useful
online intervention: correlation with a diagnostic label does not imply that
refreshing the predictor at that point improves the planner's selected action.

UAKR remains a reproducible negative/efficiency ablation. Any future attempt
must learn or calibrate the intervention effect itself, use a fresh
development split, and be compared against both fixed K=8 and the original
adaptive policy. Phase 16 remains the primary model.

## Local artifact

```text
results/phase25_uakr_residual_refresh_pilot_seed727201_t040/
```

The artifact retains effective config, source hashes, episode/step JSONL,
summary and TensorBoard events.
