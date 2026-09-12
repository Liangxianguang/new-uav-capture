# Phase 18c Tube-Width UAKR Smoke

Status: **Negative ablation; do not promote as a performance claim.**

This smoke tests whether the frozen public-context conformal tube should be
used as an additional feature in Uncertainty-Triggered Adaptive K and
Replanning (UAKR), instead of being used only as a conservative RNIC planning
surrogate. The tube is not allowed to read target truth. The run is a small
validation smoke and is not a locked-test result.

## Protocol

- Same 8 validation scenes, Diagonal SSM checkpoint and sampling seed as the
  Phase 18b paired smoke.
- Controller: distributed-delayed DN-MPC + local CBF.
- QDR: off; RNIC: on.
- Tube artifact: Phase 18b balanced-context artifact.
- UAKR tube feature: weight `0.15`, normalization scale `10.0 m`.
- Comparison: no tube, tube used by RNIC only, tube used by RNIC and UAKR.
- Locked-test: not read.

## Result

| Metric | No tube | Tube, RNIC only | Tube + UAKR feature |
| --- | ---: | ---: | ---: |
| Safe capture | 87.50% | 87.50% | 87.50% |
| Collision | 12.50% | 12.50% | 12.50% |
| Boundary violation | 0.00% | 0.00% | 0.00% |
| Mean K | 3.364 | 3.364 | 3.952 |
| Prediction refresh ratio | 45.64% | 45.64% | 52.02% |
| Forced refresh ratio | 14.63% | 14.63% | 8.46% |
| Total p50/p95/p99 (ms) | 36.16/55.99/63.40 | 36.20/56.27/64.49 | 37.06/57.87/69.14 |

Adding the tube feature shifts decisions toward more medium/high-budget
steps, but the increased compute does not change the episode outcomes in this
smoke. The p95 total-control latency increases by about 1.89 ms relative to
the no-tube run and by about 1.61 ms relative to the RNIC-only tube run. The
sample is too small for an inferential performance conclusion, but it is
enough to reject the claim that this feature is a free improvement.

## Decision

Keep tube-width as an auditable diagnostic and optional ablation. Do not use it
as a promoted UAKR contribution until a fresh, pre-registered confirmation
shows ID non-inferiority and an OOD benefit while satisfying a compute/latency
budget. The original UAKR result remains a compute-savings-only result, and
RNIC remains a No-Go performance component.

## Reproduction artifacts

Configuration:
`configs/phase18c_tube_uakr_smoke.yaml`

Local ignored results:

```text
results/phase18c_tube_uakr_smoke_seed727201/
results/phase18b_conformal_tube_online_smoke_seed727201/
results/phase18b_conformal_tube_online_smoke_seed727201_no_tube/
```

The new smoke retains effective adaptive-budget configuration, per-step K and
refresh decisions, tube diagnostics, summary JSON and TensorBoard events.
