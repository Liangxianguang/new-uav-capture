# Phase 23 Risk-Calibrated UAKR Confirmation

Status: **Improves the original adaptive policy on the confirmation block,
but fails the fixed-budget non-inferiority and efficiency gates.**

## Protocol

The monotone risk map was fitted only on 132 calibration episodes from the
canonical mirror-group split. It uses the first public uncertainty score and
the episode failure label from retained development logs. The derived UAKR
thresholds are:

- low/medium: `0.1175714995`;
- medium/high: `0.6579849200`.

The untouched confirmation scene manifest contains 46 episodes. It was run
with Diagonal-SSM checkpoints from seeds `727201`, `727202`, and `727203`
(138 closed-loop episodes total), distributed delayed DN-MPC, local CBF,
`K_max=8`, and no QDR/RNIC. The evaluator consumed the frozen calibration
artifact through `--adaptive-risk-calibration`, recorded its hash, and did not
accept explicit threshold overrides.

## Confirmation result

| Policy | Safe capture | Collision | Boundary | Timeout | Mean K | Refresh ratio | Total p50 / p95 / p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed K=8 | 97.83% | 2.17% | 0.00% | 0.00% | 8.00 | 100.00% | 111.03 / 135.23 / 142.05 |
| Original UAKR | 94.20% | 5.80% | 0.72% | 0.00% | 2.58 | 35.98% | 59.80 / 132.01 / 141.36 |
| Risk-calibrated UAKR | **95.65%** | 4.35% | 0.00% | 0.00% | 4.05 | 51.77% | 79.97 / 141.98 / 153.67 |

The risk-calibrated policy improves safe capture over original UAKR by
`+1.45 pp`; the hierarchical paired bootstrap 95% CI is `[0.00, +4.35] pp`.
Against fixed K=8, the point delta is `-2.17 pp` with 95% CI
`[-5.07, 0.00] pp`, so the predeclared `-2 pp` non-inferiority gate is not
passed. Its p95 total latency is also higher than both references.

## Interpretation

The result supports a narrower claim: a frozen, monotone calibration map can
make UAKR more responsive than its original hand-set thresholds. It does not
yet provide a better accuracy--compute Pareto point, because the added K and
refresh activity costs more latency and still does not match fixed K=8.

The calibration label is an episode-level development label. It is not used
online; online decisions use only the current public uncertainty score. The
map is not a safety certificate and must not be applied to locked-test model
selection.

## Decision

Do not promote risk-calibrated UAKR as the final method. Retain the artifact,
the evaluator integration, and the confirmation result as a reproducible
ablation. A future attempt must calibrate a more local label (prediction miss
or next-step safety-margin violation) and demonstrate both non-inferior safe
capture and a lower compute/latency budget. The Phase 16 model remains the
primary reference.

## Artifacts

```text
results/phase23_uakr_failure_risk_calibration.json
results/phase23_uakr_failure_risk_calibration.md
results/phase23_uakr_calibrated_aggregate.json
results/phase23_uakr_calibrated_aggregate.md
results/phase23_uakr_calibrated_closed_loop_seed727201/
results/phase23_uakr_calibrated_closed_loop_seed727202/
results/phase23_uakr_calibrated_closed_loop_seed727203/
```

All generated artifacts retain config, hashes, episode/step JSONL and
TensorBoard events and remain Git-ignored.
