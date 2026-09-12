# Phase 24 UAKR Residual-Triggered Budget Pilot

Status: **No-Go for promotion; retain as a negative ablation.**

## Motivation

The local reliability audit found that the public prediction-residual signal
was more informative than instantaneous uncertainty for next-state safety
diagnostics on one confirmation block (AUROC `0.652`). This pilot tests a
minimal online policy change: when the cached prediction-to-current-belief
residual reaches `0.40 m`, immediately force high-K prediction and replanning.

The residual is public and available online. The pilot does not use target
truth or a future episode label. The threshold was evaluated only on a small
validation prefix and was not opened on locked-test.

## Result

Protocol: Diagonal-SSM seed `727201`, first 20 Phase 16 validation scenes,
distributed delayed DN-MPC, local CBF, K schedule `{1,4,8}`, no QDR and no
RNIC, fixed sampling seed `745102`.

| Policy | Safe capture | Collision | Boundary | Timeout | Mean K | Refresh ratio | Total p50 / p95 / p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original UAKR | 90.0% | 10.0% | 0.0% | 0.0% | 2.67 | 37.65% | 60.15 / 133.44 / 139.84 |
| Residual trigger `0.40 m` | 85.0% | 15.0% | 0.0% | 0.0% | 3.02 | 37.27% | 65.30 / 141.61 / 151.57 |

The trigger increases candidate budget but changes two outcomes in the wrong
direction. It also increases the upper latency quantiles. This is a pilot,
not a significance claim, but it fails the predeclared direction-of-benefit
gate and does not justify additional residual-threshold scanning.

## Decision

Do not promote residual-triggered high-K escalation. The result demonstrates
why local risk ranking must be evaluated in the closed loop: a residual can
correlate with a diagnostic safety label while the corresponding planner
intervention still degrades capture. UAKR remains a compute-efficiency and
negative-ablation direction; Phase 16 remains the primary model.

The next valid improvement would need a calibrated intervention policy (for
example, separate refresh-only and K-escalation actions) selected on an
independent development split, followed by untouched confirmation.

## Local artifact

```text
results/phase24_uakr_residual_trigger_pilot_seed727201_t040/
```

The artifact retains effective config, source hashes, episode/step JSONL,
summary and TensorBoard events.
