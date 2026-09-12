# Phase 19 Gated RNIC Pilot

Status: **No-Go for promotion; implementation retained as a reproducible
diagnostic ablation.**

## Motivation

The original Reachability-Normalized Interception Cost (RNIC) applies a
penalty to every margin violation.  The Phase 17 validation and OOD runs
showed no reliable capture gain, increased latency, and a pooled collision
AUROC of only `0.504` for its logged slack.  This pilot tests a narrower
variant: mild negative slack remains available for diagnosis, but the RNIC
penalty is activated only when the best arrival-time slack is below an
explicit severe-unreachability threshold.

This is a planner objective heuristic.  It is not a CBF certificate, a
reachability proof, or a collision predictor.

## Reproducible protocol

- scenes: the first 20 records of the frozen Phase 16 validation scene file;
- predictor: validation-selected Diagonal-SSM checkpoint, seed `727201`;
- planner: distributed delayed DN-MPC;
- safety: local CBF;
- prediction: `K=8`, 8 sampling steps, fixed sampling seed `745102`;
- queue-aware rollout and adaptive K: disabled to isolate RNIC;
- execution/delay configuration: unchanged from the Phase 17 validation
  contract;
- thresholds explored: `-0.75`, `-0.50`, and `-0.25 s`;
- locked-test: not accessed.

The explicit candidate configuration is
`configs/phase19_gated_rnic_validation.yaml`; the evaluator switch is
`--rnic-activation-slack-s`.

## Pilot result

| Variant | Safe capture | Collision | Timeout | Total p50 / p95 / p99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| RNIC off | 95.0% | 5.0% | 0.0% | 36.58 / 51.43 / 59.87 |
| Gated RNIC, `-0.75 s` | 95.0% | 5.0% | 0.0% | 48.44 / 67.97 / 74.27 |
| Gated RNIC, `-0.50 s` | 95.0% | 5.0% | 0.0% | 48.58 / 67.07 / 75.06 |
| Gated RNIC, `-0.25 s` | 95.0% | 5.0% | 0.0% | 49.03 / 66.15 / 72.98 |

All three gated variants produced the same episode-level outcome as the
paired RNIC-off run on all `20/20` scenes.  The RNIC diagnostic path was
active in the gated runs and added approximately `0.97--1.00 ms` mean RNIC
latency per step; the total latency increase is larger because the gated
objective changes planner execution and accounting overhead.

The pilot is intentionally too small for a significance claim.  It is,
however, sufficient for the engineering decision: no observed benefit and a
consistent latency penalty across the tested thresholds.

## Decision and next steps

The gated-RNIC repair is **not promoted** to the main model and must not be
used to rewrite the Phase 17 No-Go result.  Keep it as a negative ablation
because it provides an explicit, testable failure mode: thresholding the
penalty does not make the current slack signal useful.

If RNIC is revisited, the next valid experiment must first introduce an
independent reachable-set target or calibrated risk label and evaluate its
ranking ability on a fresh validation block.  A larger planner sweep without
that signal-level repair is not justified.  The primary model remains the
Phase 16 GRU `both` + distributed delayed DN-MPC + local CBF configuration.

## Local artifacts

The generated pilot directories are retained under:

```text
results/phase19_rnic_gated_pilot_seed727201_off/
results/phase19_rnic_gated_pilot_seed727201_m075/
results/phase19_rnic_gated_pilot_seed727201_m050/
results/phase19_rnic_gated_pilot_seed727201_m025/
```

Each directory contains the effective configuration, source hashes,
episode/step JSONL, summary JSON, and TensorBoard event files.  The generated
results remain Git-ignored.
