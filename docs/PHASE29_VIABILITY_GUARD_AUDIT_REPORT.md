# Phase 29 Viability-Guard Development Audit

> Status: complete as a development audit; recovery gate remains No-Go.

## Purpose

This phase evaluates the existing short-horizon braking/viability guard under
the hard `flush_pending` execution contract. The guard is enabled with a
0.15 m warning margin, 0.0 m abort margin and one-step braking probe. The
experiment is a development audit only; it does not provide a safety proof and
does not tune or replace any locked-test result.

## Frozen protocol

| Item | Value |
| --- | --- |
| Configuration | `configs/phase29_viability_guard_diagnostic.yaml` |
| Variant | `hard_flush_pending` |
| Methods | `robust_cbf_qp` |
| Episodes | 8 predefined robust-safe seeds |
| Horizon | 250 steps per episode |
| Delay / authority | 2-step command delay / `flush_pending` |
| Tube multiplier | 2.1 |
| Guard | warning 0.15 m, abort 0.0 m, horizon 1 |
| TensorBoard | retained in the run directory |

The complete local artifact is
`results/phase29_viability_guard_full/`, including configuration snapshots,
summary, episode/step JSONL, source hashes and TensorBoard events. The result
was not used to tune a locked-test model.

## Results

| Metric | Result |
| --- | ---: |
| Safe capture | 12.50% (1/8) |
| Ordinary capture | 12.50% |
| Collision / boundary violation | 0.00% / 0.00% |
| Timeout | 37.50% (3/8) |
| Safety abort | 50.00% (4/8) |
| Robust-contract violation | 50.00% |
| Continuous-contract violation | 100.00% |
| Command certificate valid | 100.00% |
| Execution-rollout certificate valid | 18.31% |
| Swept-volume certificate valid | 16.11% |
| Continuous-segment certificate valid | 18.31% |
| Actual post-state safe | 100.00% |
| Actual post robust-state safe | 99.58% |
| Prefix admissible | 100.00% |
| Solver fallback count | 1,182 |
| Viability-guard brake decisions | 964 |
| Emergency-brake count | 1,147 |
| Mean queue depth | 2.0 |

The safety-filter latency was `544.55/1207.35/1379.22 ms` at p50/p95/p99.
The 100 ms value is not a hard acceptance gate, but this tail is still a
material systems limitation. The fallback breakdown was `964`
`viability_guard_brake`, `211` `progress_qp`, `3` `nominal_clipped` and `4`
normal steps.

## Interpretation

The guard successfully prevents physical collision/boundary violation in this
8-seed audit and keeps the measured actual post-state safe. It does so by
replacing progress actions with repeated braking/recovery, which produces
50.0% episode abort, 37.5% timeout and only 12.5% safe capture. Continuous
certificate coverage remains low, so the result is not evidence of
forward-invariance or execution-invariant safety.

This is a safety--liveness Pareto diagnostic, not a promoted improvement. The
dominant next repair is not a lower threshold scan: it is a queue-aware
recovery tube that reserves stopping distance and task progress, followed by
an independent development calibration. Until that repair passes, do not
reopen the robust CBF-QP end-to-end composition or the QDR×UAKR×RNIC Full
matrix.

## Reproduction

```powershell
python scripts/evaluate_safety_execution.py `
  --config configs/phase29_viability_guard_diagnostic.yaml `
  --output-dir results/phase29_viability_guard_full `
  --variants hard_flush_pending `
  --methods robust_cbf_qp
```
