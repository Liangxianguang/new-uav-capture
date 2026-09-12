# Phase 18 Delay-Aware Conformal Reachable-Tube Pilot

Status: **No-Go for promotion; implementation and validation-only smoke are
complete.**

This phase tests a reproducible repair candidate for the failed RNIC/UAKR
robustness path: calibrate a horizon-dependent split-conformal radius from
public-belief prediction candidates, align that radius with queue length and
prediction age, and expose it to the planner as a conservative tube-width
diagnostic. The artifact is empirical. It is not a reachable-set proof, a CBF
certificate, or a claim of closed-loop safety.

## Information boundary and protocol

- Predictor: Phase 16 Diagonal SSM conditional-diffusion checkpoint, action
  conditioning `both`.
- Calibration data: first half of the Phase 16 validation episode seeds.
- Confirmation data: second half of the same validation archive, untouched
  during fitting.
- Calibration coverage target: `0.90`.
- Candidate set: `K=8`, sampling steps `4`, projection iterations `4`.
- Runtime inputs: frozen radius schedule, queue length, prediction age and
  policy-safe uncertainty metadata.
- Runtime target truth: forbidden.
- Locked-test data: not read and not used for selection.

The executable calibration entry point is
`scripts/calibrate_delay_aware_conformal_tube.py`. The runtime artifact is
loaded by `evaluate_minimax_mpc.py` and `evaluate_s4_closed_loop.py` through
`--reachable-tube-calibration`. The configuration template is
`configs/phase18_delay_aware_conformal_tube.yaml`.

## Offline calibration result

| Quantity | Result |
| --- | ---: |
| Calibration windows | 835 |
| Confirmation windows | 1,087 |
| Calibration full-trajectory coverage | 90.18% |
| Confirmation full-trajectory coverage | 85.92% |
| Confirmation mean per-horizon coverage | 92.83% |
| Calibration mean nearest-candidate error | 4.804 m |
| Confirmation mean nearest-candidate error | 2.988 m |
| Simultaneous multiplier | 1.0347 |

The fitted radius schedule, in metres for horizons 1–12, is:

```text
[8.5892, 8.5072, 8.3865, 8.2286, 8.0370, 7.8391,
 7.5680, 7.2911, 6.9633, 6.6826, 6.3901, 6.0927]
```

The marginal horizon coverages are high, but the complete-trajectory
confirmation coverage misses the 90% target by 4.08 percentage points. This
is the relevant failure: per-step coverage does not establish trajectory-level
coverage. The large radii also make the candidate tubes overly conservative
for the current scene scale.

## Online smoke result

The artifact was injected into an 8-episode validation smoke using
distributed-delayed DN-MPC, UAKR, RNIC and local CBF. This run only checks
schema, horizon alignment, logging and runtime integration; it is not a
statistical comparison.

| Metric | Result |
| --- | ---: |
| Safe capture | 87.50% |
| Collision | 12.50% |
| Boundary violation | 0.00% |
| Tube enabled rate | 100.00% |
| Mean tube radius | 8.616 m |
| Maximum tube radius | 10.110 m |
| Tube budget score | 1.000 |
| Total-control p50/p95/p99 | 36.81 / 57.88 / 66.22 ms |

The smoke produced complete episode/step logs, a summary and TensorBoard
events. The `100 ms` value is not used as a hard gate; the latency percentiles
are retained for engineering comparison.

## Decision

The candidate is **not promoted** because the untouched confirmation split
does not meet the predeclared full-trajectory coverage target and the tube is
too wide to be a useful interception cost by itself. The implementation is
retained because it establishes the required experimental boundary and gives
the planner an auditable uncertainty object.

## Next repair plan

1. Freeze this failed artifact and never tune it on the locked test.
2. Refit on a new development-only split with conditional calibration by
   target motion mode and delay/observation regime; preserve a separate
   confirmation split.
3. Compare trajectory-level scalar conformal scores against the current
   horizon schedule, reporting coverage and average tube volume together.
4. Add a validation-only radius-cap/volume gate chosen before confirmation;
   if coverage and compactness cannot both pass, reject the candidate.
5. Run a controlled single-process comparison with tube disabled, tube as
   RNIC-only, and tube as UAKR diagnostic feature. Report safe capture,
   collision, timeout, clearance, and predictor/planner/RNIC/total p50/p95/p99.
6. Only if the new confirmation gate passes should a small paired OOD block be
   opened. Do not run the full QDR × UAKR × RNIC matrix before that gate.

## Reproduction artifacts

Generated artifacts are intentionally ignored by Git and retained locally:

```text
results/phase18_conformal_tube_dev_seed727201/
results/phase18_conformal_tube_online_smoke_seed727201_retry2/
```

The calibration artifact records effective configuration, source hashes,
calibration/confirmation score JSONL files, summary JSON and TensorBoard
events. The online smoke records the runtime source hashes and per-step tube
diagnostics.
