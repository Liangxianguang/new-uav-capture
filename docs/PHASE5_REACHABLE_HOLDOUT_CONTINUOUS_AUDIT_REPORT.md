# Phase 5 Held-Out Reachable Margin and Continuous Segment Audit

> Version: v1.0 (2026-09-08)
> Hold-out run: `results/phase5_execution_reachable_holdout_formal_v1/`
> Continuous run: `results/phase5_continuous_segment_formal_v1/`
> Decision: `P5 execution-invariant safety No-Go; empirical contract audit complete`

## 1. Executive conclusion

This stage completed an independent held-out execution-parameter audit,
one-factor sensitivity checks, an explicit out-of-calibration policy, and a
conservative continuous-segment checker for the benchmark's piecewise-linear
position update.

The execution-invariant safety gate remains **No-Go**:

1. The frozen 99% margin reached only `98.05--98.93%` simultaneous coverage
   on the independent in-domain hold-out split. It is empirical evidence, not
   a sufficient contract for the main safety filter.
2. Any changed execution contract is marked `reject_or_fallback`, even when a
   few sampled values overlap the original range.
3. Explicit out-of-calibration cases have only `3.91--16.41%` observed
   simultaneous coverage under the frozen margin and are not actionable.
4. The continuous-segment lower-bound checker is stricter than the sampled
   swept-volume check; its valid rates are only `17.68--43.30%`.

The project remains `Conditional Go` at the research level. Learned
CLBF/R-CLBF-QP remains paused.

## 2. Held-out margin protocol

Each variant uses separate calibration and hold-out random streams. The
calibration split freezes a simultaneous radius from the upper 99th order
statistic of the maximum normalized five-step position error. The hold-out
split evaluates that frozen radius without refitting it.

| Variant | Calibration coverage | Hold-out coverage | In-domain rate | Frozen multiplier | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| mild immutable | 99.02% | 98.93% | 100% | 0.864 | below 99% |
| hard randomized immutable | 99.02% | 98.44% | 100% | 1.958 | below 99% |
| mild flush pending | 99.02% | 98.93% | 100% | 0.853 | below 99% |
| hard flush pending | 99.02% | 98.05% | 100% | 1.964 | below 99% |

Each split contains 1024 samples per variant, with four defenders and a
five-step execution preview. The hold-out drop means the calibrated radius
cannot yet be treated as a verified 99% contract for the declared domain.

## 3. Sensitivity and out-of-calibration policy

The audit varies command noise, tracking time constant, mass, drag, and delay.
Only the exact baseline setting in each factor is actionable. Changed settings
are retained for diagnosis, but their policy is `reject_or_fallback`.

| Variant | OOD observed coverage | OOD samples rejected | Policy |
| --- | ---: | ---: | --- |
| mild immutable | 3.91% | 1024/1024 | reject/fallback |
| hard randomized immutable | 14.06% | 1024/1024 | reject/fallback |
| mild flush pending | 4.00% | 1024/1024 | reject/fallback |
| hard flush pending | 16.41% | 1024/1024 | reject/fallback |

These values explain why extrapolation is refused; they do not claim that the
fallback itself is safe. The main filter has not been modified to use these
radii.

## 4. Continuous-segment audit

The new checker uses the shared execution rollout and lower-bounds every point
on each piecewise-linear position segment. Obstacle and boundary signed
clearances are treated as 1-Lipschitz in position; inter-agent separation is
bounded using the sum of the two segment lengths. This is conservative evidence
for the benchmark interpolation contract, not a proof for unmodeled vehicle
dynamics or a real airframe.

| Variant | Safe capture | Collision | Actual post robust state | Sampled swept valid | Continuous segment valid | Fallbacks | Filter p95 (ms) | Min continuous barrier (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mild immutable | 12.5% | 87.5% | 83.12% | 59.98% | 43.30% | 93 | 1208.81 | -2.247 |
| hard randomized immutable | 25.0% | 62.5% | 68.01% | 44.40% | 36.62% | 506 | 1361.64 | -2.073 |
| mild flush pending | 62.5% | 0.0% | 100.0% | 99.85% | 39.09% | 18 | 810.06 | -0.193 |
| hard flush pending | 50.0% | 12.5% | 92.03% | 54.75% | 17.68% | 701 | 2367.80 | -1.321 |

`Actual post robust state` is a step-level empirical rate, not an episode
guarantee. The flush rows show why endpoint safety alone is insufficient:
post-step rates can be strong while the all-segment lower bound still fails.

## 5. Artifacts and reproducibility

The held-out run preserves `config.yaml`, source hashes,
`calibration_samples.jsonl`, `holdout_samples.jsonl`, `sensitivity.jsonl`,
`out_of_calibration.jsonl`, `summary.json`, and TensorBoard events. The
continuous run preserves per-variant `episodes.jsonl`, `steps.jsonl`,
`summary.json`, configuration snapshots, source hashes, and TensorBoard events.

The continuous checker is implemented in
`src/encirclement3d/safety_certificate.py` and is called by
`scripts/evaluate_safety_execution.py`; its scalar metrics and hparams are
also written to TensorBoard.

## 6. Decision boundary

This stage completes the empirical held-out and continuous-segment audit
tasks, but not a formal forward-invariance proof. The remaining requirements
for any R-CLBF-QP claim are analytic or theorem-backed reachable-set bounds,
a verified safe fallback for out-of-calibration states, a true continuous-time
execution-dynamics argument, multi-step forward invariance, and a solver that
does not rely on the observed fallback and latency failure rates.

The correct current claim is `empirically audited robust CBF-QP execution
filter under a benchmark contract`, not `execution-invariant safety` and not a
closed-loop formal safety guarantee.
