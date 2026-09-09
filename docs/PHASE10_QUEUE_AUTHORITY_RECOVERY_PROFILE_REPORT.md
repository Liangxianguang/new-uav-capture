# Phase 10 Queue Authority and Progress-Recovery Profile

> Status: profiling only, not a locked result
>
> The runs in this report use two of the retained hard-execution seeds and a
> 60-step cap. They are intended to detect implementation regressions after
> the continuous-segment and fallback changes. They do not replace the
> eight-seed authority matrix or a 30--50-seed holdout.

## 1. Changes under test

The safety filter now has two explicit controls:

- a HiGHS linear-feasibility recovery is attempted only after the sequential
  linearized projection reaches its iteration limit;
- `execution_emergency_brake_on_full_horizon_failure` controls whether a
  full-preview certificate failure alone is enough to request a queue clear.

The nonlinear execution, swept-volume, and continuous-segment certificates
remain the acceptance gates. The linear-feasibility recovery does not relax
any certificate.

The progress-oriented h3 smoke additionally uses `fallback_policy:
progress_qp` and keeps a queue prefix when the prefix is admissible. Its
one-step candidate is accepted only after an independent certificate check.

## 2. Hard authority profiling

Command:

```powershell
python scripts/evaluate_safety_execution.py `
  --config configs/phase9_queue_authority.yaml `
  --output-dir results/phase10_hard_authority_profile_v1 `
  --variants hard_immutable hard_replace_nonexecuting hard_flush_pending `
  --max-episodes 2 --max-steps 60
```

| Authority | Safe capture | Abort required | Prefix admissible | Actual post-state safe | Continuous certificate | Fallbacks | p95 / p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| immutable | 50.0% | 25.5% | 74.5% | 80.8% | 58.0% | 46 | 1837.9 / 2053.2 |
| replace nonexecuting | 0.0% | 9.2% | 90.8% | 92.5% | 56.7% | 51 | 2834.8 / 4133.3 |
| flush pending | 0.0% | 16.7% | 83.3% | 100.0% | 0.0% | 120 | 3258.8 / 3591.0 |

The sample is too small for performance claims. It does show that lower abort
rate does not imply higher capture: the flush variant has perfect actual
post-state safety in this profile but no capture and zero full-preview
certificate validity. Queue authority must therefore be reported as an
executor capability, not as a software-only safety improvement.

## 3. Continuous-segment recovery smoke

On one hard flush seed, horizon 3, 20 steps:

| Recovery setting | Continuous certificate | Actual post-state safety | Fallbacks | Emergency brakes | Mean belief progress | p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| barrier recovery, brake on full failure | 70.0% | 100.0% | 6 | 4 | 0.0172 m/step | 1687.3 |
| barrier recovery, brake disabled for full failure | 80.0% | 100.0% | 4 | 0 | 0.0238 m/step | 1138.3 |
| progress QP, brake disabled for full failure | 85.0% | 100.0% | 3 | 0 | 0.0685 m/step | 1075.5 |

The progress-QP smoke is the best current recovery candidate, but it still
fails the desired 95--99% continuous-certificate gate and has a very high
tail latency. It must not be presented as a completed robust controller.

## 4. Decision

- Keep `execution_emergency_brake_on_full_horizon_failure: true` as the
  conservative default for the general execution audit.
- Treat the no-brake and progress-QP settings as explicit experimental
  variants only.
- Do not select an authority mode from this two-seed profile.
- Re-run the full eight locked seeds after the recovery rule is frozen, then
  evaluate 30--50 unseen seeds with Wilson 95% episode intervals.
- Keep the empirical reachable-tube multiplier at `2.1` until a new holdout
  justifies a change; this profile is not a calibration set.

Evidence directories:

- `results/phase10_hard_authority_profile_v1/`
- `results/phase10_continuous_segment_h3_smoke_v1/`
- `results/phase10_continuous_segment_h3_no_brake_on_full_failure_v1/`
- `results/phase10_continuous_segment_h3_progress_no_brake_v1/`
