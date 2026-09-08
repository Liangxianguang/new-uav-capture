# Phase 5 Certified Recovery Fallback Audit

> Version: v1.0 (2026-09-08)
> Run: `results/phase5_execution_recovery_certified_fallback_v1/`
> Decision: `fallback contract improved; execution-invariant safety remains No-Go`

## 1. Scope

The execution-aware safety filter previously generated a barrier-recovery
command, but did not independently certify that command before returning it as
a fallback. This stage adds two pieces:

1. the recovery action uses the analytic execution rollout Jacobian instead of
   zero constraint rows;
2. the fallback checks a deterministic candidate list with the same nonlinear
   execution rollout certificate used for normal acceptance, and records
   `fallback_certificate_valid` in JSONL and TensorBoard.

The candidate order is recovery action, zero action, and clipped desired action
when emergency braking is not requested. A candidate is called certified only
when it satisfies the current execution contract, action-change limit, preview
horizon, and robust barrier checker.

## 2. Fixed formal protocol

The audit reuses the locked Phase 5 protocol:

- seeds: `649118, 649121, 649122, 649129, 649101, 649104, 649107, 649111`;
- four execution variants: mild/hard immutable and mild/hard flush-pending;
- robust CBF-QP, five-step execution preview, four swept-volume subdivisions;
- 8 episodes per variant, with episode/step JSONL, config, source hashes and
  TensorBoard events.

## 3. Results

| Variant | Safe capture | Collision | Boundary | Actual post robust state | Fallbacks | Certified fallback | Continuous segment | Safety p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mild immutable | 37.5% | 50.0% | 37.5% | 87.90% | 312 | 8.33% | 29.49% | 126.23 ms |
| hard randomized immutable | 25.0% | 75.0% | 75.0% | 79.74% | 485 | 1.44% | 32.46% | 127.98 ms |
| mild flush pending/brake | 75.0% | 0.0% | 0.0% | 100.0% | 22 | 0.0% | 35.48% | 141.27 ms |
| hard flush pending/brake | 50.0% | 0.0% | 0.0% | 99.55% | 720 | 0.0% | 17.59% | 460.16 ms |

Compared with the same fixed protocol before the barrier-direction fallback
fix, actual post robust-state safety improved from `83.02%` to `87.90%` for
mild immutable and from `68.01%` to `79.74%` for hard randomized immutable;
flush variants retained `100.0%` and `99.55%` respectively. These are step
rates, not episode-level safety guarantees.

## 4. Interpretation

The fallback is now auditable: the run distinguishes a recovery command that
passes the independent checker from a best-effort command returned because the
current state or pending queue is already outside the certifiable set. The low
certified-fallback rates show that the current execution contract is still too
restrictive for most failure states.

The immutable variants remain unsafe at the episode level, with collision rates
of `50%` and `75%`. Flush authority removes collisions in this small locked
matrix, but hard flush has `720` fallback steps, a `50%` timeout rate, low
continuous-segment validity, and `460.16 ms` safety p95. The result is therefore
not a safety gate pass and does not justify promoting the filter to an
execution-invariant controller.

The additional candidate certification also increases latency relative to the
previous recovery run. This is an explicit safety--latency trade-off and is
recorded rather than hidden.

## 5. Decision

- recovery fallback now has an independent certificate path and TensorBoard
  metric;
- actual post-step safety improves in the immutable variants but remains below
  the project gate;
- P5 execution-invariant safety remains **No-Go**;
- continuous-time forward invariance and real-airframe safety remain unproved;
- learned CLBF/R-CLBF-QP remains paused.

## 6. Reproducibility

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_safety_execution.py `
  --config configs/innovation_safety_execution.yaml `
  --output-dir results/phase5_execution_recovery_certified_fallback_v1 `
  --methods robust_cbf_qp
```
