# Phase 5 Analytic Execution Jacobian Audit

> Version: v1.0 (2026-09-08)
> Formal run: `results/phase5_execution_analytic_formal_v1/`
> Decision: `solver Pareto improvement; execution-invariant safety No-Go`

## 1. Scope and conclusion

This stage replaces the execution-aware safety projection's default finite-
difference Jacobian with a selectable analytic local Jacobian. The derivative
propagates the shared execution contract through command delay, tracking,
drag, acceleration limiting and velocity clipping. Geometry barriers use
local signed-distance gradients, including obstacle, boundary and inter-agent
constraints.

The original finite-difference backend remains available for comparison. Every
accepted action is still checked by the independent nonlinear rollout
certificate; the analytic Jacobian is only a local projection model and is not
itself a safety proof.

The formal audit shows a useful solver improvement, especially under flush
authority, but does not pass the execution-invariant safety gate. Immutable
queue variants remain unsafe, and continuous-segment validity remains far below
the required level. Learned CLBF/R-CLBF-QP therefore remains paused.

## 2. Verification of the Jacobian

The analytic barrier values use the same queue, rollout, uncertainty and swept
penalty convention as `execution_barrier_values`. A persistent unit test
compares every analytic derivative against central finite differences. An
additional random test with cylinder and box obstacles, two queued commands,
tracking, drag and acceleration saturation produced a maximum absolute error
of approximately `2.1e-10` over eight cases.

The derivative is local at branch changes and zero-distance normals. These
points remain subject to the independent nonlinear certificate, which is the
authoritative acceptance check.

## 3. Formal protocol

The formal run reuses the locked Phase 5 protocol:

- 8 fixed episode seeds: `649118, 649121, 649122, 649129, 649101, 649104, 649107, 649111`;
- mild immutable, hard randomized immutable, mild flush-pending and hard flush-pending variants;
- nominal, local CBF and robust CBF-QP methods;
- five-step execution preview and four swept-volume subdivisions;
- per-step JSONL, per-episode JSONL, configuration snapshot, source hashes and TensorBoard events.

The generated results are intentionally ignored by Git because they contain
large reproducible artifacts. The formal command was:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_safety_execution.py `
  --config configs/innovation_safety_execution.yaml `
  --output-dir results/phase5_execution_analytic_formal_v1
```

## 4. Robust CBF-QP results

Rates are step or episode rates as named. `actual post robust state` is an
empirical post-step rate, not an episode guarantee.

| Variant | Safe capture | Collision | Actual post robust state | Rollout certificate | Continuous segment | Fallbacks | Safety p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mild immutable | 12.5% | 87.5% | 83.02% | 61.44% | 43.87% | 90 | 819.64 |
| hard randomized immutable | 25.0% | 62.5% | 68.01% | 44.40% | 36.62% | 506 | 876.94 |
| mild flush pending | 75.0% | 0.0% | 100.0% | 96.63% | 35.48% | 22 | 501.45 |
| hard flush pending | 50.0% | 12.5% | 93.43% | 57.19% | 17.59% | 647 | 1776.21 |

Compared with the previous sequential finite-difference audit, the flush
filter p95 decreased from `810.06` to `501.45 ms` for mild flush and from
`2367.80` to `1776.21 ms` for hard flush. The hard flush actual post robust
state rate increased from `92.03%` to `93.43%`. These are engineering
improvements, not a safety pass. The immutable variants remain the closest
representation of the locked actuator contract and remain a No-Go.

## 5. Decision boundary

The following claims remain unsupported:

- execution-invariant safety under delayed/noisy/randomized execution;
- continuous-time or multi-step forward-invariance proof;
- a formal zero-slack R-CLBF-QP guarantee;
- real-airframe or flight-controller safety.

The supported claim is narrower: the benchmark has an empirically audited
execution-aware robust CBF-QP filter with an analytic local Jacobian option,
an independently checked nonlinear rollout certificate, and a measured
safety--capture--latency trade-off. The 100 ms value remains only a strict
10 Hz deployment reference, not a method-validity gate.

## 6. Provenance and retained artifacts

The formal directory retains `config.yaml`, per-variant/method
`episodes.jsonl`, `steps.jsonl`, `summary.json`, TensorBoard event files and
the source hashes recorded at run start. The run records the analytic backend
rate and linearization evaluations in JSONL, summary and TensorBoard.

The post-run backend provenance serialization fix is kept in the current
source tree; the locked numeric artifacts remain the authoritative results for
the source hashes recorded in their own `config.yaml`.
