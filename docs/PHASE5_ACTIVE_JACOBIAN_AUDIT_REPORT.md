# Phase 5 Active-Constraint Analytic Jacobian Audit

> Version: v1.0 (2026-09-08)
> Formal run: `results/phase5_execution_active_formal_v1/`
> Decision: `solver latency improvement; execution-invariant safety No-Go`

## 1. Scope and conclusion

This follow-up stage adds a conservative active-constraint screen to the
analytic execution rollout Jacobian introduced in
`docs/PHASE5_ANALYTIC_JACOBIAN_AUDIT_REPORT.md`. Only barriers at or below the
configured `0.5 m` active margin enter the linearized Dykstra projection. The
full nonlinear execution barriers are still evaluated, and the independent
rollout certificate remains the acceptance criterion.

The formal audit shows a substantial reduction in safety-filter latency without
changing the locked safety outcomes or fallback counts. It still does not pass
the execution-invariant safety gate: immutable queued execution remains unsafe,
hard flush has frequent fallback, and the continuous-segment certificate rate
is low. The result supports a measured solver Pareto improvement, not an
R-CLBF-QP proof.

## 2. Active-set contract

The active set is selected from the current full robust-barrier vector at each
linearization point. A barrier is active when its robust value is no larger
than `execution_linearization_active_margin_m`; if the set would be empty, all
barriers are retained. The projected action is then checked against the full
nonlinear rollout and continuous audit paths before it is accepted.

This makes the screen conservative with respect to acceptance: a constraint
omitted from one local projection cannot bypass the independent certificate.
The screen is a solver optimization, not a reachable-set argument.

## 3. Formal protocol

The run reuses the locked Phase 5 protocol:

- fixed seeds: `649118, 649121, 649122, 649129, 649101, 649104, 649107, 649111`;
- four variants: mild immutable, hard randomized immutable, mild flush-pending and hard flush-pending;
- robust CBF-QP method, five-step execution preview and four swept-volume subdivisions;
- per-step/per-episode JSONL, configuration snapshot, source hashes and TensorBoard events.

Formal command:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_safety_execution.py `
  --config configs/innovation_safety_execution.yaml `
  --output-dir results/phase5_execution_active_formal_v1 `
  --methods robust_cbf_qp
```

## 4. Robust CBF-QP results

`Actual post robust state` is an empirical post-step rate, not an episode
guarantee. `Active constraints` is the mean number used by each local
linearized projection.

| Variant | Safe capture | Collision | Actual post robust state | Rollout certificate | Continuous segment | Fallbacks | Active constraints | Safety p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mild immutable | 12.5% | 87.5% | 83.02% | 61.44% | 43.87% | 90 | 131.66 | 106.27 |
| hard randomized immutable | 25.0% | 62.5% | 68.01% | 44.40% | 36.62% | 506 | 70.46 | 80.98 |
| mild flush pending | 75.0% | 0.0% | 100.0% | 96.63% | 35.48% | 22 | 110.95 | 139.85 |
| hard flush pending | 50.0% | 12.5% | 93.43% | 57.19% | 17.59% | 647 | 73.08 | 411.43 |

## 5. Comparison with the previous analytic run

The prior analytic-Jacobian run used the same seed and execution protocol but
projected all linearized barriers. The active screen leaves the episode and
certificate rates unchanged in this formal run while reducing p95 latency:

| Variant | Previous p95 (ms) | Active p95 (ms) | Reduction |
| --- | ---: | ---: | ---: |
| mild immutable | 819.64 | 106.27 | 87.0% |
| hard randomized immutable | 876.94 | 80.98 | 90.8% |
| mild flush pending | 501.45 | 139.85 | 72.1% |
| hard flush pending | 1776.21 | 411.43 | 76.8% |

The active run records `analytic_rollout_jacobian` for all robust-filter steps,
the active-constraint count, linearization evaluations, fallback and
certificate fields in both JSONL and TensorBoard.

## 6. Decision boundary

The execution-invariant safety gate remains **No-Go**. In particular:

- immutable variants have only `83.02%` and `68.01%` actual post robust-state safety;
- hard flush has `647` fallback steps and only `93.43%` actual post robust-state safety;
- continuous-segment validity is `17.59%--43.87%`;
- the independent held-out reachable-margin coverage remains `98.05%--98.93%`.

The supported claim is an empirically audited execution-aware robust CBF-QP
with analytic local Jacobians and conservative active-set projection. This
does not establish continuous-time forward invariance, a formal R-CLBF-QP
certificate, or real-airframe safety. The 100 ms value remains a strict 10 Hz
deployment reference rather than a method-validity gate.
