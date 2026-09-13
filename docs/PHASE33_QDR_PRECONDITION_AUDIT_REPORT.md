# Phase 33 QDR Prefix/Suffix Precondition Audit

## Decision

This phase is an **engineering and causal-audit pass**, not a QDR promotion.
It adds an explicit distinction between an already unsafe immutable command
prefix and an unsafe newly planned suffix. The default execution authority and
the physical simulator contract are unchanged.

The distinction is necessary because a planner cannot repair a queued command
that is already committed to an immutable actuator. Under an explicitly
authorized `replace_nonexecuting` or `flush_pending` contract, the audit may
recommend recovery; it never escalates authority by itself.

## Implemented contract

The new `qdr_precondition` module provides:

- nominal suffix rollout through the shared execution dynamics;
- suffix geometry/barrier diagnostics using public bounds and obstacles;
- four auditable states:
  - `prefix_safe_suffix_safe`;
  - `prefix_safe_suffix_unsafe`;
  - `prefix_unsafe_recoverable`;
  - `prefix_unsafe_unrecoverable`;
- explicit `recovery_allowed` and `recovery_recommended` fields;
- evaluator, episode summary and TensorBoard integration.

The immutable case is intentionally labeled unrecoverable at the current
decision point. This is a statement about the declared command authority, not
a formal safety certificate.

## Schema smoke

| Item | Value |
| --- | --- |
| Scenes | first 2 records of `results/phase27_rnic_fresh_validation_scenes/scenes.jsonl` |
| Checkpoint | GRU `both`, seed `727201` |
| Planner | delayed distributed DN-MPC |
| QDR | enabled, 2-step immutable queue |
| UAKR/RNIC | disabled |
| Safety | local CBF |
| Safe capture | 50.0% (diagnostic only) |
| Boundary violation | 0.0% |
| Precondition counts | `prefix_safe_suffix_safe: 13`, `prefix_safe_suffix_unsafe: 29`, `prefix_unsafe_unrecoverable: 3` |
| First logged step | `prefix_safe_suffix_safe`, suffix barrier `0.6778 m` |

The output is retained locally at:

```text
results/phase33_qdr_precondition_smoke_seed727201/
```

The evaluator wrote step-level fields for suffix clearance/barrier,
precondition status, recovery recommendation and reason. TensorBoard contains:

```text
Summary/QDR/suffix_minimum_clearance_m
Summary/QDR/suffix_minimum_barrier_m
Summary/QDR/suffix_admissible_rate
Summary/QDR/precondition_recovery_recommended_rate
Summary/QDR/precondition_status_counts/text_summary
```

## Interpretation

The smoke exposes a meaningful distinction: many steps have a safe queued
prefix but an unsafe nominal suffix, while a smaller number are already
unrecoverable under immutable authority. It does not establish that either
class predicts episode collision, because the smoke is too small and the
suffix rollout is nominal rather than a reachable-tube calculation.

The next QDR experiment must use a fresh validation block and compare the
following frozen strata under one factor change at a time:

1. zero, one, two and four command-delay steps;
2. immutable authority versus an explicitly configured recovery authority;
3. nominal versus bounded execution noise/tracking dynamics.

The primary analysis should report collision and timeout conditional on the
four precondition states. A recovery authority must be evaluated as a plant
contract arm, with safe-capture non-inferiority and latency gates. No QDR
threshold or authority should be tuned on the locked test.

## Verification

- focused QDR/belief/planner/S4 tests: `42 passed`;
- Python compilation: passed;
- `git diff --check`: passed;
- target-truth information is not used by the audit;
- the full QDR behavior remains a diagnostic until a fresh validation
  confirmation passes the pre-registered gate.

The source implementation is
`src/encirclement3d/qdr_precondition.py`; evaluator integration is in
`scripts/evaluate_minimax_mpc.py` and
`scripts/evaluate_s4_closed_loop.py`.
