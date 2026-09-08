# Phase 5 Queue-Aware Recoverability Contract Audit

> Version: v1.1 (2026-09-08)
> Formal run: `results/phase5_recoverability_contract_audit_v2/`
> Decision: `recoverability contract auditable; execution-invariant safety remains No-Go`

## 1. Scope

The execution-aware filter previously reported a fallback action without
explicitly classifying whether a queued command had already become
unrecoverable. This stage adds a shared recoverability check used by the
execution certificate, robust CBF-QP diagnostics, step JSONL and TensorBoard.

The check evaluates only commands that cannot be changed under the current
command-authority contract. It emits `prefix_admissible`, `abort_required`,
`immutable_prefix_horizon_steps` and the minimum committed-prefix robust
barrier. `abort_required` means that the current state or committed prefix is
already outside the contract; it does not claim that a future command can
repair an already executed or immutable command.

This is a diagnostic and supervisory contract, not a continuous-time safety
proof. `flush_pending` is retained only as an explicit simulated authority
variant and is not evidence of physical actuator authority.

## 2. Fixed Protocol

The audit uses the same locked Phase 5 matrix as the previous fallback audit:

- seeds: `649118`, `649121`, `649122`, `649129`, `649101`, `649104`,
  `649107`, `649111`;
- four execution variants: mild/hard immutable and mild/hard
  flush-pending-brake;
- robust CBF-QP only, five-step execution preview and four swept-volume
  subdivisions;
- eight episodes per variant;
- every run retains `config.yaml`, episode/step JSONL, `summary.json`,
  TensorBoard events, HParams and source hashes.

## 3. Results

| Variant | Safe capture | Collision | Timeout | Actual post robust state | Abort required | Prefix admissible | Certified fallback | Safety p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mild delay/noise (immutable) | 37.5% | 0.0% | 62.5% | 55.50% | 61.19% | 38.81% | 2.28% | 519.49 ms |
| hard randomized (immutable) | 25.0% | 0.0% | 75.0% | 47.85% | 71.84% | 28.16% | 0.23% | 615.91 ms |
| mild flush pending/brake | 75.0% | 0.0% | 25.0% | 100.0% | 0.0% | 100.0% | 0.0% | 857.73 ms |
| hard flush pending/brake | 50.0% | 0.0% | 50.0% | 91.70% | 12.04% | 87.96% | 0.0% | 2489.98 ms |

Continuous-segment certificate valid rates were `16.79%`, `16.65%`,
`35.48%` and `17.59%` in the same order. The immutable variants produced
`1,007` and `1,327` solver fallback steps; the flush variants produced `22`
and `720`. These are step-level diagnostics, not episode guarantees.

## 4. Interpretation

The new signal makes the failure mode explicit. Under immutable authority,
more than 60% of mild steps and 70% of hard steps require an abort-level
response because the current state or committed execution prefix is already
outside the robust contract. A newly selected action cannot retroactively
change that prefix. This explains why increasing projection iterations or
changing the recovery direction alone cannot establish execution-invariant
safety.

The flush-pending variants show the authority trade-off: simulated queue
control removes most prefix failures and improves actual post-step safety, but
hard flush has `2489.98 ms` safety p95, `50%` timeout and only `17.59%`
continuous-segment certificate validity. It is therefore not a deployable
safety gate.

## 5. Decision

- queue-aware recoverability is now represented in the runtime contract and
  independently auditable;
- immutable execution remains outside the execution-invariant safety gate;
- `abort_required` is a truthful supervisory signal, not a certificate that
  the abort itself will succeed;
- learned CLBF/R-CLBF-QP training remains paused;
- the next safety task is to define and validate a formally supported abort or
  recoverable-set contract for the actual actuator authority, then repeat the
  same locked matrix.

## 6. Reproducibility

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_safety_execution.py `
  --config configs/innovation_safety_execution.yaml `
  --output-dir results/phase5_recoverability_contract_audit_v2 `
  --methods robust_cbf_qp
```

The four variant subdirectories contain the TensorBoard event files and the
configuration/source-hash snapshot used to generate this report.
