# Phase 5 Command Authority and Linearized Execution Audit

> Version: v1.0 (2026-09-08)
> Run: `results/phase5_authority_linearized_v1/`
> Decision: `execution-invariant-safety No-Go; authority tradeoff quantified`

## 1. Executive conclusion

This stage implements and audits two missing pieces from the previous P5
No-Go: an explicit pending-command authority contract and a bounded sequential
linearized projection for the execution-aware filter.

The authority contract distinguishes three simulated actuator semantics:

- `immutable`: all queued commands remain fixed;
- `replace_nonexecuting`: the command due this tick remains fixed, while later
  queued commands may be replaced by a braking command;
- `flush_pending`: the supervisor may cancel all queued commands before the
  execution model advances.

The default locked variants remain `immutable`. The `flush_pending` variants
are a diagnostic upper-bound experiment for command authority, not a claim
about a physical flight controller. The filter and environment reject a
directive that escalates beyond the configured authority.

The nonlinear SLSQP execution projection was replaced by a bounded sequential
linearization with finite-difference barrier Jacobians and Dykstra projections
onto speed, acceleration, and linearized execution-barrier halfspaces. Each
accepted command is still checked by the original rollout and swept-volume
certificates.

The result is informative but does not pass P5:

```text
Command authority contract: IMPLEMENTED AND UNIT-TESTED
Sequential linearized bounded projection: IMPLEMENTED AND AUDITED
Flush-pending mild Pareto point: 100% actual post robust-state safety, 62.5% safe capture
Flush-pending hard result: 92.03% actual post robust-state safety, 50.0% safe capture
Execution-invariant safety gate: NO-GO
R-CLBF-QP / learned CLBF: REMAINS PAUSED
```

The `flush_pending` improvement comes with substantial conservatism and
solver cost. The hard variant has `701` fallback steps, `689` emergency-brake
requests, and a filter p95 of `2554.09 ms`. The immutable variants remain
unsafe under delayed execution. The result therefore supports a safety--
capture--latency Pareto analysis, not a deployment-ready safety layer.

## 2. Protocol and artifacts

| Item | Value |
| --- | --- |
| Fixed seeds | `649118, 649121, 649122, 649129, 649101, 649104, 649107, 649111` |
| Variants | mild immutable, hard immutable, mild flush, hard flush |
| Methods | nominal, local CBF, robust CBF-QP |
| Preview horizon | 5 execution steps |
| Swept samples | 4 subdivisions per step |
| Linearization iterations | 3 maximum |
| Projection iterations | 160 maximum |
| TensorBoard | per variant/method directory |
| JSONL | `episodes.jsonl` and `steps.jsonl` per variant/method |

Every run preserves a configuration snapshot, source hashes, TensorBoard
event files, episode JSONL, step JSONL, and summary JSON. The formal command
is:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_safety_execution.py `
  --config configs/innovation_safety_execution.yaml `
  --output-dir results/phase5_authority_linearized_repro
```

The output directory must be new; the evaluator refuses to overwrite an
existing run.

The numeric table is tied to the source hashes stored in the run's
`config.yaml`. After the formal run completed, only diagnostic serialization
was enriched: fallback barriers/queue counts and TensorBoard hparams now carry
additional fields. The full test suite was rerun after that enrichment
(`112 passed`); the formal numeric artifacts remain untouched and are kept as
the authoritative locked-run record.

## 3. Robust filter results

Rates are episode-level unless explicitly labeled as a step rate.
`actual post robust state` is the fraction of post-step states inside the
contracted set; it is not a formal episode-level guarantee.

| Variant | Safe capture | Collision | Boundary | Timeout | Actual post robust state | Rollout cert. | Filter p95 (ms) | Fallbacks | Emergency brakes | Queue overrides |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mild immutable | 12.5% | 87.5% | 0.0% | 0.0% | 83.12% | 59.98% | 1240.51 | 93 | 0 | 0 |
| hard immutable | 25.0% | 62.5% | 37.5% | 12.5% | 68.01% | 44.40% | 1243.86 | 506 | 0 | 0 |
| mild flush | 62.5% | 0.0% | 0.0% | 37.5% | 100.0% | 97.42% | 841.27 | 18 | 396 | 396 |
| hard flush | 50.0% | 12.5% | 12.5% | 37.5% | 92.03% | 54.55% | 2554.09 | 701 | 689 | 1378 |

The immutable rows are the closest representation of the current actuator
contract and remain a No-Go. Flush authority improves the actual safety state,
but it does not produce a passing gate because hard-case fallback is frequent,
capture is reduced by conservative braking, and the bounded projection is too
slow for an online safety claim.

## 4. Baseline comparison

The nominal and local CBF methods use the same scenes and execution variants.
They are diagnostic references only: they do not project commands against the
execution-state contract.

| Variant | Method | Safe capture | Collision | Actual post robust state | Filter p95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| mild immutable | nominal | 87.5% | 12.5% | 84.17% | 0.001 |
| mild immutable | local CBF | 87.5% | 12.5% | 84.17% | 1.389 |
| hard immutable | nominal | 100.0% | 0.0% | 87.30% | 0.001 |
| hard immutable | local CBF | 100.0% | 0.0% | 87.54% | 1.498 |

The baseline capture advantage must not be interpreted as robust safety because
the commands are not checked against the queued execution contract.

## 5. Interpretation and limitations

1. Queue authority is a model assumption. `flush_pending` is useful as an
   upper-bound diagnostic, but requires an actuator or supervisor interface
   that can actually cancel pending commands.
2. Emergency braking is zero-velocity command generation under the configured
   acceleration and mass limits. It cannot retroactively change an immutable
   command that is already due this tick.
3. The sequential projection linearizes a piecewise execution rollout. The
   independent nonlinear rollout and swept-volume checks remain authoritative.
4. The hard flush p95 and fallback count show that the current finite-difference
   Jacobian is not a deployable online solver. The next implementation should
   use analytic piecewise Jacobians, a smaller certified active set, or a
   dedicated QP backend before another full locked audit.
5. The 100 ms value is retained only as a strict 10 Hz deployment reference.
   Exceeding it is an engineering limitation and does not invalidate the
   research method by itself.

## 6. Next tasks

- Calibrate reachable-set margins on an independent execution-parameter set.
- Derive analytic Jacobians for the non-saturated and saturated execution
  branches; explicitly record branch changes as certificate invalidation.
- Add a short-horizon emergency-braking feasibility test before solving the
  full projection.
- Re-audit immutable, replace-nonexecuting, and flush authority with the same
  locked seeds after the solver optimization.
- Keep learned CLBF/R-CLBF-QP paused until the immutable or explicitly
  justified actuator contract passes a multi-step gate.
