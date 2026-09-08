# Phase 5 State-Aware Execution and Swept-Volume Audit

> Version: v1.0 (2026-09-08)
> Repository: `https://github.com/Liangxianguang/new-uav-capture`
> Decision: `execution_contract_repaired_but_gate_no_go`
> Scope: delayed command queue, bounded execution, multi-step state-aware filter, and sampled swept-volume audit

## 1. Executive conclusion

This follow-up addresses the main contract failure identified in the previous
execution perturbation audit. The environment, safety filter, and independent
certificate now share one execution-dynamics implementation covering the
pending command queue, command noise bound, velocity tracking, drag,
acceleration limit, mass scale, and speed clipping. The filter observes the
pending queue and projects commands against a five-step execution preview. The
independent audit also checks four samples per executed motion segment and
applies a Lipschitz displacement allowance between samples.

The contract repair is verified in unit tests and in the preserved evaluation
artifacts, but the P5 execution gate still does not pass. With the repaired
contract, robust CBF-QP safe capture is `37.5%` under the mild variant and
`25.0%` under the hard randomized variant. The robust filter fallback counts
are `54` and `505`, and filter p95 latency is `671.36 ms` and `1945.74 ms`.
These results are not suitable for a deployable safety layer and do not prove
execution-invariant safety.

The correct current claim is:

```text
Shared execution-state contract: IMPLEMENTED AND UNIT-TESTED
Sampled swept-volume audit: IMPLEMENTED AND LOGGED
P5 execution-invariant safety gate: NO-GO
R-CLBF-QP / learned CLBF: PAUSED
Overall research status: CONDITIONAL GO
```

The failure is informative rather than a silent solver failure. In several
episodes the already queued delayed command drives the predicted state outside
the contracted robust set before a newly issued action can change it. The
current filter reports this as an execution precondition or rollout infeasible
failure and preserves it in step-level JSONL and TensorBoard.

## 2. Contract and protocol

| Item | Value |
| --- | --- |
| Complete run | `results/phase5_execution_state_aware_v5/` |
| Episodes | 8 fixed robust-safe reset seeds |
| Seeds | `649118, 649121, 649122, 649129, 649101, 649104, 649107, 649111` |
| Methods | nominal, local CBF, execution-aware robust CBF-QP |
| Prediction horizon for filter | 5 execution steps |
| Swept samples | 4 subdivisions per execution step |
| Time step | `0.1 s` |
| Safety margin | `0.35 m + 0.46 m` contracted margin |
| Noise bound | `3 sigma` command-noise bound |
| TensorBoard | preserved under every `variant/method/tensorboard/` directory |

The new shared implementation is `src/encirclement3d/execution_dynamics.py`.
The environment exposes the pending queue and execution parameters in the
observation. `safety_qp.py` consumes that state, while
`safety_certificate.py` independently rolls it forward and checks both
endpoint and swept-volume barriers.

## 3. Results

Rates are episode-level unless explicitly labeled as a step rate. The
`actual post robust state safe rate` is the mean fraction of post-step states
inside the contracted set; it is not an episode-level safety guarantee.

| Variant | Method | Safe capture | Collision | Boundary | Timeout | Rollout cert. | Swept cert. | Actual post robust state | Filter p95 (ms) | Fallbacks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mild | nominal | 87.5% | 12.5% | 0.0% | 0.0% | 0.0% | 44.43% | 84.17% | 0.001 | 0 |
| mild | local CBF | 87.5% | 12.5% | 0.0% | 0.0% | 0.0% | 45.21% | 84.17% | 1.442 | 0 |
| mild | robust CBF-QP | 37.5% | 62.5% | 25.0% | 0.0% | 79.43% | 79.43% | 90.78% | 671.355 | 54 |
| hard | nominal | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 50.63% | 87.30% | 0.001 | 0 |
| hard | local CBF | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 49.77% | 87.54% | 1.437 | 0 |
| hard | robust CBF-QP | 25.0% | 62.5% | 37.5% | 12.5% | 45.03% | 45.03% | 67.67% | 1945.742 | 505 |

The nominal and local CBF rows are diagnostic references. Their higher capture
rates do not constitute robust-certificate evidence because their commands are
not projected against the execution-state contract.

## 4. Comparison with the previous contract

The previous audit used the same broad eight-seed and two-variant structure,
but filtered the current command under a velocity-level contract and checked
execution only after the command was selected. The new run adds queue-aware
preview and swept-volume logging. It therefore exposes failure earlier and
uses a substantially more expensive nonlinear projection.

| Run | Mild robust safe capture | Hard robust safe capture | Mild filter p95 (ms) | Hard filter p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Previous contract (`phase5_execution_perturbation_full_v4`) | 50.0% | 0.0% | 6.670 | 7.749 |
| State-aware v2, short preview | 37.5% | 0.0% | 38.210 | 51.407 |
| State-aware v5, 5-step + swept audit | 37.5% | 25.0% | 671.355 | 1945.742 |

The hard-variant safe-capture increase from `0%` to `25%` is not sufficient to
claim a passing gate, because collision remains `62.5%`, fallback is frequent,
and the latency cost is very large. The comparison also shows that increasing
the preview horizon alone is not a complete solution.

## 5. Failure analysis

The main failure categories are:

- `execution_rollout_infeasible`: no newly issued command satisfied the
  five-step contracted rollout under the current queued command and actuator
  bounds.
- `execution_precondition_invalid`: the actual state had already left the
  contracted robust set before a new command could recover it.
- `swept_volume_outside_robust_safe_set`: an interpolated motion segment or
  its conservative displacement tube violated the robust barrier.

Static margin inflation is therefore not an adequate next step. A valid
recovery design needs an explicit authority model for pending commands, a
braking or queue-intervention policy, or a reachable-set contract that proves
when the delayed command is already unrecoverable. The nonlinear SLSQP
projection also needs an analytically linearized or dedicated QP formulation
before it can be considered an online safety layer.

## 6. Reproducibility and TensorBoard artifacts

The formal run preserves:

- root and per-method `config.yaml` snapshots;
- source hashes, including `execution_dynamics.py`;
- `episodes.jsonl` and `steps.jsonl` with queue-aware certificate fields;
- `summary.json` with safe capture, certificate, swept-volume, fallback, and
  latency aggregates;
- TensorBoard event files with episode, summary, and hparam metrics.

Reproduction command:

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_safety_execution.py `
  --config configs/innovation_safety_execution.yaml `
  --output-dir results/phase5_execution_state_aware_repro
```

The output directory must be empty or new; the evaluator refuses to overwrite
an existing run.

## 7. Decision and next actions

1. Keep the shared execution-state contract and swept-volume checker as the
   new P5 audit baseline.
2. Do not train or enable learned CLBF/R-CLBF-QP while the execution gate is
   No-Go.
3. Design a queue-aware emergency braking/command-authority contract and
   prove when pending actions are recoverable.
4. Replace the current finite-difference nonlinear projection with a
   sequential linearized QP or another bounded solver, then repeat the same
   locked seeds and variants.
5. Calibrate the noise and tracking reachable-set margin independently before
   making any forward-invariance claim.

The follow-up command-authority and sequential-linearized projection audit is
reported in `docs/PHASE5_AUTHORITY_LINEARIZED_AUDIT_REPORT.md`.
