# Phase 4 Distributed DN-MPC Validation Report

> Version: v1.0 (2026-09-07)
> Repository: `https://github.com/Liangxianguang/new-uav-capture`
> Decision: `diagnostic_only`
> Scope: finite-candidate sequential best-response DN-MPC with bounded communication

## 1. Executive conclusion

Phase 4 is partially successful. The distributed implementation, information boundary, communication fault injection, fallback accounting and reproducible three-seed evaluation are working. On the fixed S3 validation set, all four distributed communication modes achieved 100% safe capture and 0% collision, matching the centralized worst-case oracle and exceeding the DynamicEncirclement baseline (91.7% safe capture, 8.3% collision).

The overall P4 gate is **not passed** because the distributed planner is not real-time under the current 100 ms control-cycle budget. The mean planner p95 across the three checkpoint seeds is 257.79 ms for no communication, 445.06 ms for delayed communication, 451.07 ms for dropout communication and 623.73 ms for ideal communication. The current evidence supports a communication-robustness diagnostic, not a complete deployable DN-MPC claim.

The current main result remains the centralized Scenario MPC from Phase 3. P4 should next focus on reducing local search cost or validating an explicit low-frequency planner plus high-frequency safety execution architecture.

## 2. Locked protocol for this validation

| Item | Value |
| --- | --- |
| Scene split | `validation` |
| Scene family | S3 random mixed obstacles |
| Shared scenes | 12 |
| Checkpoint seeds | `745101`, `745201`, `745301` |
| Candidate source | prediction checkpoint |
| Candidate status | `dynamics-projected` only |
| Planner | finite-shooting sequential best-response |
| MPC horizon | 8 steps |
| Control horizon | 3 steps |
| Maximum distributed iterations | 3 |
| Local timeout | 100 ms |
| Communication interval | every step |
| Delayed mode | 2 steps |
| Dropout mode | probability 0.10 |
| Control-cycle budget | 100 ms |
| Safety post-filter | existing local CBF |

The planner receives defender state, belief-level target information, projected candidate trajectories, local geometry and permitted peer messages. It does not read future target truth. All methods use the same scene file and the same checkpoint-specific candidate generation protocol.

The SHA-256 of `scenes.jsonl` is identical for all three runs:

```text
FD883350CBF126731443EFE7796A5FA8DC6C8D1BA08BEA03A6B445B43A90CA5C
```

## 3. Three-seed aggregate results

Values are arithmetic means across the three checkpoint seeds. Percentages are reported as percentages; latency is in milliseconds.

| Method | Safe capture | Collision | Valid plan | Effective plan | Convergence | Planner p95 | Planner p99 | Total p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| worst-case centralized oracle | 100.00% | 0.00% | 100.00% | 100.00% | n/a | 13.17 | 14.26 | 56.76 |
| distributed ideal | 100.00% | 0.00% | 100.00% | 99.05% | 99.05% | 623.73 | 685.85 | 660.85 |
| distributed delayed | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% | 445.06 | 503.93 | 484.34 |
| distributed dropout | 100.00% | 0.00% | 100.00% | 99.93% | 100.00% | 451.07 | 545.02 | 490.02 |
| distributed none | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% | 257.79 | 272.67 | 294.91 |
| DynamicEncirclement | 91.67% | 8.33% | n/a | n/a | n/a | n/a | n/a | 1.88 |

The per-seed safety result is directionally identical: every distributed mode is 12/12 safe captures with no collision for seeds `745101`, `745201` and `745301`. The ideal mode has effective-plan rates of 98.90%, 99.34% and 98.90%; the remaining modes are at or above 99.78% effective planning in each seed. Valid planner rate and solver success rate are 100% for all distributed modes.

## 4. Communication audit

| Mode | Mean attempted | Mean sent | Mean received | Mean dropped | Mean bytes sent | Mean message age |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ideal | 17,156 | 17,156 | 17,156 | 0 | 4,117,440 | 0.00 steps |
| delayed | 15,756 | 15,756 | 14,892 | 0 | 3,781,440 | 1.89 steps |
| dropout | 15,760 | 14,180 | 13,397 | 1,580 | 3,403,120 | 2.04 steps |
| none | 0 | 0 | 0 | 0 | 0 | 0.00 steps |

The dropout counts are deterministic under the recorded evaluation seeds. Delayed messages show the configured two-step maximum age in the step logs. The no-communication mode still runs the local planner; it only disables peer-message exchange.

## 5. Gate review

| Gate | Result | Assessment |
| --- | --- | --- |
| Explicit distributed information contract | implemented | Passed for this implementation |
| Sequential best-response local planner | implemented | Passed as a first mechanism |
| Same-scene centralized oracle comparison | completed | Safety metrics match the oracle on S3 |
| Delayed/noisy and dropout safety robustness | completed | No safe-capture or collision degradation observed |
| Message, age, failure and fallback audit | completed | JSONL, TensorBoard and summary artifacts present |
| Valid planner or recorded fallback | passed | Valid plan rate and solver success are 100%; dropout fallback is recorded |
| Planner p95 <= 100 ms | failed | Every distributed mode exceeds the budget |
| Locked-test and unseen adaptive target generalization | not run | Cannot support a generalization claim |

The communication robustness part of P4 is therefore positive, but the real-time gate is a hard failure. This is not a complete P4 pass. No result in this report should be described as a formal distributed-game guarantee or as real UAV deployment evidence.

## 6. Why the real-time gate failed

The current implementation evaluates multiple local candidate action sequences across up to three sequential best-response iterations. Communication itself is not the dominant issue: the no-communication mode still has a mean planner p95 of 257.79 ms, while ideal communication rises to 623.73 ms. The likely bottlenecks are repeated local candidate evaluation, role-variant enumeration and per-step Python-side finite-shooting work.

The next performance experiment should be pre-registered before changing the S3 result: profile each local stage, reduce candidate branching with a fixed rule, add warm starts, evaluate parallel local subproblems where valid, and compare a low-frequency planner/high-frequency safety executor. Any new configuration must retain the same scenes, checkpoint seeds and communication audit.

## 7. Reproducibility artifacts

The three runs are stored under:

- `results/phase4_dn_mpc_s3_seed745101`
- `results/phase4_dn_mpc_s3_seed745201`
- `results/phase4_dn_mpc_s3_seed745301`

Each run contains `config.yaml`, `protocol.yaml`, the shared `scenes.jsonl`, per-method `episodes.jsonl`, `steps.jsonl`, `summary.json` and TensorBoard event files. The implementation and tests are:

- `src/encirclement3d/distributed_dn_mpc.py`
- `scripts/evaluate_minimax_mpc.py`
- `scripts/evaluate_minimax_mpc_s3.py`
- `configs/phase4_dn_mpc.yaml`
- `tests/test_distributed_dn_mpc.py`

The acceptance checklist is maintained in `docs/INNOVATIVE_METHOD_FULL_PLAN.md` and `docs/INNOVATIVE_METHOD_TODOLIST.md`.

## 8. Scope limits and next decision

This report only validates the S3 validation protocol. It does not cover locked-test scenes, unseen adaptive target policies, a unified higher-order flight dynamics model, R-CLBF-QP, an independent safety certificate checker or end-to-end three-seed deployment.

Decision: retain centralized Scenario MPC as the current primary planning contribution; retain distributed DN-MPC as an implemented and experimentally supported communication-robustness diagnostic; do not claim the complete Mamba-Diffusion + DN-MPC + R-CLBF-QP method has passed.
