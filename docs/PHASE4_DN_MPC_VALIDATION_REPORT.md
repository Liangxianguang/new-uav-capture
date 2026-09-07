# Phase 4 Distributed DN-MPC Validation Report

> Version: v1.1 (2026-09-07)
> Repository: `https://github.com/Liangxianguang/new-uav-capture`
> Decision: `diagnostic_only`
> Scope: finite-candidate sequential best-response DN-MPC with bounded communication

## 1. Executive conclusion

Phase 4 is partially successful. The distributed implementation, information boundary, communication fault injection, fallback accounting and reproducible three-seed evaluation are working. On the fixed S3 validation set, all four distributed communication modes achieved 100% safe capture and 0% collision, matching the centralized worst-case oracle and exceeding the DynamicEncirclement baseline (91.7% safe capture, 8.3% collision).

The optimization pass restores the real-time budget: the mean planner p95 across the three checkpoint seeds is 8.47 ms for no communication, 10.07 ms for delayed communication, 10.27 ms for dropout communication and 12.42 ms for ideal communication. The overall P4 gate is still **not passed** because `distributed_ideal` has effective/converged plan rates of 98.90%, 99.34% and 98.90% across the three seeds; the pre-registered per-seed threshold is 99%. The current evidence supports a communication-robust, real-time implementation diagnostic, not a complete deployable DN-MPC claim.

The current main result remains the centralized Scenario MPC from Phase 3. P4 should next focus on the marginal ideal-mode convergence gap, then validate locked-test and unseen adaptive target policies before being promoted to the main DN-MPC result.

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
| worst-case centralized oracle | 100.00% | 0.00% | 100.00% | 100.00% | n/a | 11.10 | 13.95 | 47.89 |
| distributed ideal | 100.00% | 0.00% | 100.00% | 99.05% | 99.05% | 12.42 | 14.23 | 50.30 |
| distributed delayed | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% | 10.07 | 12.39 | 50.20 |
| distributed dropout | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% | 10.27 | 13.70 | 51.52 |
| distributed none | 100.00% | 0.00% | 100.00% | 100.00% | 100.00% | 8.47 | 11.56 | 49.98 |
| DynamicEncirclement | 91.67% | 8.33% | n/a | n/a | n/a | n/a | n/a | 1.58 |

The per-seed safety result is directionally identical: every distributed mode is 12/12 safe captures with no collision for seeds `745101`, `745201` and `745301`. The ideal mode has effective-plan rates of 98.90%, 99.34% and 98.90%; delayed, dropout and no-communication modes are 100% in each seed. Valid planner rate and solver success rate are 100% for all distributed modes.

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
| Valid planner or recorded fallback | partial | Valid plan rate and solver success are 100%; ideal-mode effective planning is 98.90%, 99.34% and 98.90% |
| Planner p95 <= 100 ms | passed | Optimized distributed modes have mean p95 of 8.47--12.42 ms; mean total p95 is 49.98--51.52 ms |
| Per-seed effective/converged plan rate >= 99% | failed | Ideal mode misses the threshold in two of three seeds by 0.10 percentage points |
| Locked-test and unseen adaptive target generalization | not run | Cannot support a generalization claim |

The communication safety and real-time parts of P4 are positive, but the per-seed effective-plan gate is not fully satisfied. This is not a complete P4 pass. No result in this report should be described as a formal distributed-game guarantee or as real UAV deployment evidence.

## 6. Performance optimization and remaining gap

The optimization pass vectorizes local scenario-cost evaluation across candidate sequences, target scenarios and horizon steps while preserving worst-case/expected/CVaR aggregation, obstacle and boundary penalties, inter-agent separation, formation terms, communication limits and fallback diagnostics. On the locked S3 validation protocol, this reduces the mean planner p95 to 8.47--12.42 ms and the mean total-control p95 to 49.98--51.52 ms across distributed modes.

The remaining experiment should be pre-registered before changing the S3 result: identify the source of the ideal-mode non-convergence, add a deterministic convergence or warm-start treatment, and repeat all three seeds. Any new configuration must retain the same scenes, checkpoint seeds and communication audit. Locked-test and unseen adaptive-target evaluation remain mandatory before a generalization claim.

## 7. Reproducibility artifacts

The optimized three runs are stored under:

- `results/phase4_dn_mpc_optimized_s3_12_seed745101`
- `results/phase4_dn_mpc_optimized_s3_12_seed745201`
- `results/phase4_dn_mpc_optimized_s3_12_seed745301`

Each run contains `config.yaml`, `protocol.yaml`, the shared `scenes.jsonl`, per-method `episodes.jsonl`, `steps.jsonl`, `summary.json` and TensorBoard event files. The implementation and tests are:

- `src/encirclement3d/distributed_dn_mpc.py`
- `scripts/evaluate_minimax_mpc.py`
- `scripts/evaluate_minimax_mpc_s3.py`
- `configs/phase4_dn_mpc.yaml`
- `tests/test_distributed_dn_mpc.py`

The acceptance checklist is maintained in `docs/INNOVATIVE_METHOD_FULL_PLAN.md` and `docs/INNOVATIVE_METHOD_TODOLIST.md`.

## 8. Scope limits and next decision

This report only validates the S3 validation protocol. It does not cover locked-test scenes, unseen adaptive target policies, a unified higher-order flight dynamics model, R-CLBF-QP, an independent safety certificate checker or end-to-end three-seed deployment.

Decision: retain centralized Scenario MPC as the current primary planning contribution; retain distributed DN-MPC as an implemented, real-time and experimentally supported communication-robustness diagnostic whose ideal-mode effective-plan gate is not yet fully passed; do not claim the complete Mamba-Diffusion + DN-MPC + R-CLBF-QP method has passed.
