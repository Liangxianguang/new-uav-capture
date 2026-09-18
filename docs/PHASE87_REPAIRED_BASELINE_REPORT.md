# Phase 87 repaired baseline validation

This report records the first Phase 87 baseline diagnostic after all six Phase
86 expert-calibration blocks passed. The evaluation remains development-only;
no locked-test scenes were used.

## Gate status entering Phase 87

The six repaired blocks satisfy the Phase 86 target contract and the
oracle/public-belief acceptance threshold. `route_exercise` passed at 84% and
82.5%; `execution_delay` passed at 95% and 96% after the queue-aware projected
state repair.

## CBF-off diagnostic

The `target_speed` fixed manifest was rerun with local CBF disabled, using the
same scene file and environment contract as the repaired expert evaluation.

| Controller | Acceptance | Safe capture in pursuit | Physical collision | Defender boundary | Target invalid | Target boundary | Target obstacle |
|---|---:|---:|---:|---:|---:|---:|---:|
| Oracle route | 20% | 20% | 67% | 13% | 0% | 0% | 0% |
| Public-belief route | 19% | 19% | 76% | 5% | 0% | 0% | 0% |

The diagnostic confirms that target validity is independent from defender
safety failure: all target-contract rates remain zero while removing the local
empirical CBF causes defender collisions. The CBF-on repaired results remain
the primary Phase 87 baseline; the filter is an empirical safety filter only,
not a formal robust-CBF/QP proof.

The CBF-on expert matrix is represented by the Phase 86 repaired summaries:
the four ordinary blocks, the fixed 200-scene route-exercise block, and the
fixed 100-scene execution-delay block. Their acceptance rates are all at
least 80%, and their target-invalid, target-boundary, and target-obstacle
rates are zero for both controllers.

Artifacts:

- `results/phase87_repaired_baseline/target_speed_cbf_off/oracle_route/summary.json`
- `results/phase87_repaired_baseline/target_speed_cbf_off/public_belief_route/summary.json`

The Phase 83 and Phase 79 learned checkpoints require a separate adapter to
consume the repaired Phase 86 scene contract. Their historical reports are not
substituted for this validation. Legal-trajectory training, Hard/Stress
promotion, CBF ablations beyond this diagnostic, and locked-test diagnostics
remain closed until the current-checkpoint adapter is run and recorded.
