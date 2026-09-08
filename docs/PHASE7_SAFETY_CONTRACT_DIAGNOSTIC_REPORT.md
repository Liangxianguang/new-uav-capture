# Phase 7.1 Safety Contract Diagnostic

## Decision

The direct `worst_case DN-MPC + robust CBF-QP` composition remains **No-Go**.
The diagnostic rerun shows that the dominant failures are not planner failures:
the planner success, valid-plan, and effective-plan rates remain 100%. The
robust filter rejects or cannot solve a large fraction of states under the
current margin and action-change contract.

The formal diagnostic run is:

```text
results/phase7_joint_safety_locked_seed745101_robust_p4protocol_v2_diagnostics/
```

It uses the same 100 locked-test episodes, scene file, checkpoint, P4 MPC
configuration, and 20-step prediction cache as the formal P7 comparison. The
only source change relative to the first P7 run is richer failure logging.

## Failure Breakdown

| Quantity | Result |
| --- | ---: |
| Total control steps | 5,289 |
| Safety certificate valid | 64.87% |
| Safety fallback | 35.13% |
| Robust precondition valid | 73.45% |
| Planner solver/valid/effective | 100% / 100% / 100% |
| Safe capture | 50.00% |
| Collision | 49.00% |
| Boundary violation | 16.00% |
| Safety p95 | 11.95 ms |
| Total-control p95 | 135.75 ms |

Fallback categories over the 5,289 steps are:

| Category | Steps | Fraction |
| --- | ---: | ---: |
| `precondition_invalid` | 1,225 | 23.16% |
| `qp_infeasible` | 350 | 6.62% |
| `inconsistent_action_bounds` | 283 | 5.35% |

The category counts sum to all 1,858 fallback steps. The largest individual
solver message is `Positive directional derivative for linesearch` (344
steps), while the precondition failures are distributed across obstacle,
boundary, and inter-agent barriers. This means that simply increasing the
solver iteration limit is not a defensible fix: most failures occur before a
QP is attempted, and the state is already outside the contracted robust safe
set.

## Action-Bound Ablation

An exploratory run replaced the inscribed component-wise speed box with exact
per-defender L2 speed constraints implemented as SLSQP nonlinear constraints:

```text
results/phase7_joint_safety_locked_seed745101_robust_p4protocol_v3_l2speed/
```

This was rejected as an implementation path. It produced only `1%` safe
capture and `99%` collision, with `42.33%` fallback and `149.00 ms` total
control p95. The experiment is retained as a failed ablation, not as a formal
method result. The source was restored to the validated linear-QP path after
the regression was confirmed.

The lesson is precise: replacing the conservative speed-box approximation
with a generic nonlinear solver is not sufficient. A future repair must use a
dedicated convex speed-constrained formulation or a rigorously certified
projection while preserving the independent certificate and action-change
contract.

## Reproducibility and Tracking

Both diagnostic runs retain `config.yaml`, `summary.json`, episode/step JSONL,
source hashes, and TensorBoard events. The richer run records safety status,
solver message, failure category, fallback reason, precondition validity,
recovery usage, slack, active-constraint count, and recoverability fields in
the step records. Category counts are also written as TensorBoard text
summaries.

The P7 conclusion therefore stays unchanged: retain P4 prediction and DN-MPC
as modular evidence, stop direct end-to-end stacking, and repair the safety
reset/margin/fallback contract before reopening a multi-seed end-to-end gate.
