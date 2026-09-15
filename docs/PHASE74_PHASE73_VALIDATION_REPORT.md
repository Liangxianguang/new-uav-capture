# Phase 74: validation of prediction backbones and prior ablations on target-crossing scenes

## Scope and protocol

Phase 74 evaluates the released prediction backbones and the previously
implemented planner/safety ablations on the same frozen Phase 73
`development_validation` block. The block contains 200 scenes and 100
complete mirror groups. In every scene the direct target route is blocked by a
central obstacle and at least two lateral bypass routes are certified during
generation. The `external_holdout` block and every locked-test artifact remain
closed.

The source dataset manifest SHA-256 is
`10c7ac7f67f55bf077d7b1218229e26afacf28e855762e7efdd6f513a3160f02`. The
closed-loop evaluator normalizes the nested Phase 73 records into a run-local
scene file; the resulting in-run scene hash used by the aggregate artifacts
is `7a89920d9c53310cd1857064ec6c7472cea6713b61fd8d8199dce92029e01c32`.
All intervals below use mirror-group bootstrap with 10,000 samples and the
three predictor seeds `727201/727202/727203`, unless a one-seed ablation is
explicitly marked.

The nominal diagnostic uses the Phase 70 environment with physical execution
disabled, so it isolates the obstacle-crossing and communication/planning
behavior. The strict execution diagnostic enables a 4-step command delay and
bounded command noise with standard deviation `0.08 m/s` and bound `3 sigma`.
The strict condition is the relevant test for delayed execution robustness;
it is not a hard 100 ms requirement.

## Backbone validation

### Nominal physical-execution diagnostic

| Backbone | Safe capture | Collision | Boundary violation | Timeout | Total latency p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU `both` | 10.83% [8.16, 13.83] | 89.17% | 89.17% | 0% | 20.50 / 31.00 / 36.73 |
| Diagonal SSM `both` | 10.67% [8.00, 13.67] | 89.33% | 89.33% | 0% | 24.71 / 35.80 / 41.06 |
| Official S4 `both` | 10.83% [8.00, 13.67] | 89.17% | 89.17% | 0% | 37.86 / 53.51 / 61.78 |

The paired safe-capture difference is `-0.17 pp [-1.00, +0.67]` for
Diagonal SSM versus GRU and `0.00 pp [-0.83, +0.83]` for Official S4 versus
GRU. Thus this scene block provides no evidence that changing the temporal
backbone repairs the crossing failure.

The stage latency quantiles are:

| Backbone | Predictor p50/p95/p99 | Planner p50/p95/p99 | QDR/tube p50/p95/p99 | Safety p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| GRU | 7.64 / 14.71 / 16.41 | 8.82 / 15.19 / 17.90 | 0 / 0 / 0 | 1.61 / 3.10 / 4.04 |
| Diagonal SSM | 8.68 / 14.04 / 16.52 | 12.76 / 20.43 / 24.97 | 0 / 0 / 0 | 1.26 / 2.32 / 3.09 |
| Official S4 | 13.57 / 21.88 / 25.53 | 19.69 / 31.03 / 36.73 | 0 / 0 / 0 | 1.99 / 3.79 / 4.75 |

### Strict delayed-execution diagnostic

| Backbone | Safe capture | Collision | Boundary violation | Timeout | Total latency p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU `both` | 0% | 100% | 55.17% [50.00, 60.50] | 0% | 25.77 / 38.41 / 44.12 |
| Diagonal SSM `both` | 0% | 100% | 55.50% [50.17, 60.83] | 0% | 39.18 / 56.81 / 64.61 |
| Official S4 `both` | 0% | 100% | 55.50% [50.17, 60.67] | 0% | 27.50 / 38.64 / 43.95 |

The strict condition produces the same outcome for all three backbones. The
main bottleneck is therefore the coupled planner--execution--safety contract,
not a demonstrated lack of capacity in GRU, SSM, or S4. Official S4 uses the
upstream implementation, but the local CPU run reports the existing slow
structured-kernel fallback warning; these numbers must not be presented as
deployment throughput.

## Ablation validation

### Safety-layer ablation

On the nominal diagnostic, GRU `both` with local CBF has `10.83%` safe capture
and `89.17%` collision. Removing the safety layer gives `0%` safe capture and
`100%` collision. The paired safe-capture change for no-CBF is
`-10.83 pp [-13.67, -8.17]`, and the collision change is
`+10.83 pp [+8.00, +13.83]`. This is evidence that the local filter changes
failure behavior, not a formal safety guarantee. The local CBF remains an
empirical filter; no R-CLBF-QP or robust-CBF-QP safety proof is claimed.

### Action conditioning

This is a one-seed paired diagnostic (`727201`) on the same nominal block.

| Condition | Safe capture | Collision | Total latency p50/p95/p99 (ms) | Paired safe-capture delta vs `both` |
| --- | ---: | ---: | ---: | ---: |
| `both` | 10.5% [6.0, 15.5] | 89.5% | 20.41 / 31.06 / 37.09 | reference |
| `none` | 11.5% [7.0, 17.0] | 88.5% | 25.39 / 36.27 / 42.10 | +1.00 pp [0.00, +2.50] |
| `history` | 11.0% [6.5, 16.0] | 89.0% | 25.12 / 36.02 / 42.52 | +0.50 pp [0.00, +1.50] |
| `future` | 11.0% [6.5, 16.0] | 89.0% | 26.19 / 37.01 / 43.48 | +0.50 pp [0.00, +1.50] |

The intervals touch zero and the comparison has only one training seed, so
action conditioning is implemented and audited but does not establish a
closed-loop benefit here.

### QDR, UAKR, and RNIC under strict delayed execution

These arms use Diagonal SSM `both`, all three predictor seeds, and the strict
4-step/0.08 m/s execution condition.

| Arm | Safe capture | Collision | Boundary violation | Total latency p50/p95/p99 (ms) | Module diagnostic |
| --- | ---: | ---: | ---: | ---: | --- |
| Fixed distributed delayed | 0% | 100% | 55.17% | 39.18 / 56.81 / 64.61 | reference strict backbone |
| QDR `M6` | 7.00% [5.00, 9.17] | 93.00% | 93.00% | 51.03 / 82.15 / 94.60 | queue length 4; suffix-gate exhaustion about 14–15% |
| UAKR | 0% | 100% | 55.50% | 43.47 / 66.13 / 75.61 | mean K about 7.00; refresh interval about 1.25 steps |
| RNIC | 0% | 100% | 55.50% | 54.68 / 79.90 / 91.18 | unreachable-slot ratio about 0.858; RNIC p50/p95/p99 0.99/1.50/1.72 ms |

QDR is the only new module with a positive point estimate relative to the
strict fixed distributed arm, but `93%` collision remains far from a usable
interceptor. UAKR is almost always in the high-K bucket and therefore does not
deliver its intended compute saving. RNIC adds cost while its feasibility
diagnostic is poor. These are validation diagnostics, not promotion evidence.

### Delayed/tube/synchronous/asynchronous planner baselines

This one-seed (`727201`) nominal diagnostic uses the same SSM checkpoint and
200-scene block.

| Baseline | Safe capture | Collision | Boundary violation | Total latency p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| M0 current-state delayed | 11.5% | 88.5% | 88.5% | 19.74 / 39.51 / 51.63 |
| M2 fixed tube | 11.5% | 88.5% | 88.5% | 34.72 / 54.32 / 63.02 |
| M4 synchronous distributed | 11.0% | 89.0% | 89.0% | 24.92 / 37.46 / 42.22 |
| M5 asynchronous distributed | 11.0% | 89.0% | 89.0% | 18.47 / 26.13 / 33.73 |

The stronger baseline interfaces execute successfully, but none solves the
crossing task in this block. Pure-pursuit and dynamic-encirclement belief
baselines are also weak (`12.0%` and `10.5%` safe capture respectively) and
therefore do not explain the failure as a predictor-only artifact.

## Decision

1. The Phase 73 scene contract is effective at testing genuine obstacle
   bypass, and it exposes a large generalization gap relative to the earlier
   same-side/locked-test distribution. The historical locked-test results are
   preserved and are not overwritten by this validation.
2. No backbone or current ablation meets a reasonable effective-interception
   claim on the strict delayed-execution crossing task. QDR is a promising
   failure-reduction direction but is not yet a successful solution.
3. Do not open `external_holdout` or retune on the locked test. First perform
   development-only failure attribution (target contact vs obstacle/boundary,
   queue state at failure, selected bypass side, and first unsafe command),
   then retrain/fine-tune the defender on crossing scenes and re-run the same
   validation protocol.
4. Only after the strict validation passes pre-registered capture, collision,
   timeout, and queue-liveness gates should the untouched external holdout be
   opened once for confirmation.

Generated JSONL, summaries, checkpoints, and TensorBoard files remain under
ignored `results/phase74*` directories. The versioned reproducibility matrix
is `configs/phase74_phase73_validation_matrix.yaml`.
