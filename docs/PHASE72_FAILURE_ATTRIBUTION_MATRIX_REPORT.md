# Phase 72: Maneuvering Adversary v2 failure-attribution matrix

## Scope and preservation

This is a validation-only diagnostic. It does not read, modify, tune, or
replay the locked test. All previous Phase 16--71 result directories and
checkpoints remain intact. New outputs use `results/phase72_*` directories and
are ignored by Git.

The matrix reuses the three cached Phase 71 scene blocks (`711201/711202/711203`,
20 episodes per block), so each arm sees the same geometry, target behavior,
observation condition, and seeds. The four arms are:

| Arm | Defender | local CBF |
| --- | --- | --- |
| A | released V5 | on |
| B | released V5 | off |
| C | dynamic rule expert | on |
| D | dynamic rule expert | off |

The original Phase 71 V5+CBF outputs are retained as arm A. Arms B--D are
newly evaluated in Phase 72. A second CBF-enabled replay adds explicit failure
attribution fields without changing the dynamics.

## Results

| Arm | Episodes | Safe capture | Collision/safety failure | Boundary failure | Mean minimum clearance |
| --- | ---: | ---: | ---: | ---: | ---: |
| V5 + CBF | 60 | 6.67% [1.67, 13.33] | 93.33% | 93.33% | 0.411 m |
| V5 - CBF | 60 | 1.67% [0, 5.00] | 98.33% | 28.33% | -0.028 m |
| Rule + CBF | 60 | 5.00% [0, 11.67] | 95.00% | 93.33% | 0.404 m |
| Rule - CBF | 60 | 0.00% [0, 0] | 100.00% | 36.67% | -0.063 m |

The intervals are episode-bootstrap 95% intervals from the existing
validation-only aggregation utility. In this evaluator, `collision` is an
aggregate safety-failure flag and includes world-boundary violations; it does
not by itself mean physical target-obstacle contact.

## Explicit failure attribution

The CBF-enabled attribution replay reports the following across the three
seed blocks:

| CBF arm | Physical defender/obstacle collision | Target boundary contact | Defender boundary contact | Target-obstacle collision |
| --- | ---: | ---: | ---: | ---: |
| V5 + CBF | 0/60 | 56/60 | 0/60 | 0/60 |
| Rule + CBF | 1/60 | 56/60 | 0/60 | 0/60 |

Thus the dominant Phase 71 failure is target escape to the world boundary,
not target hitting an obstacle and not the defenders being pushed through the
boundary by local CBF. Disabling CBF exposes a second failure mode: the
defender formation loses separation, with negative minimum clearance and
nearly universal safety failure.

The CBF correction is materially larger for the rule expert (approximately
`1.00--1.25` mean action-norm correction per seed) than for V5
(approximately `0.23` per seed). This indicates a controller--filter
interface mismatch, but the target-boundary failure remains the first-order
task bottleneck.

## Interpretation

1. Disabling local CBF is not a solution. It slightly changes descriptive
   capture counts but produces physical separation violations.
2. The current randomized Phase 71 contract places the target on the fleeing
   side near the `x=+10` boundary for left-side defenders. The target can
   reach that boundary before an intercept, while the scene generator only
   certifies a short transit route, not a finite-horizon adversary rollout.
3. The current `adaptive_maneuvering` target selects an obstacle-bypass label
   in some episodes, but `target_crossing_required=false` means many episodes
   do not require the target to cross the central obstacles. They are not a
   clean test of target-side obstacle bypass.
4. The expert data bottleneck is therefore downstream of a task-contract
   problem. More behavior-cloning epochs on the current failed labels should
   not be used as the next intervention.

## Reproducible artifacts

New validation-only artifacts include:

- `results/phase72_matrix_v5_seed711201_no_cbf/` through `711203`;
- `results/phase72_matrix_rule_seed711201_cbf/` through `711203`;
- `results/phase72_matrix_rule_seed711201_no_cbf/` through `711203`;
- `results/phase72_matrix_*_aggregate.json`;
- `results/phase72_attribution_v5_seed*_cbf/` and
  `results/phase72_attribution_rule_seed*_cbf/`;
- `results/phase72_attribution_*_aggregate.json`.

The source-side diagnostic fields are implemented in
`scripts/run_mixed_obstacle_showcase.py`:
`physical_collision`, `target_boundary_violation`,
`defender_boundary_violation`, and their step counts. The local CBF remains an
empirical filter only; no R-CLBF-QP, forward-invariance, or real-flight safety
claim is made.

## Decision and next action

Phase 72 P1 is complete. Do not retrain on the current mixed failed archive.
The next experiment is a new validation-only route-contract block:

1. set `target_crossing_required=true` for a fixed single-wall curriculum;
2. require the direct target path to be blocked and at least two bypass routes
   to be finite-horizon feasible;
3. add target-boundary and defender-boundary certificates to scene generation;
4. use a route-aware teacher and retain only safe/cooperative demonstrations;
5. repeat the same CBF on/off matrix before training a new defender.

Locked-test remains closed until this repaired validation contract produces a
quality-gated checkpoint across three seeds.
