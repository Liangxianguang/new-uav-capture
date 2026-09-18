# Phase 86 repaired single-factor development report

Date: 2026-09-18

Scope: development calibration only. No external-holdout or locked-test scene was read for this evaluation. Local CBF remains an empirical filter; this report makes no robust-CBF/QP or real-flight claim.

## Inputs and contract

Five newly generated blocks are retained in `results/phase86_single_factor_development/`: 100 episodes / 50 complete mirror groups per block. They use `configs/phase85_target_contract_repaired_environment.yaml` and `configs/phase86_single_factor_development.yaml`. Every generated scene passed the initial escape certificate.

The route-exercise block uses the retained Phase 73 **development-calibration** split only, adapted without geometry changes to the Phase 86 evaluator schema: 200 episodes / 100 mirror groups, target crossing required, direct route blocked, and at least two certified lateral bypass routes. Its source scene SHA-256 is `e06da0641f3154ef6cec96667144af84d25b8768e8a63c05324351f64deb1605`.

For all rows below, both oracle and public-belief rollouts record:

- `target_invalid_episode = 0%`;
- `target_boundary_violation = 0%`;
- `target_obstacle_collision = 0%`.

## Oracle and public-belief gate audit

| Block | Episodes / groups | Oracle acceptance | Public acceptance | Target contract | Decision |
|---|---:|---:|---:|---|---|
| target_speed (0.55/0.65) | 100 / 50 | 96% | 98% | pass | Go |
| obstacle_near (4.5/5.0 m) | 100 / 50 | 99% | 96% | pass | Go |
| formation_tight (y scale 0.65) | 100 / 50 | 99% | 99% | pass | Go |
| command_noise (0.030 m/s) | 100 / 50 | 95% | 95% | pass | Go |
| execution_delay (2 steps) | 100 / 50 | 76% | 73% | pass | No-Go |
| route_exercise | 200 / 100 | 97.5% | 97.5% | pass | Conditional No-Go |

The `execution_delay` failure is not a target-contract failure. It is caused by defender safety failures: oracle/public defender physical collision is 16%/19%, defender-boundary violation is 8%/8%, and the 80% expert acceptance gate is missed.

The route-exercise raw expert acceptance passes, but its stronger exercise-specific rate is 80% for oracle and 79% for public belief. Public belief therefore misses the predeclared 80% route-exercise gate by one episode. It may not be relabelled as a passed route-exercise block and it may not be mixed into training data without a separate, predeclared repair experiment.

## Decision

Four ordinary single-factor blocks pass the development expert gate and preserve the repaired target contract. `execution_delay` and strict public-belief `route_exercise` do not. Therefore Phase 87 current-checkpoint validation, legal-trajectory training, Hard/Stress closed-loop promotion, CBF ablations, external holdout, and locked-test diagnostics remain closed.

The next experiment is a targeted diagnosis of the two No-Go blocks, holding their scene manifests fixed and separating defender collision, defender boundary, queue delay, and target-validity fields. Do not retune the target boundary contract based on these defender-side failures.
