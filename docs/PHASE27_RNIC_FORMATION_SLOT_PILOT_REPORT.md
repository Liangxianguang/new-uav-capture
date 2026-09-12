# Phase 27 RNIC Formation-Slot Pilot

> Status: multi-seed ID confirmation complete; the non-inferiority direction is
> supported, but superiority and OOD transfer remain unconfirmed.

## Purpose

The original RNIC term scored the earliest reachable interception of any one
defender. That definition does not explicitly represent encirclement: it can
prefer a fast single interceptor while leaving the remaining defenders unable
to occupy useful perimeter positions. Phase 27 adds a reproducible
`formation_slot` mode. At every prediction step, each defender is assigned to
one target-relative 3-D slot, and the assignment minimizing the sum of
normalized arrival-time shortfalls is selected by exact permutation
enumeration.

The implementation is intentionally bounded to at most six defenders and uses
the same acceleration/speed surrogate as the original RNIC. It is a planning
heuristic, not a dynamic reachable-set computation or a safety certificate.

## Protocol

- 20 frozen validation episodes from the Phase 17 comparison manifest;
- one fixed GRU `both` checkpoint, seed `727201`;
- identical scene order, episode seeds, checkpoint, sampling seed `745102`,
  one candidate sample, eight diffusion steps, local CBF and CPU execution;
- queue-aware rollout and UAKR disabled to isolate the RNIC cost definition;
- two-step command delay retained in the simulator contract;
- `worst_case` centralized DN-MPC, with RNIC disabled, interceptor RNIC, and
  formation-slot RNIC arms;
- one process per arm; predictor, planner, RNIC, safety and total-control
  p50/p95/p99 latency are retained;
- generated results remain Git-ignored and retain their own config, source
  hashes, JSONL logs and TensorBoard event files.

The frozen scene manifest SHA-256 is
`5ba54afc62d2cf123014f1c5e598a0e3e914e9067870bc7a199ba3393d8f8d99`.

## Closed-loop results

| Arm | Safe capture | Capture | Collision | Boundary | Timeout | Mean clearance (m) | Worst clearance (m) | Total p50 / p95 / p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RNIC off | 95.0% | 95.0% | 5.0% | 0.0% | 0.0% | 0.411 | -0.162 | 62.89 / 68.49 / 72.59 |
| Interceptor RNIC | 95.0% | 95.0% | 5.0% | 0.0% | 0.0% | 0.411 | -0.162 | 65.64 / 71.66 / 77.33 |
| Formation-slot RNIC | **100.0%** | **100.0%** | **0.0%** | 0.0% | 0.0% | **0.448** | **0.158** | 78.33 / 86.10 / 98.23 |

The formation arm rescued the same paired episode that collided under the
other two arms: 1/20 paired collision was removed, corresponding to a
descriptive `+5.0` percentage-point safe-capture change. The pilot is too small
and uses one checkpoint to make a significance or generalization claim.

The formation assignment was nontrivial: the mean step-level assignment
switch rate was `17.16%`. Its RNIC calculation latency was `14.07 / 15.52 /
16.68 ms` at p50/p95/p99, compared with `2.72 / 3.27 / 3.86 ms` for the
interceptor term. The formation arm's component latency was:

| Component | p50 / p95 / p99 (ms) |
| --- | ---: |
| Predictor | 34.57 / 38.34 / 44.48 |
| Planner | 35.61 / 38.82 / 43.76 |
| RNIC diagnostic | 14.07 / 15.52 / 16.68 |
| Local CBF | 3.75 / 4.26 / 4.74 |
| Total control | 78.33 / 86.10 / 98.23 |

The 100 ms value is only a deployment reference, not a hard acceptance gate.
The result must nevertheless be reported with all quantiles because the
formation enumeration has a clear computational cost.

## Fresh multi-seed ID confirmation

To avoid reusing the historical pilot block, a new validation-selection
manifest was generated before the confirmation runs. It contains 60 episodes
and 30 complete mirror groups, uses seed block `781101`, remains inside the
training-support geometry and motion ranges, and excludes locked-test. Its
SHA-256 is
`5aaaa79c6dbef2d346ff57d48d238a34fe911d3534ebb576dff0252ba0593022`.

The confirmation uses three frozen GRU `both` checkpoints (`727201`,
`727202`, `727203`) and the same scenes, sampling seed, local CBF, two-step
command delay and centralized `worst_case` protocol for all arms. The pooled
episode-level result is:

| Arm | Safe capture | Capture | Collision | Boundary | Timeout | Mean clearance (m) | Total p50 / p95 / p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RNIC off | 87.22% [81.11%, 92.78%] | 90.00% | 12.78% | 0.0% | 0.0% | 0.405 | 63.04 / 69.85 / 82.06 |
| Interceptor RNIC | 88.89% [82.22%, 95.00%] | 91.67% | 11.11% | 0.0% | 0.0% | 0.413 | 65.48 / 71.57 / 77.54 |
| Formation-slot RNIC | **89.44% [83.33%, 94.44%]** | **92.22%** | **10.56%** | 0.0% | 0.0% | **0.416** | **77.94 / 85.24 / 91.75** |

Relative to RNIC off, the formation-slot arm changes safe capture by
`+2.22 pp` with paired bootstrap 95% CI `[-0.56, +5.00] pp`, collision by
`-2.22 pp` with CI `[-5.00, +0.56] pp`, and mean clearance by `+0.0106 m`
with CI `[-0.0030, +0.0263] m`. The lower bound for the predeclared `-2 pp`
ID non-inferiority criterion is above `-2 pp`, so the ID non-inferiority
direction is supported. Because the safe-capture CI crosses zero, the data do
not establish superiority or a statistically reliable collision reduction.

The formation assignment remains nontrivial, with per-seed assignment-switch
rates around 13--15%. Its pooled component latency is:

| Component | p50 / p95 / p99 (ms) |
| --- | ---: |
| Predictor | 34.16 / 37.69 / 41.10 |
| Planner (including formation RNIC) | 35.58 / 39.11 / 42.54 |
| Local CBF | 3.74 / 4.24 / 4.87 |
| Total control | 77.94 / 85.24 / 91.75 |

The formation RNIC calculation itself is approximately 14.2 / 15.5 / 16.9
ms at p50/p95/p99 across the three runs. The latency increase is therefore a
real trade-off that must be included in the final ablation rather than hidden
inside the planner timing.

This confirmation is still centralized and `worst_case`; it is not a
distributed delayed-planning confirmation and it does not test geometry,
target-speed or delay/execution transfer. Those experiments remain open.

## Fresh distributed-delayed confirmation

The same frozen 60-episode manifest, three checkpoints, sampling seed and
two-step command-delay contract were then evaluated with
`distributed_delayed`. This isolates whether the formation objective survives
delayed peer messages rather than only centralized access to all defenders.

| Arm | Safe capture | Capture | Collision | Boundary | Timeout | Mean clearance (m) | Total p50 / p95 / p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RNIC off | 96.67% [93.89%, 98.89%] | 96.67% | 3.33% | 0.0% | 0.0% | 0.268 | 75.10 / 82.40 / 89.86 |
| Interceptor RNIC | 96.67% [93.89%, 98.89%] | 96.67% | 3.33% | 0.0% | 0.0% | 0.268 | 102.11 / 111.98 / 122.65 |
| Formation-slot RNIC | 96.67% [93.89%, 98.89%] | 96.67% | 3.33% | 0.0% | 0.0% | 0.268 | 205.76 / 220.87 / 235.10 |

All three arms produced the same episode outcomes on this confirmation:
formation-slot RNIC has paired safe-capture delta `0.00 pp` with CI
`[0.00, 0.00] pp` and paired clearance delta `+0.0000004 m` with CI
`[-0.0000011, +0.0000024] m`. Thus the centralized ID direction does not
transfer to a measurable distributed-delayed improvement under the current
team-score and message-availability contract. The formation assignment is
being evaluated—the mean assignment-switch rate is `24.51%`—but its effect is
not reflected in the selected controls or episode outcomes.

The pooled component latency is:

| Arm | Predictor p50/p95/p99 (ms) | Planner p50/p95/p99 (ms) | Safety p50/p95/p99 (ms) | Total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| RNIC off | 34.50 / 38.06 / 41.40 | 32.60 / 36.27 / 39.53 | 3.78 / 4.34 / 5.15 | 75.10 / 82.40 / 89.86 |
| Interceptor RNIC | 34.51 / 37.89 / 43.69 | 59.40 / 65.60 / 72.90 | 3.79 / 4.36 / 4.94 | 102.11 / 111.98 / 122.65 |
| Formation-slot RNIC | 34.58 / 37.90 / 42.14 | 162.87 / 176.11 / 188.12 | 3.74 / 4.29 / 4.93 | 205.76 / 220.87 / 235.10 |

For the formation arm, the pooled RNIC calculation latency is
`13.08/15.19/17.10 ms` at p50/p95/p99. Since the user does not impose a hard
100 ms gate, these numbers are reported as a systems trade-off; nevertheless,
the formation objective currently adds about `138.48 ms` to pooled total p95
relative to RNIC off in the distributed setting without a behavioral gain.

This is a valid negative result: the implementation and audit path work, but
the current distributed formation-slot objective should not be promoted as a
main performance contribution. Future work would need a communication-aware
local surrogate or an explicit consensus/assignment protocol, calibrated on a
development block before another confirmation.

## Interpretation

The pilot and fresh confirmation together support five limited conclusions:

1. The formation-slot RNIC definition is executable, deterministic, unit
   tested and auditable through assignment and slack logs.
2. On the historical pilot block, the new objective changed the selected
   rollout in a favorable direction and improved clearance while the
   interceptor-only term was behavior-neutral.
3. On fresh ID scenes and three seeds, formation-slot RNIC satisfies the
   predeclared safe-capture non-inferiority direction, but not a superiority
   claim; its computational cost is measurable.
4. In fresh distributed-delayed confirmation, all arms have identical episode
   outcomes while formation-slot RNIC adds substantial planning latency; this
   is a negative distributed transfer result under the current contract.
5. The result is not evidence of OOD robustness or safety. In particular, it
   does not repair the previously observed QDR and robust CBF-QP contract
   failures.

## Decision and next experiments

- **Implementation gate: pass.** Formation-slot RNIC has a bounded solver-free
  assignment rule, invalid-input checks, centralized planner integration,
  distributed team-score integration, evaluator CLI overrides and TensorBoard
  fields.
- **ID non-inferiority gate: directional pass.** The fresh three-seed result
  is not below the fixed reference by the predeclared 2 pp margin.
- **Superiority gate: no evidence.** Do not replace the Phase 16 main
  reference or claim an RNIC gain from the current confirmation.
- **Distributed transfer gate: no-go for the current implementation.** The
  formation objective produced no paired episode-level change and increased
  total p95 substantially.
- **Next:** retain this result as a negative ablation; only reopen RNIC after
  designing and calibrating a communication-aware local assignment surrogate.
  OOD transfer is optional diagnostic work, not a promotion step for the
  current formation objective.
- **Required ablations:** fixed assignment versus adaptive assignment,
  distance-only versus time-only versus RNIC, slot radius sensitivity chosen
  only on calibration, and centralized versus distributed delayed planning.
- **Required statistics:** paired safe-capture and collision differences,
  clearance, timeout, assignment-switch rate, planner/RNIC/total p50/p95/p99,
  and a failure decomposition for episodes where slot assignment is
  unreachable.

## Reproduction

The committed configuration is
`configs/phase27_rnic_formation_slot_pilot.yaml`.

The three validation arms were run with the same command template and only
the RNIC switch changed:

```powershell
python scripts/evaluate_s4_closed_loop.py `
  --scenes results/phase17_validation_qdr_off_diagonal_seed727201/scenes.jsonl `
  --protocol configs/phase15_s4_branching_pilot.yaml `
  --environment-config configs/capture_radius_pursuit_central_v4_flee.yaml `
  --mpc-config configs/phase27_rnic_formation_slot_pilot.yaml `
  --checkpoint results/phase16_gru_both_seed727201/checkpoint.pt `
  --candidate-source checkpoint --methods worst_case --episodes 20 `
  --num-samples 1 --sampling-steps 8 --sampling-seed 745102 `
  --projection-iterations 4 --prediction-refresh-interval-steps 1 `
  --device cpu --safety-layer local_cbf --decision validation_selection `
  --rnic --rnic-cost-mode formation_slot --rnic-slot-radius-m 1.4
```

Use `--no-rnic` for the reference and
`--rnic --rnic-cost-mode interceptor` for the original RNIC arm. The retained
local result directories are:

```text
results/phase27_rnic_off_seed727201/
results/phase27_rnic_interceptor_seed727201/
results/phase27_rnic_formation_seed727201/
```
