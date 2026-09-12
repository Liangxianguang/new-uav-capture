# Phase 27 RNIC Formation-Slot Pilot

> Status: implementation and validation pilot complete; promotion as a main
> performance claim is pending multi-seed confirmation.

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

## Interpretation

This pilot supports three limited conclusions:

1. The formation-slot RNIC definition is executable, deterministic, unit
   tested and auditable through assignment and slack logs.
2. On this frozen validation block, the new objective changed the selected
   rollout in a favorable direction and improved clearance while the
   interceptor-only term was behavior-neutral.
3. The result is not yet evidence that RNIC improves the three-seed main
   model, the distributed delayed setting, OOD robustness, or safety. In
   particular, it does not repair the previously observed QDR and robust
   CBF-QP contract failures.

## Decision and next experiments

- **Implementation gate: pass.** Formation-slot RNIC has a bounded solver-free
  assignment rule, invalid-input checks, centralized planner integration,
  distributed team-score integration, evaluator CLI overrides and TensorBoard
  fields.
- **Promotion gate: pending.** Do not replace the Phase 16 main reference or
  claim an RNIC gain from this 20-episode pilot.
- **Next:** freeze the slot radius and weight on a separate calibration block;
  run three seeds on fresh validation scenes with RNIC off, interceptor RNIC
  and formation-slot RNIC; then run the same frozen choice on geometry,
  target-speed and delay/execution OOD blocks.
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
