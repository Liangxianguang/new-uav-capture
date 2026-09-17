# Phase 86 nominal-repaired calibration report

Date: 2026-09-17

Scope: development calibration only. No locked-test or external-holdout result was read or used for tuning. This is a nominal-block milestone, not completion of all Phase 86 single-factor blocks.

## Contract and fixed inputs

- Scene pool: `results/phase86_nominal_repaired_development_calibration/scenes.jsonl`
- Episodes: 300 in 150 complete mirror groups
- Scene SHA-256: `a631c960873be35dafd025eaf575a3f5798514e70775468212cdf6d885c58b71`
- Manifest SHA-256: `af22742aaadb5026a7515201eef8fff7ba9a2c9180f00a94e1207e305c64b6d0`
- Environment: `configs/phase85_target_contract_repaired_environment.yaml`
- Experiment protocol: `configs/phase86_nominal_repaired_formal.yaml`
- Target mode: `adaptive_maneuvering`, target speed scales 0.50 and 0.60
- Obstacles: three per scene
- Local CBF: enabled as an empirical filter only; this result makes no formal robust-CBF/QP claim.

The raw scene pool was never removed or rewritten. Failed result directories were retained. Target-invalid episodes are excluded by the acceptance contract and are not eligible policy data.

## Failed iterations and diagnosis

The first 300-episode pass reported 12/300 target-invalid episodes for the oracle controller and 19/300 for public belief. These were `target_candidate_invalid` events rather than observed boundary crossings. The candidate evaluator incorrectly promoted a long-horizon 0.60 m preference margin into an immediate physical-invalid label even when the next realized step had positive obstacle and boundary clearance.

After separating preference-margin rejection from physical validity, v2 reduced the failure but did not close the contract: oracle had 5/300 target-obstacle violations and public belief had 6/300. Trace inspection showed that all regular candidates could become infeasible while the target still had low enough speed to brake. Because the candidate set had no braking action, the target continued accelerating evasively until collision became unavoidable. Predictive boundary handling also over-pruned the candidate set in this state.

The final repair therefore:

- replans after every infeasible selection instead of holding a rejected maneuver;
- ranks physically safe fallbacks by feasible prefix and immediate obstacle clearance;
- forces boundary recovery only when that recovery trajectory is itself feasible;
- injects a zero-speed `obstacle_recovery` braking candidate only when every normal non-trivial maneuver is horizon-infeasible;
- replans every step during obstacle recovery so normal evasion resumes as soon as feasible.

Focused regression on historical episodes 3, 70, 91, and 166 produced zero target-invalid, target-boundary, and target-obstacle events under both controllers. Public-belief episode 70 still ended in defender physical collision, which is intentionally accounted separately and does not invalidate the target contract.

## Final v3 calibration result

| Metric | Oracle route | Public-belief route | Gate |
|---|---:|---:|---:|
| Episodes / mirror groups | 300 / 150 | 300 / 150 | >=300 / >=150 |
| Expert acceptance | 98.33% (295/300) | 96.00% (288/300) | >=80% |
| Safe capture | 98.33% | 96.00% | reported |
| Target-invalid | 0.00% | 0.00% | 0% |
| Target boundary violation | 0.00% | 0.00% | 0% |
| Target obstacle collision | 0.00% | 0.00% | 0% |
| Defender physical collision | 1.33% | 3.67% | separate diagnostic |
| Defender boundary violation | 0.00% | 0.00% | separate diagnostic |
| Timeout | 0.33% | 0.33% | reported |
| Target maneuver fallback | 0.67% | 1.00% | diagnostic |
| Route exercise accepted | 13.33% | 17.67% | diagnostic |

The public-belief acceptance gap from oracle is 2.33 percentage points. Both controllers pass the nominal calibration target-validity and acceptance gates. Low fallback incidence, adaptive-maneuvering configuration, and tests that prohibit recovery injection when a regular feasible candidate exists establish that braking is exceptional rather than the default target policy. The current episode schema does not store the complete maneuver-mode sequence, so this is not claimed as a full quantitative diversity certificate.

## Reproduction

```powershell
python scripts\evaluate_phase77_expert_calibration.py `
  --scenes results\phase86_nominal_repaired_development_calibration\scenes.jsonl `
  --environment-config configs\phase85_target_contract_repaired_environment.yaml `
  --episodes 300 --controller oracle_route `
  --experiment-name phase86_nominal_repaired_calibration_oracle_v3 `
  --output-dir results\phase86_nominal_repaired_development_calibration\oracle_v3

python scripts\evaluate_phase77_expert_calibration.py `
  --scenes results\phase86_nominal_repaired_development_calibration\scenes.jsonl `
  --environment-config configs\phase85_target_contract_repaired_environment.yaml `
  --episodes 300 --controller public_belief_route `
  --experiment-name phase86_nominal_repaired_calibration_public_v3 `
  --output-dir results\phase86_nominal_repaired_development_calibration\public_v3
```

## Decision and next gate

`nominal_repaired` development calibration is **Go**.

## Independent development validation

The same fixed environment and target parameters were evaluated without tuning on the separately seeded validation pool:

- Scene pool: `results/phase86_nominal_repaired_development_validation/scenes.jsonl`
- Episodes: 300 in 150 complete mirror groups
- Scene SHA-256: `b903cf5fb6819a1ad970ce035b9cc9b25236e316c30fcf2b4e8e19401758869b`
- Manifest SHA-256: `86cf8e451f06f2d96bb961c66cabb9d2da24ae8302b35250ce7d75292deb689c`

| Metric | Oracle route | Public-belief route | Gate |
|---|---:|---:|---:|
| Expert acceptance | 97.33% (292/300) | 96.33% (289/300) | >=80% |
| Target-invalid | 0.00% | 0.00% | 0% |
| Target boundary violation | 0.00% | 0.00% | 0% |
| Target obstacle collision | 0.00% | 0.00% | 0% |
| Defender physical collision | 2.00% | 3.00% | separate diagnostic |
| Defender boundary violation | 0.00% | 0.00% | separate diagnostic |
| Timeout | 0.67% | 0.67% | reported |
| Target maneuver fallback | 0.67% | 0.00% | diagnostic |
| Route exercise accepted | 16.67% | 21.67% | diagnostic |

The public-belief acceptance gap from oracle is 1.00 percentage point. The independent validation pool therefore passes the nominal target-validity and acceptance gates without a parameter change.

`nominal_repaired` development calibration and validation are **Go**. External holdout remains closed because the other required Phase 86 single-factor blocks (`target_speed`, `obstacle_near`, `formation_tight`, `route_exercise`, `execution_delay`, and `command_noise`) remain open work. No defender training should begin until those development blocks have been generated and validated under the same zero-target-invalid contract.
