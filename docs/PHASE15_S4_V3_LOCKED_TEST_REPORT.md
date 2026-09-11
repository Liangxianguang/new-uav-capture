# Phase 15 S4-v3 Preliminary Locked-Test Report

> Historical one-seed, three-epoch pilot only. It is superseded for all Phase
> 15 decisions by `PHASE15_S4_V3_FORMAL_MULTISEED_REPORT.md`, which contains
> the 30-epoch, three-seed offline and causal online closed-loop results.

## Protocol

- Dataset: `phase15_s4_branching_predictor_v3`
- Scenes: 420 train / 90 validation / 90 locked-test
- Locked-test windows: 1,539
- Validation calibration: first half of validation episode seeds, 835 windows
- History: 16 steps; prediction horizon: 12 steps
- Action contract: history action features + future planned-action conditions
- Sampling: 4 candidates, 4 diffusion steps, fixed sampling seed `745102`
- Candidate projection: 4 iterations using the checkpoint dynamics constraints
- Device: CPU

The results below are frozen evaluation artifacts under
`results/phase15_s4_v3_locked_test/`. They are not training-set metrics.

## Results

| Model | Backend | Raw minADE (m) | Raw minFDE (m) | Projected minADE (m) | Projected minFDE (m) | Projected coverage | Projected feasible candidates | p50 (ms) | p95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GRU action | `gru_gaussian` | 0.5630 | 0.9404 | **0.4936** | **0.9286** | **92.66%** | 77.71% | 0.88 | 1.52 |
| Diagonal diffusion action | `portable_diagonal_ssm` | 1.0993 | 1.3946 | 0.6398 | 1.3480 | 82.78% | **84.11%** | 2.80 | 3.70 |
| S4-DPLR action | `s4_dplr_dense_reference` | 1.0989 | 1.3890 | 0.6425 | 1.3551 | 83.37% | 84.10% | 3.67 | 4.71 |
| Official S4 action | `official_state_spaces_s4_dplr` | 1.0939 | 1.3800 | 0.6401 | 1.3492 | 83.17% | 83.97% | 2.04 | 2.54 |

All three models use `both` action conditioning. The current S4-DPLR
reference is not the upstream `state-spaces/s4` implementation.

## Interpretation

1. The action feature contract is active: the evaluator reports 12 history
   action features and 12 future action-condition features for every model.
2. Projection makes the diffusion candidates dynamically feasible, but the
   resulting feasible-candidate rate is only 84.1% for both diffusion models;
   the recommended 95% feasibility gate is not met.
3. The current GRU checkpoint is the strongest predictor in this preliminary
   locked-test run. This is not evidence that GRU is generally superior because
   all models were only trained for three CPU epochs with hidden size 32.
4. The dense S4-DPLR reference is statistically indistinguishable from the
   diagonal diffusion in this single-seed run and is slightly worse after
   projection. An S4 contribution cannot be claimed from this result.
5. The official upstream S4 adapter has passed forward, backward, matched
   three-epoch training, checkpoint reload, and locked-test evaluation. With
   this single seed it is close to the dense S4-DPLR reference and does not
   beat the Diagonal SSM on projected minFDE or feasibility.

## Next gates

- Train GRU, Diagonal SSM, dense S4-DPLR, and upstream S4 with at least three
  independent seeds and retain the upstream source hash for every run.
- Run `none/history/future/both` action-conditioning ablations using validation
  only for selection.
- Add stratified metrics for observation condition, speed scale, branch sign,
  rollout policy, and mirror group.
- Improve or explicitly report projected candidate feasibility before DN-MPC
  integration; raw diffusion candidates remain diagnostic only.
- Integrate only the selected projected predictor into DN-MPC after the
  prediction gate is passed. Do not use this report as an end-to-end capture
  result.
