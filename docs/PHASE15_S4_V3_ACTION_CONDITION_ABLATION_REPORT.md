# Phase 15 S4-v3 Action-Conditioning Ablation

> Decision: `both` is selected as the primary action-conditioned GRU
> configuration on validation. This selection uses validation histories only;
> no locked-test artifact is read by the aggregation command.

## Protocol

- Dataset: `phase15_s4_branching_predictor_v3`
- Train/validation scenes: 420 / 90; the 90-scene locked test is held out.
- Model: GRU Gaussian predictor, hidden width 32, one recurrent layer.
- Training budget: 30 epochs, batch size 128, learning rate `1e-3`.
- Conditions: `none`, `history`, `future`, and `both`.
- Matched training seeds: `727201`, `727202`, `727203`.
- Target normalization: training-split per-horizon coordinate standardization.
- Validation selection: final epoch metrics, with seed-level percentile bootstrap
  intervals.

`history` appends historical executed defender actions to the observation
sequence. `future` supplies the logged future planned-action condition. `both`
uses both. The logged future condition is an offline training/evaluation field;
it is not future target truth and it is not automatically available to a real
online controller. The online evaluator therefore uses the previous causal
DN-MPC plan as its future condition and records its availability separately.

## Validation Results

| Condition | minFDE (m) | minADE (m) | Energy score | Calibrated full-trajectory coverage | Predictor p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 1.0288 +/- 0.0101 | 0.6039 +/- 0.0021 | 2.4212 +/- 0.0104 | 90.56% +/- 2.11% | 1.786 |
| history | 0.9994 +/- 0.0076 | 0.5876 +/- 0.0040 | 2.3559 +/- 0.0139 | 92.09% +/- 0.79% | 1.215 |
| future | 0.8388 +/- 0.0099 | 0.5000 +/- 0.0078 | 2.0017 +/- 0.0289 | 93.68% +/- 0.54% | 1.812 |
| both | **0.8000 +/- 0.0064** | **0.4793 +/- 0.0049** | **1.9212 +/- 0.0186** | 93.38% +/- 1.12% | 1.764 |

The intervals in this table are descriptive bootstrap intervals over three
training seeds. With only three seeds they should not be treated as a
population-level uncertainty estimate.

## Paired Selection Evidence

The paired difference is candidate minus `both`; positive minFDE/minADE means
that the candidate is worse than the selected condition.

| Candidate | minFDE delta (m) | minADE delta (m) | Coverage delta |
| --- | ---: | ---: | ---: |
| none | +0.2288 `[+0.2150, +0.2409]` | +0.1246 `[+0.1179, +0.1283]` | -2.82 pp `[-4.05, -2.02]` |
| history | +0.1994 `[+0.1985, +0.2008]` | +0.1083 `[+0.1053, +0.1127]` | -1.29 pp `[-3.13, +0.64]` |
| future | +0.0388 `[+0.0340, +0.0477]` | +0.0207 `[+0.0159, +0.0262]` | +0.31 pp `[-1.10, +1.29]` |

The result supports retaining historical executed actions and the causal
future-plan interface together. It does not isolate a S4 contribution, because
this experiment uses a GRU and only changes the action feature contract.

## Feasibility and Calibration Caveat

The reported `candidate_feasible_fraction` in the training histories is a raw,
unprojected trajectory diagnostic. The direct GRU output can violate the
acceleration constraint even when its prediction error is low. The selected
`both` checkpoint must therefore be projected before it enters DN-MPC, and the
projected feasibility rate must be reported in the next locked-test and closed-
loop runs. Conformal coverage is a separate marginal calibration diagnostic; it
is not a collision or safety certificate.

The machine-readable validation artifact is generated locally at
`results/phase15_s4_v3_validation_action_condition_ablation.json` and can be
reproduced with:

```powershell
python scripts\aggregate_prediction_validation_results.py `
  --condition "none=models/phase15_s4_v3_ablation_gru_none_seed*" `
  --condition "history=models/phase15_s4_v3_ablation_gru_history_seed*" `
  --condition "future=models/phase15_s4_v3_ablation_gru_future_seed*" `
  --condition "both=models/phase15_s4_v3_ablation_gru_both_seed*" `
  --reference-condition both `
  --output results\phase15_s4_v3_validation_action_condition_ablation.json `
  --bootstrap-samples 10000 `
  --bootstrap-seed 20260911
```

## Locked-Test Confirmation

The frozen three-seed locked-test confirmation is complete. With projected
candidates, `both` obtains minFDE `0.7000 +/- 0.0057 m`, versus `0.7348 +/-
0.0101 m` for `future`, `1.0266 +/- 0.0140 m` for `history`, and `1.0615 +/-
0.0067 m` for `none`. The corresponding full-trajectory coverage is `94.59%`;
the candidate feasibility rate remains only `77.60%`. The all-condition
comparison is a pre-registered secondary result, not a second model-selection
step. The machine-readable artifact is
`results/phase15_s4_v3_action_condition_locked_test_aggregate.json`.

## Remaining Gates

1. Freeze `both` as the primary GRU action-conditioning contract; do not use
   the locked test to make further architecture or action-feature choices.
2. Run the selected predictor with no safety layer, local CBF, and robust
   CBF-QP diagnostic mode on identical scenes. Report ordinary capture,
   safe capture, collision, boundary, timeout, candidate age, certificate and
   component latency separately.
3. Only after this condition gate, compare single-mode versus genuinely
   multi-candidate diffusion. The current GRU result has one repeated candidate
   and therefore does not demonstrate multimodality.
