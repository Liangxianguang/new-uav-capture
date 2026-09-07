# Phase 2 Formal Prediction Analysis

## 1. Decision

**Decision: Conditional Go.**

The portable SSM-conditioned diffusion predictor is reproducible and useful as a multimodal candidate generator, but it does not yet satisfy the stronger prediction gate required to claim a complete Phase 2 success. It may be used as input to a diagnostic centralized scenario MPC. DN-MPC and R-CLBF-QP remain unvalidated.

The main reason for the conditional decision is that the locked-test minFDE improvement over the GRU baseline is below the preregistered 10% threshold. The diffusion candidates are also not executable before a dynamics-consistency projection. Both facts are retained as explicit results rather than hidden by post-processing.

## 2. Protocol

- Dataset version: `v3_multimodal`, regenerated after the target hard-speed-limit fix.
- Train/validation/locked-test samples: `15,232 / 5,712 / 7,616`.
- Target modes: `flee_persistence`, `random_turn`, `s_curve`, `burst`, `boundary_escape`.
- Training seeds: `745101`, `745201`, `745301`.
- Models: 2-layer, 128-hidden GRU Gaussian baseline and 2-layer, 128-hidden portable diagonal SSM diffusion model.
- Training: 40 epochs, batch size 128, learning rate `1e-3`, train-split-only standardization.
- Diffusion sampling: 8 candidates, 8 denoising steps, fixed evaluation sampling seed `745102`.
- Calibration: split conformal full-trajectory maximum-distance radius fitted only on the first half of validation episodes.
- Physical candidate limits: target speed `3.6 m/s`, target acceleration `4.0 m/s^2`, recorded obstacle geometry and world bounds.
- Locked-test evaluation never uses locked-test labels for model selection or calibration.

## 3. Locked-Test Aggregate Results

Values are mean +/- sample standard deviation over the three training seeds. Lower is better for ADE, FDE and energy score; higher is better for coverage and candidate feasibility.

| Model and candidate stage | minADE | minFDE | Energy score | Full-trajectory coverage | Feasible candidate fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU, raw | 0.4187 +/- 0.0122 | 0.6825 +/- 0.0163 | 1.6402 +/- 0.0461 | 0.9006 +/- 0.0124 | 0.0782 +/- 0.0112 |
| Portable SSM diffusion, raw | 0.4325 +/- 0.0030 | **0.6379 +/- 0.0078** | 1.7242 +/- 0.0243 | 0.8926 +/- 0.0028 | **0.0000** |
| GRU, dynamics projected | 0.4128 +/- 0.0112 | 0.7075 +/- 0.0206 | 1.6456 +/- 0.0469 | 0.8962 +/- 0.0123 | 0.9884 +/- 0.0024 |
| Portable SSM diffusion, dynamics projected | **0.4161 +/- 0.0032** | **0.6802 +/- 0.0085** | 1.6918 +/- 0.0239 | 0.8874 +/- 0.0018 | **0.9971 +/- 0.0005** |

Relative to raw GRU, raw diffusion minFDE improves by `6.54%`. Relative to dynamics-projected GRU, projected diffusion minFDE improves by `3.87%`. Neither reaches the preregistered `10%` prediction gate. The diffusion candidate set is genuinely non-degenerate, with locked-test candidate spread approximately `0.172 +/- 0.001` in physical coordinates.

The average single-sample p95 latency is `1.85 +/- 0.51 ms` for GRU and `21.05 +/- 2.64 ms` for portable SSM diffusion on the RTX 4060 Laptop GPU. Both fit the 100 ms control-cycle budget in isolation; total planner and safety-layer latency is not yet measured.

## 4. Target-Mode Analysis

Projected minFDE averaged over the three seeds is:

| Target mode | GRU | Portable SSM diffusion | Observation |
| --- | ---: | ---: | --- |
| `flee_persistence` | 0.7032 +/- 0.0549 | 0.6932 +/- 0.0090 | Small diffusion gain and lower seed variance |
| `random_turn` | 0.6444 +/- 0.0198 | 0.6378 +/- 0.0097 | Small diffusion gain |
| `s_curve` | 0.8785 +/- 0.0501 | **0.7416 +/- 0.0120** | Main diffusion gain |
| `burst` | **0.5710 +/- 0.0529** | 0.6939 +/- 0.0045 | Diffusion loses on this mode |
| `boundary_escape` | 0.7517 +/- 0.0517 | **0.6392 +/- 0.0085** | Diffusion gain and lower seed variance |

The mode split rules out a blanket claim that diffusion dominates. Its benefit is concentrated on curved and boundary-seeking motion, while the GRU remains better on burst behavior after projection. This motivates mode-conditioned planner evaluation and a future target-strategy generalization test.

## 5. Dynamics Projection Finding

The raw diffusion samples satisfy speed and boundary checks at high rates but fail the acceleration check almost universally. The cause is structural: the denoiser samples future positions in parallel and does not enforce a velocity-state recurrence.

The new `project_candidate_trajectories` function reconstructs each candidate with the shared speed, acceleration and world-bound constraints. It is deliberately a runtime contract, not a safety certificate and not obstacle avoidance. On the locked test it raises diffusion feasible-candidate fraction from `0%` to `99.71%` on average. The report keeps both raw and projected metrics because projection changes the prediction distribution and increases projected minFDE by about `6.6%` relative to raw diffusion.

Obstacle clearance remains an independent check. Projected candidates are not allowed to bypass the future R-CLBF-QP safety layer.

## 6. Reproducibility Artifacts

Each formal run contains:

- `checkpoint.pt`
- `metadata.json` and `config.yaml`
- `training.json`
- `tensorboard/` training event files
- `tensorboard_hparams/` hyperparameter event files
- `locked_test_metrics.json`
- `locked_test_config.yaml`
- `tensorboard_locked_test/` evaluation event files

Formal run directories are:

```text
results/phase2_formal_gru_seed745101
results/phase2_formal_gru_seed745201
results/phase2_formal_gru_seed745301
results/phase2_formal_diffusion_seed745101
results/phase2_formal_diffusion_seed745201
results/phase2_formal_diffusion_seed745301
```

The old `docs/PHASE2_PREDICTION_REPORT.md` is retained as the historical smoke NO-GO report. This document records the subsequent formal v3 experiment and does not overwrite the historical result.

## 7. Gate Evaluation

- [x] Three independent training seeds completed.
- [x] Train/validation/locked-test episode seeds are disjoint.
- [x] Training, validation, calibration and locked-test artifacts are reproducible and TensorBoard tracked.
- [x] Multimodal candidate spread is nonzero and stable across seeds.
- [x] Conformal coverage is close to the 90% target on the locked test.
- [x] Dynamics-projected candidates exceed the 95% feasibility requirement.
- [x] p95 prediction latency is below the isolated 100 ms budget.
- [ ] Raw diffusion candidates satisfy the dynamics contract without post-processing.
- [ ] Diffusion improves locked-test minFDE by at least 10% over GRU.
- [ ] Prediction-to-planner and total end-to-end latency are validated.

## 8. Next Step

Implement a centralized scenario min-max MPC as a diagnostic consumer of the **projected** candidate set. Compare expected-cost, worst-case and CVaR objectives on identical scene seeds against `DynamicEncirclementController`, while logging solver status, fallback reasons, latency and safe-capture metrics. Do not implement the final distributed DN-MPC claim until the centralized planner shows a measurable benefit without target-truth access.
