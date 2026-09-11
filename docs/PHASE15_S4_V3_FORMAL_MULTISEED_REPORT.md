# Phase 15 S4-v3 Formal Multi-Seed Report

> Decision: the S4-v3 data contract and the causal online conditioning path are
> complete. The current official S4 and S4-diffusion predictors do **not**
> outperform the GRU predictor on the frozen offline test. A mixed closed-loop
> result does not support an S4-superiority claim. The robust CBF-QP remains a
> diagnostic safety branch and is not part of this decision.

## 1. Frozen Data Contract

The S4-v3 raw data are frozen at
`results/phase15_s4_branching_train_v3_splits/`, with 420 train, 90
validation, and 90 locked-test scenes. There are 300 mirror groups, each with
two scenes. Groups are assigned as units, so neither a layout nor its mirrored
counterfactual crosses a split. The audit passed with 13,224 train, 1,922
validation, and 1,539 locked-test windows.

Each prediction window contains target/defender observation history, historical
executed defender actions, future planned defender actions, commanded/delayed/
executed action streams, and action/communication timestamps. The complete
audit command is:

```powershell
python scripts\audit_s4_dataset.py `
  --dataset-root results\phase15_s4_branching_train_v3_splits `
  --output results\phase15_s4_branching_train_v3_splits\audit.json `
  --expected-train-scenes 420 `
  --expected-validation-scenes 90 `
  --expected-locked-test-scenes 90
```

The data audit does not establish predictor usefulness by itself. Model and
risk settings must be selected on the validation split only.

## 2. Offline Predictor Protocol

All four models use action-history features and a future action-condition
input. They were trained for 30 CPU epochs with hidden width 32, matched seeds
`727101`, `727102`, and `727103`, and evaluated on the 1,539 frozen locked-test
windows. Diffusion uses four candidates and four sampling steps. Candidate
projection uses the checkpoint dynamics constraints.

`Diagonal SSM` is the project lightweight baseline. `S4-DPLR` is a dense local
reference. `Official S4` calls the upstream `state-spaces/s4` DPLR kernel; its
upstream source hashes are retained in checkpoint metadata. CPU evaluation uses
the upstream naive fallback because the optional CUDA structured kernel is not
installed.

| Model | Projected minFDE (m) | Projected minADE (m) | Coverage | Feasible candidates | Predictor p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU action-conditioned | **0.7244 +/- 0.0352** | **0.3873 +/- 0.0169** | **93.85%** | 76.93% | **0.91** |
| Diagonal SSM diffusion | 1.3332 +/- 0.0228 | 0.6525 +/- 0.0140 | 82.76% | 82.96% | 3.24 |
| Dense S4-DPLR diffusion | 1.3303 +/- 0.0077 | 0.6511 +/- 0.0031 | 83.58% | 82.89% | 4.07 |
| Official S4 diffusion | 1.3373 +/- 0.0275 | 0.6561 +/- 0.0163 | 82.93% | **83.04%** | 2.37 |

The official S4 minus GRU projected-minFDE difference is `+0.6129 m`, with a
matched-seed bootstrap 95% CI of `[+0.5718, +0.6739] m`. This rejects an
offline S4 advantage on this protocol. Diffusion/S4 improves projected
candidate feasibility by roughly six percentage points, but every candidate
family remains below the 95% feasibility target. Therefore neither the
diffusion multimodality claim nor a confidence-calibrated risk claim is ready.

Full machine-readable summaries are retained in
`results/phase15_s4_v3_formal_locked_test/aggregate.json` and are regenerated
with:

```powershell
python scripts\aggregate_prediction_seed_results.py `
  --root results\phase15_s4_v3_formal_locked_test `
  --output results\phase15_s4_v3_formal_locked_test\aggregate.json `
  --bootstrap-samples 10000 `
  --bootstrap-seed 20260911
```

## 3. Causal Action-Conditioned Closed Loop

The 90 original raw locked-test scenes are replayed without geometry or target
seed changes. Prediction refreshes every control step. At the first control
step, the future-action condition is all zeros. Afterwards it is the previous
DN-MPC action sequence, shifted one step and padded causally. Thus no future
target trajectory or offline future defender rollout is supplied to the online
predictor. The availability field records whether a non-zero previous planned
sequence was available at each control step.

This online condition is deliberately not conflated with the logged offline
future planned-action field. It is a causal warm-start proxy and requires a
`none/history/future/both` ablation before attributing a gain to action
conditioning.

The formal protocol uses four candidates, four diffusion steps, local CBF, and
three matched predictor training seeds. Percentile intervals below are
hierarchical bootstrap 95% CIs resampling both matched training seeds and
locked-test episode indices. The Dynamic Encirclement baseline is deterministic
under the fixed scene manifest and was run once; it is shown without a
training-seed interval.

| Predictor | Controller | Safe capture | Collision / boundary | Timeout | Condition available | Total p95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Shared baseline | Dynamic Encirclement | 87.78% | 0.00% / 0.00% | 12.22% | n/a | 1.57 |
| GRU | Worst-case MPC | 79.63% [74.07%, 84.81%] | 0.00% / 0.00% | 20.37% | 95.32% | 40.68 |
| GRU | Distributed delayed DN-MPC | **95.56% [92.96%, 97.78%]** | 0.00% / 0.00% | 4.44% | 92.06% | 49.17 |
| Official S4 | Worst-case MPC | **84.44% [79.63%, 88.89%]** | 0.00% / 0.00% | 15.56% | 95.05% | 57.13 |
| Official S4 | Distributed delayed DN-MPC | 94.44% [91.48%, 97.04%] | 0.00% / 0.00% | 5.56% | 92.17% | 68.12 |

Against GRU, Official S4 changes safe capture by `-1.11` percentage points
`[-2.59, 0.00]` for distributed delayed DN-MPC, and by `+4.81` percentage
points `[+0.74, +9.63]` for worst-case MPC. This reversal by planner is a
predictor--planner coupling observation, not evidence that S4 is superior. It
is incompatible with the offline error result and must be resolved by a
validation-selected risk/refresh/action-conditioning ablation before any
architecture claim.

The aggregate artifacts are retained locally under
`results/phase15_s4_v3_closed_loop_action_online_aggregate.{json,md}` and can
be reproduced with:

```powershell
python scripts\aggregate_closed_loop_seed_results.py `
  --group "gru=results/phase15_s4_v3_closed_loop_action_online_gru_seed*" `
  --group "official_s4=results/phase15_s4_v3_closed_loop_action_online_official_s4_seed*" `
  --reference-group gru `
  --output results\phase15_s4_v3_closed_loop_action_online_aggregate.json `
  --bootstrap-samples 10000 `
  --bootstrap-seed 20260911
```

The `100 ms` value is recorded only as an engineering reference. It is not a
method-selection gate in this project. Predictor, planner, safety, and total
control percentiles remain mandatory report fields.

## 4. Next Formal Experiments

1. Select `none`, history-only, causal future-plan-only, and both action
   conditions on validation. Repeat the winner once on locked test; do not use
   the locked test to choose a condition.
2. Compare a single trajectory with four and eight candidates. Report
   minADE/minFDE, branch accuracy, coverage, feasible-candidate rate, pairwise
   diversity, NLL or energy score, and calibration error. Calibrate candidate
   weights before interpreting expected-risk or CVaR results probabilistically.
3. Select `expected`, worst-case, and CVaR DN-MPC plus refresh intervals
   `1/5/10/20` on validation. Re-run only the selected protocol against Dynamic
   Encirclement, Pure Pursuit, GRU-MPC, diagonal SSM-MPC, and the selected
   diffusion controller on frozen test scenes.
4. Construct three untouched external OOD blocks, each with at least 100
   episodes: new obstacle geometry, new adaptive target policy, and new
   delay/dropout/execution condition. Do not mix these scenes into v3 training
   or validation after they are frozen.
5. Keep no safety layer, local CBF, and robust CBF-QP as explicitly separate
   comparisons. Robust CBF-QP may re-enter an end-to-end claim only after its
   execution-aware post-state and continuous-segment safety audits pass; until
   then it is a diagnostic result.

## 5. Reproducibility Status

The source code records source hashes, configuration snapshots, scene manifests,
per-episode JSONL, per-step JSONL, and summaries. Generated models and result
artifacts are intentionally Git-ignored; this report, evaluators, and tests are
version controlled. The former one-seed preliminary report remains at
`PHASE15_S4_V3_LOCKED_TEST_REPORT.md` for historical context and is superseded
for Phase 15 decision-making by this report.
