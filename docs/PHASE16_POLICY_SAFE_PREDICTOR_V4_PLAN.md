# Phase 16 Policy-Safe Predictor-v4 Experiment Plan

> Status: dataset conversion and audit complete; observation-only baseline
> complete on validation only; learned predictor training has not started.
>
> Decision: the Phase 15 raw S4-v3 collection is retained, but Phase 15
> predictor numbers that learned targets relative to simulator truth at the
> forecast origin are historical results. They must not be used to select a
> Phase 16 architecture or to support a fair online-prediction claim.

## 1. Question and Contract

This phase asks whether an action-conditioned SSM/diffusion predictor improves
upon an observation-only predictor when its forecast origin is reconstructed
from information available to the defenders. It is deliberately narrower than
the full Mamba/SSM + DN-MPC + robust-CBF-QP claim.

At every prediction origin, the v4 dataset constructs the reference position
and velocity using only:

- the four public local policy observations;
- each defender's public position;
- target-belief relative position, belief velocity, confidence, and message
  age; and
- confidence-and-age weighting, `max(confidence, 1e-3) / (1 + message_age)`.

Simulator target state is used only as the future supervised label and for a
physical future-velocity diagnostic. It is not an input, reference origin, or
baseline velocity. Future planned defender actions remain an offline condition
only; the later closed-loop experiment must replace them with the causal
previous DN-MPC plan.

## 2. Frozen Data and Current Evidence

The raw archive contains 600 scenes from five defender rollout policies,
seven target speed scales (0.50--0.80), upper/lower evasive branches, three
observation conditions, mirrored counterfactual layouts, and randomized wall,
spawn, altitude, and defender geometry. The 300 mirror groups are assigned as
whole groups, so no mirrored layout crosses a split.

| Split | Scenes | Predictor windows | Branch -1 / +1 windows |
| --- | ---: | ---: | ---: |
| Train | 420 | 13,224 | 5,589 / 7,635 |
| Validation | 90 | 1,922 | 543 / 1,379 |
| Locked test | 90 | 1,539 | 560 / 979 |

`results/phase16_s4_policy_safe_predictor_v4/raw_split_audit.json` verifies
the raw scene split, field presence, finite values, and split disjointness.
The three derived archives carry schema version 2, the public-reference
contract, geometry context, historical executed actions, future planned action
conditions, and label-only branch annotations.

| Artifact | SHA-256 |
| --- | --- |
| `train/dataset.npz` | `ae3f391a1bd441b181b2d5077e675c4ea50481a44891def1d23ce312a8dcc401` |
| `validation/dataset.npz` | `045986870a8a87fc14405c95e4b0310a23c37145062990345c936deac86a024d` |
| `locked_test/dataset.npz` | `c0788092d5e0af66c79df8aa49a5068d62f3d6e7504c1376f3b4ea5405193759` |
| `raw_split_audit.json` | `3ec6db4a79cb72ef5d5d3d4ea0061c3514121a3b29d21964715868468bbc86cb` |

The window counts are intentionally not branch-balanced because episodes yield
different numbers of valid windows. Every branch result must therefore report
both branch-wise values and a macro average over the two branches; a raw
window-weighted aggregate alone is insufficient.

### Validation-only lower bound

The constant-velocity baseline was run before any v4 learned model on the
second half of validation episodes. Split-conformal calibration used only the
first half (835 windows), and no locked-test file was read.

| Method | Projected minADE (m) | Projected minFDE (m) | Full trajectory coverage | Candidate feasible | Predictor p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Public-belief constant velocity, K=1 | 3.539 | 4.058 | 85.83% | 47.01% | 0.038 ms |

The baseline has no target-truth velocity and no multimodality (`candidate
spread = 0`, `bimodal candidate fraction = 0`). Its high branch top-1 score
(97.74% after the labelled decision step) does not demonstrate branch
coverage: it emits exactly one branch. Later diffusion results must report
top-1 accuracy, uniform-vote accuracy, any-candidate branch coverage, and the
bimodal-candidate fraction separately. Candidate weights are currently
uniform and uncalibrated; neither vote nor coverage may be called a probability.

## 3. Pre-registered Selection Protocol

The following choices are fixed before opening the Phase 16 locked test.

| Item | Fixed protocol |
| --- | --- |
| Training split | 13,224 v4 train windows only |
| Validation split | 90 v4 validation scenes; first half of episode seeds for conformal calibration and second half for selection metrics |
| Test split | 90 v4 locked-test scenes; not read until the winner, K, projection, and risk settings are recorded |
| Input condition | Historical executed actions and future planned action condition (`both`) for the primary model; `none/history/future` are GRU-only attribution ablations |
| Training budget | 30 epochs, batch 128, hidden width 32, one recurrent/SSM layer, learning rate `1e-3`, train-split standardization, matched seeds 727201/727202/727203 |
| Diffusion sampling | Evaluate K = 1, 4, and 8 candidates with four sampling steps; retain raw and dynamics-projected output side by side |
| Calibration | Split conformal full-trajectory maximum-distance radius at 90% nominal coverage; calibration never uses the selection half or locked test |
| Runtime | Record CPU p50/p95/p99 for predictor, then component and total latency in closed loop. The 100 ms reference is not a pass/fail gate. |

All models see the same v4 inputs and normalization contract:

1. GRU Gaussian predictor: single-mode error and latency reference.
2. Portable diagonal SSM conditional diffusion: linear-complexity project model.
3. Local dense S4-DPLR conditional diffusion: architectural reference.
4. Official upstream `state-spaces/s4` DPLR conditional diffusion: external
   S4 implementation check.

The installed PyTorch build is CPU-only even though an RTX 4060 is visible to
the host. The official S4 implementation is usable through its CPU naive
fallback; it is valid for an equal offline comparison, but GPU runtime claims
must wait for a separately recorded CUDA-enabled environment.

## 4. Execution Batches

### A. Freeze and smoke-test the v4 contract

- [x] Audit raw scene/mirror split and generate v4 train, validation, and
  locked-test archives.
- [x] Add branch-label persistence and validation-only evaluation isolation.
- [x] Run the observation-only constant-velocity baseline.
- [x] Run targeted regression tests (31 passed), Python compilation, and diff
  whitespace validation.
- [x] Record SHA-256 hashes for the three derived archives and the raw audit,
  including branch/window counts and the public-reference contract.
- [ ] Perform one 2-epoch smoke run per model to confirm input shape, official
  S4 loading, checkpoint metadata, TensorBoard output, and causal-action
  interface. Smoke outputs cannot enter any result table.

### B. Attribute action conditioning before architecture selection

Train GRU `none`, `history`, `future`, and `both`, each with the three matched
seeds. Use the fixed 30-epoch budget; do not alter hyperparameters after seeing
individual seed curves. Aggregate only validation-selection artifacts.

- [ ] Compare projected minADE/minFDE, energy score, full-trajectory coverage
  error, geometry feasibility, and p50/p95/p99 latency.
- [ ] Publish per-branch and macro branch metrics, with episode/seed-stratified
  bootstrap intervals rather than a raw-window interval.
- [ ] Lock the GRU action condition. `both` remains primary only if it is not
  worse than the other conditions on the pre-specified projected error and
  coverage summary.

### C. Compare SSM/S4 families on the locked action contract

Train the portable diagonal SSM, dense local S4-DPLR, and official upstream S4
with `both` and the same three seeds. GRU `both` from Batch B is the matched
reference, so this batch produces 21 total learned runs across B and C.

- [ ] Evaluate each diffusion checkpoint at K = 1, 4, and 8 on the
  validation-selection half only.
- [ ] For every K report raw/projected minADE/minFDE, top-1 ADE/FDE, energy
  score, spread, conformal coverage error, bounds/speed/acceleration/obstacle
  feasibility, branch metrics, and p50/p95/p99 latency.
- [ ] Select one K and one model family using an aggregate of the three seeds.
  A diffusion or S4 claim requires improvement relative to GRU `both` in a
  primary forecast metric without loss of calibration or feasibility. If none
  meet this condition, GRU remains the selected model and the negative S4
  result is retained.
- [ ] Freeze selected checkpoint hashes, K, sampling seed/steps, projection
  settings, conformal radius protocol, and decision table before opening test.

The following are reporting gates, not targets that may be tuned on the locked
test: full-trajectory coverage error should be within 5 percentage points of
the nominal 90%; projected feasible fraction should materially exceed the
47.01% constant-velocity baseline and be reported even when error is good;
and a multi-candidate method must show nonzero branch diversity rather than
only an oracle minimum-distance gain.

### D. One-time Phase 16 locked-test prediction evaluation

- [ ] Run the public-belief constant-velocity baseline and every frozen
  primary learned comparison on the 1,539 locked-test windows.
- [ ] Calibrate only from validation; do not refit normalization, conformal
  radius, architecture, K, projection iterations, or safety margin.
- [ ] Aggregate the three learned seeds with paired hierarchical bootstrap and
  report the same per-branch/macro metrics and latency distribution.
- [ ] Produce a failure gallery for missed turns, stale beliefs, obstacle
  projection changes, and single-mode collapse. Do not delete failures.

### E. Causal DN-MPC closed loop, separate from safety claims

Only the selected predictor(s) enter the existing DN-MPC integration. The
online action condition is the previous causal plan shifted by one step; it is
not the logged future action sequence from a completed rollout.

- [ ] Run no-safety, local-CBF, and robust-CBF-QP diagnostic branches on the
  same frozen scenes, with all planner and predictor seeds paired.
- [ ] Report ordinary capture and safe capture separately, plus collision,
  boundary violation, timeout, capture time, minimum clearance, planner
  valid/effective rate, action-condition availability, candidate age,
  fallback, certificate status, and predictor/planner/filter/total latency.
- [ ] Retain P4 as modular DN-MPC evidence. Do not represent robust-CBF-QP as
  a closed-loop certificate until the queue-authority, reachable-tube, and
  fallback gates in the safety workstream pass.

### F. Three frozen OOD blocks after the in-distribution test

Create, audit, and freeze three new 100-scene blocks before evaluating any
method. They are diagnostic generalization blocks, not substitute test data.

| OOD block | Shift relative to v4 training data | Required strata |
| --- | --- | --- |
| Geometry | Novel obstacle layouts/placements and higher topology complexity without changing the physical workspace contract | obstacle shape/count, clearance, altitude |
| Target policy | A previously unseen adaptive switching evader with branch reversals | speed band, branch direction, switch timing |
| Observation/execution | Stronger delay, dropout, noise, and recorded action-execution mismatch | delay, dropout, message age, action age |

- [ ] Keep the current target speed/acceleration/world bounds explicit; a
  difficulty increase must be physically feasible, not a hidden constraint
  violation designed to make baselines fail.
- [ ] Run the selected model, GRU reference, constant-velocity baseline, and
  non-predictive DN-MPC/encirclement baselines on identical scene manifests.
- [ ] Report all results by stratum and aggregate. A higher average capture
  rate alone cannot establish robustness if any realistic stratum collapses.

## 5. Safety Workstream Remains a Blocker for a Full Claim

The existing direct joint run of DN-MPC plus robust CBF-QP remains No-Go
(50% safe capture, 49% collision, 16% boundary violation). Dataset expansion
does not repair that failure. It remains a parallel contract-repair task:

1. complete the independent safety artifacts;
2. align reset set, margins, action limits, pending-command authority, and
   fallback with the execution dynamics;
3. pass multi-step reachable-tube, queue-prefix, and continuous-segment audits
   under delay/noise/tracking perturbations; then
4. reopen three-seed end-to-end testing.

Until then, the defensible output is a public-belief trajectory predictor plus
modular DN-MPC evaluation, not a proven R-CLBF-QP end-to-end system.

## 6. Reproduction Commands

The completed validation baseline is retained locally at
`results/phase16_validation_constant_velocity/` and is regenerated with:

```powershell
python scripts\evaluate_constant_velocity_baseline.py `
  --validation-dataset results\phase16_s4_policy_safe_predictor_v4\validation\dataset.npz `
  --environment-config configs\capture_radius_pursuit_central_v4_flee.yaml `
  --validation-only `
  --output results\phase16_validation_constant_velocity
```

The primary GRU run template is:

```powershell
python scripts\train_prediction_models.py `
  --model gru `
  --train-dataset results\phase16_s4_policy_safe_predictor_v4\train\dataset.npz `
  --validation-dataset results\phase16_s4_policy_safe_predictor_v4\validation\dataset.npz `
  --output results\phase16_gru_both_seed727201 `
  --action-conditioning both `
  --epochs 30 --batch-size 128 --hidden-dim 32 --num-layers 1 `
  --learning-rate 1e-3 --seed 727201 --device cpu
```

For every trained checkpoint, selection evaluation must include
`--validation-only`; only the frozen winner can be passed a locked-test
dataset. The official model additionally requires
`--official-s4-root tmp\official_s4_archive\s4-main`.
