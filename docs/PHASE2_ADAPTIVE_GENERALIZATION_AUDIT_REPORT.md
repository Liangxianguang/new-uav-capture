# Phase 2 Unseen Adaptive-Policy Generalization Audit

> Version: v1.0 (2026-09-08)
> Evaluation runs: `results/phase2_adaptive_prediction_audit_v1/`
> Decision: `Conditional Go`; diffusion generalizes to this unseen policy, but its candidate weights remain uncalibrated and dynamics projection remains mandatory.

## 1. Scope

This audit evaluates the six frozen Phase 2 checkpoints without retraining or
model selection.  It answers a narrower question than the original
five-mode locked test: do the frozen predictors retain useful candidate-set
coverage when the target switches to the unseen `adaptive_adversarial`
policy?

The audit is not a new training result and does not use its labels for
calibration.  Each evaluation reads only the checkpoint-specific training
metadata, the fixed validation split for conformal radii, and a separately
generated locked test split.  The actor histories contain visible delayed
belief features only; target truth is used solely as the future supervised
label and final evaluator reference.

## 2. Frozen Protocol

- checkpoints: formal GRU and portable diagonal SSM diffusion checkpoints for
  seeds `745101`, `745201`, and `745301`;
- unseen locked-test data: 32 `adaptive_adversarial` episodes, seeds
  `648201--648232`, 7,616 windows;
- history and prediction horizon: 16 and 12 steps;
- candidates: 8, with 8 diffusion denoising steps and fixed sampling seed
  `745102`;
- calibration: 90% split-conformal full-trajectory radius from the first half
  of validation episodes only;
- physical contract: speed, acceleration, world bounds and obstacle-clearance
  checks, followed by four dynamics-projection iterations;
- traceability: every evaluation directory retains
  `locked_test_metrics.json`, `locked_test_config.yaml`, source hashes and a
  `tensorboard_locked_test/` event file.  The associated training directories
  retain checkpoints, training events, HParams events, configs and metadata.

Candidate weights are still `uniform_uncalibrated`; conformal coverage is a
set-coverage diagnostic and is not calibrated per-candidate probability.

## 3. Aggregate Results

Values are mean +/- sample standard deviation across the three frozen training
seeds.  Lower is better for minADE, minFDE and energy score; higher is better
for coverage and feasible-candidate fraction.

| Model and stage | minADE (m) | minFDE (m) | Energy score | Full-trajectory coverage | Feasible candidates |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU, raw | 0.5039 +/- 0.0242 | 0.8908 +/- 0.0668 | 1.9831 +/- 0.1051 | 82.38% +/- 5.03% | 12.36% +/- 2.94% |
| Portable SSM diffusion, raw | **0.3235 +/- 0.0019** | **0.5051 +/- 0.0072** | **1.1879 +/- 0.0192** | **89.94% +/- 0.25%** | 0.0033% +/- 0.0016% |
| GRU, dynamics projected | 0.4976 +/- 0.0222 | 0.9033 +/- 0.0648 | 1.9903 +/- 0.1003 | 82.22% +/- 4.86% | **99.991% +/- 0.008%** |
| Portable SSM diffusion, dynamics projected | **0.3155 +/- 0.0016** | **0.5363 +/- 0.0071** | **1.1747 +/- 0.0152** | **90.31% +/- 0.03%** | 99.914% +/- 0.025% |

For dynamics-projected candidates, diffusion lowers minFDE by 40.63% and
energy score by 40.98% relative to the projected GRU, while increasing full
trajectory coverage by 8.09 percentage points.  This is a strong result for
this particular unseen-policy shift, and it is materially more stable across
the three seeds than the GRU baseline.

The raw diffusion candidates remain almost universally acceleration-infeasible.
They must not be supplied directly to MPC or safety filtering; planner inputs
must continue to be explicitly marked `projected` and evaluated against the
dynamics contract.

## 4. Latency

The CPU single-sample p95 latency is `8.08 +/- 0.96 ms` for GRU and
`39.37 +/- 8.77 ms` for portable SSM diffusion.  These are isolated predictor
measurements on the audit machine, not end-to-end control-cycle measurements.
The project treats 100 ms as a deployment reference rather than a hard Phase 2
method gate; planner, safety and execution latency must still be reported
separately.

## 5. Interpretation and Decision

The audit closes the previously missing unseen-policy prediction check.  The
portable SSM diffusion model has retained approximately 90% conformal set
coverage and substantially outperforms the GRU on minFDE, energy score and
coverage under this adaptive target policy.  The result supports continued use
of projected diffusion candidates in the already completed planner studies.

It does not change the overall Phase 2 decision from `Conditional Go`:

- the original multi-mode frozen test still misses its preregistered 10%
  minFDE improvement gate;
- diffusion has no calibrated sample-level or mode probability, only
  `uniform_uncalibrated` weights and conformal set coverage;
- raw diffusion trajectories remain acceleration-infeasible;
- this audit has not tested delayed-noisy or dropout observation conditions
  for the prediction model itself; and
- prediction success does not repair the Phase 5 execution-invariant safety
  No-Go result.

## 6. Reproduction

Run `scripts/collect_prediction_dataset.py` with
`--target-motion-mode adaptive_adversarial` to create a separate locked test
dataset.  Evaluate each frozen checkpoint with
`scripts/evaluate_prediction_models.py --training-output <formal-training-run>`
and an empty independent `--output` directory.  The evaluator refuses to
overwrite a nonempty independent evaluation directory, preserving both
training and prior evaluation evidence.
