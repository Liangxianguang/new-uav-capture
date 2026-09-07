# Phase 1 Foundation Report

## Scope

This phase made the execution model and prediction-data contract executable. It
does not claim that Mamba, diffusion, DN-MPC, or R-CLBF-QP has been validated.

## Completed

- Added an opt-in defender execution model with command delay, command noise,
  first-order velocity tracking, drag, mass scaling, acceleration limits, and
  per-episode parameter randomization.
- Preserved the historical ideal velocity contract when
  `dynamics.execution.enabled: false`.
- Added execution error, desired/delayed/executed action norms, and effective
  execution parameters to observations and step-level info.
- Added policy-safe target trajectory windows with explicit separation between
  input observations and hidden future labels.
- Added reproducible dataset metadata containing seed ranges, effective
  configuration, input/label contracts, environment details, and source hashes.
- Added a stationary rollout option for long-horizon prediction data. Expert
  rollouts remain available, but short expert episodes are not silently padded.

## Data contract

The collector saves:

```text
history_observations:          [samples, history, defenders, features]
future_target_displacements:   [samples, horizon, 3]
reference_positions:            [samples, 3]
episode_indices / timesteps / episode_seeds / target_motion_modes
```

The observation tensor is produced by
`encirclement3d.observation_encoding.policy_observations` and does not use
target truth. Future target positions are used only as supervised labels and
are represented relative to the published, confidence-weighted team belief.

## Reproduction smoke test

```powershell
conda run --no-capture-output -n uav-encirclement-gpu `
  python scripts/collect_prediction_dataset.py `
  --output results/phase1_prediction_dataset_smoke `
  --episodes 1 --seed 745101 --history-length 4 --horizon-steps 12 `
  --obstacle-count 2 --target-speed-scale 0.45 `
  --target-motion-mode s_curve --rollout-policy stationary
```

Observed result: 250 frames, 238 valid windows, 4 defenders, and 63 local
observation features. The smoke output was intentionally removed after
verification because generated experiment artifacts are ignored by Git.

## Verification

Executed in the repository environment `uav-encirclement-gpu`:

```text
69 passed in 36.27s
```

The default execution configuration is disabled, and dedicated tests verify
that this preserves the historical action-to-velocity behavior. Enabled-mode
tests cover delay, acceleration/tracking, randomized execution parameters,
seed reproducibility, and execution diagnostics.

## Limitations and next gate

- The execution model is still a lightweight bounded model, not a flight
  controller, PyBullet model, or hardware validation.
- The dataset currently uses stationary and scripted target rollouts; an
  adaptive adversarial target policy is still required for a strong game-
  theoretic claim.
- The prediction model has not been retrained on this dataset yet.
- The next gate is a TensorBoard-tracked GRU baseline versus an SSM/diffusion
  predictor on disjoint train/validation/test episode seeds.
