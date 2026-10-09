# CWM-v1: reversible causal target-response experiment

The retained baseline is Phase2 GRU seed745101 + distributed-delayed DN-MPC +
local CBF, evaluated on 2026-10-05. It is NOT the old strict Joint CBF system.
The baseline is never fine-tuned or replaced. Local CBF remains empirical.

Stages: P0 portable historical capsule and trajectory replay; P1 paired-action
response diagnostic; P2 train-only intervention branches; P3 matched plain
action-conditioned and structured response predictors; P4 shadow evaluation;
P5 independently calibrated guarded control. No locked/holdout tuning.

All collection, training and replay outputs use new directories. Active RL
jobs in the original checkout are untouched. First pilots use one CPU thread.
Learned predictions cannot relax the physical task or the existing CBF.

## Baseline preservation

```powershell
python experiments/cwm_v1/freeze_baseline.py --source D:/uav-capture/new-uav-capture --output experiments/cwm_v1/baseline/capsule.zip
python experiments/cwm_v1/replay_baseline.py --capsule experiments/cwm_v1/baseline/capsule.zip --output results/cwm_v1/baseline_replay --all
```

The capsule checks every historical provenance hash before capturing exact
source bytes, the predictor, frozen scenes, original metrics and all 256
reference trajectories. Extraction checks every member hash. Replay compares
physical outcomes and defender/target trajectories, excluding wall-clock
latency. Runtime package versions are recorded, not silently inferred from the
older environment.yml. Cross-platform bitwise reproducibility is not promised.

The frozen validation records are regression references only, never training
data. Source-level off mode is the restored historical evaluator itself.
New model artifacts and histories must not be stored in the baseline capsule.

## Research boundaries

Action conditioning alone does not identify a causal model. The experiment
uses paired simulator interventions under an explicitly defined command ->
CBF -> execution -> target response pathway. Any finding is conditional on
the simulator and its hidden-noise assumptions, not proof of real-world SCM
identifiability or universal robustness.

Plain and structured predictors receive the same train-only data and candidate
budget. A useful prediction result does not authorize online action changes;
closed-loop evidence and a separately calibrated guard are required.

## Completed pilot: NO-GO

Six models (plain/structured x seeds 101/202/303) were trained for 40 fixed
epochs on 108 paired windows from 36 original training episodes. Both failed
the constant-velocity comparison. Guarded control remains rejected; off is
the default. See `docs/CWM_V1_PILOT_REPORT_20261009.md` and `reports/`.

`artifacts/pilot_20261009.zip` contains the exact paired dataset, six weights,
histories, saved predictions and source snapshot with member SHA256 hashes.
Extract to a NEW directory, never over the baseline or an existing run:

```powershell
python experiments/cwm_v1/verify_release.py
Expand-Archive experiments/cwm_v1/artifacts/pilot_20261009.zip results/cwm_v1/pilot_release
python experiments/cwm_v1/verify_checkpoints.py --dataset results/cwm_v1/pilot_release/dataset/pairs.npz --run results/cwm_v1/pilot_release/training
python experiments/cwm_v1/train.py --dataset results/cwm_v1/pilot_release/dataset/pairs.npz --output results/cwm_v1/retrain_new
python experiments/cwm_v1/audit_pilot.py --dataset results/cwm_v1/pilot_release/dataset/pairs.npz --reference results/cwm_v1/pilot_release/training --rerun results/cwm_v1/retrain_new --capsule experiments/cwm_v1/baseline/capsule.zip --output results/cwm_v1/retrain_audit_new.json
python experiments/cwm_v1/shadow.py --capsule experiments/cwm_v1/baseline/capsule.zip --run results/cwm_v1/pilot_release/training --mode shadow --output results/cwm_v1/shadow_new
python -m pytest experiments/cwm_v1/test_cwm_v1.py -q
```

Use the recorded capsule runtime. Retraining numerical predictions can be
checked on that runtime; checkpoint ZIP bytes and elapsed times need not match.
The archived source is the exact training snapshot; new edits are not a
substitute for that snapshot. The pilot dataset is portable, but recollection
of the full original train pool also requires access to that original pool:

```powershell
python experiments/cwm_v1/collect_pairs.py --capsule experiments/cwm_v1/baseline/capsule.zip --training-scenes D:/uav-capture/new-uav-capture/results/phase91_zero_cbf_curriculum_v5/train/scenes.jsonl --output results/cwm_v1/pairs_new --groups-per-variant 12 --seed 910910
```

The full original train pool is not silently inferred from the selected
36 scenes. The manifest records its hash. Recollection and retraining write
only new directories; the baseline and existing experiments stay untouched.
