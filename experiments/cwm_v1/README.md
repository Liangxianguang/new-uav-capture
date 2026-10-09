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
