# Preregistered fresh task-effect training stage

Data and training protocols were fixed BEFORE collection/training. Fresh
episode seeds973010–973073/layout1973010–1973041,32 mirror groups assigned
24 train/8 development. No V10 development reuse or reserved holdout collection.
Fixed later snapshots extend temporal coverage without future-label sampling.
Target/scene ranges and baseline safety contracts remain unchanged.

Fixed training: five configurations x three seeds x80epochs, same
initialization/sampler order for ordinary-MSE/task and structured-MSE/task pairs.
Preserve independently trained motion-only, GRU, CV and fixed-library controls.
The V11 exact cost-effect loss supplements physical supervised losses only;
it is never a low-predicted-cost reward. Ordinary and structured task models
are evaluated separately; graph superiority alone is not causal usefulness.

```powershell
$taskPython = 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe'
& $taskPython experiments/cwm_v10/collect_local.py --capsule experiments/cwm_v1/baseline/capsule.zip --source experiments/cwm_v7/artifacts/paired_training_20261010.zip --geometry experiments/cwm_v7/artifacts/geometry_qualification_20261010.zip --v9 experiments/cwm_v9/artifacts/local_shadow_20261010.zip --protocol experiments/cwm_v12/data_protocol.json --output results/cwm_v12/fresh_data
& $taskPython experiments/cwm_v10/replay_public.py --data results/cwm_v12/fresh_data --capsule experiments/cwm_v1/baseline/capsule.zip --output results/cwm_v12/fresh_public
& $taskPython experiments/cwm_v12/train_task_effect.py --data results/cwm_v12/fresh_data --public results/cwm_v12/fresh_public --output results/cwm_v12/fresh_training
& $taskPython experiments/cwm_v12/train_task_effect.py --data results/cwm_v12/fresh_data --public results/cwm_v12/fresh_public --output results/cwm_v12/fresh_retraining
```

Output must be a new directory. Scene construction performs original route
validation before episode progress and may take several minutes; lack of an
early progress line does not mean the live process is stuck. Authoritative
collection completed at `results/cwm_v12/data_20261010`, independent replay at
`public_replay_20261010`. All64 original episodes safely captured, all saved
trajectory arrays and1472 public local contexts/histories/backbones matched.
These are ORIGINAL controller outcomes, not enhancement performance.
960 eligible calls (716train/244development),521 full-support calls
(392train/129development). Partial calls keep physical masked labels, never
fabricated full-horizon task labels. Added later snapshots are not a guarantee
of more full-support task data: near-terminal calls are often incomplete.

Both fixed training processes completed at `training_20261010` and
`training_repeated_20261010`. Both ordinary-task and structured-task gates
FAILED. Keep the original controller; do not promote these checkpoints.
See `docs/CWM_V12_TASK_EFFECT_TRAINING_20261010.md` for complete results.
The complete V1–V12 suite passed:141 tests in150.28s; arithmetic/archive
verification passes while model qualification FAILS. These are distinct claims.
The entry independently audits geometry, every
public252 encoding, full histories, original scores/candidates and masks before
training, verifies task labels against the original engine, and records full
source closure, optimizer/RNG, predictions and every learned full-cost ranking.
No model is promoted or control activated. Do not change the protocol after
seeing development outcomes; record any follow-up hypothesis in a new stage.

The release verifier reaudits data,
reloads all30 checkpoints and original-engine scores, recomputes qualification,
checks matched initialization/batch RNG and independent optimizer/RNG/history
equality. Use new output directories and never overwrite an existing archive.

```powershell
& $taskPython experiments/cwm_v12/task_training_release.py --verify experiments/cwm_v12/artifacts/task_effect_training_20261010.zip
& $taskPython -m pytest experiments/cwm_v12 -q
```

Published archive creation was exclusive:

```powershell
& $taskPython experiments/cwm_v12/task_training_release.py --data results/cwm_v12/data_20261010 --public results/cwm_v12/public_replay_20261010 --primary results/cwm_v12/training_20261010 --retrained results/cwm_v12/training_repeated_20261010
```
