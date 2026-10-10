# Preregistered fresh task-effect training stage

Data and training protocols were fixed BEFORE collection/training. Fresh
episode seeds973010–973073/layout1973010–1973041,32 mirror groups assigned
24 train/8 development. No V10 development reuse or reserved holdout collection.
Fixed later snapshots extend temporal coverage without future-label sampling.
Target/scene ranges and baseline safety contracts remain unchanged.

Planned fixed training: five configurations x three seeds x80epochs, same
initialization/sampler order for ordinary-MSE/task and structured-MSE/task pairs.
Preserve independently trained motion-only, GRU, CV and fixed-library controls.
The V11 exact cost-effect loss supplements physical supervised losses only;
it is never a low-predicted-cost reward. Ordinary and structured task models
are evaluated separately; graph superiority alone is not causal usefulness.

```powershell
$taskPython = 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe'
& $taskPython experiments/cwm_v10/collect_local.py --capsule experiments/cwm_v1/baseline/capsule.zip --source experiments/cwm_v7/artifacts/paired_training_20261010.zip --geometry experiments/cwm_v7/artifacts/geometry_qualification_20261010.zip --v9 experiments/cwm_v9/artifacts/local_shadow_20261010.zip --protocol experiments/cwm_v12/data_protocol.json --output results/cwm_v12/fresh_data
```

Output must be a new directory. Scene construction performs original route
validation before episode progress and may take several minutes; lack of an
early progress line does not mean the live process is stuck. Authoritative
collection started at `results/cwm_v12/data_20261010`. This protocol milestone
does not contain completed V12 data or newly trained models. Training entry and
independent source/public replay audits are still to be implemented/finished.
No model is promoted, no control is activated. Do not change the protocol after
seeing development outcomes; record any follow-up hypothesis in a new stage.
