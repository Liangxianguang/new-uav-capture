# Fresh delayed-local two-head training (2026-10-10)

New32 S4 mirror groups, preassigned24 train/8 development; reserved V7 holdout
remains untouched. Qualified wall geometry and existing target/controller rules
are unchanged. Data use actual local calls, delayed peer states/plans and a
fixed V9-generated library independent of V10 weights/future truth.

Three architectures x three seeds x80 fixed epochs: independently trained
motion-only, ordinary two-head and graph-pooled two-head. Original GRU weights
are never in an optimizer. Both heads are optional residuals around its supplied
forecast. Response is exactly zero at the reference command; motion correction
need not be zero, so only disabling the *whole module* preserves the baseline.
No model is promoted. Code in this stage is offline, not an active controller.

The complete V1–V10 experiment test suite passed: 106 tests in 98.94s.
This is not a closed-loop qualification of the optional controller.

```powershell
$taskPython = 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe'
& $taskPython experiments/cwm_v10/collect_local.py --capsule experiments/cwm_v1/baseline/capsule.zip --source experiments/cwm_v7/artifacts/paired_training_20261010.zip --geometry experiments/cwm_v7/artifacts/geometry_qualification_20261010.zip --v9 experiments/cwm_v9/artifacts/local_shadow_20261010.zip --output results/cwm_v10/fresh_data
& $taskPython experiments/cwm_v10/replay_public.py --data results/cwm_v10/fresh_data --capsule experiments/cwm_v1/baseline/capsule.zip --output results/cwm_v10/fresh_public
& $taskPython experiments/cwm_v10/train_two_head.py --data results/cwm_v10/fresh_data --output results/cwm_v10/fresh_training
& $taskPython experiments/cwm_v10/train_two_head.py --data results/cwm_v10/fresh_data --output results/cwm_v10/fresh_retraining
& $taskPython experiments/cwm_v10/two_head_release.py --verify experiments/cwm_v10/artifacts/two_head_training_20261010.zip
& $taskPython experiments/cwm_v10/decision_release.py --verify experiments/cwm_v10/artifacts/two_head_decision_20261010.zip
& $taskPython -m pytest experiments/cwm_v1 experiments/cwm_v2 experiments/cwm_v3 experiments/cwm_v4 experiments/cwm_v5 experiments/cwm_v6 experiments/cwm_v7 experiments/cwm_v8 experiments/cwm_v9 experiments/cwm_v10 -q
```

Directories must be new. Scene construction performs unchanged route validation
before the first progress line and can take several minutes. Do not restart a
live process just because it has not printed episode progress yet.

Authoritative local runs: `data_20261010`, `public_replay_20261010`,
`training_20261010`, `training_repeated_20261010`. Collection, independent public
replay and both training processes all completed. Creation of the published
training archive was exclusive:

```powershell
& $taskPython experiments/cwm_v10/two_head_release.py --data results/cwm_v10/data_20261010 --public results/cwm_v10/public_replay_20261010 --primary results/cwm_v10/training_20261010 --retrained results/cwm_v10/training_repeated_20261010
```

Do not recreate over existing artifacts. The verifier reconstructs deterministic
declared geometry and all252 public features, full history from independent
original replay frames, original local generation/scores/inputs, masks, train
normalization, all18 checkpoints and deterministic retraining including optimizer
and sampler/Torch RNG states. No stored pickle is loaded.

Exploratory same-library diagnostics were run independently twice, after training
results, on already-inspected fresh development data. Never override the failed
training gate with these diagnostics:

```powershell
& $taskPython experiments/cwm_v10/decision_two_head.py --data results/cwm_v10/data_20261010 --training results/cwm_v10/training_20261010 --capsule experiments/cwm_v1/baseline/capsule.zip --output results/cwm_v10/fresh_decision
& $taskPython experiments/cwm_v10/decision_two_head.py --data results/cwm_v10/data_20261010 --training results/cwm_v10/training_20261010 --capsule experiments/cwm_v1/baseline/capsule.zip --output results/cwm_v10/fresh_decision_repeated
```

Published diagnostic creation:

```powershell
& $taskPython experiments/cwm_v10/decision_release.py --primary results/cwm_v10/decision_20261010 --repeated results/cwm_v10/decision_repeated_20261010
```

This archive verifier recomputes rankings/group attribution from the paired
recorded scores; it does not independently recalculate new-head scores. Both
complete diagnostic processes recalculate them from frozen local public context
and reloaded weights. See `docs/CWM_V10_TWO_HEAD_TRAINING_20261010.md` for
results, qualification failure and remaining work.
