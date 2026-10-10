# Actual local-candidate shadow (2026-10-10)

Optional public response provider is lazy/default-off. A failed qualification
gate prevents even optional weight loading. Shadow mode loads the frozen V7
median models, copies actual first local calls, then does all alternate work
after the original plan returns. It generates alternative candidates using
the original local generator and compares full original local costs on copies.
No path in this entry point transfers action authority to an enhanced model.

All diagnostic truth is collected on cloned environments after public inputs
and candidate pools are frozen. Missing delayed peer plans cause a skip, never
a fabricated central plan. The private target state is never a model input.
The changed delayed-relative context/local reference anchor is a transport
shift from V7 training, not a qualified distributed deployment.

```powershell
$taskPython = 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe'
& $taskPython experiments/cwm_v9/local_shadow.py --capsule experiments/cwm_v1/baseline/capsule.zip --archive experiments/cwm_v7/artifacts/paired_training_20261010.zip --output results/cwm_v9/fresh_off
& $taskPython experiments/cwm_v9/local_shadow.py --capsule experiments/cwm_v1/baseline/capsule.zip --archive experiments/cwm_v7/artifacts/paired_training_20261010.zip --mode shadow --output results/cwm_v9/fresh_primary
& $taskPython experiments/cwm_v9/local_shadow.py --capsule experiments/cwm_v1/baseline/capsule.zip --archive experiments/cwm_v7/artifacts/paired_training_20261010.zip --mode shadow --output results/cwm_v9/fresh_repeated
& $taskPython experiments/cwm_v9/local_shadow.py --capsule experiments/cwm_v1/baseline/capsule.zip --archive experiments/cwm_v7/artifacts/paired_training_20261010.zip --mode load_failed --output results/cwm_v9/fresh_failed
& $taskPython experiments/cwm_v9/local_shadow.py --capsule experiments/cwm_v1/baseline/capsule.zip --archive experiments/cwm_v7/artifacts/paired_training_20261010.zip --mode guarded_blocked --output results/cwm_v9/fresh_blocked
& $taskPython experiments/cwm_v9/local_shadow_release.py --verify experiments/cwm_v9/artifacts/local_shadow_20261010.zip
& $taskPython -m pytest experiments/cwm_v1 experiments/cwm_v2 experiments/cwm_v3 experiments/cwm_v4 experiments/cwm_v5 experiments/cwm_v6 experiments/cwm_v7 experiments/cwm_v8 experiments/cwm_v9 -q
```

Fresh run directories must not exist. Restore checks the unchanged historical
capsule. Default mode is off; optional providers have zero load/predict calls.
`load_failed` is an injected unavailable optional-checkpoint fault, not a
destructive test of baseline files. `guarded_blocked` uses the actual failed V7
gate and has zero optional load calls. Shadow is explicitly not guarded control.

Authoritative five runs: `shadow_20261010_v3`,
`shadow_repeated_20261010`, `off_20261010_v2`,
`failed_20261010_v2`, `blocked_20261010_v2`. All remain in local results.
The first shadow draft had no independently reconstructable public context;
the second added context for scored calls only. Those drafts are retained.
Version3 stores public context for *every* captured invocation, including skips.
No model, scene, probe-selection or eligibility threshold changed between drafts.

Exclusive release creation was:

```powershell
& $taskPython experiments/cwm_v9/local_shadow_release.py --primary results/cwm_v9/shadow_20261010_v3 --repeated results/cwm_v9/shadow_repeated_20261010 --off results/cwm_v9/off_20261010_v2 --failed results/cwm_v9/failed_20261010_v2 --blocked results/cwm_v9/blocked_20261010_v2
```

Do not run creation against the existing archive; use `--verify`. The verifier
reconstructs original local generation/scores from safe JSON public context,
reloads exact public model outputs, checks physics/terminal masks and independent
record/array equality, and verifies all five modes' original replay fields.
No persisted pickle is loaded. Reconstructing scores does not rerun all cloned
target dynamics; those are independently replayed by the two full shadow runs.

Result:320 captured calls;128 missing-peer skips;61 incomplete-support calls;
131 jointly complete calls. All16 original trajectories in each of five modes
match V7 bytes. The motion and exact-response exploration signals are positive,
but learned response gains remain negative versus GRU on the same enlarged pool.
Original-GRU enlarged-pool gains also show candidate-library expansion effects
that must not be attributed to causal learning. No V7 gate is overridden.
Report: `docs/CWM_V9_ACTUAL_LOCAL_SHADOW_20261010.md`.
