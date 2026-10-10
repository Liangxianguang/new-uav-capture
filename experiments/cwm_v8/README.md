# Frozen S4 decision-value diagnostic (2026-10-10)

Development-only, no new training or holdout collection, enhanced control off.
Use the previously published V7 archive and predeclared median seeds, not best
seeds. Nine information conditions score the same eight original/pressure/brake
command probes with the complete frozen distributed DN-MPC team cost.
Truth paths are offline labels, never online forecasts or certified bounds.

```powershell
$taskPython = 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe'
& $taskPython experiments/cwm_v8/s4_value.py --capsule experiments/cwm_v1/baseline/capsule.zip --archive experiments/cwm_v7/artifacts/paired_training_20261010.zip --output results/cwm_v8/fresh_primary
& $taskPython experiments/cwm_v8/s4_value.py --capsule experiments/cwm_v1/baseline/capsule.zip --archive experiments/cwm_v7/artifacts/paired_training_20261010.zip --output results/cwm_v8/fresh_repeated
& $taskPython experiments/cwm_v8/s4_value_release.py --verify experiments/cwm_v8/artifacts/s4_decision_value_20261010.zip
& $taskPython -m pytest experiments/cwm_v1 experiments/cwm_v2 experiments/cwm_v3 experiments/cwm_v4 experiments/cwm_v5 experiments/cwm_v6 experiments/cwm_v7 experiments/cwm_v8 -q
```

Output directories must not exist. All archived run/source files are preserved.
The initial draft used inherited bootstrap metadata incorrectly identifying
development groups as TRAIN. It remains in local `value_20261010`; only metadata
was repaired before two new complete runs. Authoritative runs are
`value_20261010_v2` and `value_repeated_20261010`.

Archive creation was exclusive, not overwrite:

```powershell
& $taskPython experiments/cwm_v8/s4_value_release.py --primary results/cwm_v8/value_20261010_v2 --repeated results/cwm_v8/value_repeated_20261010
```

Do not repeat that creation command against an existing release archive. The
verification command is read-only. The release verifies the source V7 training
archive, actual support masks/population and original replay arrays, recomputes
all ranks/group statistics from recorded costs, and requires independent
records to be byte-identical. It does not independently rerun each cost from a
serialized mid-planning state, which this artifact does not contain.

Result: 80 development windows, 69 jointly complete, 11 jointly excluded. All
non-CV conditions choose exactly the same probe as GRU in 69/69. CV changes
two choices and has negative mean diagnostic gain. Both exploratory signal
flags are false. Prior V7 gate stays failed; no controller is promoted.
See `docs/CWM_V8_S4_DECISION_VALUE_20261010.md` for interpretation/next tasks.
