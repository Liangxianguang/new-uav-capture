# V13: TRAIN-only true mechanism diagnosis, not another training claim

Two independent original-controller replays finished: 48 TRAIN episodes,
716 eligible original local calls, 1,450 non-anchor candidate pairs. Both
reproduce V12 original episode arrays and proposed/CBF-command/executed/target
branch arrays; only private diagnostic labels are added. No network training,
new control authority, development selection, holdout use, or prior gate override.

The V13 protocol was recorded before the formal TRAIN-only replay. Exploratory
V12 development aggregates preceded it; this is not untouched validation.
Do not use terminal target-y sign as branch truth. Actual simulator branch
state is recorded after each defender-update/target-action transition, alongside
commitment, branch scores and avoidance exposure. These are labels, not inputs.

```powershell
$taskPython = 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe'
& $taskPython experiments/cwm_v13/mechanism_replay.py --output results/cwm_v13/fresh_primary
& $taskPython experiments/cwm_v13/mechanism_replay.py --output results/cwm_v13/fresh_repeated
& $taskPython experiments/cwm_v13/mechanism_release.py --verify experiments/cwm_v13/artifacts/mechanism_diagnosis_20261010.zip
& $taskPython -m pytest experiments/cwm_v13 -q
```

Each replay output must be new; no directory is overwritten. Authoritative
runs are `mechanism_qualified_20261010` and `mechanism_repeated_20261010`.
To package two completed replays in a fresh clone with no existing archive:

```powershell
& $taskPython experiments/cwm_v13/mechanism_release.py --primary results/cwm_v13/fresh_primary --repeated results/cwm_v13/fresh_repeated
```

The packager audits before exclusive archive creation and verifies again after
creation. It independently recomputes metrics from the pinned V12 arrays and
V13 labels, checks complete TRAIN identities/groups, terminal suffix padding,
branch commitment consistency, source closure, all48 trajectories and two-run
bytes. The verifier does not independently rerun the simulator; the two replay
processes do. Label-cohort means are descriptive, not mediation identification.

44/1,450 candidate pairs flip true branches, all at snapshot steps2/4.
Group-equal response means: flipped0.460440m, same0.012854m,
pending0.000147m. CBF changes46,176/54,352 observed defender command points
(including anchor candidates), max3.328207m/s. Execution is disabled in this
distribution; command-to-execution max1.88e-15m/s, not a noisy-execution result.
V12 response/task/decision gates stay failed. See the Chinese report and
candidate plan; no claims about enhanced original-eight-Level coverage.
Full V1–V13 regression:149 passed in167.28s; V13 alone:8 passed in12.29s.

Failed original draft replay and failed verifier archive are retained locally
under ignored `results/cwm_v13`; neither is released as valid evidence.
