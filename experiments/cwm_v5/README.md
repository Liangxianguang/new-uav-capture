# Offline decision value of action response

This is a diagnostic, not training or an enabled controller. It reuses the
V4 original-TRAIN archive after its failed data screen, without overriding
that failure or claiming fresh validation. It asks whether perfect response
information alone could change probe choices under the original 8-step
distributed DN-MPC team diagnostic objective.

Four target-information conditions are scored using unchanged commands and
copies of the original planner's post-plan delayed-message state:

1. Original K=1 projected GRU path, shared by all 13 probes.
2. That path plus the simulator's actual paired target response.
3. The reference branch's actual target path, shared by all probes.
4. Each probe's actual target path.

The `_team_scenario_costs` function and original risk aggregation are used
without editing the frozen source. Truth paths/corrections are explicitly
`raw` diagnostic containers, not certified projected or deployable forecasts.
No private label enters an actual planning or execution call.

All 13 probes must have full valid 8-step support for a window to be ranked.
All scoring occurs on planner copies. Complete parent environment/planner
fingerprints are checked. Every one of the ten original TRAIN episodes is
replayed and compared with V4 target/defender trajectory bytes and outcomes.
Two separate process runs must reproduce every window score/metric byte-exactly.

From the repository research checkout:

```powershell
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v5/decision_value.py --capsule experiments/cwm_v1/baseline/capsule.zip --archive experiments/cwm_v4/artifacts/asymmetric_diagnostic_20261009.zip --output results/cwm_v5/decision_value_20261009
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v5/decision_value.py --capsule experiments/cwm_v1/baseline/capsule.zip --archive experiments/cwm_v4/artifacts/asymmetric_diagnostic_20261009.zip --output results/cwm_v5/decision_value_replay_20261009
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' -m pytest experiments/cwm_v5 -q
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v5/release.py --run results/cwm_v5/decision_value_20261009 --repeated results/cwm_v5/decision_value_replay_20261009
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v5/release.py --verify experiments/cwm_v5/artifacts/decision_value_20261009.zip
```

Use new output directories; never overwrite historical evidence. The release
ZIP contains both runs, score arrays/metrics, trajectory files, protocol and
source closure. It references the unchanged V4 ZIP and baseline capsule
already shipped in the repository. For an extracted source rerun, invoke
`source/cwm_v5/decision_value.py` with those two existing archive paths.

Oracle probe ranking is NOT a formal bound on learned control performance.
These 13 interventions are not the full optimizer library. The team score
does not reproduce all local sequential best responses; its proposed-command
rollout also differs from future CBF-filtered/executed commands. Descriptive
bootstrap samples independent scene groups, not correlated windows, and is
not fresh statistical validation. No capture-rate improvement, terminal-value
extension, 24-step planner or enhanced action execution is introduced here.
