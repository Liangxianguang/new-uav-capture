# V16 new-scene single-axis diagnostic

User authorized new scenarios. Preserve original DN-MPC+CBF, weights, target
rules and reserved holdout. Reference plus wall translation, wider initial
encirclement, shorter wall:8paired layout groups×2mirrors×4variants=64episodes.
One geometry axis changes per variant. Noise episode seeds are matched to
reference group/mirror; unique episode_index values identify artifacts only.

Freeze protocol before collection. Keep route-invalid scenes and all terminal
prefixes, never replace an unsuccessful group. No new model/control enabled.
Eight unchanged V6 joint probes at fixed2/4/6/8,80step held-command stress;
8/16step response is diagnostic, not actual local-MPC candidate evaluation.
Report per-variant legality, original captures/collisions/boundaries/timeouts,
valid prefix, response support and signal groups. All signal/legality gates must
pass before separate fresh confirmation; no best-scene selection by response.

Original eight Levels remain a distinct task benchmark. These new scenes do not
prove coverage, safety or transferable causal mechanisms.

```powershell
python experiments/cwm_v16/collect_scenes.py --output results/cwm_v16/scenes_new
python experiments/cwm_v16/collect_scenes.py --output results/cwm_v16/scenes_repeated_new
python experiments/cwm_v16/replay_scenes.py --data results/cwm_v16/scenes_new --output results/cwm_v16/public_new
python experiments/cwm_v16/scene_release.py --primary results/cwm_v16/scenes_20261010 --repeated results/cwm_v16/scenes_repeated_20261010 --public results/cwm_v16/public_20261010
python experiments/cwm_v16/scene_release.py --verify experiments/cwm_v16/artifacts/new_scene_diagnostic_20261010.zip
```

Use matching capsule runtime for exact comparisons. All run outputs and archive
creation are exclusive; published artifacts/reports are never overwritten.
For independent fresh outputs call the audit API or verify the released ZIP.
No model archive or failed V15 checkpoint is loaded by these scene diagnostics.

Results:all64original episodes safe capture; reference finite scene gate passes,
three new variants fail due to1/2/1target boundary invalid stress branches.
First8/16steps target valid; long80step predeclared gate is NOT relaxed.
Full report:`docs/CWM_V16_NEW_SCENE_DIAGNOSTIC_20261010.md`.
