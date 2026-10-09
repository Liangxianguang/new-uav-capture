# Asymmetric original-TRAIN interaction diagnostic

This stage does not train or enable a controller. The previous response models
failed to beat zero response. The predeclared follow-up tests public-geometry
single-agent left/right/brake commands on fresh Level5/6/7/8 TRAIN groups, with
the original target rules, encoder, GRU, DN-MPC, local CBF and execution intact.

Eight planned commands are unchanged for the reference branch. Steps 9-24
repeat the eighth command, not a longer DN-MPC plan. All branches pass through
the original local CBF. Private target fields are diagnostic labels only.
No windows are selected by private target mode or observed response strength.

Every window checks deterministic repeated reference branches and hashes the
complete parent environment dictionary before/after cloned intervention.
Each of the ten original TRAIN episodes is rerun without the observer;
actual target/defender trajectory arrays must be byte-identical and all
physical outcomes equal. This proves observer isolation on these scenes,
not safety or success of an enhanced controller.

Run from the research checkout with the recorded Python environment:

```powershell
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v4/diagnose.py --capsule experiments/cwm_v1/baseline/capsule.zip --training-scenes D:/uav-capture/new-uav-capture/results/phase91_zero_cbf_curriculum_v5/train/scenes.jsonl --exclude-scenes results/cwm_v1/pilot_pairs_20261009/selected_scenes.jsonl --exclude-scenes results/cwm_v2/scout_level7_8_20261009_v2/selected_scenes.jsonl --exclude-scenes results/cwm_v2/confirmation_level7_8_20261009/selected_scenes.jsonl --output results/cwm_v4/asymmetric_20261009
```

The released ZIP includes the original TRAIN pool, all excluded groups,
selected scenes, raw paired arrays, observed/unobserved trajectories,
protocol, statistics and source closure. To rerun after extraction,
use `source/cwm_v4/diagnose.py`, `inputs/train/scenes.jsonl`, each
`inputs/excluded/*.jsonl` as an explicit `--exclude-scenes` argument,
and the unchanged baseline capsule from the repository. Use a new output
directory; original artifacts are never overwritten.

```powershell
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' -m pytest experiments/cwm_v4/test_diagnostic.py -q
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v4/analyze_latency.py --run results/cwm_v4/asymmetric_20261009
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v4/release.py --run results/cwm_v4/asymmetric_20261009 --training-scenes D:/uav-capture/new-uav-capture/results/phase91_zero_cbf_curriculum_v5/train/scenes.jsonl
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v4/release.py --verify experiments/cwm_v4/artifacts/asymmetric_diagnostic_20261009.zip
```

The 5cm/5%-points/two-groups signal screen is a data heuristic, not a
physical necessity, significance test, promotion gate or safety contract.
Common-prefix statistics and complete-horizon endpoint support must be
read together. Any apparent signal still requires independently sampled
confirmation; no S4 scene, new adversary, holdout tuning or enhanced
action selection is authorized by this experiment.

The post-hoc latency analysis reports the first observed 5cm threshold
crossing, not an identified physical response delay. It never changes the
fixed screen or promotes a failed run. Release auditing recomputes both the
screen metrics and this diagnostic from the raw arrays.
