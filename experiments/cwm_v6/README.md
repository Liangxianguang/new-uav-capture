# Authorized S4 mechanism experiment

This is a separate new-scene benchmark, not evidence of Level1-8 improvement.
User authorization: 2026-10-10. Original baseline and local empirical CBF
remain unchanged; no enhanced actions or new weights are enabled here.

Two frozen protocols use disjoint scene/episode seed blocks. Each includes
four independent groups with two mirrored members, two target speeds and
two observation conditions. Mirrors/windows/action probes are not separate
independent scenes. Private branch state is saved only as an audit label.

The original eight-step plan is the anchor. Seven public-geometry pressure
or brake interventions pass through the original local safety filter.
Common step-key RNG and exact clone repeats verify paired interventions;
observed/unobserved full original episodes verify read-only isolation.
Steps 9-16 hold the eighth command; they are not a longer MPC plan.

Both experiments pass the predeclared short-horizon response signal screen.
However, 4/8 scout and 5/8 confirmation original episodes have target-wall
violations. Conservative route existence does not guarantee that the
legacy adaptive-branching target can execute those routes under the preserved
environment dynamics. This is not a validated closed-loop benchmark, and
no new model was trained or promoted. See the stage report for exact results.

Run from the research repository, using the recorded Python/runtime:

```powershell
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v6/s4_collect.py --capsule experiments/cwm_v1/baseline/capsule.zip --output results/cwm_v6/scout_fresh
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v6/s4_collect.py --capsule experiments/cwm_v1/baseline/capsule.zip --protocol experiments/cwm_v6/s4_confirmation_protocol.json --output results/cwm_v6/confirmation_fresh
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v6/s4_release.py --verify experiments/cwm_v6/artifacts/s4_mechanism_20261010.zip
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' -m pytest experiments/cwm_v1 experiments/cwm_v2 experiments/cwm_v3 experiments/cwm_v4 experiments/cwm_v5 experiments/cwm_v6 -q
```

Output paths must not already exist. The archive includes actual paired data,
all scene metadata and trajectories, and the exact source bytes for each run.
The scout source predates a relative-path resolution/snapshot-storage fix;
its preserved hash is verified separately, never silently substituted.
Archive audit recomputes statistics, checks terminal masks and full snapshot
coverage, verifies scene-group disjointness and compares actual trajectory
arrays. Package generation uses exclusive creation and does not overwrite
milestone evidence. Exact reproducibility is local CPU/runtime scoped.
