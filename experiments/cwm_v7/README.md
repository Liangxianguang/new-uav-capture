# S4 geometry qualification and matched response-training preparation

User authorized new scenes. Original Levels, GRU seed745101, 252-feature
encoder, distributed-delayed eight-step DN-MPC and empirical local CBF remain
unchanged. New-scene geometry only: move the existing S4 wall to x=1.5m,
including its declared zone; all agents, exits and target rules are unchanged.

The geometry milestone is complete. Its development grid x=0,1.5,3.0 selects
the smallest passing translation using target validity only, never capture
or response strength. Fresh qualification uses separate seeds and twice the
groups. Finite stress branches stop on terminal conditions and expose censoring
and defender failures; they are not certified safe behavior or a longer MPC.

The training protocol is separately frozen. It preassigns 24 mirror groups
(48 episodes) to train and 8 (16 episodes) to development validation. Another
8 groups remain reserved and uncollected for a future frozen-model holdout.
Geometry development/qualification data never enter model validation or test.
Ordinary and graph-structured predictors use identical public inputs, paired
labels, auxiliary branch-label supervision, optimizer, seeds and fixed epochs.
Private branch labels are not inputs. Both return the original supplied GRU
trajectory exactly at the anchor command. This is not a strict CBF certificate,
SCM-identification proof or controller promotion.

Use new output directories, recorded local CPU/runtime, and repository root:

```powershell
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v7/diagnose_geometry.py --capsule experiments/cwm_v1/baseline/capsule.zip --v6-archive experiments/cwm_v6/artifacts/s4_mechanism_20261010.zip --output results/cwm_v7/geometry_fresh
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v7/diagnose_geometry.py --capsule experiments/cwm_v1/baseline/capsule.zip --v6-archive experiments/cwm_v6/artifacts/s4_mechanism_20261010.zip --protocol experiments/cwm_v7/geometry_confirmation_protocol.json --output results/cwm_v7/qualification_fresh
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v7/geometry_release.py --verify experiments/cwm_v7/artifacts/geometry_qualification_20261010.zip
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v7/collect_training.py --capsule experiments/cwm_v1/baseline/capsule.zip --development results/cwm_v7/geometry_fresh --qualification results/cwm_v7/qualification_fresh --output results/cwm_v7/data_fresh
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v7/train_response.py --data results/cwm_v7/data_fresh --output results/cwm_v7/training_fresh
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' -m pytest experiments/cwm_v1 experiments/cwm_v2 experiments/cwm_v3 experiments/cwm_v4 experiments/cwm_v5 experiments/cwm_v6 experiments/cwm_v7 -q
```

The immutable geometry archive contains complete traces, actual trajectory/
stress arrays and source bytes as run. Later training preparation sources
are maintained in this repository; the geometry archive does not pretend
to contain model-training results. The completed training milestone is now
`artifacts/paired_training_20261010.zip`, containing the actual64-episode data,
two independent six-model runs and exact sources. The development gate fails:
response error improves over zero, but the structured model does not beat the
ordinary model and total target ADE does not improve. No model is promoted.
See `docs/CWM_V7_PAIRED_TRAINING_20261010.md`.

```powershell
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v7/train_response.py --data results/cwm_v7/data_fresh --output results/cwm_v7/retraining_fresh
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v7/training_release.py --verify experiments/cwm_v7/artifacts/paired_training_20261010.zip
& 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe' experiments/cwm_v7/forecast_support.py --data results/cwm_v7/data_fresh --training results/cwm_v7/training_fresh --output results/cwm_v7/support_diagnostic_fresh.json
```

The post-training support/oracle report is a separately versioned development
diagnostic, not a changed training gate or holdout claim. Exact replays are scoped
to the recorded CPU/runtime, not arbitrary hosts or deterministic latency.
