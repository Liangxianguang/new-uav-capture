# V14 preregistered public mechanism pilot

Fresh32 mirror groups,48 train/16 development episodes,975010–975073;
layouts1975010–1975041. Preserve holdout966010–966025. Fixed snapshots2,3,4,5,6,8.
No future-label window selection, target-rule changes, or control activation.
The existing qualified wallx1.5 finite distribution and frozen V9 library are
retained. New sample seeds and windows do not establish geometry or8Level transfer.

Train3 configurations x3seeds x40epochs: snapshot-only branch, public-history
branch, public-history CBF-command estimator. True3-class branch state and
filtered commands are supervision ONLY. Matched per-snapshot/offset training
prior, zero paired-branch response and raw proposed commands are controls.
History/geometry features remain original public252 and delayed4x6 context.
No simulator-private labels, target state, future execution or terminal masks
are network inputs. Neither model is a new executable controller.

Protocols are fixed before fresh collection/labels/training. All gates are
offline mechanism gates, not performance promotion. Two independent training
runs must reproduce checkpoint/optimizer/RNG/history/prediction evidence.
If unsuccessful, keep original GRU + delayed DN-MPC + empirical local CBF.

Collection reuses the unmodified V10 collector with this data protocol; all
outputs must be exclusive fresh directories. Fixed commands:

```powershell
$taskPython = 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe'
& $taskPython experiments/cwm_v10/collect_local.py --capsule experiments/cwm_v1/baseline/capsule.zip --source experiments/cwm_v7/artifacts/paired_training_20261010.zip --geometry experiments/cwm_v7/artifacts/geometry_qualification_20261010.zip --v9 experiments/cwm_v9/artifacts/local_shadow_20261010.zip --protocol experiments/cwm_v14/data_protocol.json --output results/cwm_v14/fresh_data
& $taskPython experiments/cwm_v10/replay_public.py --data results/cwm_v14/fresh_data --capsule experiments/cwm_v1/baseline/capsule.zip --output results/cwm_v14/fresh_public
& $taskPython experiments/cwm_v14/replay_mechanism_labels.py --data results/cwm_v14/fresh_data --output results/cwm_v14/fresh_mechanism
& $taskPython experiments/cwm_v14/replay_mechanism_labels.py --data results/cwm_v14/fresh_data --output results/cwm_v14/fresh_mechanism_repeated
& $taskPython experiments/cwm_v14/train_public_mechanisms.py --data results/cwm_v14/fresh_data --public results/cwm_v14/fresh_public --mechanism results/cwm_v14/fresh_mechanism --output results/cwm_v14/fresh_training
& $taskPython experiments/cwm_v14/train_public_mechanisms.py --data results/cwm_v14/fresh_data --public results/cwm_v14/fresh_public --mechanism results/cwm_v14/fresh_mechanism --output results/cwm_v14/fresh_retraining
```

Authoritative completed collection/public/two-label runs:
`results/cwm_v14/data_20261010`, `public_20261010`, `mechanism_20261010`,
`mechanism_repeated_20261010`. 64/64 original safe captures, zero collision,
boundary or target-invalid episodes. All original episode arrays equal;
1536 actual public252 histories/backbones and candidate branches reproduced.
These are original-controller outcomes, not enhanced-controller performance.
1152 train calls/24 groups and384 development calls/8 groups; all1536 labels
and records byte-equal between independent label processes. Both fixed
training processes finished, with weights/optimizer/RNG/history equality
and independent18-checkpoint prediction/gate recomputation verified.
Snapshot/history branch gates FAILED; CBF-command estimator passed this
finite offline mechanism gate: median error0.198601 versus proposed0.647881
m/s,69.35% improvement, all3seeds improve, group95% lower gain0.408563m/s.
No response/decision/closed-loop or deployment promotion follows from this.
See `docs/CWM_V14_PUBLIC_MECHANISM_PILOT_20261010.md` for failure details.
Artifact47,485,035bytes, SHA256
`9ebac503f906c94be2185ec0121319cf10085635253a69a27651c1070c52f59f`.

Release creation in a fresh clone with no existing artifact:

```powershell
& $taskPython experiments/cwm_v14/public_mechanism_release.py --data results/cwm_v14/fresh_data --public results/cwm_v14/fresh_public --mechanism results/cwm_v14/fresh_mechanism --mechanism-repeated results/cwm_v14/fresh_mechanism_repeated --primary results/cwm_v14/fresh_training --retrained results/cwm_v14/fresh_retraining
& $taskPython experiments/cwm_v14/public_mechanism_release.py --verify experiments/cwm_v14/artifacts/public_mechanism_training_20261010.zip
& $taskPython -m pytest experiments/cwm_v14 -q
```

Models have21072/35016/39696 parameters respectively. Snapshot/history are
explicit differing-capacity baselines; performance differences cannot isolate
memory or causality. CBF output is an estimator and never replaces the frozen
safety filter. Branch labels use target-valid prefix, command labels observed
prefix; duplicate anchor candidate plus separate reference are disclosed and
excluded from paired-effect metrics. Calibration uses ten fixed confidence
bins with group/call weighting. All priors/normalization are TRAIN-only.
Fixed forty epochs and all three seeds are retained, even when gates fail.
Complete V1–V14 regression:165 passed in352.01s. Artifact verification passed
again independently after packaging. Baseline capsule SHA remains unchanged.
