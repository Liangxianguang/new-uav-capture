# CWM-v2 training-data adequacy screen

No new online controller or promoted weights. Original DN-MPC/local CBF is
preserved by the verified CWM-v1 capsule. This stage screens fresh train groups
before spending a larger training budget. It is not a formal Level1-8 test.

The first scout and a disjoint confirmation together contain 12 groups and
51 windows from fast_target/agile_target/sensor_degraded. All failed the
predeclared adequacy gate. See `docs/CWM_V2_LEVEL7_8_SCOUT_20261009.md`.

## Verify and inspect

```powershell
python experiments/cwm_v2/verify_scout.py
Expand-Archive experiments/cwm_v2/artifacts/level7_8_scout_20261009.zip results/cwm_v2/release_new
python -m pytest experiments/cwm_v1/test_cwm_v1.py experiments/cwm_v1/test_release_and_diagnostics.py experiments/cwm_v2/test_scout.py -q
```

The archive contains exact data, scenes, source and hashes for each collection.
Target truth is labels only. Target private mode/route fields are diagnostic
logs, not model inputs. Original GRU predictions are copied without an extra
sampling call; the original predictor and controller are not changed.

## Recollect (requires the original train pool)

```powershell
python experiments/cwm_v2/scout.py --capsule experiments/cwm_v1/baseline/capsule.zip --training-scenes D:/uav-capture/new-uav-capture/results/phase91_zero_cbf_curriculum_v5/train/scenes.jsonl --exclude-scenes results/cwm_v1/pilot_release/dataset/selected_scenes.jsonl --output results/cwm_v2/scout_new
python experiments/cwm_v2/prepare_confirmation.py --scenes results/cwm_v1/pilot_release/dataset/selected_scenes.jsonl --scenes results/cwm_v2/scout_new/selected_scenes.jsonl --output results/cwm_v2/exclusions_new.jsonl
python experiments/cwm_v2/scout.py --capsule experiments/cwm_v1/baseline/capsule.zip --training-scenes D:/uav-capture/new-uav-capture/results/phase91_zero_cbf_curriculum_v5/train/scenes.jsonl --exclude-scenes results/cwm_v2/exclusions_new.jsonl --protocol experiments/cwm_v2/confirm_protocol.json --output results/cwm_v2/confirmation_new
python experiments/cwm_v2/check_tap.py --capsule experiments/cwm_v1/baseline/capsule.zip --output results/cwm_v2/tap_new
```

Use fresh directories and the recorded CWM-v1 runtime. The current collector
adds diagnostic metadata fields compared with scout #1; its exact old source
is archived. Numerical recollection can be compared; compressed archive bytes,
metadata elapsed time and a newer source hash need not match.

Continuation: seek a mechanism-rich existing scenario with user approval,
then train an action-response residual around the frozen backbone with matched
plain and zero-response controls. No adequacy pass or model name authorizes
guarded control; independent prediction, calibration and closed-loop gates
remain required.
