# Anchored target-response diagnostic: NO-GO

Six tiny matched residual models were trained around the frozen original GRU
path. No baseline weights are trainable or included in their optimizer.
Identical candidate and anchor commands return the backbone bytes directly;
other responses are bounded per coordinate. This is not a safety guarantee.

V2 data adequacy remains failed. This is a limited offline correction experiment,
not a larger training run, promotion, SCM identification, or causal benefit.
The previously inspected original-train scout groups are split for exploratory
development only. Neither learned response beats the zero-response control.
See `docs/CWM_V3_ANCHORED_RESPONSE_20261009.md`.

## Reproduce on the recorded baseline runtime

```powershell
Expand-Archive experiments/cwm_v3/artifacts/anchored_diagnostic_20261009.zip results/cwm_v3/release_new
python experiments/cwm_v3/train_residual.py --archive experiments/cwm_v2/artifacts/level7_8_scout_20261009.zip --output results/cwm_v3/retrain_new
python experiments/cwm_v3/verify_training.py --run results/cwm_v3/release_new/training --rerun results/cwm_v3/retrain_new --output results/cwm_v3/retrain_audit_new.json
python experiments/cwm_v3/shadow_residual.py --capsule experiments/cwm_v1/baseline/capsule.zip --run results/cwm_v3/release_new/training --output results/cwm_v3/shadow_new
python -m pytest experiments/cwm_v3/test_residual.py -q
```

Use new output directories. The archive contains training data, six weights,
histories, protocol, exact source snapshot and member SHA256 checks. It relies
on the retained CWM-v1 capsule and CWM-v2 scout archive; neither is overwritten.
Checkpoint container bytes/time need not match when retraining; compare actual
model tensors, predictions, histories, groups and metrics using the verifier.
No new online action path or guarded mode is enabled.
