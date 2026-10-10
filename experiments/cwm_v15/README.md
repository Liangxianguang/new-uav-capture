# V15 preregistered frozen public CBF-mediated target response

Fresh32 mirror groups,24train/8development; episodes978010–978073,
layouts1978010–1978041. Preserve holdout966010–966025. Same qualified
wallx1.5 scenes and fixed steps2,3,4,5,6,8; no future-label sampling.

V14 cbf_history seed976102 is fixed from its error-median selection. It is a
public-input estimator, never the safety layer. Cached estimated commands
are inputs only to the optional two-head core. Raw proposed commands remain
unchanged in the original controller, label masks and full local cost engine.

Three configs x3seeds x80epochs: independent motion-only, raw plain two-head,
mediated plain two-head. Match raw/mediated init, batch RNG and trainable
budget. No private command oracle, branch head, task loss or candidate changes.
Response error must improve over zero and raw; original full-cost gains must
have positive group lower bounds versus own motion/CV/motion-only/raw.
Offline gates never activate control. Two independent runs and full audit
required; failed checkpoints remain unpromoted and original model remains.

Collection uses original V10 collector with this data protocol. Public replay
uses V10 replay_public. Protocol committed and pushed before new data/training
at896c06a721215db2e574c5a430451db89d30aba7. All output paths must be exclusive
fresh directories. Frozen original capsule and dependency archives are required;
use the recorded baseline Python/PyTorch/NumPy environment for byte equality.

```powershell
python experiments/cwm_v10/collect_local.py --capsule experiments/cwm_v1/baseline/capsule.zip --source experiments/cwm_v7/artifacts/paired_training_20261010.zip --geometry experiments/cwm_v7/artifacts/geometry_qualification_20261010.zip --v9 experiments/cwm_v9/artifacts/local_shadow_20261010.zip --protocol experiments/cwm_v15/data_protocol.json --output results/cwm_v15/data_new
python experiments/cwm_v10/replay_public.py --data results/cwm_v15/data_new --capsule experiments/cwm_v1/baseline/capsule.zip --output results/cwm_v15/public_new
python experiments/cwm_v15/train_mediated_response.py --data results/cwm_v15/data_new --public results/cwm_v15/public_new --output results/cwm_v15/training_new
python experiments/cwm_v15/train_mediated_response.py --data results/cwm_v15/data_new --public results/cwm_v15/public_new --output results/cwm_v15/training_repeated_new
```

Independent audit before/after exclusive archive creation:

```powershell
python experiments/cwm_v15/mediated_release.py --data results/cwm_v15/data_20261010 --public results/cwm_v15/public_20261010 --primary results/cwm_v15/training_20261010 --retrained results/cwm_v15/training_repeated_20261010
python experiments/cwm_v15/mediated_release.py --verify experiments/cwm_v15/artifacts/mediated_response_training_20261010.zip
```

The packaging command intentionally refuses to overwrite the published ZIP or
reports. For a new reproduction use fresh run paths and the audit API, or verify
the published artifact directly. The release archive references, rather than
duplicates, the SHA-pinned V14 mediator archive. Tests require published V1–V15
dependency archives. Report: `docs/CWM_V15_FROZEN_CBF_MEDIATED_RESPONSE_20261010.md`.

Completed:64episodes,1152train/384dev calls;840/259complete-support calls.
Two9core×80epoch runs reproduced. Mediated median ADE.325555m versus GRU
.618854m/CV.339228m, response.033658m versus zero.025939m/raw.035159m.
Response and decision gates FAIL. All checkpoints remain unpromoted; control
enhancement stays disabled. Archive43,361,522bytes, SHA256
`4cb80ba58cbd149afc23b9bd1c5733382f13db352167201a6f62ceab71cb978b`.
