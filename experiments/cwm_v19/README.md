# V19 frozen-common-motion causal geometry response

Preregister before collection or training. Fresh32 mirror groups, first24 train
and last8 development, separate from V18 pilot and reserved holdout. Original
controller and CBF are never modified; all new candidates are shadow only.

V18 showed actionable paired-response labels but also that exact response added
to biased motion can worsen ranking. Stage1 calibrates common reference motion;
stage2 trains raw/mediated response using a separate core, with the shared
stage1 reference prediction fully frozen. Within-seed response comparisons
therefore cannot attribute a changed common-motion model to response effects.

All three seeds, final fixed epochs and two deterministic runs are required.
Candidate library gain and learned response gain remain separate. Offline
gates do not activate a controller; previous failures remain recorded.

## Reproduction

Use separate Python processes and exclusive output directories. The standalone
experiment scripts share historical helper modules; combined pytest imports
are isolated by the local conftest to avoid V12/V19 `data_audit` cache collision.
The runtime experiment sources and the historical experiments are unchanged by
that test-only import isolation.

```powershell
python experiments/cwm_v19/collect_geometry_data.py --output results/cwm_v19/fresh_data
python experiments/cwm_v19/replay_geometry_public.py --data results/cwm_v19/fresh_data --output results/cwm_v19/public_replay
python experiments/cwm_v19/train_frozen_response.py --data results/cwm_v19/fresh_data --public results/cwm_v19/public_replay --output results/cwm_v19/primary
python experiments/cwm_v19/train_frozen_response.py --data results/cwm_v19/fresh_data --public results/cwm_v19/public_replay --output results/cwm_v19/retrained
python experiments/cwm_v19/frozen_training_release.py --data results/cwm_v19/fresh_data --public results/cwm_v19/public_replay --primary results/cwm_v19/primary --retrained results/cwm_v19/retrained --output results/cwm_v19/reproduced_training.zip
python -m pytest -q experiments/cwm_v19
```

The collector admits training only after overall and split data gates. The
trainer independently audits public frames/features and frozen original costs
before entering its optimizer. The release verifier reloads all18 trained
models across two runs, reproduces public forecasts and original full-cost
rankings, confirms matched frozen motion weights and compares entire checkpoint
trees including optimizer and RNG states. No learned model is promoted by it.
