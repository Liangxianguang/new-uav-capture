# V27: bounded public self-generated research candidates

The fixed failed V23 `cv_cost_l2 seed993102` is a public proposal mechanism,
NOT a qualified controller. Default off/refusal/failure returns the original
candidate order and multiplicity without permitting any new action selection.

Each supported sequential local context starts from the unchanged V18 geometry
pool. Public CV / common motion / action-specific target forecasts go through
the original local candidate generator on independent copies. At most 24 unique
finite `[8,3]` local UAV sequences and two expansion rounds are permitted. Target
paths `[N,8,3]` are not joint actions `[N,8,4,3]`. Original candidates and selected
anchor remain unchanged; final pool is re-forecast with actual received delayed
peer plans. Missing contexts are skipped, never filled using unreceived plans.

The replay independently re-encodes public history, reobserves ALL sequential
invocations in the two V26 historical records, and compares them to the audited
V26 public trace. Expansion runs after original plan, not inside an active local
selector; no generated action is executed. Two-process comparisons cover final
actions, public inputs, predictions and proposal provenance. These are candidate
availability and reproducibility evidence, not ranking/capture/latency benefits
or untouched holdout/full-Level validation.

```powershell
python -m pytest experiments/cwm_v27 -q
python experiments/cwm_v27/run_sequential_candidates.py --output results/cwm_v27/NEW_FIRST
python experiments/cwm_v27/run_sequential_candidates.py --output results/cwm_v27/NEW_SECOND
python experiments/cwm_v27/run_sequential_candidates.py --primary results/cwm_v27/NEW_FIRST --repeated results/cwm_v27/NEW_SECOND --artifact results/cwm_v27/NEW_RELEASE.zip --report results/cwm_v27/NEW_REPORT.json
```

Do not overwrite evidence/output directories. V28's fresh candidate-ranking
experiment separately freezes data identities and the new loss before collection.
Original GRU/DN-MPC/local CBF, safety settings and V23 failure remain unchanged.
