# V22: diagnose V21 decision failure without promoting a model

This is explicitly **posthoc** analysis of already inspected train/development.
It cannot rescue V21 qualification, tune thresholds or report fresh validation.
The original controller, target, safety contract and reserved holdout stay fixed.
The V21 archive and all final models/seeds are pinned; full V21 semantic audit
is required before diagnosis. Publish this protocol before executing analysis.

Compare actual unique and geometry-union candidates on identical complete-union
support. Replace the common-motion prediction or anchored response with truth
separately, using the unmodified original complete cost. Average both replacement
orders to decompose regret; negative contributions remain negative, not clipped.
Truth paths are diagnostic oracles, never controller inputs or deployable bounds.
Report every seed/configuration, candidate costs, help/harm and margins, train/dev
generalization, group-equal summaries and paired descriptive bootstrap intervals.
Do not change V21's predeclared primary, selection, gates or failed qualification.

## Reproduction

Use the same recorded Windows Python environment as V21. The initial protocol
was published as `518e81d` before analysis, after V21 results were inspected.

```powershell
python -m pytest -q experiments/cwm_v22
python experiments/cwm_v21/loss_training_release.py --verify experiments/cwm_v21/artifacts/response_loss_factorial_20261010.parts.json
python experiments/cwm_v22/decision_diagnostics.py --output results/cwm_v22/primary
python experiments/cwm_v22/decision_diagnostics.py --output results/cwm_v22/repeated
```

Each diagnostic process first verifies the exact full archive and re-infers
all30models, train/dev forecasts, full original costs, gates, optimizer/RNG and
public252 frames via V21's unchanged full audit. It reconstructs both splits'
contexts through the original frozen engine, checks833/272complete-union calls,
then scores all diagnostic paths. An exact-path cache shares repeated scores
only within an identical call; no approximation or alternative cost is used.
Actual candidate scores and selected action must reproduce the copied original
solver. Every development cost/choice/regret shared with V21 must match exactly.
Each split/model/population saves full decision records, not only a summary.
Two independent outputs must match before reporting reproducible conclusions.

Archive **all**61evidence files from each run (60decision sets plus summary),
not just summary metrics. The record audit checks original complete costs,
choices, tie flags, regrets, order-averaged repairs, group summaries and paired
bootstrap values, and compares every evidence byte across the two runs.
Full release verification then independently re-executes the complete pinned
V21 audit and every V22 oracle-path original-engine calculation, comparing all
recomputed record bytes with the release. Matching hashes alone is insufficient.

```powershell
python experiments/cwm_v22/diagnostic_release.py --primary results/cwm_v22/primary --repeated results/cwm_v22/repeated --output results/cwm_v22/reproduced_diagnosis.zip
python experiments/cwm_v22/diagnostic_release.py --verify results/cwm_v22/reproduced_diagnosis.zip --recompute-output results/cwm_v22/verify_recomputed
```

Both output paths must be new. The archive contains no baseline overwrite or
new trained model; it retains V21's failed qualification and requires the pinned
published V21 artifact parts to re-execute verification.
