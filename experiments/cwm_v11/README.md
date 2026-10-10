# Full frozen local MPC score and task-effect audit

V10 gate remains failed. This stage does not train/promote a controller. Numerical
score parity uses all768 calls; mechanism diagnosis uses TRAIN ONLY full support.
Reserved holdout is untouched. See `docs/CWM_V11_TASK_SCORE_AUDIT_20261010.md`.

`task_score.py` preserves every target-independent original safety/control term
and implements the exact target-dependent frozen K=1 objective. Unsupported
QDR/reachability/escape-gap/FC extensions raise rather than silently disappear.
Discrete interceptor roles and norm/abs kinks only permit piecewise derivatives.
The auxiliary loss matches observed paired COST EFFECTS with true reference
labels, not low predicted costs. It must supplement physical response loss,
not replace it. Incomplete terminal support is rejected for this auxiliary term.

```powershell
$taskPython = 'D:/miniconda3/envs/uav-encirclement-gpu/python.exe'
& $taskPython experiments/cwm_v11/audit_task_score.py --output results/cwm_v11/fresh_primary
& $taskPython experiments/cwm_v11/audit_task_score.py --output results/cwm_v11/fresh_repeated
& $taskPython experiments/cwm_v11/task_score_release.py --verify experiments/cwm_v11/artifacts/full_local_score_audit_20261010.zip
& $taskPython -m pytest experiments/cwm_v11 -q
```

Output directories must be new. Authoritative runs are
`audit_qualified_20261010` and `audit_repeated_20261010`; initial draft audit
was retained, not overwritten. Published archive was created exclusively:

```powershell
& $taskPython experiments/cwm_v11/task_score_release.py --primary results/cwm_v11/audit_qualified_20261010 --repeated results/cwm_v11/audit_repeated_20261010
```

The archive verifier reloads V10 models and recalculates full original-engine
costs, gradients and train-only response attribution; it does not only audit
stored claims. It checks run-used source closure. Baseline source comes from
the pinned original capsule. Reverification on another runtime may require
fresh qualification; this is empirical arithmetic evidence, not a theorem,
multi-scenario generalization, a latency result or a safety certificate.
