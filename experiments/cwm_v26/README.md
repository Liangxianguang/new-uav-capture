# V26: real-checkpoint sequential public shadow, default off

This implementation regression retains the FAILED V23 primary `cv_cost_l2`
seed993102 (the previously fixed ADE-median seed, not best-seed selection).
It does not train a new model, override qualification or enable a controller.

The optional loader verifies the complete V23 archive/summary and checkpoint
identity, preserves CPU RNG and applies the original train-only normalization.
Every local-selection invocation is copied, including subsequent DN-MPC rounds.
Real original252 history, received delayed peer context and original GRU/CV
paths are passed to the actual checkpoint. Missing peers are skipped, never
filled with zero or privileged current plans. Input preparation and inference
run after the original plan returns; snapshot copy cost stays in the original
timeout window. No deadline, planner, candidate pool, CBF or safety margin changes.

Two independent six-mode runs on two fixed historical records have 412 sequential
calls per fault/shadow mode: 388 complete calls and 24 missing-peer calls. All388
complete calls produce real forecasts in shadow. Off/refusal never load; actual
checkpoint load/forward then controlled failures also preserve the original
historical trajectories, DN-MPC plans and post-CBF commands.

The release verifier INDEPENDENTLY captures all sequential calls using the
original public pre-step observer (not the runtime history hook), reconstructs
received messages/candidates/costs/choice, and reinfers every real forecast from
the hash-pinned actual checkpoint. It is not a checksum-only or synthetic test.

Still NOT proved: active local-selector integration, self-generated candidates,
new-data decision gain, holdout, full original Levels/new-scene enhanced
capture/safety/latency qualification. These two inspected regression records
cannot become training or model-selection evidence. local CBF remains empirical.

```powershell
python experiments/cwm_v26/real_shadow_entry.py --output results/cwm_v26/fresh_primary
python experiments/cwm_v26/real_shadow_entry.py --output results/cwm_v26/fresh_repeated
python experiments/cwm_v26/release_real_shadow.py --primary results/cwm_v26/fresh_primary --repeated results/cwm_v26/fresh_repeated --output results/cwm_v26/fresh_release.zip --report results/cwm_v26/fresh_release.json --audit-output results/cwm_v26/fresh_release_audit
python experiments/cwm_v26/release_real_shadow.py --verify experiments/cwm_v26/artifacts/real_sequential_shadow_20261010.zip --audit-output results/cwm_v26/fresh_verify
python experiments/cwm_v26/check_shadow_evidence.py --output results/cwm_v26/fresh_semantic_checks
python -m pytest experiments/cwm_v21 experiments/cwm_v22 experiments/cwm_v23 experiments/cwm_v24 experiments/cwm_v25 experiments/cwm_v26 -q
```

Run replay/release in separate Python processes. Use fresh output directories;
do not overwrite original or existing results. The small V26 release includes
the exact failed primary checkpoint for verification, not as a qualified model.
