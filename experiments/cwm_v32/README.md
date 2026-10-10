# V32: fixed TRAIN proposed-command prefix sensitivity

This is a post-hoc diagnostic, NOT another trained model or an active capture
result. Existing V28 cv_l2 and cv_rank_l2 ADE-median seed994101 stay fixed; the
comparison cannot select a new seed/configuration or override failed gates.

Problem-first/failure-boundary decomposition asks whether ALL8-action flattening
lets later commands influence earlier predicted responses. Native collection
`cwm_v7/collect_training.py:s4_rollout` feeds only the current command at each
offset to local CBF and the original step. The original `pursuit_env.step`
accepts a4x3 current command, updates defenders then target. No future command
is passed into that physical step. Existing V10 plain response features, in
contrast, flatten the entire8x4x3 sequence before predicting all8 offsets.
This motivates an invariance diagnostic, NOT a demonstrated explanation of
V28's decision failure or a guaranteed novel/improving replacement.

For all2304 TRAIN calls/all24 groups and all saved candidates, replace only
suffix k:8 with the unchanged public anchor commands, k=1..8. Preserve every
other public input. Both scored models and identical-input roundoff controls
run in the same batch. k=8 must be reported as the no-change control. Raw
differences are never adjusted by control noise. Valid masks describe original
support only; no private labels select interventions or enter the model.
Results retain unchanged candidates, zero-valid rows, masks and group support.

Full data-manifest integrity includes development file hashing, but development
arrays are not opened for normalizers or statistics. Full TRAIN normalization
is recomputed and matched to both exact checkpoint identities/provenance.
Sources are checked against the frozen data/training snapshots, then actual
run-source snapshots are saved and rechecked. No changes to old scientific
dependencies, original252 encoding, GRU, target rules, CBF or running jobs.

Suffix splicing may be outside the original candidate-library distribution.
This stage does NOT reroll these synthetic sequences in the original engine.
Sensitivity is not predictive accuracy, causal identification or capture gain.
Even large sensitivity would require a fresh matched prefix-causal vs full
sequence training experiment; more independent geometry coverage is another
unresolved hypothesis. No promising TRAIN diagnostic may promote a module.

Run only AFTER exact source/protocol HEAD is actually pushed and verified.
Use a NEW short output; do not change used sources while it is running:

```powershell
python -m pytest experiments/cwm_v32 -q
python experiments/cwm_v32/prefix_invariance.py --data results/cwm_v28/fresh_sequential_ranking_20261010 --audit results/cwm_v28/independent_sequential_ranking_audit_20261010 --primary results/cwm_v28/primary_fixed_ranking_20261010 --output D:/uav-capture/cwm32-prefix-20261011
```

Tests use explicitly synthetic causal/noncausal models and masks. They are
implementation checks, not the real fixed-model TRAIN diagnostic. No actual
V32 completion or improvement is claimed by this source/protocol publication.

All45 fixtures passed before publication: all8 prefixes, invalid shape/dtype/
nonfinite inputs, exact prefix preservation, no private forward features,
causal/noncausal toy distinctions, k=8 and repeated-input controls, masked
group-point statistics, exclusive output, unpublished-source refusal, complete
toy IO with development arrays never opened, all prior file-identity rejection,
and in-run data/manifest/normalizer/checkpoint/protocol mutation rejection.
The toy full-workflow report marks SYNTHETIC/MOCKED scope; it is not actual
V28 inference. No scientific experiment starts if exact GitHub publication
cannot be confirmed.

When Git HTTPS transport fails, exact branch-HEAD verification may use the
official authenticated GitHub Git-data API with the existing noninteractive
Git credential. It still requires remote SHA == local HEAD and exact committed
source/protocol/publisher bytes; the publisher dependency is also snapshotted.
This does not waive prepublication or any scientific gate. No credentials are
printed/stored and no network/Git settings are changed. Fallback publication
accepts only exact matching blob/tree/commit objects and a nonforce ref update.
