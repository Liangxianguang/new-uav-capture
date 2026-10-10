# V35 bounded-memory training preparation (no trainer / no new weights)

This package is separate from the already-running, preregistered V34 source
closure. It implements prerequisites and batch primitives for the full fixed
V34 comparison, NOT a smaller replacement experiment or deployment entry.
The scientific trainer, repeats, reload and closed-loop qualification remain
required. No V35 native preparation or scientific optimizer has run.

`AuditedCorpus(data, audit)` accepts only COMPLETE512episode V34 collection
plus terminal native independent audit, exact sources, all declared data
evidence, ALL sequential-call/augmented native-cost records and TRAIN48 /
development16 disjoint families. It streams EVERY supported call for physical
prefix/input checks before becoming usable; corrupted, missing or partial
evidence is refused. Each later load rechecks file identity. No physical arrays,
contexts or tensors are cached; only bounded16-call batches are returned.

`corpus.normalization()` uses ALL supported TRAIN histories in recorded order,
the existing fixed float64 block-Welford algorithm and original0.01 floor.
No development histories, candidate multiplicity or truth masks weight it.
`population_weights(corpus)` computes full-TRAIN equal-group / augmented-point
physical weights and BASE-only complete-call cost weights. Prefix probe suffix
invalidity cannot silently remove an otherwise complete original base-library
ranking call. Weight normalization uses fsum for its numerical CHECK only;
the original scalar weight formula and fixed batch denominator are unchanged.

`public_batch` reuses original six public tensors (five neural tensors plus
analytic CV). Target truths/costs never enter predictor inputs.
`batch_objective` reuses original vectorL2 +0.001 energy on ALL augmentation,
but applies0.1 original-cost KL ONLY to the leading unchanged base candidates.
No larger-library decision gain, truth-driven proposal selection, model/gate
switching, new target rules or original GRU/CBF modification is introduced.

The caller STILL must implement/test/publish the full fixed trainer and perform
all original-cost/finite-difference audits at public GRU/CV references before
the motion optimizer and at the frozen deployed motion before the response
optimizer. It must retain three fixed seeds / final80 epochs / two full9model
runs and independently reload them. Fixed development gates, original8Levels,
new-scene paired capture/safety/latency, fallback and untouched holdout remain.
The V34 protocol's `training_entry_implemented=false` is not changed while it
is a live data-stage source; V35 currently ALSO has no scientific trainer.

Tests are synthetic/mocked evidence authorities and toy tensor gradients,
not native trajectories, neural optimization, causal identification or effects.
No deployment eligibility is granted by this package or its passing tests.
