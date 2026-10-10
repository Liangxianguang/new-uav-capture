# V37 independent full streamed model reload (not scientifically executed)

This entry requires COMPLETE V34 native data/independent audit and BOTH complete
V36 nine-model runs. It never trains, updates an optimizer, changes a seed/epoch,
selects a favorable model, relaxes a gate or enables a controller. The existing
scientific sources remain unchanged, including live V34 collection closure.

`evidence.py` inventories ALL declared source/protocol snapshots, checkpoints,
80epoch histories, per-call canonical references/score audits/predictions,
decision JSONL/population files, normalizer and weights. It rechecks actual
preregistered Git source bytes and individual checkpoint serialized identities.
Only final checkpoint serialization SHA is excluded from cross-run summary
semantics AFTER each actual file is verified. ALL12 final/motion-stage checkpoint
contents must match weights/Adam/RNG/provenance, and complete history bytes
must match. Canonical/cache/prediction/decision/audit evidence bytes must match;
summary/preregistration wrapper bytes may differ and are validated individually.
No scientific fields are removed or redefined to rescue a failed model.

`checkpoint_reload.py` creates the original matched architectures, reconstructs
fixed initialization, loads strict model tensors and actually restores real
Adam state plus sampler/Torch RNG, without optimizer.step. ALL trained moments
and fixed80epoch steps/options must exist. The unused motion-model response
head is the only optimizer member without trained moments. Frozen common weights
must match each per-seed motion stage in all models. Temporary RNG restore is
isolated and leaves the caller RNG unchanged.

`reload_streamed.py` independently reexecutes every supported-call prediction
for ALL18 final models in the same fixed global16-call blocks. Every tensor
dtype/shape/byte must match saved arrays and canonical common references. It
reruns original-engine true/local/decision costs and all prescribed derivative
checks at GRU/CV and deployed common references, checks all detached cache
tensors, and independently rebuilds ALL three original-library decision rows.
The extra augmented probes remain physical evidence, not a new ranking library.

It recomputes TRAIN-only normalizer/weights, public-input precision, GRU/CV
controls, all point/group/axis/ranking metrics, original-library contributions,
fixed final ADE-median selection and strict original/matched-control gates.
JSONL decisions and arrays stay on disk; bounded batches and scalar statistics
keep physical/model outputs out of full-population RAM caches. Complete source,
run and native data evidence is reverified before issuing the final certificate.

## Entry

Use the corresponding Git checkout/preregistration history and original Python
runtime. Run ONLY after actual completion of full native data audit and both
fixed training runs; output must be NEW:

```powershell
python experiments/cwm_v37/reload_streamed.py --data <complete_V34_data> --audit <complete_V34_audit> --primary <complete_primary_run> --repeated <complete_repeated_run> --output <NEW_reload_directory>
```

No native training/reload has run at this stage. Tests use toy costs, mocked
audit authorities and tiny synthetic80epoch neural optimizers, solely checking
restoration/plumbing/mutation refusal. They are not trained V34 models, native
effect measurements, causal identification, capture/safety or latency evidence.

Controlled packaging/extracted full replay, original8Levels/new-scene paired
closed-loop capture/safety/latency, active fault fallback and untouched original
holdout still remain. Passing reload does not promote a model, certify the
whole project, or change original V28 failed gates. Enhancement stays off.
