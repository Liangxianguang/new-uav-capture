# V31: exact public-input ambiguity, TRAIN only

This is a post-hoc information diagnostic of the failed V28 model, not a new
training stage or a new model qualification. The earlier development failure
informed the question; the protocol is published before the diagnostic run.
No claimed novelty or guaranteed explanation of the failure.

The existing V28 core flattens public history/context/action features and
returns an anchored continuous response. Prior graph-structured, branch and
CBF-mediator experiments remain failed; they are not relabeled as new ideas.
Following failure-boundary/decomposition guidance from the research ideation
skill, this diagnostic first checks one falsifiable information bottleneck:
can identical actual public feature tensors have incompatible response labels?

Only ALL2304 V28 TRAIN calls/24groups enter the statistics. The complete prior
data manifest, including development files, is hashed to preserve scientific
integrity; development arrays are never loaded into diagnostic statistics.
Original train-only normalization is recomputed and matched to the existing
fixed selected checkpoint. Labels, masks, true branches, executed actions and
original cost values cannot affect feature keys. Float32 rounding is exactly
the already-used neural conversion, not a swept binning parameter.

Two keys separate response-core information from the extra public analytic CV
origin available to the whole optional model. Identical hash buckets require
explicit feature byte equality. Candidate/terminal/group support is retained.
Weighted conditional coordinateMSE measures an empirical lower bound for an
ideal deterministic finite-key predictor on these saved observations. It does
not bound vectorL2, unknown scenes, latent causal identification or capture.
Singletons necessarily have zero empirical variance: no collision is NOT
proof that public information suffices. No truth-based deployment gate.

The command must run AFTER exact source/protocol HEAD is pushed. Use a NEW
absolute output path; do not edit sources while a run is active:

```powershell
python experiments/cwm_v31/public_input_ambiguity.py --data results/cwm_v28/fresh_sequential_ranking_20261010 --data-audit results/cwm_v28/independent_sequential_ranking_audit_20261010 --primary results/cwm_v28/primary_fixed_ranking_20261010 --output D:/uav-capture/cwm31-input-audit-20261011
python -m pytest experiments/cwm_v31 -q
```

22 fixture tests pass before the run: private-label exclusion, every actual
public input, CV-vs-core distinction, exact torch dtype/scaling, float32 anchor
collapse, weighted conditional least squares, singleton limits, terminal
padding, hash-collision rejection and nonfinite inputs. These do NOT substitute
for the complete actual TRAIN audit or model/native-engine re-inference.

An additional12 synthetic workflow tests exercise the complete file-loading
path with publication mocked: development arrays are never opened for
statistics, original normalizers are recomputed, all inputs remain unchanged,
manifest/array/normalizer/checkpoint/summary tampering is refused, mutations
during analysis do not sign completion, and user-owned output is never
overwritten. The full34 V31 tests pass. Synthetic fixture summaries explicitly
identify their toy scope; they are NOT real V28 audits or new model results.

The original GRU/controller/CBF/252 encoding, live jobs, V28 fixed selection,
scientific failure gates and reserved holdout stay unchanged. This stage does
not generate new scenes, start training, qualify active capture or complete
the persistent project goal.
