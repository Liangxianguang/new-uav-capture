# S4 paired-response models trained reproducibly, not promoted

Date:2026-10-10. Status: offline training completed; development gate failed;
original DN-MPC + empirical local CBF retained; enhanced control disabled.

## What was actually trained

The separately qualified translated S4 scene uses the unchanged existing
adaptive-branching target, world dynamics, safety margins and frozen GRU.
The wall's x center is1.5m; this is new-scene geometry, never original Level
modification. Geometry qualification is documented in the preceding report.

New data:64 original episodes from32 independent mirror groups. Before
collection,24 groups/48 episodes are assigned train and8 groups/16 episodes
development validation. All64 original episodes achieve safe capture, with
no target-invalid episode, defender collision or boundary violation. All
observed/unobserved stored trajectory arrays are byte-equal and seven physical
outcomes match. This is original-controller performance in this small S4
distribution, not evidence of an enhanced-model gain or Level coverage.

There are320 snapshots:240 train and80 development. Eight command probes and
common step-key noise per state yield17,112 common-valid paired target points
within8 steps.47.90% of those points have response magnitude>5cm;848/2240
both-committed probe pairs flip the target exit (37.86%). All32 groups support
response/flips.16-step data contain no target-invalid or defender safety
termination; capture censors later frames. Training uses the original8 steps.
No data are generated after termination, and padded zeros are not valid labels.

Both models take public8x252 history,4x6 defender state relative to belief,
proposed/reference8x4x3 commands and supplied frozen GRU trajectory8x3.
The ordinary model flattens agent/action features; the structured model pools
shared per-agent and relative-agent graph messages. Parameter counts31,696
and31,920 differ by0.71%. Both use the same auxiliary branch-label supervision;
private labels are targets only. This is paired interventional-response
learning, not a claim that graph pooling establishes an identified SCM.

Same three seeds, optimizer, minibatch orders and80 fixed epochs. Train-only
normalization. Paired-effect MSE uses common valid masks and group-balanced
window weights; both get the same branch auxiliary loss and energy penalty.
No sweep, early stopping, best-epoch search, new original-GRU optimizer or
baseline checkpoint fine-tuning occurs. At the reference command, correction
is exactly zero and the original double-precision backbone bytes are retained.
For other commands, corrections are bounded per coordinate at2.5m; that is
not a dynamics, safety or strict-CBF certificate.

## Development results

All values below are group-equal means across8 development groups. Paired
response error uses common-reference-valid non-anchor points. Training ADE
uses all valid branch target points, including some frames whose reference
branch already terminated. Neither metric is a capture or safety outcome.

| Predictor / seed | Paired response error (m) | Target trajectory ADE (m) |
| --- | ---: | ---: |
| Frozen GRU + zero response | 0.230262 | 0.633731 |
| Ordinary /967101 | 0.168383 | 0.636952 |
| Ordinary /967102 | 0.162002 | 0.651752 |
| Ordinary /967103 | 0.172213 | 0.641002 |
| Structured /967101 | 0.168829 | 0.643159 |
| Structured /967102 | 0.172386 | 0.649954 |
| Structured /967103 | 0.179344 | 0.648507 |
| Constant velocity + zero response | 0.230262 | 0.523268 |

Structured median paired-response error improves25.13% over zero response,
but is2.38% worse than ordinary median. Each structured seed improves more
than20% over zero, satisfying the first predeclared condition; the second
requires a5% advantage over ordinary median and fails. The published gate
remains failed. No model-selection rule or threshold changes after results.

Predeclared median-error seeds for later diagnostics: ordinary967101 and
structured967102. These are not the best seeds. Branch classification accuracy
is about90-92% on committed valid points, but is auxiliary prediction, not
counterfactual correctness, causal identification or a controller metric.

## Why response learning is not yet overall prediction benefit

A post-training support-matched oracle diagnostic is explicitly exploratory
and does not override the failed gate. The two support populations are4872
common-reference-valid points across80 states, and4416 jointly complete8-step
points across69 states. All eight development groups have support.

| Forecast information, same common-valid support | Group-equal ADE (m) |
| --- | ---: |
| Original frozen GRU, no response | 0.621530 |
| GRU + exact paired response labels | 0.631812 |
| Ordinary learned forecast | 0.623115 |
| Structured learned forecast | 0.637324 |
| Exact reference-motion truth, no response | 0.201479 |
| Exact reference-motion truth + ordinary learned response | 0.147335 |
| Exact reference-motion truth + structured learned response | 0.150838 |

The jointly complete8-step population has the same pattern:0.649931m original,
0.664115m with exact response,0.652593m ordinary,0.668968m structured;
reference-motion truth alone0.202286m, with ordinary/structured response
0.152209/0.156871m. Full action-specific truth naturally gives0m and is not
a deployable model or performance bound.

Thus even exact response addition does not repair the frozen GRU's common
motion-forecast error on these populations. The response mechanism is
learnable, but assuming the original forecast is an accurate counterfactual
reference is not supported. Constant velocity also outperforms frozen GRU
in all-branch ADE on this new target strategy. These are new-domain diagnostics,
not a claim that the original GRU is poor on its original Levels.

Next task: evaluate information value using the complete frozen DN-MPC
objective and these fixed median models/oracle conditions on development data.
If common-motion correction has decision value, build it as a separate optional
enhancement head around the retained frozen GRU, distinguish its benefit from
action response, and use matched ordinary/structured controls. Do not simply
increase epochs or call ordinary calibration a causal gain. Any changed
design needs new fixed development protocols; reserved test seeds remain
untouched until final model/protocol selection. Default-off replay, latency,
physical safety and closed-loop capture still require explicit validation.

## Reproducibility and preservation

Two independent processes each train all six models. All model tensors,
optimizer states, normalizers, sampler and Torch RNG states match exactly;
all80-epoch loss-history bytes match. Reloading each checkpoint reproduces
its development prediction/response/logit array dtype, shape and raw bytes,
and recomputes every reported metric and failed gate. The verifier checks
actual scene-group splits, train-only normalization, terminal support, physical
target-wall/boundary validity, original trajectory arrays and all source hashes.

65 focused V1-V7 tests pass, including rejection of forged gate success,
validation-contaminated normalization, altered masks with updated digests,
and changed optimizer/RNG states. The original capsule hash remains
`c83a27a4360ad055b69dbced4c28370cb31a32ae89ea9fbab13523fa6218a329`.
The previous full historical256/256 trajectory replay is retained; this stage
does not extend exact-runtime or latency claims to arbitrary hosts.

Training archive: `experiments/cwm_v7/artifacts/paired_training_20261010.zip`.
14,806,822 bytes, SHA256
`d14dcca4dd2b9eb4d8609c1eb906413d4dbe7a76d4bd3c68f4ae17b959f97d69`.
It includes actual64-episode scenes/data/trajectories, both full training runs,
six weights per run, optimizer/RNG snapshots, loss histories, predictions,
protocols and exact source closure as run. The post-training support diagnostic
and script are versioned separately; they do not pretend to be preregistered
training results. Commands are in the stage README.

No new prediction/action path is enabled in the original evaluator, no
holdout or Level holdout is used, no safety margin is lowered, no original
training job is stopped, and no user-owned dirty source is staged. Both
geometry and training milestones use scoped Conventional Commits on the
independent research branch, not main.
