# V33: UNTRAINED prefix-response architecture hypothesis

This directory contains an untrained optional response core and a capacity/
compute-shape-matched full-sequence query control. No training protocol, trained
weights, fixed-model qualification, active selector integration, new-scene
capture/safety/latency result or default promotion is supplied by this prototype.
The original GRU + distributed-delayed DN-MPC + local CBF remains unchanged.

Both cores use the original five public tensor contracts: 8x252 past history,
4x6 received context, 8x4x3 proposed and anchor commands, and8x3 original GRU
forecast. Neither receives labels/costs/actual CBF execution/future true state.
The additional history GRU is newly initialized optional network, not the
original frozen predictor. Motion output is identically zero; a trained core
would later wrap the SAME frozen common motion component as its matched control.

Both compute8 offset queries with identical widths/heads/neural tensor shapes.
An8-dimensional fixed query code names the offset. The prefix mode masks ALL
proposed AND anchor commands after the query's offset; the matched full mode
uses all8 commands. Both apply the SAME public prefix-anchor zero constraint.
Each shared24-output head supplies the diagonal query offset. All parameters
and initial states can be copied identically across the pair. This is not an
assertion of matching the older one-query V28 network's compute or parameter
count; comparisons to that older model must be separately labeled.

Prefix invariance is an architectural constraint for fixed public context,
not causal identification, a guarantee of accurate hidden target dynamics, or
evidence of better action selection. The fixed GRU forecast is known decision
context, not realized future truth. The existing common motion calibration can
still depend on its fixed anchor sequence; this prototype does NOT certify
whole-module invariance when that reference anchor/context changes. Tests do
check this response core's proposed and anchor suffix invariance independently.
Mixed command sequences may leave old data support; V32 sensitivity alone
cannot prove an improvement or qualify a new architecture.

Before scientific training: finish fixed V32 diagnostic, freeze a NEW explicit
data/architecture/loss/seed/epoch protocol, collect independently assigned
broader geometry/matched public action-prefix support using the original
engine, audit ALL private branches and original costs, recompute TRAIN-only
normalization, train BOTH controls with the same frozen common motion/loss/
budget, retain all seeds/epochs/failures, independently reload and confirm on
fresh development and untouched holdout only under its original contract.
Original8Levels + new-scene paired closed-loop/safety/latency/fallback evidence
remains required. This directory has NO train/deploy command; it cannot be used
as an easier substitute for the complete user goal or override old gates.

```powershell
python -m pytest experiments/cwm_v33 -q
```

Tests randomize synthetic heads to check nonzero sensitivity and gradients;
these are explicitly not learned weights or native environment evaluations.

All29 synthetic fixtures pass: matched state/parameter counts, exact zero init,
all8 proposed+anchor suffix invariance checks, noncausal matched-control
sensitivity, exact anchor/prefix-zero, past/future gradient separation, invalid
shapes/nonfinite inputs/scales/modes, and compatibility with the UNCHANGED
FrozenOriginResponse wrapper. The wrapper fixture uses newly created toy cores
and a toy optimizer step only to check the common reference remains bitwise
frozen; it is NOT new scientific training or a real original-model replay.
