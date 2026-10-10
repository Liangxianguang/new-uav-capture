# V21: fresh matched 2x2 response-loss experiment

V20 diagnosed weak-response false positives offsetting strong-response gains.
This experiment freezes its protocol before collection/training and uses new
episode990010–990073/layout1990010–1990041. First24 mirror groups train, last8
development, both mirror members together, four speed/sensor variants balanced.
The reserved holdout and all old data are excluded. Geometry distribution and
public actual/CV/radial/tangential/stop probes remain unchanged V18/V19 shadow
probes. Original GRU + distributed-delayed DN-MPC + local CBF stays frozen.

Four response objectives cross coordinate MSE/vector L2 and call/point weighting.
Each seed shares one fully frozen common-motion model. Raw action inputs,
response architecture/initialization, batch order,80epoch budget, optimizer,
gradient clip, residual penalty and final ADE-median selection are matched.
The primary is fixed as raw_point_l2, not whichever configuration later wins.
There is no mediator, threshold search, truth-bin gate or best-epoch selection.

Full-training group normalizers determine importance weights before training.
Point weighting includes every valid nonanchor point, matching the response
gate across groups. Minibatch denominator is batch size, not a random sum of
importance weights. This isolates weighting without silently adding a ratio
estimator. Loss units differ for MSE and L2; findings remain conditional on the
specified optimizer/budget, not a universal statement about loss geometry.

All5configurations x3seeds x2independent runs must be retained. All response and
original complete-cost decision gates remain enforced. The primary also needs
to improve over the call-MSE control. A passed offline gate cannot enable the
controller. Whole-module fallback, sequential candidates, untouched holdout,
original8Levels/new-scene closed-loop capture, safety and latency remain needed.

The collector must retain any data gate failure; do not replace failed scenes
or reassign splits. Training starts only after independent public replay and
full data audit. Standalone scripts run in fresh Python processes with exclusive
output directories. Reproduction entry points are added with implementation;
this protocol/README is committed and published before collection/training.
