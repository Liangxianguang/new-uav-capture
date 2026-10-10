# V23: deployed-reference response-cost contrast, default off

V21 improved average response prediction, but all controller research gates
failed. V22's posthoc truth replacements indicate response-ranking errors on
expanded candidates plus common-motion generalization errors; they are not
causal identification or deployable performance bounds. V12's previous task
loss also failed. Do not claim this experiment is effective before evaluation.

New32mirror groups are assigned before collection: episode993010-993073,
layout1993010-1993041, first24train/last8development, four speed/sensor variants.
No V18/V19/V21 data, inspected development windows or reserved holdout reused.
Original GRU seed745101, public252encoding, distributed-delayed DN-MPC, local
CBF safety margin, target rules and shadow candidate generator stay unchanged.
Original controller alone collects data and remains available. All optional
calibration/response is default off; this is not replacing the frozen model.

Matched2x2: frozen GRU-origin/CV-origin calibrated common motion crossed
with point-vector-L2 alone/plus normalized L1 original-cost response contrast.
Primary=cv_cost_l2. New wrapper preserves original GRU as backbone and neural
context for BOTH origins. CV is a public optional reference origin, followed by
its own motion calibration, using only original public belief quantities.
Identical raw-action response cores, initialization, sample order,80epochs,
optimizer/clip/scales/energy penalty. Per seed both matched motion cores train
80epochs, then completely freeze before response training. Scalar cost
labels and true reference enter training supervision only, never model inputs.

Unlike failed V12, cost contrast is measured at the actual deployed frozen
reference, not a true reference, with L1 weight.1 rather than squared weight1.
All valid nonanchor points still get physical supervision. Only complete calls
get full-cost supervision; no invented terminal future, truth-bin gate, seed
search or more favorable data replacement. Cost weights fixed across the full
train population, with fixed minibatch denominator, not random normalization.

All original response/ADE/decision gates remain. Primary must additionally
beat matched CV-L2 in normalized cost contrast, not worsen its response median,
and show positive decision-gain interval lower bounds vs CV-L2 and GRU-cost
L2. No secondary may replace the primary. Costs use unchanged full original
engine; differentiable score must independently reproduce costs and gradients.

Publish protocol before collection/training. Implementation, qualified data
collection/public replay and full independent audit precede optimization.
Retain both complete training runs/all18models per run, failed gates and
training/development evidence. Even a passed offline experiment cannot enable
the controller without original-entry fallback, sequential candidates, holdout,
original8Levels/new-scene closed-loop, safety and latency validation.

## Fresh data entry points

```powershell
python -m pytest -q experiments/cwm_v23
python experiments/cwm_v23/fresh_cost_data.py --output results/cwm_v23/fresh_data
python experiments/cwm_v23/replay_cost_public.py --data results/cwm_v23/fresh_data --output results/cwm_v23/public
```

The collector uses the unchanged qualified V18 public geometry library and
original-controller shadow branches; it does not instantiate a new learned
controller. Split assignment is fixed and failed data gates retained. Public
replay retains full original observation/252feature traces. The independent
`cost_data_audit.py` reconstructs geometry, candidates, delays, all252history,
paired masks/labels, train/dev support and complete original cost/selection
before training. Training implementation is now supplied below; full results
and complete independent retraining/release verification are still pending;
data collection or a protocol alone does not mean a new model has been trained.

## Precollection amendment

The initial `a309cd6` protocol had bare public CV as a motion reference. Before
ANY V23 collection/replay/training, already inspected V21 reference ADE(.374364)
vs the fixed .95CV total-ADE gate(.366607) showed accurate anchored response alone
cannot guarantee sufficient motion correction. Both GRU/CV origins now retain
matched80epoch motion calibration; both completely freeze before response
training. The primary, new seeds, splits and gates stay fixed. The prior commit
is retained, no V23 outcome inspected, and revised protocol must be published
before collecting. No claim is made that this amendment ensures a passed gate.

## Fixed training implementation (not a trained-model success report)

```powershell
python -m pytest -q experiments/cwm_v21 experiments/cwm_v22 experiments/cwm_v23
python experiments/cwm_v23/train_cost_response.py --data results/cwm_v23/data_20261010 --public results/cwm_v23/public_20261010 --output results/cwm_v23/primary_20261010
python experiments/cwm_v23/train_cost_response.py --data results/cwm_v23/data_20261010 --public results/cwm_v23/public_20261010 --output results/cwm_v23/retrained_20261010
```

No optimizer is instantiated until independent full data audit and original
cost/piecewise-gradient audits at both initial public origins have passed.
After matched80epoch motion calibration, both common models are frozen and
gradients cleared; deployed-reference full-cost/gradient audits must then pass
again before response optimization. The V11 numerical tolerances and explicit
nonsmooth-coordinate accounting are reused unchanged. Scores include all
target-independent original terms and reject unsupported objectives.

`cost_origin_model.py` adds an analytic PUBLIC CV path as a sixth wrapper tensor,
but all unchanged neural cores still receive only the original five tensors
and original GRU context. Learned-calibration energy does not penalize the
deterministic CV-minus-GRU offset. Both motion origins use unchanged V10 masked
coordinate-MSE/group-call pretraining; response uses fixed full-population
point-L2/energy weights, and task variants use the fixed .1 deployed L1 loss.
Only complete UNION calls enter full-cost supervision; incomplete calls get
zero cost population weight, with a fixed batch-size denominator.

The deployed reference cache is generated with canonical per-split batch32
evaluation, and final forecasts must match cached references exactly. This
avoids treating response-minibatch-dependent floating-point rounding as a new
motion reference. All candidate responses remain action-anchored, and private
future/labels/costs are supervision only, never forward inputs.

All18 models/run retain final checkpoints, optimizer/RNG, history, train/dev
predictions, magnitude/offset diagnostics, deployed-cache/cost-gradient audits,
and both original-engine actual/union costs/choices/regrets on the SAME complete
union support. `cost_qualification.py` preserves all gates and uses the matching
origin's independently median-selected motion control. CV means uncalibrated
public CV. Primary requires both extra response-control decision improvements;
actual-library diagnostics cannot replace union gates or online validation.

Tests include synthetic two-independent-run optimization checks and archived
historical TRAIN-fixture original-engine checks. Those fixtures are implementation
tests only, never new V23 model fitting, normalization or scientific validation.
Complete two-run research artifact auditing/public release and all online
requirements remain necessary; no new controller is enabled by these scripts.
