# Phase 16 Locked-Test Closed-Loop Report

> Status: frozen locked-test diagnostic after validation-only configuration
> selection. No configuration, checkpoint, or safety-layer choice was made
> from these results.

## Protocol

The evaluation uses the frozen 90-scene locked-test manifest, with the same
scene pairing for all predictor seeds. The predictor origin is the public team
belief and the causal `both` action condition. Predictions are refreshed at
every control step. The controller uses distributed delayed DN-MPC and the
local CBF safety layer selected on validation. Results aggregate seeds
`727201`, `727202`, and `727203` with matched episode-level hierarchical
bootstrap (`10,000` samples, seed `20260911`).

The locked-test manifest SHA-256 is
`4340cefd0dbd9c68359d31b42554129e4ff84a38677e4596763f675d2fa48636`.

## Three-Seed GRU Results

| Planner branch | Safe capture | Collision | Boundary | Timeout | Capture time (s) | Min clearance (m) | Total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Distributed delayed DN-MPC + local CBF | **94.81% [91.85%, 97.41%]** | 0.00% | 0.00% | 5.19% [2.59%, 8.52%] | 1.226 [1.202, 1.255] | 0.526 [0.518, 0.534] | 25.34 / 43.58 / 47.67 |
| Worst-case DN-MPC + local CBF | 79.26% [74.44%, 84.07%] | 0.00% | 0.00% | 20.74% [15.93%, 25.56%] | 2.288 [2.079, 2.512] | 0.456 [0.445, 0.468] | 19.97 / 37.21 / 39.57 |

The distributed branch is the primary locked-test result. Across its three
seeds, safe capture was `95.56%`, `93.33%`, and `95.56%`; every observed
non-capture was a timeout, with no collision or boundary violation. The
future-action condition was available for `92.12%` of distributed steps, and
the mean prediction age was zero because of per-step refresh.

## Frozen SSM Comparison

The diagonal SSM and official S4 checkpoints were evaluated with the same
three seeds, 90-scene manifest, local CBF, per-step refresh, and K=8 candidate
protocol. GRU remains the K=1 accuracy reference. The following comparison is
therefore an architecture-and-candidate-cardinality comparison, not an
architecture-only causal ablation.

| Predictor | Distributed safe capture | Distributed total p50/p95/p99 (ms) | Worst-case safe capture | Worst-case total p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| GRU, K=1 | 94.81% [91.85%, 97.41%] | 25.34 / 43.58 / 47.67 | 79.26% [74.44%, 84.07%] | 37.21 |
| Diagonal SSM, K=8 | 95.56% [92.96%, 97.78%] | 71.46 / 90.81 / 100.71 | 82.96% [78.15%, 87.41%] | 89.56 |
| Official S4, K=8 | 95.56% [92.96%, 97.78%] | 69.82 / 90.02 / 99.73 | 82.96% [78.52%, 87.41%] | 91.37 |

Matched seed-and-scene paired bootstrap gives a distributed-safe-capture delta
of `+0.74` percentage points for each SSM family relative to GRU, with 95% CI
`[0.00, 2.59]` percentage points. This does not establish a stable main-branch
improvement. For the worst-case branch, diagonal SSM has a `+3.70` point delta
with CI `[0.00, 7.41]`, while official S4 has `+3.70` points with CI
`[0.37, 7.04]`. That auxiliary finding is confounded by K=8 versus K=1 and
does not overturn the offline prediction selection, where GRU remains the
error reference.

Both SSM variants retained zero collision and boundary violation. The official
S4 run uses the upstream implementation through its CPU naive/slow-kernel
fallback because the structured CUDA kernel is unavailable. Its runtime is
therefore a valid CPU measurement for this environment, not a deployment
throughput claim.

## Candidate-Count Ablation

The K=1 versus K=8 ablation fixes checkpoint, training seed, locked scene,
sampling seed, refresh policy, local CBF, and planner. It changes only the
number of diffusion trajectories passed to DN-MPC. GRU is deliberately
excluded: it produces one deterministic mean trajectory, so duplicating it
would not constitute a multimodal comparison.

| Predictor | Branch | K=1 safe capture | K=8 safe capture | Paired K=8 minus K=1 | K=1 collision | K=8 collision |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Diagonal SSM diffusion | Distributed delayed | 95.56% | 95.56% | 0.00 pp [0.00, 0.00] | 0.00% | 0.00% |
| Official S4 diffusion | Distributed delayed | 95.56% | 95.56% | 0.00 pp [0.00, 0.00] | 0.00% | 0.00% |
| Diagonal SSM diffusion | Worst-case | 75.19% | 82.96% | **+7.78 pp [+3.70, +12.22]** | 0.74% | 0.00% |
| Official S4 diffusion | Worst-case | 77.41% | 82.96% | **+5.56 pp [+1.85, +9.26]** | 0.37% | 0.00% |

Here the primary distributed branch is already saturated at 86/90 captures per
seed, so additional candidate diversity cannot improve its observed capture
count. In contrast, the conservative worst-case objective makes the candidate
set consequential: K=8 reduces timeout and removes the small K=1
collision/boundary-failure incidence. This establishes a bounded, planner-
specific closed-loop benefit of multimodal diffusion candidates. It does not
prove that K=8 improves all planners or geometries.

Observed K=1 total p95 latency was `80.16 ms` (diagonal SSM distributed) and
`89.20 ms` (official S4 distributed), compared with `90.81 ms` and `90.02 ms`
for K=8. These runs used different parallel CPU contention levels, so the
figures are retained as runtime profiles rather than a controlled speedup
claim. The K=1/K=8 aggregate artifacts are
`results/phase16_locked_closed_loop_diagonal_ssm_multimodal_aggregate.json`
and `results/phase16_locked_closed_loop_official_s4_multimodal_aggregate.json`.

## Interpretation

This confirms the validation-selected GRU + distributed delayed DN-MPC + local
CBF configuration on the locked scene block. The result supports a robust
modular closed-loop baseline under the current simulator assumptions. The SSM
comparisons do not establish a main-branch advantage over GRU, nor an
architecture-only S4 gain. They also do not establish that robust CBF-QP
provides a closed-loop safety certificate or that the method is ready for
real-flight deployment. The current total latency is reported with p50/p95/p99;
the previous 100 ms reference remains an engineering reference rather than a
hard acceptance gate.

The locked-test aggregate artifact is
`results/phase16_locked_closed_loop_gru_aggregate.json` and its generated
summary is `results/phase16_locked_closed_loop_gru_aggregate.md`. The complete
three-family paired artifact is
`results/phase16_locked_closed_loop_predictor_comparison_aggregate.json`.

## Remaining Work

1. Run a controlled single-process runtime benchmark for K=1/K=8 if throughput
   becomes a method claim; do not infer it from the parallel evaluation runs.
2. Run pre-registered no-safety, action-conditioning, and
   planner/safety ablations without reopening model selection.
3. Evaluate OOD geometry, unseen target policies, and stronger delay/dropout/
   tracking-error blocks.
4. Keep robust CBF-QP as a separately labelled diagnostic until its reset,
   queue-authority, fallback, and multi-step certificate gates pass.
