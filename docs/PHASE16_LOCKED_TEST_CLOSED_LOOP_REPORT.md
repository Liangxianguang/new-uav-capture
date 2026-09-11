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

1. Run a matched candidate-cardinality ablation (GRU K=8 and SSM/S4 K=1) to
   separate architecture, diffusion, and candidate-count effects.
2. Run pre-registered no-safety, multimodality, action-conditioning, and
   planner/safety ablations without reopening model selection.
3. Evaluate OOD geometry, unseen target policies, and stronger delay/dropout/
   tracking-error blocks.
4. Keep robust CBF-QP as a separately labelled diagnostic until its reset,
   queue-authority, fallback, and multi-step certificate gates pass.
