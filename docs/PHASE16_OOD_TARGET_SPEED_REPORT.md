# Phase 16 OOD Target-Speed Closed-Loop Report

> Status: frozen diagnostic on an unseen target-speed axis. This report does
> not reopen model selection and does not establish general robustness.

## Protocol

The Phase 16 training archive uses target speed scales from `0.50` through
`0.80`. This diagnostic freezes 100 new S4 episodes in 50 upper/lower mirror
groups and evaluates the unseen scales `0.82` and `0.90` (50 episodes each).
Obstacle geometry remains inside the training ranges, and nominal and
delayed-noisy observation conditions are retained. The scene manifest SHA-256
is `273023b070b0ba1cbebcd2073b17a89222b790238c4b541469e0f443825cb9d8`.

The validation-selected GRU `both` checkpoints for training seeds `727201`,
`727202`, and `727203` are evaluated with public-belief origins, causal action
conditions, one candidate, eight sampling steps, per-step prediction refresh,
distributed delayed DN-MPC, and local CBF. The auxiliary worst-case DN-MPC
branch is retained for comparison. Every seed executes all 100 episodes in
both branches. Intervals are hierarchical 95% bootstrap intervals over
matched training seeds and frozen episode indices (`10,000` samples; seed
`20260911`).

This block changes target speed only. It is not paired with the original
in-distribution locked-test manifest, so the difference from the ID result is
descriptive rather than a paired significance estimate.

## Results

| Planner branch | Safe capture | Collision | Boundary | Timeout | Capture time (s) | Min clearance (m) | Future action available | Total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Distributed delayed DN-MPC + local CBF | **42.67% [37.00%, 48.33%]** | 0.00% | 0.00% | **57.33% [51.67%, 63.00%]** | 1.471 [1.342, 1.630] | 0.422 [0.411, 0.432] | 96.57% [96.15%, 96.97%] | 83.94 / 106.72 / 130.87 |
| Worst-case DN-MPC + local CBF | 55.00% [47.33%, 63.00%] | 0.00% | 0.00% | 45.00% [36.67%, 52.67%] | 3.903 [3.604, 4.231] | 0.369 [0.364, 0.375] | 98.03% [97.72%, 98.32%] | 69.74 / 86.33 / 104.66 |

All branches refresh prediction every control step, so mean prediction age is
zero. The distributed branch has 44,882 pooled step-latency samples and the
worst-case branch has 40,168. The distributed p95 exceeds the 100 ms
engineering reference and its p99 is 130.87 ms; 100 ms is not used as a hard
acceptance gate, but the three quantiles remain part of the result.

For context, the same distributed GRU configuration achieved `94.81%`
safe capture on the separate in-distribution locked-test manifest. Since the
scene manifests differ, the comparison is not a paired causal effect. The
observed OOD degradation is dominated by timeout rather than physical
collision or boundary failure under local CBF.

## Interpretation

The predictor-controller stack does not transfer well to target speeds above
the training support. At the main operating branch, safe capture falls to
`42.67%` and more than half of episodes time out, despite zero observed
collision and boundary violations. The higher speed also raises the total
latency profile, which can further reduce interception margin, although this
experiment does not identify how much of the loss is due to latency versus
target reachability and prediction error.

The worst-case objective is less timeout-prone in this block but still gives
only `55.00%` safe capture. It should not replace the distributed branch as a
general solution: the result is an auxiliary diagnostic, and its objective
has already shown poor transfer on the geometry OOD block.

These results expose a meaningful generalization boundary rather than a
software failure. The next experiment should hold speed and geometry fixed
and vary communication/execution delay and tracking noise. In parallel, the
speed block should be decomposed by speed level and observation condition,
and the offline predictor metrics should be measured at the same unseen
speeds before considering speed augmentation or calibration.

## Artifact Completeness

Each of the three seed directories contains 100 episodes for both methods,
method-level `episodes.jsonl` and `steps.jsonl`, root `summary.json`, source
hashes, and configuration snapshots. The aggregate artifacts are:

```text
results/phase16_ood_target_speed_gru_aggregate.json
results/phase16_ood_target_speed_gru_aggregate.md
```

Generated result artifacts remain local and are intentionally Git-ignored.
