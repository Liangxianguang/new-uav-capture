# Phase 16 OOD Target-Behavior Closed-Loop Report

> Status: frozen single-axis diagnostic. This is evidence for two held-out
> adaptive-branch decision rules, not for arbitrary adversarial behavior.

## Protocol

This 100-episode block contains 50 upper/lower mirror groups. Target speed
(`0.65` and `0.75`), geometry, nominal execution dynamics, and the
nominal/delayed-noisy observation settings remain inside their previous
supports. Only the hidden target branch-decision policy changes:

| Behavior | Defender lookahead | Commit margin |
| --- | ---: | ---: |
| `short_lookahead` | 0.20 s | 0.00 s |
| `long_lookahead_margin` | 1.20 s | 0.20 s |

The target still makes its exit choice internally from defender state. Branch
labels and scores remain excluded from controller observations. Both behavior
conditions have 50 episodes; upper/lower biases and the two speed levels each
have 50 episodes. The frozen manifest SHA-256 is
`3cb04e784b499ba99e6f90bdc2a23b91e8bb9c60d6669caee94dc683006b5f1c`.

Three GRU `both` checkpoints (`727201`, `727202`, `727203`) are evaluated with
public-belief prediction origins, causal action conditions, one candidate,
eight sampling steps, per-step refresh, local CBF, and distributed-delayed or
auxiliary worst-case DN-MPC. Every seed runs all 100 episodes. Intervals are
hierarchical 95% bootstrap intervals over matched seeds and frozen episodes
(`10,000` samples; seed `20260911`).

## Results

| Planner branch | Safe capture | Collision | Boundary | Timeout | Capture time (s) | Min clearance (m) | Future action available | Total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Distributed delayed DN-MPC + local CBF | **94.33% [91.33%, 97.33%]** | 0.00% | 0.00% | 5.67% [3.00%, 9.00%] | 1.220 [1.211, 1.228] | 0.520 [0.512, 0.529] | 92.22% [91.99%, 92.45%] | 80.48 / 97.83 / 115.97 |
| Worst-case DN-MPC + local CBF | 62.67% [56.67%, 69.00%] | 0.00% | 0.00% | 37.33% [31.33%, 43.33%] | 2.380 [2.128, 2.648] | 0.426 [0.415, 0.436] | 96.23% [95.83%, 96.61%] | 70.26 / 88.85 / 111.57 |

All target branches made a decision (`100%` decision rate). Prediction age is
zero due to per-step refresh. The distributed branch uses 7,702 pooled
step-latency samples; the worst-case branch uses 32,474. The p99 values remain
above the 100 ms engineering reference, which is reported rather than used as
a hard acceptance gate.

## Interpretation

The main distributed configuration transfers to these two held-out branch
decision rules without a visible safety or capture regression relative to the
separate in-distribution locked-test result (`94.81%` safe capture). The scene
manifests are different, so this is descriptive consistency rather than a
paired significance claim. It demonstrates only that altering lookahead and
commit margin does not by itself break the present target-behavior interface.

This contrasts with the other OOD axes: geometry reduces capture primarily
through timeout, target speed severely reduces capture through timeout, and
communication/execution stress causes physical safety failures. The behavior
block therefore narrows the current failure diagnosis: the dominant unresolved
issues are speed generalization and execution-aware safety, not these two
adaptive branching parameters.

## Artifact Completeness

Every seed directory contains 100 episodes for both methods, method-level
episode/step logs, root summaries, configuration snapshots, and source hashes.
The aggregate artifacts are:

```text
results/phase16_ood_target_behavior_gru_aggregate.json
results/phase16_ood_target_behavior_gru_aggregate.md
```

Generated results remain local and Git-ignored.
