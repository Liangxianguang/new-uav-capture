# Phase 16 OOD Geometry Closed-Loop Report

> Status: frozen diagnostic on one held-out geometry axis. This report does
> not reopen model selection and does not establish general robustness.

## Protocol

The training archive samples one wall with half-x extent in `[0.45, 0.65] m`,
half-y extent in `[3.85, 4.35] m`, and height in `[9.25, 9.55] m`. This OOD
block freezes 100 new episodes in 50 upper/lower defender mirror groups, with
all three geometry variables outside those training intervals:

| Geometry field | Training archive (m) | OOD diagnostic (m) |
| --- | ---: | ---: |
| Wall half-x extent | `[0.45, 0.65]` | `[0.70, 0.85]` |
| Wall half-y extent | `[3.85, 4.35]` | `[4.38, 4.42]` |
| Wall height | `[9.25, 9.55]` | `[9.60, 9.80]` |

The scenario generator rejects a candidate layout unless both mirror routes
are valid, then records the accepted geometry, layout seed, mirror group, and
protocol metadata. The frozen scene manifest SHA-256 is
`47c393d1d678f25aad14c978734aae5d89105d826ea3e7b32b16a46cc1654501`.

The test reuses the validation-selected GRU `both` checkpoints for training
seeds `727201`, `727202`, and `727203`, with public-belief origins, causal
action conditions, one candidate, eight sampling steps, per-step prediction
refresh, delayed distributed DN-MPC, and local CBF. It also retains the
auxiliary worst-case DN-MPC branch. Each seed executes all 100 episodes in
both branches. Hierarchical 95% bootstrap intervals resample matched training
seeds and frozen episode indices (`10,000` samples; bootstrap seed
`20260911`).

This block changes obstacle geometry only. It is not paired with the original
locked-test scenes, so the reported degradation is descriptive rather than a
paired causal effect estimate.

## Results

| Planner branch | Safe capture | Collision | Boundary | Timeout | Capture time (s) | Min clearance (m) | Future action available | Total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Distributed delayed DN-MPC + local CBF | **84.00% [79.33%, 88.33%]** | 0.00% | 0.00% | 16.00% [11.67%, 20.67%] | 1.633 [1.536, 1.734] | 0.471 [0.461, 0.480] | 94.05% [93.70%, 94.41%] | 72.93 / 93.60 / 111.26 |
| Worst-case DN-MPC + local CBF | 46.33% [39.67%, 53.33%] | 1.67% [0.33%, 3.67%] | 1.67% [0.33%, 3.67%] | 52.00% [45.00%, 58.67%] | 3.580 [3.066, 3.988] | 0.368 [0.357, 0.382] | 98.15% [97.80%, 98.43%] | 61.77 / 77.13 / 97.40 |

Both branches use fresh predictions at every step: the mean prediction age is
zero. The distributed branch has 16,117 pooled step-latency samples; the
worst-case branch has 44,200. Latency remains a measured CPU profile, not a
strict 100 ms acceptance gate. The distributed p95 is below 100 ms whereas
its p99 is 111.26 ms; report all three quantiles rather than relying on an
average.

For context only, the same distributed GRU configuration achieved 94.81%
safe capture on the separate in-distribution locked-test manifest. The OOD
geometry block therefore reveals a substantial transfer gap, expressed mainly
as timeout rather than observed physical failure for the distributed branch.
Because the geometry and scene manifests differ, this is not a significance
claim against the locked-test result.

## Interpretation

The validation-selected distributed configuration remains physically safe in
this one geometry-shift diagnostic: no collision or boundary violation was
observed across three seeds and 300 executed distributed episodes. That is
not a robust-CBF or real-flight safety certificate. The lower capture rate and
higher timeout rate show that the planner/predictor combination is sensitive
to longer, taller blocking walls outside its training support.

The auxiliary worst-case branch does not transfer satisfactorily: its
conservative objective both greatly reduces capture and retains nonzero
collision/boundary incidence. It should not be promoted as the geometry-OOD
operating point. This is consistent with the main locked-test evidence that
local CBF creates a safety--capture Pareto rather than a free safety gain.

## Artifact Completeness

Each seed directory retains `config.yaml`, source hashes, `scenes.jsonl`,
method-level `episodes.jsonl` and `steps.jsonl`, TensorBoard events, and root
`summary.json`. All six method logs contain 100 episodes. The aggregate
artifact is:

```text
results/phase16_ood_geometry_gru_aggregate.json
results/phase16_ood_geometry_gru_aggregate.md
```

Generated result artifacts remain local and are intentionally Git-ignored.

## Next OOD Blocks

1. Freeze an unseen target-behavior/speed block while holding geometry at the
   in-distribution range.
2. Freeze a stronger communication, execution-delay, and tracking-noise block
   while holding target behavior and geometry fixed.
3. For each block, retain three seeds, mirror pairing where feasible, all
   safety/timeout/clearance metrics, and p50/p95/p99 latency.
4. Only after the individual axes are measured, test a deliberately labelled
   combined-stress diagnostic. Do not use any OOD diagnostic to select model
   checkpoints or tune the locked-test configuration.
