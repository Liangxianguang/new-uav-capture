# Phase 16 OOD Delay and Execution Closed-Loop Report

> Status: frozen diagnostic. This block is a deliberate stress test and is a
> No-Go result for the current local-CBF closed-loop safety contract.

## Protocol

This block keeps target speed in the training support (`0.65` and `0.75`) and
keeps S4 geometry in the training ranges. It changes only sensing,
communication, and execution conditions. The 100 episodes form 50 upper/lower
mirror groups, split evenly between two stress conditions:

| Condition | Detection dropout | Observation noise | Message delay | Message dropout |
| --- | ---: | ---: | ---: | ---: |
| `delay6_dropout20` | 0.35 | 0.08 | 6 steps | 0.20 |
| `delay8_dropout25` | 0.40 | 0.10 | 8 steps | 0.25 |

All episodes additionally use the frozen execution mapping:

- command delay: 2 steps;
- command noise standard deviation: 0.08;
- velocity time constant: 0.40 s;
- drag coefficient: 0.10;
- per-episode max-speed scale: `[0.85, 0.98]`;
- per-episode max-acceleration scale: `[0.70, 0.95]`;
- mass scale: `[0.90, 1.10]`;
- drag range: `[0.05, 0.20]`.

The execution overrides are now part of the frozen scene contract and are
applied by `config_for_spec`; the test suite verifies that they reach the
environment configuration rather than remaining metadata only. The scene
manifest SHA-256 is
`6478d0d51053b0eb0e9d5fbfbbce9c39e6be2c365c0ad28d2b5abce903f132aa`.

The validation-selected GRU `both` checkpoints for seeds `727201`, `727202`,
and `727203` are evaluated with public-belief origins, causal action
conditions, one candidate, eight sampling steps, per-step refresh, local CBF,
and both distributed-delayed and auxiliary worst-case DN-MPC. Each seed
executes all 100 episodes. Intervals are hierarchical 95% bootstrap
intervals over matched training seeds and frozen episode indices (`10,000`
samples; seed `20260911`).

## Results

| Planner branch | Safe capture | Ordinary capture | Collision | Boundary | Timeout | Min clearance (m) | Total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Distributed delayed DN-MPC + local CBF | **47.00% [41.00%, 53.00%]** | 47.00% [41.33%, 53.00%] | **53.00% [47.33%, 59.00%]** | **19.00% [14.33%, 24.00%]** | 0.00% | 0.136 [0.112, 0.159] | 86.09 / 122.82 / 178.46 |
| Worst-case DN-MPC + local CBF | 53.67% [46.67%, 60.33%] | 55.33% [49.33%, 61.33%] | **46.33% [39.67%, 53.33%]** | **24.00% [19.33%, 29.00%]** | 0.00% | 0.193 [0.160, 0.222] | 69.50 / 85.76 / 106.57 |

Prediction age remains zero because predictions are refreshed every control
step. The distributed branch has 14,377 pooled step samples; the worst-case
branch has 13,647. The distributed p95 and p99 exceed the 100 ms engineering
reference, but 100 ms is not used as a hard acceptance gate.

The current local-CBF filter does not prevent physical failures under this
execution model. Independent safety diagnostics also degrade substantially;
the step-level certificates cannot be interpreted as a multi-step safety
proof when the executed command is delayed and tracked with error.

## Interpretation

This is a clear No-Go for claiming execution-robust safety. The failure is
not primarily timeout: all episodes terminate through capture or safety-unsafe
physical outcomes. The distributed branch has 53% collision and 19% boundary
violation; the worst-case branch has 46.33% collision and 24% boundary
violation. This directly supports the earlier conclusion that robust CBF-QP
and local CBF must be evaluated with queue authority, execution dynamics,
multi-step reachable tubes, and certified fallback rather than one-step
nominal filtering.

The result is also a useful separation from the target-speed and geometry OOD
blocks. Those blocks primarily produced timeout degradation with zero
observed collision in the distributed branch. Here the newly activated
execution dynamics expose a safety-contract failure. The current model should
not be deployed under these conditions and should not be reported as having a
closed-loop safety certificate.

## Artifact Completeness

All three seed directories contain 100 episodes for both methods, method-level
episode/step logs, root summaries, source hashes, and configuration snapshots.
The aggregate artifacts are:

```text
results/phase16_ood_delay_execution_gru_aggregate.json
results/phase16_ood_delay_execution_gru_aggregate.md
```

Generated result artifacts remain local and are intentionally Git-ignored.

## Required Follow-up

1. Do not tune the locked-test model or safety margins on this OOD block.
2. Decompose failures by execution delay, tracking error, barrier family,
   queue prefix, and fallback action.
3. Re-run a certified fallback/robust-CBF diagnostic only after the queue and
   execution contract is repaired.
4. Keep this No-Go result in the paper limitations and safety discussion.
