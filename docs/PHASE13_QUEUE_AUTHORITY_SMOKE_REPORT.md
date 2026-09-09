# Phase 13 Queue-Authority Smoke Report

This report is a short regression smoke after the fallback prefix-screening
optimization. It is not a locked authority result: each variant uses two
retained seeds, a 30-step cap, and the continuous-segment constraint remains
disabled in the authority configuration.

| Authority | Prefix admissible | Abort required | Actual post-state safety | Continuous certificate | Safe capture | Collision | Boundary | p50 / p95 / p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| immutable | 100.0% | 0.0% | 100.0% | 80.0% | 0/2 | 0% | 0% | 417.8 / 1,569.2 / 1,828.8 |
| replace nonexecuting | 100.0% | 0.0% | 100.0% | 68.3% | 0/2 | 0% | 0% | 644.0 / 3,073.0 / 3,691.2 |
| flush pending | 91.7% | 8.3% | 100.0% | 0.0% | 0/2 | 0% | 0% | 2,148.0 / 3,443.1 / 3,748.0 |

The flush row is not evidence that queue clearing is unsafe in general; it
shows that, under this short sample and the current hard execution contract,
the prefix check can fail even when the actual post-state remains safe. Queue
authority must still be treated as an executor capability. A real-system
claim requires confirmation that the actuator can replace pending commands or
flush them atomically.

The observed pattern is consistent with the existing locked evidence: queue
feasibility, discrete post-state safety, and task capture are different
metrics. No authority mode meets the hard-scene 80% safe-capture target in
this smoke, so no mode is selected as the final controller.

Evidence:

- `results/phase13_queue_authority_smoke_v1/summary.json`
- `results/phase9_queue_authority_locked_v2/summary.json`

