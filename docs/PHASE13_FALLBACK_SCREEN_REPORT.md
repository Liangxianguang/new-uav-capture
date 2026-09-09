# Phase 13 Fallback Certificate Screening

The fallback candidate evaluator now checks a one-step certificate before
attempting the full preview horizon when receding fallback is enabled. A
candidate that fails the first step cannot pass a longer rollout containing
that same first step. Full-horizon candidates remain preferred whenever any
full-horizon candidate is certified, so this optimization does not relax the
safety gate.

The change was tested with the existing execution fallback tests and a hard
flush progress-QP smoke:

| Metric | Phase 12 smoke | Phase 13 smoke |
| --- | ---: | ---: |
| Episodes / steps | 2 / 40 | 2 / 40 |
| Continuous certificate | 92.5% | 92.5% |
| Actual post-state safety | 100% | 100% |
| Collision / boundary | 0% / 0% | 0% / 0% |
| Safe capture | 0/2 | 0/2 |
| Fallback certificate validity | 100% | 100% |
| p50 latency | 212.8 ms | 195.5 ms |
| p95 latency | 1,227.8 ms | 1,497.3 ms |
| p99 latency | 1,345.7 ms | 1,814.5 ms |

The tail-latency difference is noisy and is not evidence of a stable
improvement. The optimization is retained because it preserves all safety
metrics and removes logically impossible full-horizon evaluations after a
failed one-step screen. The progress-QP path still has no locked-block capture
evidence and remains an experimental recovery policy.

Evidence:

- `results/phase13_fallback_screen_smoke_v1/summary.json`
- `src/encirclement3d/safety_qp.py`

