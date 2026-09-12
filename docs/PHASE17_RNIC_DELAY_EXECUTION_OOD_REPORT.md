# Phase 17 RNIC Delay/Execution OOD Report

## Protocol correction

The first attempt to evaluate this block exposed a configuration precedence
bug: the Phase 17 default execution mapping could overwrite a frozen scene's
`execution_overrides`. Commit `bc43941` changes the evaluator so an explicit
scene override has priority. All results in this report are post-fix reruns;
the earlier potentially contaminated artifacts are not used.

The frozen manifest is
`results/phase16_ood_delay_execution_shift/scenes.jsonl`, SHA-256
`6478d0d51053b0eb0e9d5fbfbbce9c39e6be2c365c0ad28d2b5abce903f132aa`.
It contains 100 episodes/50 mirror groups with message delay 6, message
dropout 20%, observation noise 0.08, command noise 0.08, tracking lag,
randomized drag/mass and immutable pending commands.

## Per-seed outcomes

| Seed | Cost | Safe capture | Capture | Collision | Boundary | Timeout | Total p50/p95/p99 (ms) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 727201 | fixed distance | 43.00% | 43.00% | 57.00% | 17.00% | 0.00% | 117.72/137.97/144.14 |
| 727201 | RNIC | 41.00% | 41.00% | 59.00% | 20.00% | 0.00% | 146.05/176.18/183.92 |
| 727202 | fixed distance | 40.00% | 40.00% | 60.00% | 22.00% | 0.00% | 115.03/136.95/142.84 |
| 727202 | RNIC | 40.00% | 40.00% | 60.00% | 19.00% | 0.00% | 123.44/180.34/218.21 |
| 727203 | fixed distance | 41.00% | 41.00% | 59.00% | 21.00% | 0.00% | 116.45/139.20/147.19 |
| 727203 | RNIC | 40.00% | 40.00% | 60.00% | 20.00% | 0.00% | 45.54/66.00/74.80 |

## Aggregate decision

Across the three matched seeds, fixed distance gives safe capture
`41.33% [35.67%, 47.33%]`; RNIC gives `40.33% [35.00%, 46.00%]`.
The paired safe-capture delta is `-1.00 pp` with 95% CI `[-3.00,+0.33] pp`.
Collision delta is `+1.00 pp` with CI `[-0.33,+3.00] pp`; timeout is zero
in both arms. Total p95 increases from `138.34` to `173.92 ms`.

RNIC diagnostics show the candidates are genuinely hard to intercept:
minimum slack is approximately `-1.609 s`, mean slack is `0.082--0.088 s`,
unreachable-slot ratio is `35.40--35.79%`, and the earliest all-scenario
feasible step is `5.31--5.38`. These diagnostics explain the failure axis but
the current cost does not convert them into safer actions under immutable
execution.

**Decision: No-Go.** RNIC is retained as a transparent negative ablation. It
does not meet the pre-registered improvement requirement and must not be
presented as a validated robustness contribution.

## Next method direction

The results motivate a delay-aware, conformally calibrated reachable tube:
calibrate prediction error on a development split, inflate the tube according
to queue length and uncertainty, then use the calibrated radius consistently
for candidate budgeting, interception cost and post-hoc coverage checks. This
is the next candidate innovation; it should be implemented and evaluated as a
new protocol rather than obtained by retuning RNIC on this OOD block.
