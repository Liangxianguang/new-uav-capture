# Phase 17 M0/QDR Smoke Report

Status: **diagnostic smoke only; not a locked-test result**.

This report records the first executable Phase 17 result.  The experiment
used the existing `distributed_delayed` DN-MPC + local CBF stack, the belief
candidate source, CPU execution, eight episodes, seeds `648301..648308`, and
the same delayed execution model in both arms:

- execution enabled;
- immutable action queue with two command steps;
- zero command noise;
- planner horizon 8 and control horizon 3;
- local CBF retained;
- no adaptive K and no RNIC.

The only intended treatment difference was QDR state alignment.

| Arm | Safe capture | Capture event | Collision | Boundary | Mean capture time (s) | Total p50/p95/p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| QDR off | 100.0% | 100.0% | 0.0% | 0.0% | 1.275 | 73.31 / 104.53 / 107.45 |
| QDR on | 87.5% | 100.0% | 12.5% | 0.0% | 1.400 | 77.68 / 96.23 / 109.77 |

QDR was exercised in every control step with mean and maximum queue length 2.
Its additional rollout latency was 1.61 / 2.13 / 2.50 ms at p50/p95/p99.
Planner latency for the QDR arm was 62.67 / 81.77 / 92.85 ms and local-CBF
latency was 6.45 / 8.09 / 9.48 ms at p50/p95/p99.

## Interpretation

The adapter and TensorBoard logging path are functioning, but this smoke test
does **not** support a QDR improvement claim.  In this first implementation,
QDR reduced the total p95 relative to the paired arm but lost one safe capture
and one collision in only eight episodes.  The result is consistent with the
known limitation that the current adapter aligns the delayed planning state
and candidate time index, while the planner/safety layer does not yet score a
fully explicit immutable prefix with a separate recovery decision.

The next QDR gate is therefore a validation-scale paired experiment over
delay `0/1/2/4`, tracking time constant, and bounded command noise.  It must
also add prefix clearance/error diagnostics before QDR can be promoted to the
main branch.  No threshold was tuned on locked-test data.

### Follow-up instrumentation check

The evaluator now records the nominal immutable-prefix geometry separately
from the post-action episode metrics.  The logged fields are prefix minimum
clearance, boundary margin, inter-agent distance, maximum safety-margin
violation, and pending-command authority.  A one-episode schema smoke using
the Phase 17 configuration produced prefix minimum clearance `5.3255 m`,
boundary margin `0.6000 m`, inter-agent distance `1.4405 m`, and zero nominal
prefix violation.  This run only verifies the audit path; it is not added to
the performance table and does not establish QDR safety.

## Reproducibility artifacts

The source configuration is
`configs/phase17_queue_adaptive_reachability.yaml`.  Both run directories
retain `config.yaml`, `episodes.jsonl`, `steps.jsonl`, `summary.json`, and
TensorBoard event files:

- `results/phase17_m0_qdr_delayed_smoke/distributed_delayed/tensorboard/`
- `results/phase17_m0_no_qdr_delayed_smoke/distributed_delayed/tensorboard/`
- `results/phase17_qdr_prefix_smoke_check_v2/distributed_delayed/`

The generated `results/` directory is intentionally ignored by Git; source
code, tests, and the configuration/report are versioned.
