# Phase 17 QDR Validation Report

Status: **validation-selection diagnostic; QDR promotion gate is No-Go**.

This report records the first validation-scale paired QDR comparison.  It is
not a locked-test result and no parameter was tuned on locked-test data.

## Protocol

- 90 frozen Phase 16 validation scenes, 45 mirror groups;
- identical scene file and episode seeds in both arms;
- Diagonal-SSM conditional-diffusion checkpoint from seed `727201`;
- distributed delayed DN-MPC + local CBF;
- two-step immutable execution queue, zero command noise;
- fixed `K=8`, refresh every step, RNIC disabled;
- one CPU process per arm, with component and total p50/p95/p99 latency;
- result label: `validation_selection`.

The scene manifest SHA-256 is
`5ba54afc62d2cf123014f1c5e598a0e3e914e9067870bc7a199ba3393d8f8d99`.

## Paired result

| Arm | Safe capture | Capture event | Collision | Boundary | Total p50/p95/p99 (ms) |
|---|---:|---:|---:|---:|---:|
| QDR off | 94.44% (85/90) | 94.44% | 5.56% | 0% | 111.75 / 135.95 / 144.66 |
| QDR on | 88.89% (80/90) | 90.00% | 11.11% | 0% | 110.65 / 134.77 / 140.99 |

The paired QDR-on minus QDR-off safe-capture difference is `-5.56` percentage
points, bootstrap 95% CI `[-13.33, +2.22]` percentage points.  The paired
collision difference is `+5.56` percentage points, bootstrap 95% CI
`[-2.22, +13.33]` percentage points.  The interval is not a proof of a
significant effect, but the direction is inconsistent with the preregistered
QDR improvement hypothesis and the gate is not passed.

QDR-on component latency was:

| Component | p50 / p95 / p99 (ms) |
|---|---:|
| Predictor | 51.33 / 55.88 / 59.44 |
| Planner | 48.25 / 71.17 / 77.88 |
| QDR rollout | 1.96 / 2.38 / 2.73 |
| Local CBF | 3.63 / 4.17 / 4.94 |
| Total control | 110.65 / 134.77 / 140.99 |

## Immutable-prefix diagnosis

The QDR-on arm records the nominal prefix independently of the post-action
episode metrics.  Across its step log:

- minimum nominal prefix obstacle clearance: `-4.9816 m`;
- minimum nominal prefix boundary margin: `1.0714 m`;
- minimum nominal prefix inter-agent distance: `0.5926 m`;
- maximum safety-margin violation: `5.3316 m`;
- nominal prefix violation rate: `2.27%` of control steps.

This identifies the current failure mechanism: an immutable queued command
can already drive the nominal prefix into an unsafe geometry before a newly
computed suffix becomes controllable.  The current QDR adapter aligns the
planner state and candidate time index, but it does not cancel the queue or
provide a separately verified recovery action.  These diagnostics use public
geometry and nominal dynamics; they are not a reachable-set or CBF proof.

The evaluator now also performs a post-hoc endpoint audit: when the delayed
time `t+d` is reached, it compares the saved nominal endpoint with simulator
truth.  In a one-episode zero-noise schema check, the mean and maximum
position/velocity errors were all `0`, with `22/24` checks completed before
termination.  This confirms the audit path only; it does not alter the
90-scene result above or establish safety under noise.

## Decision and next experiment

QDR remains **No-Go for a main-method claim**.  The next experiment must add a
validation-only recovery/precondition branch and compare it against the
current QDR adapter under delay `0/1/2/4`, tracking time constant and bounded
command noise, one factor at a time.  The endpoint audit is now available and
must remain post hoc; it must not silently upgrade immutable authority.  If
the paired gate still fails, retain QDR as a transparent negative/diagnostic
result and keep the Phase 16 controller as the main baseline.

## Reproducibility artifacts

| Arm | Artifact directory |
|---|---|
| QDR off | `results/phase17_validation_qdr_off_diagonal_seed727201/` |
| QDR on | `results/phase17_validation_qdr_on_diagonal_seed727201/` |

Each directory contains `config.yaml`, copied `scenes.jsonl`, episode/step
JSONL, summaries, source hashes, and TensorBoard events.  Generated results
are Git-ignored; the evaluator, tests, configuration and this report are
versioned.
