# Phase 26 QDR Delayed-State Safety Projection Pilot

> Status: diagnostic pilot; the new safety-projection variant is No-Go for
> promotion as a closed-loop performance contribution.

## Motivation

The first QDR implementation aligned the planner state and target-candidate
time index to the end of the pending command queue, but the local CBF still
filtered the newly selected command at the pre-queue state. This pilot adds an
explicit `queue-aware-safety-projection` switch. When enabled, the local CBF
is evaluated at the first controllable delayed state. The old behavior remains
available by default and is not silently changed.

This experiment does not grant the simulator any additional command
authority. The pending queue remains immutable, and the new action is still
appended behind it.

## Protocol

- 20 frozen episodes selected from the Phase 17 validation manifest;
- one Diagonal-SSM checkpoint, training seed `727201`;
- distributed delayed DN-MPC, two-step immutable command queue, zero command noise;
- fixed `K=8`, prediction refresh every step, RNIC off, local CBF on;
- identical scene order, episode seeds, sampling seed `745102` and projection iterations;
- one CPU process per arm; all component and total latency quantiles are retained;
- aggregate bootstrap uses 10,000 episode resamples with seed `20260912`.

The parent Phase 17 validation manifest SHA-256 is
`5ba54afc62d2cf123014f1c5e598a0e3e914e9067870bc7a199ba3393d8f8d99`. The
20-episode copied comparison manifest SHA-256 is
`46f042628f0d24673abe660b596830515a14ad9f68079651b05d289c637118fa`.

## Results

| Arm | QDR | Delayed-state safety projection | Safe capture | Capture event | Collision | Boundary | Timeout | Mean clearance (m) | Prefix minimum clearance (m) | Total p50 / p95 / p99 (ms) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QDR-off reference | off | off | 95.0% | 95.0% | 5.0% | 0.0% | 0.0% | 0.444 | n/a | 115.85 / 141.40 / 158.98 |
| Existing QDR | on | off | 80.0% | 80.0% | 20.0% | 0.0% | 0.0% | 0.291 | -4.844 | 119.74 / 148.61 / 160.58 |
| Aligned safety projection | on | on | 80.0% | 85.0% | 20.0% | 0.0% | 0.0% | 0.324 | -0.229 | 120.74 / 149.90 / 162.94 |

Paired differences relative to QDR-off are:

| Candidate | Safe-capture delta | Collision delta | Mean-clearance delta (m) |
| --- | ---: | ---: | ---: |
| Existing QDR | `-15.0 pp [-35.0, +5.0]` | `+15.0 pp [-5.0, +35.0]` | `-0.154 [-0.296, -0.021]` |
| Aligned safety projection | `-15.0 pp [-35.0, +5.0]` | `+15.0 pp [-5.0, +35.0]` | `-0.120 [-0.246, +0.007]` |

The sample is a validation pilot and not a significance claim. Both QDR arms
fail the predeclared ID non-inferiority direction. The new projection reduces
the worst recorded nominal prefix penetration, but does not reduce collision
rate or improve safe capture and adds approximately `1.0--1.5 ms` to total
latency relative to existing QDR.

The aligned arm's prefix endpoint audit coverage is `91.08%`, compared with
`90.62%` for existing QDR. This verifies the audit path but does not establish
an execution safety certificate; endpoint checks that are not reached before
episode termination are not counted as valid certificates.

### Prefix-risk classification rerun

The aligned arm was rerun with the released prefix-classifier logging schema,
keeping the same 20 scenes, checkpoint, sampling seed, queue contract and
controller settings. Across `465` QDR control steps, `92.47%` were classified
as prefix-admissible and `7.53%` contained at least one public-geometry
violation. The minimum prefix barrier was `-0.579 m`. The first violating
control-step causes were counted as:

| Cause | Violating control steps |
| --- | ---: |
| obstacle | 30 |
| inter-agent | 3 |
| boundary | 2 |

These counts are control-step diagnostics rather than episode-level failure
counts. They show that a non-trivial portion of the QDR prefix is already
outside the declared geometric safe set before the newly planned action can
take effect. The result supports treating queue execution as a precondition
and does not justify claiming that a delayed-state CBF projection repairs the
immutable queue.

The machine-readable audit is retained at
`results/phase26_qdr_prefix_audit_aligned_seed727201.json` and the rerun
itself at `results/phase26_qdr_prefix_audit_aligned_seed727201/`.

### Explicit queue-authority diagnostic

To test whether the prefix failure is caused by immutable execution rather
than by the planner itself, two additional 20-episode paired diagnostics were
run with the same frozen scenes, checkpoint, sampling protocol and local CBF.
The authority was changed only through the explicit simulator contract, and
the emergency brake was requested only when the prefix classifier reported an
unsafe prefix.

| Authority | Safe capture | Collision | Boundary | Timeout | Prefix admissible | Recovery request rate | Total p50 / p95 / p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| immutable / aligned projection | 80.0% | 20.0% | 0.0% | 0.0% | 92.47% | 0.0% | 120.74 / 149.90 / 162.94 |
| replace_nonexecuting | 75.0% | 20.0% | 0.0% | 5.0% | 90.15% | 9.85% | 114.89 / 139.07 / 148.43 |
| flush_pending | 70.0% | 5.0% | 0.0% | 25.0% | 74.74% | 25.26% | 115.13 / 137.96 / 149.95 |

The authority arms are diagnostic pilots, not significance claims. They show
that `flush_pending` can trade collision for timeout, while
`replace_nonexecuting` does not repair collision on this block. Neither arm
meets the safe-capture non-inferiority direction, so neither is promoted as a
QDR improvement. The key result is methodological: execution authority must
be reported as part of the plant contract, not hidden inside a “safety layer”.

Artifacts are retained at:

```text
results/phase26_qdr_authority_replace_seed727201_retry1/
results/phase26_qdr_authority_flush_seed727201/
```

## Decision

The new delayed-state safety projection is retained as an explicit, tested
contract option and diagnostic. It is **not** promoted as a QDR performance
improvement. The result indicates that correcting the CBF evaluation state is
necessary for semantic consistency, but it is insufficient to repair the
immutable queue's already unsafe prefix or the closed-loop planner outcome.

Do not continue scanning safety-projection margins on this 20-episode prefix.
The next valid QDR experiment must model the prefix as a precondition and
separate three cases: prefix already unsafe, prefix safe but suffix unsafe,
and both safe. Any recovery branch must use explicitly authorized
`replace_nonexecuting` or `flush_pending` execution contracts and must be
compared against immutable authority; it must not silently change the
physical execution model.

## Reproducibility artifacts

```text
results/phase26_qdr_pilot_off_seed727201/
results/phase26_qdr_pilot_old_seed727201/
results/phase26_qdr_pilot_aligned_seed727201/
results/phase26_qdr_pilot_aggregate.json
results/phase26_qdr_pilot_aggregate.md
```

Each run retains the effective configuration, source hashes, copied scene
manifest, episode/step JSONL, summary and TensorBoard event files. Generated
results remain Git-ignored.
