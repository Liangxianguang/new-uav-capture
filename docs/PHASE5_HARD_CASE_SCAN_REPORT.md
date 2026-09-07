# Phase 5 Hard-Case Scan Report

> Date: 2026-09-07
> Run: `results/phase5_hard_case_scan_v1/`
> Status: pressure-test evidence; not a formal P5 pass.

## Protocol

The scan evaluates five deterministic one-step states:

- narrow channel between two wall boxes;
- box corner approach;
- upper world boundary approach;
- close inter-agent separation;
- an altitude reset state intentionally outside the robust contracted set.

Each state is evaluated with hard-barrier QP, soft-slack diagnostic, local CBF,
and direct zero-action fallback. All actions are checked by the independent
one-step certificate. There are 20 case-method rows, and all configuration and
source hashes are stored in the output directory and TensorBoard event file.

## Results

| Method | Independent valid rows | Next-state safe rows | QP infeasible | Notable failure |
| --- | ---: | ---: | ---: | --- |
| hard-barrier | 4/5 | 4/5 | 0 | 1 precondition failure |
| soft-slack diagnostic | 4/5 | 4/5 | 0 | 3 nonzero-slack rows, 1 precondition failure |
| local CBF | 5/5 | 5/5 | N/A | uses the smaller local margin |
| zero fallback | 4/5 | 4/5 | N/A | unsafe reset remains unsafe |

For the four robust-safe cases, hard-barrier maximum constraint violation was
at numerical tolerance (`1.1e-15 m` worst in this scan). The hard-barrier
latencies were `2.96`--`6.90 ms` per one-step solve. Soft-slack returned
nonzero slack of approximately `4.5e-5`--`4.9e-5 m` in the narrow-channel,
upper-boundary, and close-agent cases. Those rows are retained as diagnostic
results and are not counted as QP certificates.

The altitude case has a current robust barrier of `-0.46 m`. Both hard and
soft paths classify it as `precondition_invalid`; no solver infeasibility is
claimed. The local CBF row is valid only under its zero robust-margin contract,
so it is not evidence that the robust contract holds for that state.

## Interpretation

The scan supports three limited conclusions:

1. The current linearized hard-barrier QP handles the selected wall, corner,
   boundary, and close-agent states when the robust-safe precondition holds.
2. Slack is visible and auditable, but soft-slack success cannot be promoted
   to a safety certificate.
3. Initial-set membership remains a first-class gate. A fallback action cannot
   provide an invariant-set certificate after the rollout starts outside that
   set.

The scan does not cover continuous-time swept-volume collision, multi-step
forward invariance, execution noise, delayed commands, or adaptive target
policies. Those remain required before P6 or an end-to-end safety claim.

## Reproduction and artifacts

```powershell
python scripts/scan_safety_hard_cases.py `
  --config configs/innovation_safety.yaml `
  --output-dir results/phase5_hard_case_scan_v1
```

Artifacts:

```text
results/phase5_hard_case_scan_v1/config.yaml
results/phase5_hard_case_scan_v1/steps.jsonl
results/phase5_hard_case_scan_v1/summary.json
results/phase5_hard_case_scan_v1/tensorboard/
```
