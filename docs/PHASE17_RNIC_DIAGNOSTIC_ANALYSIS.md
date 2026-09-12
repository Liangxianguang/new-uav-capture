# Phase 17 RNIC Diagnostic Analysis

Status: **No-Go as a risk-prediction subclaim**.

This post-hoc analysis asks a narrower question than the closed-loop RNIC
comparison: does the logged nominal `minimum_best_slack_s` separate episodes
that later collide from episodes that remain safe? The answer is no for the
current delay/execution-OOD block. The analysis never feeds simulator outcomes
back into planning and therefore cannot inflate the closed-loop result.

## Protocol

- Three frozen RNIC runs, seeds `727201`, `727202`, and `727203`.
- The same 100-episode delay/execution-OOD manifest was used for every seed.
- The only score is `-minimum_best_slack_s`; larger values mean more nominal
  reachability risk.
- Outcomes are episode-level collision and unsafe (`collision OR boundary`).
- Ties use the mid-rank AUROC definition.

The executable analysis is
`scripts/analyze_phase17_rnic_diagnostics.py`, and it writes both JSON and
Markdown summaries plus optional TensorBoard scalars.

## Result

| Group | Episodes | Mean slack (s) | Unsafe | Collision | Unsafe AUROC | Collision AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seed 727201 | 100 | -0.864 | 59.00% | 59.00% | 0.489 | 0.489 |
| seed 727202 | 100 | -0.863 | 60.00% | 60.00% | 0.515 | 0.515 |
| seed 727203 | 100 | -0.864 | 60.00% | 60.00% | 0.506 | 0.506 |
| pooled | 300 | -0.863 | 59.67% | 59.67% | 0.504 | 0.504 |

The four pooled risk bins, ordered from low to high `-minimum_best_slack_s`,
have collision rates `53.33%`, `65.33%`, `61.33%`, and `58.67%`. They are not
monotonic. Thus the current RNIC slack is not a calibrated failure score under
the tested queue, dropout, observation-noise, command-noise, tracking-lag, and
dynamics-randomization block.

## Decision

1. Do not present RNIC as a safety certificate, collision predictor, or main
   performance contribution.
2. Retain RNIC as a transparent baseline/negative control because it exposes
   when nominal arrival-time assumptions become infeasible.
3. Do not run the full QDR × UAKR × RNIC matrix until one component passes its
   pre-registered gate.
4. The next candidate should calibrate a horizon-dependent reachable tube on
   the development split, including queue length and execution uncertainty,
   and then test the frozen calibration on untouched confirmation scenes.

## Reproduction artifact

The generated local artifacts are ignored by Git but retained for inspection:

```text
results/phase17_ood_delay_execution_rnic_diagnostics.json
results/phase17_ood_delay_execution_rnic_diagnostics.md
results/phase17_ood_delay_execution_rnic_diagnostics/tensorboard/
```
