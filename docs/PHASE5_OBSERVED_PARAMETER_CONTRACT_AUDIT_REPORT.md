# Phase 5 Observed Execution-Parameter Contract Audit

> Version: v1.0 (2026-09-08)
> Run: `results/phase5_execution_reachable_observed_q995_runtime_v2/`
> Decision: `empirical contract audit pass; execution-invariant safety remains No-Go`

## 1. Purpose

The earlier reachable-margin hold-out audit used a conservative diagnostic
contract: randomized execution parameters and command noise were compared with
a configured nominal, no-noise rollout. The main safety filter, however, uses
the execution parameters available in the observation. This report audits that
declared observation contract separately.

The new run uses `observed_execution_parameters` for the reference rollout,
`q=0.995`, 4096 calibration samples and 4096 independent hold-out samples per
variant. Each sample still contains a five-step execution preview for four
defenders.

## 2. Results

| Variant | Calibration coverage | Runtime multiplier | Hold-out coverage | Runtime hold-out coverage |
| --- | ---: | ---: | ---: | ---: |
| mild delay/noise | 99.512% | 0.899 | 99.561% | 99.561% |
| hard randomized execution | 99.512% | 0.906 | 99.292% | 99.243% |
| mild flush pending/brake | 99.512% | 0.886 | 99.561% | 99.561% |
| hard flush pending/brake | 99.512% | 0.914 | 99.609% | 99.561% |

The observed-parameter contract therefore clears the empirical 99% hold-out
coverage target for all four variants. The per-sample runtime margin also
clears the target, with the lowest observed coverage at 99.243%.

The current filter multiplier is `1.0`, which is more conservative than the
empirical runtime multipliers of approximately `0.886--0.914` under this
benchmark contract. This comparison is diagnostic only; it does not authorize
changing the filter or claiming a proof.

## 3. Interpretation against the earlier audit

The earlier `98.05--98.93%` result remains valid for its declared
`configured_nominal` contract. It is conservative because the reference
rollout does not use the randomized execution parameters that are already
available to the filter. The new `99.243--99.609%` result answers a different
question: coverage when the reference rollout uses the per-sample observed
execution parameters.

This distinction does not remove the need to model observation error,
unobserved actuator state, parameter drift, or command-authority mismatch.
Out-of-calibration and altered execution settings remain marked
`reject_or_fallback`, so their empirical coverage is not actionable evidence
for the main filter.

## 4. Safety decision

This run is an empirical contract audit, not a continuous-time or closed-loop
safety proof. It does not establish forward invariance, and it does not fix the
existing failures in actual post-step safety, continuous-segment certificates,
fallback behavior, or filter latency under the execution-perturbation matrix.

Therefore:

- the observed-parameter reachable-margin audit passes its empirical 99%
  hold-out gate;
- the margin is not promoted into the main safety filter solely from this run;
- P5 execution-invariant safety remains **No-Go**;
- learned CLBF/R-CLBF-QP remains paused until the execution contract,
  certifiable fallback, multi-step invariance, and continuous-time argument are
  all resolved.

## 5. Reproducibility

The run preserves `config.yaml`, source hashes, calibration and hold-out JSONL,
sensitivity JSONL, out-of-calibration JSONL, `summary.json`, and TensorBoard
events. The exact command was:

```powershell
$env:PYTHONPATH = "src"
python scripts/audit_execution_reachable_holdout.py `
  --config configs/innovation_execution_reachable_holdout_audit.yaml `
  --output-dir results/phase5_execution_reachable_observed_q995_runtime_v2 `
  --parameter-contract observed_execution_parameters `
  --quantile 0.995 `
  --calibration-samples 4096 `
  --holdout-samples 4096 `
  --sensitivity-samples 64
```
