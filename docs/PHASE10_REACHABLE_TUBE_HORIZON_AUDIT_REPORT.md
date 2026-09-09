# Phase 10 Reachable-Tube Horizon Audit

This audit freezes an empirical tube multiplier on an independent calibration
split and evaluates it on a separate holdout split for horizons 1, 2, 3, and
5. Each row uses 1,024 calibration samples and 2,048 holdout samples at the
0.995 calibration quantile. Wilson intervals are 95% intervals for the
holdout coverage proportion. The audit is empirical and does not prove
continuous-time reachability or forward invariance.

## 1. Observed-parameter contract

The no-noise reference rollout uses the execution parameters observed by the
filter. The calibrated multipliers are below one because the analytic
noise-only tube is already conservative for this contract; the runtime safety
configuration still requires a multiplier of at least one.

| Variant | H1 multiplier / coverage | H2 multiplier / coverage | H3 multiplier / coverage | H5 multiplier / coverage |
| --- | --- | --- | --- | --- |
| mild immutable | 0.90 / 99.27% [98.80, 99.56] | 0.89 / 99.66% [99.30, 99.83] | 0.84 / 98.34% [97.69, 98.81] | 0.88 / 99.22% [98.73, 99.52] |
| hard immutable | 0.92 / 99.80% [99.50, 99.92] | 0.91 / 99.76% [99.43, 99.90] | 0.92 / 99.56% [99.17, 99.77] | 0.91 / 99.56% [99.17, 99.77] |
| mild flush pending | 0.88 / 99.27% [98.80, 99.56] | 0.89 / 99.85% [99.57, 99.95] | 0.87 / 99.17% [98.67, 99.48] | 0.89 / 99.56% [99.17, 99.77] |
| hard flush pending | 0.92 / 99.51% [99.10, 99.73] | 0.91 / 99.46% [99.04, 99.70] | 0.92 / 99.46% [99.04, 99.70] | 0.94 / 99.80% [99.50, 99.92] |

The mild H3 row misses the 99% point estimate target. It is therefore not
valid to select a single horizon multiplier from the observed-parameter rows
without a policy for this failure case.

## 2. Configured-nominal contract

Here the nominal rollout uses configured execution parameters while the actual
rollout randomizes the execution parameters within the declared hard/mild
domain. This is the relevant contract when the filter cannot observe the true
mass, drag, or actuator time constant.

| Variant | H1 multiplier / coverage | H2 multiplier / coverage | H3 multiplier / coverage | H5 multiplier / coverage |
| --- | --- | --- | --- | --- |
| mild immutable | 0.90 / 99.27% [98.80, 99.56] | 0.89 / 99.66% [99.30, 99.83] | 0.84 / 98.34% [97.69, 98.81] | 0.88 / 99.22% [98.73, 99.52] |
| hard immutable | 2.06 / 99.80% [99.50, 99.92] | 2.06 / 99.41% [98.98, 99.66] | 2.07 / 99.17% [98.67, 99.48] | 2.04 / 99.37% [98.92, 99.63] |
| mild flush pending | 0.88 / 99.27% [98.80, 99.56] | 0.89 / 99.85% [99.57, 99.95] | 0.87 / 99.17% [98.67, 99.48] | 0.89 / 99.56% [99.17, 99.77] |
| hard flush pending | 2.01 / 98.88% [98.32, 99.25] | 2.05 / 99.66% [99.30, 99.83] | 2.05 / 99.41% [98.98, 99.66] | 2.07 / 99.66% [99.30, 99.83] |

The hard H1 flush row fails the 99% runtime-coverage gate, while the hard H2,
H3, and H5 rows pass at the point-estimate level. The existing hard multiplier
`2.1` is therefore defensible as a conservative locked configuration for the
declared hard domain, but it is still an empirical choice rather than a proof.

## 3. Decision and next experiment

The evidence supports a horizon-aware calibration table, not an unconditional
fixed multiplier:

- keep `2.1` for the current hard configured-nominal contract until an
  independent larger holdout is completed;
- treat observed-parameter and configured-nominal contracts as different
  safety assumptions;
- reject or fall back when delay, queue length, or parameter uncertainty is
  outside the calibrated domain;
- collect queue-length, speed, noise, and delay as covariates before fitting a
  dynamic multiplier `f(delay, queue_length, speed, uncertainty)`;
- do not lower the multiplier because a simulation captures more targets.

Evidence files:

- `results/phase10_reachable_tube_horizon_holdout_v1/summary.json`
- `results/phase10_reachable_tube_horizon_holdout_configured_v1/summary.json`
- `scripts/audit_execution_reachable_horizons.py`

