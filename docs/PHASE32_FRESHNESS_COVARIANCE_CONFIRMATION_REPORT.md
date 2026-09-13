# Phase 32 Freshness--Covariance Fusion Confirmation

## Decision

**Promotion No-Go.** The deterministic freshness--covariance public-belief
fusion implementation is reproducible and auditable, but its confirmation did
not establish a closed-loop benefit. It fails the pre-registered ID
non-inferiority gate on the centralized worst-case branch and does not change
the distributed episode outcome.

The module is therefore frozen as a negative/engineering ablation. It is not
added to the main QDR--UAKR--RNIC claim, and no parameter tuning or locked-test
run is opened from this result.

## Frozen protocol

| Item | Value |
| --- | --- |
| Confirmation scenes | `results/phase32_fcbf_confirmation_scenes/scenes.jsonl` |
| Episodes | 40 per seed; 20 complete mirror groups |
| Seeds | `727201`, `727202`, `727203` |
| Scene manifest SHA-256 | `67d2de08013a9dd61b160cdeffda489fc8a75d4c4625dc2d0c13f435f829ad52` |
| Source validation manifest SHA-256 | `5aaaa79c6dbef2d346ff57d48d238a34fe911d3534ebb576dff0252ba0593022` |
| Predictor | frozen GRU `both` checkpoint per seed |
| Safety layer | local CBF |
| Sampling | one sample, 8 diffusion steps, projection iterations 4 |
| Planner changes | fusion off: `legacy`; fusion on: `freshness_covariance` |
| Fusion parameters | age decay `0.15`, age inflation `0.05 m^2/step`, dropout inflation `0.25`, covariance floor `0.01 m^2`, confidence power `1.0`, minimum ESS `1.25` |
| Statistical unit | matched training seed and frozen-scene episode; 10,000 hierarchical bootstrap samples |
| Locked-test used | No |

The baseline and candidate used the same scenes, checkpoints, sampling seed,
planner horizon, execution delay, communication condition and local safety
layer. Fusion only consumed public belief fields; it did not read target truth.

## Closed-loop outcomes

Rates are means across the three training seeds. Brackets are hierarchical
bootstrap 95% CIs for the group mean. Paired deltas are candidate minus
baseline on the same seed and episode.

| Branch | Configuration | Safe capture | Collision | Boundary | Timeout |
| --- | --- | ---: | ---: | ---: | ---: |
| worst-case | baseline | 87.50% [80.83, 93.33] | 12.50% [6.67, 19.17] | 0.00% [0, 0] | 0.00% [0, 0] |
| worst-case | fusion | 86.67% [79.17, 93.33] | 13.33% [6.67, 20.83] | 0.00% [0, 0] | 0.00% [0, 0] |
| distributed delayed | baseline | 97.50% [94.17, 100.00] | 2.50% [0, 5.83] | 0.00% [0, 0] | 0.00% [0, 0] |
| distributed delayed | fusion | 97.50% [94.17, 100.00] | 2.50% [0, 5.83] | 0.00% [0, 0] | 0.00% [0, 0] |

| Branch | Safe-capture paired delta | Collision paired delta | Clearance paired delta |
| --- | ---: | ---: | ---: |
| worst-case | -0.83 pp [-5.00, +1.67] | +0.83 pp [-1.67, +5.00] | +0.0017 m [-0.0207, +0.0185] |
| distributed delayed | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | -0.0027 m [-0.0044, -0.0013] |

The worst-case safe-capture CI lower bound is below the required `-2 pp`
non-inferiority threshold. The distributed branch is behavior-neutral, not a
fusion improvement.

## Latency

Values are pooled step-level p50/p95/p99 in milliseconds. The 100 ms value is a
deployment reference, not a hard acceptance gate.

| Branch | Configuration | Predictor | Planner | Safety | Total |
| --- | --- | --- | --- | --- | --- |
| worst-case | baseline | 34.79 / 39.02 / 44.25 | 22.55 / 25.28 / 28.48 | 3.80 / 4.48 / 5.15 | 65.55 / 73.06 / 80.64 |
| worst-case | fusion | 37.63 / 41.84 / 46.86 | 24.60 / 27.59 / 30.71 | 3.73 / 4.35 / 5.11 | 70.37 / 77.86 / 87.93 |
| distributed delayed | baseline | 35.02 / 39.22 / 46.22 | 35.30 / 39.70 / 46.24 | 3.83 / 4.50 / 5.33 | 78.55 / 87.60 / 100.77 |
| distributed delayed | fusion | 29.03 / 35.34 / 41.03 | 34.57 / 41.66 / 48.19 | 2.87 / 3.77 / 4.49 | 70.28 / 83.01 / 94.81 |

The pooled distributed tail is variable across training seeds, so this
confirmation does not support a stable latency improvement. The outcome gate
already fails independently of latency; a dedicated runtime benchmark would
be needed before making a deployment-throughput claim.

## Fusion diagnostics

The fusion path was enabled on 100% of candidate steps. Averaged over seeds,
the worst-case / distributed diagnostics were respectively:

```text
effective sample size:       3.278 / 3.276
weight entropy:               1.175 / 1.174
mean message age (steps):     0.087 / 0.082
maximum message age (steps):  60 / 60
mean covariance trace (m^2): 0.0546 / 0.0501
fallback rate:                1.81% / 1.73%
```

Fallbacks were recorded as `insufficient_effective_samples`. Each of the six
runs wrote `Evaluation/config/text_summary`, source/manifest hashes and eight
`Summary/BeliefFusion/*` TensorBoard tags, in addition to episode and step
JSONL.

## Interpretation and next action

This result rejects the current fixed fusion weights as a promoted method; it
does not reject every possible public-belief fusion design. The correct next
step is not another weight scan on this holdout. The main study returns to the
three planned contributions:

1. repair and independently audit the QDR queue/execution contract;
2. calibrate UAKR using local intervention-effect labels before online gating;
3. rebuild RNIC reachable-time labels and validate its ranking with an
   independent checker;
4. only if those single-module gates pass, run the pre-registered factorial
   combinations and mixed OOD blocks.

The frozen fusion code may be retained as a baseline-support ablation or
revisited only on a newly defined development split with a new hypothesis.

## Artifacts

Aggregate artifact:

```text
results/phase32_fcbf_confirmation_aggregate.json
```

Run directories:

```text
results/phase32_fcbf_confirmation_baseline_seed727201/
results/phase32_fcbf_confirmation_baseline_seed727202/
results/phase32_fcbf_confirmation_baseline_seed727203/
results/phase32_fcbf_confirmation_fusion_seed727201/
results/phase32_fcbf_confirmation_fusion_seed727202/
results/phase32_fcbf_confirmation_fusion_seed727203/
```

These generated artifacts remain ignored by Git. The aggregation command was:

```powershell
python scripts\aggregate_closed_loop_seed_results.py `
  --group "baseline=results\phase32_fcbf_confirmation_baseline_seed*" `
  --group "fusion=results\phase32_fcbf_confirmation_fusion_seed*" `
  --reference-group baseline `
  --evaluation-split validation_confirmation `
  --report-title "Phase 32 Freshness Covariance Confirmation" `
  --output results\phase32_fcbf_confirmation_aggregate.json `
  --bootstrap-samples 10000 `
  --bootstrap-seed 20260913
```
