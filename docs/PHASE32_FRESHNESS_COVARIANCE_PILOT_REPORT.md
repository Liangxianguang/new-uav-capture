# Phase 32 Freshness--Covariance Public-Belief Fusion Pilot

## Decision

This is a **development pilot** and a **Go-to-confirmation candidate**, not a
promoted main result. The pilot uses a frozen one-seed, 20-episode validation
prefix and must not be used to tune the method or open the locked test.

The candidate implements deterministic fusion of public target beliefs. Each
source is weighted by reported confidence, message age, dropout inflation and
reported covariance trace. The fused covariance is eigenvalue-floored, and a
deterministic self-anchor fallback is used when the effective sample size is
too small. The implementation does not read simulator target truth and does
not propagate the position mean a second time because the environment already
time-aligns stale public means.

## Reproducibility contract

| Item | Frozen value |
| --- | --- |
| Development scenes | `results/phase27_rnic_fresh_validation_scenes/scenes.jsonl` |
| Episodes | 20 (10 complete mirror groups) |
| Checkpoint | `results/phase16_gru_both_seed727201/checkpoint.pt` |
| Checkpoint seed | `727201` |
| Sampling seed | `745102` |
| Prediction samples / diffusion steps | `1 / 8` |
| Projection iterations | `4` |
| Safety layer | local CBF |
| Planner branches | centralized `worst_case`, `distributed_delayed` |
| QDR/RNIC/EGC/FC-DBF | disabled |
| Decision label | `validation_selection` |
| Pilot fusion parameters | age decay `0.15`, age inflation `0.05 m^2/step`, dropout inflation `0.25`, covariance floor `0.01 m^2`, confidence power `1.0`, minimum ESS `1.25` |

The baseline uses `belief_fusion_mode: legacy`; the pilot uses
`belief_fusion_mode: freshness_covariance`. Both runs use identical scenes,
checkpoint, sampling, planner and safety settings.

## Episode-level result

The table reports descriptive rates only; no confidence interval is attached
because this is one seed and 20 episodes.

| Branch | Fusion | Safe capture | Collision | Boundary | Timeout | Mean min clearance (m) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| worst-case | off | 85.0% | 15.0% | 0.0% | 0.0% | 0.3928 |
| worst-case | on | 90.0% | 10.0% | 0.0% | 0.0% | 0.4266 |
| distributed delayed | off | 95.0% | 5.0% | 0.0% | 0.0% | 0.2504 |
| distributed delayed | on | 95.0% | 5.0% | 0.0% | 0.0% | 0.2548 |

The worst-case branch shows a descriptive `+5 pp` safe-capture change and a
`-5 pp` collision change. The distributed branch has unchanged episode
outcomes. These changes are not statistically established and may be caused
by the small development prefix.

## Latency result

All values are milliseconds. Predictor, planner, safety and total-control
latencies are reported separately as p50/p95/p99.

| Branch | Fusion | Predictor | Planner | Safety | Total |
| --- | --- | --- | --- | --- | --- |
| worst-case | off | 34.94 / 38.67 / 45.30 | 22.76 / 25.79 / 27.85 | 3.82 / 4.47 / 5.38 | 65.98 / 72.06 / 79.91 |
| worst-case | on | 38.08 / 43.07 / 47.93 | 25.01 / 27.69 / 29.75 | 3.77 / 4.51 / 5.09 | 71.28 / 79.02 / 86.66 |
| distributed delayed | off | 35.00 / 39.68 / 45.47 | 35.42 / 39.93 / 43.68 | 3.81 / 4.55 / 5.18 | 78.50 / 89.61 / 93.70 |
| distributed delayed | on | 37.81 / 41.34 / 47.97 | 45.04 / 50.53 / 55.39 | 3.78 / 4.34 / 4.81 | 91.09 / 100.28 / 111.54 |

The fusion pilot increases distributed total p95 by `10.67 ms` on this run.
The project's 100 ms value remains a deployment reference rather than a hard
gate; the tail latency must nevertheless be reported and explained.

## Fusion diagnostics

The pilot logs the following step-level and episode-level quantities and
writes them to TensorBoard under `Summary/BeliefFusion/*`:

| Branch | Enabled | Effective sample size | Weight entropy | Mean age (steps) | Max age (steps) | Mean covariance trace (m^2) | Fallback |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| worst-case | 100% | 3.312 | 1.179 | 0.008 | 60 | 0.0374 | 2.85% |
| distributed delayed | 100% | 3.322 | 1.182 | 0.006 | 60 | 0.0353 | 2.90% |

The maximum age of 60 is a diagnostic maximum rather than the average age;
it confirms that the age path is active even though most observations are
fresh. Fallbacks are currently caused by insufficient effective sample size.

## Gate for the next stage

The pilot passes only the engineering/observability gate. Before any claim of
benefit, run a fresh confirmation holdout with three checkpoint seeds and the
parameters above frozen:

1. ID safe-capture paired 95% CI lower bound must be at least `-2 pp` relative
   to the legacy baseline;
2. collision and boundary must not increase on either planner branch;
3. at least one delayed/OOD stratum must show a pre-registered improvement in
   collision, timeout or safe capture;
4. distributed total p95 increase must be reported and remain operationally
   justified; and
5. fusion diagnostics must be finite, reproducible and consistent with the
   public-belief information boundary.

If the confirmation does not satisfy these conditions, freeze the module as a
reproducible belief-fusion ablation and do not add it to the main three-module
claim. If it passes, test it as a supporting component for QDR or RNIC, with a
new factorial arm rather than silently changing the main baseline.

## Artifacts

Successful local runs:

- `results/phase32_fcbf_baseline_seed727201_smoke_retry1/`
- `results/phase32_fcbf_pilot_seed727201_smoke/`

The result directories are intentionally ignored by Git. They contain the
effective configuration, source hashes, episode/step JSONL and TensorBoard
event files. The versioned implementation is in
`src/encirclement3d/belief_fusion.py`; integration and logging changes are in
`src/encirclement3d/minimax_mpc.py`,
`src/encirclement3d/distributed_dn_mpc.py`,
`scripts/evaluate_minimax_mpc.py` and
`scripts/evaluate_s4_closed_loop.py`.
