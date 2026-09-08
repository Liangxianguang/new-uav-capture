# Phase 4 Unseen Adaptive Target Validation Report

> Version: v1.1 (2026-09-08)
> Repository: `https://github.com/Liangxianguang/new-uav-capture`
> Decision: `generalization_gate_passed_runtime_tradeoff_recorded`
> Scope: `adaptive_adversarial` unseen-target locked-test under velocity-level kinematic dynamics

## 1. Executive conclusion

The Phase 4 generalization experiment is complete. It uses 100 frozen
`adaptive_adversarial` episodes, one shared scene file, three prediction
checkpoints (`745101`, `745201`, `745301`), five checkpoint-dependent controller
methods, and one checkpoint-independent DynamicEncirclement baseline. The
non-baseline methods never receive target ground truth; they consume only the
visible observation, projected prediction candidates, local geometry, and
permitted peer messages.

The pre-registered outcome gate passes. The shared DynamicEncirclement baseline
achieves `95.00%` safe capture and `2.00%` collision. Across the three
checkpoint runs, the distributed methods achieve `98.00%` to `99.00%` safe
capture and `0.00%` to `1.00%` collision. Solver, valid-plan, effective-plan,
and distributed convergence rates are at least `99%`; all communication modes
have zero fallback in this protocol. The result supports a conditional claim
that projected diffusion candidates plus risk-sensitive distributed planning
generalize better than the current dynamic baseline on this unseen target
policy.

Runtime is reported as an engineering trade-off rather than a hard P4 outcome
gate. Prediction is refreshed every 20 control steps and candidate age is
explicitly recorded. The mean refresh rate is approximately `6.56%`, but the
maximum candidate age is `19` steps. The three-checkpoint mean total-control
p95 is `167.73--229.62 ms` depending on method. The `100 ms` value is retained
only as a strict 10 Hz deployment reference budget; it is not used to reject
the P4 generalization or encirclement result. The current measurements do not
support a claim that strict 10 Hz real-time deployment has been achieved.
Further work should use batch inference, fewer diffusion steps, distillation,
or an asynchronous prediction worker and re-evaluate candidate coverage and
safety jointly.

This report does not validate R-CLBF-QP, execution-invariant safety, a formal
zero-sum game guarantee, higher-order flight dynamics, or real UAV deployment.

## 2. Frozen protocol and provenance

| Item | Value |
| --- | --- |
| Protocol | `configs/phase4_unseen_adaptive_target.yaml` |
| Split | `locked_test` |
| Target policy | `adaptive_adversarial` |
| Episodes | 100 |
| Shared scenes | `results/phase4_unseen_adaptive_locked_seed745101_gpu_v2/scenes.jsonl` |
| Scene SHA-256 | `22C7A825CB11F0642B8C4F47671856E754982CEA818619E110740B50BB21B13F` |
| Observation strata | `nominal`, `delayed_noisy` |
| Obstacle counts | 3, 4, 5 |
| Defender sides | `left`, `right` |
| Candidate source | checkpoint |
| Candidate status | `dynamics-projected` |
| Samples / diffusion steps | 8 / 8 |
| Prediction refresh interval | 20 control steps |
| Planner | finite-shooting sequential best-response DN-MPC |
| MPC horizon / control horizon | 8 / 3 steps |
| Communication modes | ideal, 2-step delayed, 10% dropout, none |
| Local safety layer | existing local CBF |
| Strict 10 Hz deployment reference | 100 ms (reference only; not a P4 hard gate) |

All result directories preserve `config.yaml`, `protocol.yaml`,
`episodes.jsonl`, `steps.jsonl`, `summary.json`, and TensorBoard event files.
The dynamic baseline is checkpoint-independent and is reported from the
shared-scene run at
`results/phase4_unseen_adaptive_locked_seed745201_refresh20_v2_dynamic_encirclement/`.
The 745101 baseline outcome is identical; its representative total-latency
measurement is not used to claim a checkpoint-specific predictor result.

Checkpoint SHA-256 values are:

| Checkpoint | SHA-256 |
| --- | --- |
| `phase2_formal_diffusion_seed745101/checkpoint.pt` | `41AB7CEF31C3AFA53735DD96569C7A4D4247EAC9C84FB01FA874933B904A36E8` |
| `phase2_formal_diffusion_seed745201/checkpoint.pt` | `B667AFB7F9AA6292348E6DBCCB809F083191E6EBDB6B6D94D28DCF451F852D27` |
| `phase2_formal_diffusion_seed745301/checkpoint.pt` | `4947E7648675087472AC9AA7FCB493C3A7841508480D0F03FD684E20738A5EF2` |

Shared source hashes recorded by the runs include:

| Source | SHA-256 |
| --- | --- |
| `scripts/evaluate_minimax_mpc_s3.py` | `EEF16EE9F07356AB8AFED1590816FD67A9E45808C99BAB6DD84917DC1804B5BE` |
| `src/encirclement3d/minimax_mpc.py` | `A5CB98CF4FEFB1BBF69E41B9EFED2764792A16F715D5038EB5696CF3E673BF31` |
| `src/encirclement3d/distributed_dn_mpc.py` | `EC98020E9F5F703016C1B15FABD190549FFA3CB651A5708DE13A3ECCB4DB4D5D` |
| `src/encirclement3d/prediction.py` | `5F823A451E2FD7CF8306A70484E1C4259EC484CCB62537CD87F1BC6A86A3154C` |
| `src/encirclement3d/pursuit_env.py` | `4C3B5C5122608A0AA6A0A077511058B3B104F722CEB8DBD16A1E618990DC32D6` |
| `configs/phase4_unseen_adaptive_target.yaml` | `BBE90203FC2C570D08D56ED4112AFD8C1B674E436479B6273EB5F3FCCC91E724` |

The 745101 runs record evaluator hash
`7900F8F4E2F582609A358172642035E2723F8D89ECB8CD8485C026304EEE48F6`.
The 745201 and 745301 runs record evaluator hash
`DCB2E98017D6A7130745A761BA1BBB4EBCF54D0F1EDAB2F6DE866BAD2B63ADF8`.
The difference is retained as provenance; the later runs contain the same
cached-candidate behavior plus the finalized age and refresh diagnostics.

## 3. Per-checkpoint outcome matrix

`Safe`, `ordinary capture`, `collision`, `boundary`, and `timeout` are episode
rates. Capture time is in seconds and clearance is in meters. `n/a` means that
the metric is not defined for the non-MPC baseline or centralized oracle.

| Seed | Method | Safe | Ordinary | Collision | Boundary | Timeout | Mean capture time | Mean min clearance | Solver | Valid | Effective | Converged |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 745101 | dynamic_encirclement* | 95.00% | 95.00% | 2.00% | 0.00% | 3.00% | 4.096 | 0.370 | n/a | n/a | n/a | n/a |
| 745101 | worst_case | 97.00% | 97.00% | 1.00% | 0.00% | 2.00% | 4.751 | 0.309 | 100.00% | 100.00% | 100.00% | n/a |
| 745101 | distributed_ideal | 98.00% | 98.00% | 1.00% | 0.00% | 1.00% | 4.685 | 0.317 | 99.96% | 100.00% | 99.96% | 99.96% |
| 745101 | distributed_delayed | 99.00% | 99.00% | 0.00% | 0.00% | 1.00% | 4.652 | 0.303 | 100.00% | 100.00% | 100.00% | 100.00% |
| 745101 | distributed_dropout | 99.00% | 99.00% | 0.00% | 0.00% | 1.00% | 4.667 | 0.305 | 100.00% | 100.00% | 100.00% | 100.00% |
| 745101 | distributed_none | 98.00% | 98.00% | 0.00% | 0.00% | 2.00% | 4.612 | 0.296 | 100.00% | 100.00% | 100.00% | 100.00% |
| 745201 | dynamic_encirclement* | 95.00% | 95.00% | 2.00% | 0.00% | 3.00% | 4.096 | 0.370 | n/a | n/a | n/a | n/a |
| 745201 | worst_case | 98.00% | 98.00% | 1.00% | 0.00% | 1.00% | 4.804 | 0.309 | 100.00% | 100.00% | 100.00% | n/a |
| 745201 | distributed_ideal | 98.00% | 98.00% | 1.00% | 0.00% | 1.00% | 4.688 | 0.316 | 99.96% | 100.00% | 99.96% | 99.96% |
| 745201 | distributed_delayed | 99.00% | 99.00% | 0.00% | 0.00% | 1.00% | 4.649 | 0.304 | 100.00% | 100.00% | 100.00% | 100.00% |
| 745201 | distributed_dropout | 99.00% | 99.00% | 0.00% | 0.00% | 1.00% | 4.652 | 0.305 | 100.00% | 100.00% | 100.00% | 100.00% |
| 745201 | distributed_none | 98.00% | 98.00% | 0.00% | 0.00% | 2.00% | 4.606 | 0.296 | 100.00% | 100.00% | 100.00% | 100.00% |
| 745301 | dynamic_encirclement* | 95.00% | 95.00% | 2.00% | 0.00% | 3.00% | 4.096 | 0.370 | n/a | n/a | n/a | n/a |
| 745301 | worst_case | 97.00% | 97.00% | 1.00% | 0.00% | 2.00% | 4.753 | 0.310 | 100.00% | 100.00% | 100.00% | n/a |
| 745301 | distributed_ideal | 98.00% | 98.00% | 1.00% | 0.00% | 1.00% | 4.685 | 0.316 | 99.98% | 100.00% | 99.98% | 99.98% |
| 745301 | distributed_delayed | 99.00% | 99.00% | 0.00% | 0.00% | 1.00% | 4.644 | 0.303 | 100.00% | 100.00% | 100.00% | 100.00% |
| 745301 | distributed_dropout | 99.00% | 99.00% | 0.00% | 0.00% | 1.00% | 4.669 | 0.303 | 100.00% | 100.00% | 100.00% | 100.00% |
| 745301 | distributed_none | 98.00% | 98.00% | 0.00% | 0.00% | 2.00% | 4.623 | 0.296 | 100.00% | 100.00% | 100.00% | 100.00% |

\* `dynamic_encirclement` is checkpoint-independent. The repeated rows show
the same shared-scene outcome baseline, not three separate model evaluations.

## 4. Three-checkpoint aggregate

The following are arithmetic means of the three checkpoint runs for
checkpoint-dependent methods. The dynamic baseline is one shared-scene run.
Latency values are arithmetic means of the run-level percentiles, not pooled
step samples.

| Method | Runs | Safe | Collision | Boundary | Timeout | Mean capture time (s) | Mean min clearance (m) | Solver | Valid | Effective | Converged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic_encirclement* | 1 | 95.00% | 2.00% | 0.00% | 3.00% | 4.096 | 0.370 | n/a | n/a | n/a | n/a |
| worst_case | 3 | 97.33% | 1.00% | 0.00% | 1.67% | 4.769 | 0.310 | 100.00% | 100.00% | 100.00% | n/a |
| distributed_ideal | 3 | 98.00% | 1.00% | 0.00% | 1.00% | 4.686 | 0.316 | 99.97% | 100.00% | 99.97% | 99.97% |
| distributed_delayed | 3 | 99.00% | 0.00% | 0.00% | 1.00% | 4.648 | 0.303 | 100.00% | 100.00% | 100.00% | 100.00% |
| distributed_dropout | 3 | 99.00% | 0.00% | 0.00% | 1.00% | 4.662 | 0.304 | 100.00% | 100.00% | 100.00% | 100.00% |
| distributed_none | 3 | 98.00% | 0.00% | 0.00% | 2.00% | 4.614 | 0.296 | 100.00% | 100.00% | 100.00% | 100.00% |

The worst observed episode-level minimum clearance across the three runs is
`-0.020 m` for the dynamic baseline, `-0.039 m` for centralized worst-case,
`-0.028 m` for distributed ideal, `0.025 m` for delayed, `0.025 m` for
dropout, and `0.022 m` for no communication. These are diagnostic geometric
clearance values under the current simulator and should not be interpreted as
execution-robust safety certificates.

## 5. Latency, prediction age, and communication

| Method | Planner p50 / p95 / p99 (ms) | Predictor p50 / p95 / p99 (ms) | Safety p50 / p95 / p99 (ms) | Total p50 / p95 / p99 (ms) | Refresh | Mean age | Max age |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| dynamic_encirclement* | 0 / 0 / 0 | 0 / 0 / 0 | 6.44 / 9.10 / 11.38 | 7.98 / 11.08 / 13.64 | 0.00% | 0.00 | 0 |
| worst_case | 55.28 / 69.10 / 76.37 | 0 / 164.14 / 192.15 | 6.44 / 9.32 / 11.60 | 67.35 / 229.62 / 265.80 | 6.37% | 8.70 | 19 |
| distributed_ideal | 46.27 / 67.65 / 80.50 | 0 / 163.06 / 189.09 | 6.43 / 9.19 / 11.31 | 58.49 / 224.57 / 261.84 | 6.48% | 8.67 | 19 |
| distributed_delayed | 44.51 / 56.84 / 63.75 | 0 / 162.35 / 189.38 | 6.41 / 9.30 / 11.32 | 56.59 / 211.74 / 247.69 | 6.56% | 8.67 | 19 |
| distributed_dropout | 44.58 / 59.08 / 67.63 | 0 / 162.11 / 194.39 | 6.45 / 9.61 / 13.13 | 56.85 / 212.48 / 252.81 | 6.56% | 8.66 | 19 |
| distributed_none | 36.22 / 47.94 / 56.49 | 0 / 121.55 / 189.22 | 6.25 / 9.39 / 12.39 | 48.06 / 167.73 / 241.06 | 6.56% | 8.66 | 19 |

### 5.1 Fixed-scene sampling ablation

A separate 40-episode validation ablation was completed with checkpoint
`745201`. All four configurations use the same scene file
`results/phase4_adaptive_validation_ablation_seed745201_base_8x8/scenes.jsonl`,
whose SHA-256 is
`784960B8FF4980C2AAE06A3FDABD10B992D497B5F175144738AF9DA6C293DF68`.
The configuration label is `num_samples x diffusion_steps`; the planner,
safety layer, environment, and episode scenes are otherwise held fixed.

| Configuration | Safe capture | Collision | Planner p95 (ms) | Predictor p95 (ms) | Total p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8x8 | 92.5% | 7.5% | 13.65 | 37.61 | 51.42 |
| 4x8 | 92.5% | 7.5% | 9.35 | 38.96 | 48.71 |
| 2x8 | 92.5% | 7.5% | 6.70 | 37.62 | 44.86 |
| 8x4 | 92.5% | 7.5% | 27.34 | 38.76 | 53.90 |

The `2x8` configuration has the lowest total-control p95 in this validation
sample, while all four configurations have the same outcome rates. This is an
engineering trade-off result, not a replacement of the primary locked-test
configuration; selecting `2x8` as the main configuration requires a new
locked-test or a pre-registered configuration decision.

Communication aggregates over the three checkpoint runs are:

| Mode | Attempted | Sent | Received | Dropped | Sent bytes | Received bytes | Mean age | Max age |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ideal | 183,860 | 183,860 | 183,860 | 0 | 44,126,400 | 44,126,400 | 0.00 | 0 |
| delayed | 172,140 | 172,140 | 164,940 | 0 | 41,313,600 | 39,585,600 | 1.91 | 2 |
| dropout | 172,692 | 155,186 | 148,903 | 17,506 | 37,244,560 | 35,736,640 | 2.06 | 8 |
| none | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 | 0 |

All methods report `0.00%` fallback and `0.00%` local solver failure in the
formal summaries. The ideal distributed effective-plan rate is `99.97%`
because a small number of local plans did not meet the numerical effective
criterion even though valid plans were returned; this remains above the
pre-registered `99%` threshold.

## 6. Stratified results

Values below are three-checkpoint arithmetic means of safe-capture rates. They
are included to expose observation, geometry, and initialization-side effects.

| Method | Nominal | Delayed/noisy | 3 obstacles | 4 obstacles | 5 obstacles | Left | Right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dynamic_encirclement* | 95.83% | 94.23% | 96.97% | 93.94% | 94.12% | 97.96% | 92.16% |
| worst_case | 97.92% | 96.79% | 96.97% | 96.97% | 98.04% | 98.64% | 96.08% |
| distributed_ideal | 97.92% | 98.08% | 96.97% | 96.97% | 100.00% | 100.00% | 96.08% |
| distributed_delayed | 100.00% | 98.08% | 96.97% | 100.00% | 100.00% | 100.00% | 98.04% |
| distributed_dropout | 100.00% | 98.08% | 96.97% | 100.00% | 100.00% | 100.00% | 98.04% |
| distributed_none | 100.00% | 96.15% | 96.97% | 100.00% | 97.06% | 97.96% | 98.04% |

The hard-stratum requirement is met by the obstacle-count strata: delayed and
dropout reach `100%` at four and five obstacles, while the baseline is `93.94%`
and `94.12%` respectively. Right-side starts remain the weaker side for the
baseline and risk-sensitive methods, but the distributed methods do not show a
new collision or boundary-violation pattern there.

## 7. Gate review and interpretation

| Gate | Result | Evidence |
| --- | --- | --- |
| Same frozen scenes for all methods | Pass | Common scene SHA-256 and validated scene records |
| No target-truth access by non-oracle planners | Pass | Evaluator contract; truth is used only by environment termination metrics |
| Safe capture not below baseline | Pass | All checkpoint-dependent aggregates are `>=97.33%` vs `95.00%` baseline |
| Collision not worse than baseline by more than 1 pp | Pass | `0--1%` vs `2%` baseline |
| Hard stratum improvement | Pass | Four/five-obstacle delayed and dropout strata improve over baseline |
| Solver, valid, effective plan rate >=99% | Pass | Distributed rates are `99.97--100%`; no fallback |
| Communication audit | Pass | Delayed age, dropout count, bytes, and step logs are preserved |
| Strict 10 Hz deployment reference (100 ms) | Not met, non-blocking | Cached-prediction aggregate is `167.73--229.62 ms`; this is recorded as a runtime trade-off, not a P4 outcome gate |
| Execution-invariant safety | Not tested / No-Go | P5 execution perturbation audit failed under current filter contract |
| R-CLBF-QP closed-loop certificate | Not started | P6 remains paused |

The appropriate current claim is therefore:

```text
P4 unseen adaptive-target outcome/generalization: PASS
P4 cached low-frequency runtime contract: ENGINEERING TRADE-OFF RECORDED
Strict 10 Hz deployment reference: NOT YET MET
P5 execution-robust safety: NO-GO under current contract
R-CLBF-QP and complete end-to-end stack: UNVALIDATED
Overall research status: CONDITIONAL GO
```

## 8. Reproducibility artifact map

The formal results are stored locally under:

- `results/phase4_unseen_adaptive_locked_seed745101_refresh20_v2_*`
- `results/phase4_unseen_adaptive_locked_seed745201_refresh20_v2_*`
- `results/phase4_unseen_adaptive_locked_seed745201_refresh20_v2_comm_batch/`
- `results/phase4_unseen_adaptive_locked_seed745301_refresh20_v2_*`
- `results/phase4_unseen_adaptive_locked_seed745301_refresh20_v2_comm_batch/`
- `results/phase4_unseen_adaptive_locked_seed745101_refresh5/dynamic_encirclement/`

Each completed method directory contains the configuration snapshot, protocol
snapshot, episode JSONL, step JSONL, summary JSON, and TensorBoard event data.
The result directories are local artifacts and are intentionally not added to
Git. The report, evaluator, tests, and configuration are the versioned
reproducibility surface.

## 9. Next actions

1. Keep the P4 outcome result as a conditional planning/generalization claim.
2. Retain the completed fixed-scene `8x8`, `4x8`, `2x8`, and `8x4` sampling
   ablation with candidate coverage, capture, safety, age, and latency metrics;
   do not replace the primary configuration from this validation sample alone.
3. Implement batch or asynchronous prediction before claiming strict 10 Hz
   deployment.
4. Repair the P5 execution contract around delay queues, tracking state,
   shared dynamics, swept-volume checks, and multi-step invariance.
5. Do not train or enable learned CLBF until the execution-perturbation gate
   passes. The current safety contribution remains conditional robust CBF-QP.
