# Phase 31 FC-DBF Validation Confirmation

## 1. Decision

The pre-registered three-seed confirmation classifies FC-DBF as **No-Go for
promotion**.  The distributed-delayed branch is outcome-equivalent to the
off arm, but the worst-case branch misses the safe-capture non-inferiority
gate and FC-DBF adds a large distributed planning cost.  The locked test is
not opened and no additional FC-DBF weight/threshold scan is authorized.

This is a valid negative result, not evidence that the implementation is
incorrect.  It shows that the current finite formation-slot gate is
auditable and executable, but it does not yet convert into a reliable closed-
loop capture benefit.

## 2. Frozen confirmation block and artifacts

The confirmation block was created by
`scripts/select_mirror_group_scenes.py` from the Phase 27 validation manifest.
The source contained 60 episodes and 30 mirror groups.  The first 10 groups
were reserved for the earlier Phase 31 smoke; the confirmation uses the
remaining 20 complete groups (40 episodes), preserving the original episode
seeds and indices.  The selected scenes file hash is
`67d2de08013a9dd61b160cdeffda489fc8a75d4c4625dc2d0c13f435f829ad52`.

Every arm uses checkpoint seeds `727201`, `727202`, and `727203`, the same
predictor protocol, CPU device, local CBF, one diffusion sample, eight
sampling steps, four projection iterations, sampling seed `745102`, and the
same fixed FC-DBF parameters from the pilot.  The comparison is paired at the
episode level and uses 10,000-sample hierarchical bootstrap intervals over
matched seed/episode units.

| Arm | Runs |
| --- | --- |
| FC-DBF off | `results/phase31_fc_dbf_confirmation_off_seed727201/` through `...203/` |
| FC-DBF on | `results/phase31_fc_dbf_confirmation_on_seed727201/` through `...203/` |
| Aggregate | `results/phase31_fc_dbf_confirmation_aggregate.json` and `.md` |

The generated results remain Git-ignored local artifacts.  The selection
script, configs, source code and this report are versioned.

## 3. Episode outcomes

| Planner branch | FC-DBF | Safe capture | Capture | Collision | Boundary | Timeout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Worst-case | off | 87.50% [80.83%, 93.33%] | 90.83% [85.00%, 95.83%] | 12.50% [6.67%, 19.17%] | 0% | 0% |
| Worst-case | on | 86.67% [80.00%, 92.50%] | 90.00% [84.17%, 95.00%] | 13.33% [7.50%, 20.00%] | 0% | 0% |
| Distributed delayed | off | 97.50% [94.17%, 100.00%] | 97.50% [94.17%, 100.00%] | 2.50% [0.00%, 5.83%] | 0% | 0% |
| Distributed delayed | on | 97.50% [94.17%, 100.00%] | 97.50% [94.17%, 100.00%] | 2.50% [0.00%, 5.83%] | 0% | 0% |

Paired differences (on minus off) are:

| Branch | Safe-capture delta | Collision delta | Gate interpretation |
| --- | ---: | ---: | --- |
| Worst-case | -0.83 pp [-3.33, 0.00] | +0.83 pp [0.00, 3.33] | Fails the -2 pp non-inferiority and no-worsening collision gates |
| Distributed delayed | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] | Outcome gate passes, but latency gate fails |

## 4. Latency

Values are p50/p95/p99 in milliseconds; 100 ms is not treated as a hard
constraint.

| Branch | FC-DBF | Predictor | Planner | Safety | Total control |
| --- | ---: | ---: | ---: | ---: | ---: |
| Worst-case | off | 34.30 / 37.76 / 42.01 | 22.43 / 24.91 / 27.43 | 3.74 / 4.29 / 4.83 | 64.85 / 71.06 / 78.08 |
| Worst-case | on | 33.57 / 36.89 / 43.52 | 39.07 / 42.59 / 49.20 | 3.62 / 4.14 / 5.12 | 80.59 / 87.48 / 96.87 |
| Distributed delayed | off | 34.34 / 37.62 / 41.59 | 33.68 / 37.41 / 40.62 | 3.76 / 4.32 / 4.99 | 76.09 / 83.33 / 90.01 |
| Distributed delayed | on | 33.66 / 37.07 / 40.70 | 184.43 / 199.36 / 210.45 | 3.38 / 4.19 / 5.09 | 225.65 / 243.68 / 257.89 |

FC-DBF increases total p95 by about `16.43 ms` in worst-case and
`160.35 ms` in distributed delayed.  The distributed planner is therefore
the dominant unresolved bottleneck; further parameter tuning without an
algorithmic cost reduction would not be a fair innovation experiment.

## 5. Gate and implementation diagnostics

The on-arm diagnostics are the mean of the three per-seed aggregate reports:

| Branch | Feasible rate | Gate exhaustion | Mean progress (m) | Mean min slack (s) | Mean max error (m) | Switch rate | Mean cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Worst-case | 100.000% | 0.000% | 0.589 | -0.807 | 4.505 | 0.0436 | 1.347 |
| Distributed delayed | 98.778% | 1.222% | 0.590 | -0.813 | 4.554 | 0.0722 | 1.390 |

All six runs report solver success, valid-plan, effective-plan and local
solver-failure rates of `100%`, `100%`, `100%` and `0%`, respectively.  The
distributed gate exhaustion is close to, but above, the pre-registered `<1%`
engineering target.  The negative slot slack is the declared short-horizon
development contract from Phase 31, not a safety certificate.

## 6. TensorBoard and reproducibility audit

All 12 method logs (two planner branches × two arms × three seeds) contain
the nine expected `Summary/FCDBF/*` tags:

```text
Summary/FCDBF/enabled_rate
Summary/FCDBF/feasible_rate
Summary/FCDBF/mean_min_slot_slack_s
Summary/FCDBF/mean_max_slot_error_m
Summary/FCDBF/mean_slot_error_m
Summary/FCDBF/mean_slot_progress_m
Summary/FCDBF/assignment_switch_rate
Summary/FCDBF/gate_exhaustion_rate
Summary/FCDBF/mean_cost
```

Each run also stores its effective YAML configuration, source hashes,
`episodes.jsonl`, `steps.jsonl`, and TensorBoard event files.  The aggregate
manifest hash matches across all runs.  No locked-test artifact was read.

## 7. Next action

FC-DBF should now be frozen as a reproducible negative/engineering result.
Do not open its OOD gate or continue weight scans.  The next research effort
should address the unresolved QDR/UAKR/RNIC contracts: either construct a
fresh, independently labeled risk signal with a measured intervention effect,
or repair execution-delay/safety authority before attempting another
end-to-end composition.  Any future formation module must first pass a
controlled planner-cost benchmark and then the same paired non-inferiority
gate.
