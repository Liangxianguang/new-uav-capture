# Phase 56 development-calibration report

## Scope and statistical contract

This report summarizes the three-seed development-calibration closed-loop matrix
for the Phase 56 strong-baseline study. It uses the frozen
`development_calibration` manifest (`120 episodes / 60 mirror groups`), GRU
`both`, local CBF, CPU Torch `1/1` threads, common candidate/MPC/execution
settings, and B0--B7 method contracts. Confidence intervals are hierarchical
95% bootstrap intervals over matched predictor seeds and mirror groups; the
upper/lower members are averaged within each mirror group before resampling.

The complete machine-readable output is generated at
`results/phase56_dev_calibration_phase56_report.json` by
`scripts/aggregate_phase56_mirror_group_results.py`. Raw `episodes.jsonl`,
`steps.jsonl`, effective configs, manifest hashes and TensorBoard events remain
the reproducibility artifacts. No locked-diagnostic result was used for tuning.

## Overall closed-loop outcome

| Method | Safe capture | Collision | Boundary | Timeout | Capture time (s) | Minimum clearance (m) | Total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 current-state delayed-MPC | 69.72% [63.61,75.56] | 30.00% [24.17,35.83] | 17.78% [13.06,23.06] | 0.28% [0,1.11] | 2.991 | 0.299 | 65.13/73.88/82.36 |
| B1 QDR-MPC | 75.00% [68.89,80.56] | 24.72% [19.16,31.39] | 15.00% [10.56,20.00] | 0.28% [0,1.11] | 4.631 | 0.301 | 115.99/130.09/143.25 |
| B2 fixed-tube MPC | 69.72% [63.05,75.83] | 30.00% [24.17,36.11] | 17.78% [12.50,22.78] | 0.28% [0,1.11] | 2.991 | 0.299 | 63.56/79.96/107.23 |
| B3 queue-aware tube MPC | 75.00% [68.33,80.56] | 24.72% [18.89,31.11] | 15.00% [10.83,20.01] | 0.28% [0,1.11] | 4.631 | 0.301 | 109.17/139.00/172.49 |
| B4 synchronous distributed MPC | 75.56% [69.44,81.39] | 24.44% [18.61,30.56] | 14.44% [10.00,18.90] | 0.00% | 2.534 | 0.310 | 70.29/86.81/104.86 |
| B5 asynchronous distributed MPC | 67.78% [61.66,73.89] | 32.22% [26.11,38.34] | 10.00% [6.66,13.62] | 0.00% | 2.480 | 0.259 | 69.05/85.36/103.60 |
| B6 QDR + asynchronous MPC | 93.33% [87.78,98.33] | 5.28% [1.39,9.44] | 5.00% [1.67,9.17] | 1.39% [0,3.61] | 5.822 | 0.567 | 112.58/145.74/184.98 |
| B7 fixed-K=8 QDR | 75.00% [68.61,80.56] | 24.72% [19.16,31.11] | 15.00% [10.28,20.56] | 0.28% [0,1.11] | 4.631 | 0.301 | 106.11/130.55/160.63 |

The B6 result is a strong descriptive composite result, but it does not by
itself identify whether the gain comes from QDR, asynchronous scheduling, or
their interaction. B1 is the isolated QDR contract required for that causal
comparison and is therefore the primary promotion gate.

## Runtime decomposition

Values are p50/p95/p99 in milliseconds. `QDR/tube` is zero for methods that do
not execute a QDR/tube stage; it is still reported explicitly rather than
omitted.

| Method | Predictor | Planner | QDR/tube | Safety | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 | 34.36/39.87/45.85 | 22.36/26.50/30.54 | 0.00/0.00/0.00 | 3.71/4.89/6.05 | 65.13/73.88/82.36 |
| B1 | 34.56/40.40/46.46 | 62.10/70.64/79.78 | 3.28/5.75/6.96 | 3.69/4.98/6.22 | 115.99/130.09/143.25 |
| B2 | 33.16/42.43/56.59 | 21.63/28.55/39.03 | 0.00/0.00/0.00 | 3.56/5.18/7.20 | 63.56/79.96/107.23 |
| B3 | 32.44/42.71/54.64 | 58.60/75.85/97.82 | 3.09/5.68/7.59 | 3.49/5.36/7.02 | 109.17/139.00/172.49 |
| B4 | 32.28/40.05/49.69 | 30.01/40.58/48.34 | 0.00/0.00/0.00 | 3.48/5.04/6.62 | 70.29/86.81/104.86 |
| B5 | 32.04/40.29/50.09 | 29.10/38.62/47.39 | 0.00/0.00/0.00 | 3.46/4.89/6.48 | 69.05/85.36/103.60 |
| B6 | 32.38/42.63/55.61 | 61.82/83.74/107.98 | 2.99/5.76/7.68 | 3.49/5.33/7.10 | 112.58/145.74/184.98 |
| B7 | 31.70/40.36/51.05 | 57.31/71.67/89.36 | 2.90/5.28/6.58 | 3.44/4.98/6.71 | 106.11/130.55/160.63 |

The 100 ms figure is descriptive only, not a hard requirement. B6 has a total
p95 of 145.74 ms on this CPU contract, so a paper must report the complete
latency distribution and the safety--runtime trade-off.

## Paired gate results

| Gate | Result | Evidence |
| --- | --- | --- |
| ID non-inferiority for isolated B1 QDR | **FAIL** | Safe-capture delta `-10.00 pp [-23.33,+1.69]`; collision delta `+10.00 pp [-3.33,+23.33]`. |
| Stress safety for B1 QDR | **PASS** | On delay `>=6` or noise `>=0.08`, collision delta `-35.71 pp [-50.00,-21.43]`. |
| Liveness | **PASS** | B1 timeout delta `0.00 pp [-1.11,+1.11]`; mean maximum exhaustion streak `12.25 <= 24`. |
| Asynchronous efficiency | **FAIL** | B5 total p95 is `83.92 ms` versus B4 `88.01 ms`, only `4.64%` reduction; safe-capture delta CI lower is `-15.00 pp`. |
| Promotion | **FAIL** | The pre-registered promotion rule requires all primary gates. |

The result is not a reason to retune on the locked split. It means that this
development block supports a conditional stress-safety observation, while the
isolated QDR ID claim and asynchronous efficiency claim are not yet promoted.
B6 should be treated as an interaction diagnostic until a new calibration study
pre-registers a causal decomposition.

## Interpretation and safety boundary

QDR's deterministic time-index checker is an independent implementation audit:
queue lengths `0/2/4/8` have zero position/velocity equivalence error and no
double-delay index. This validates the stated queue composition contract; it is
not a closed-loop safety certificate.

`local_cbf` is an empirical action filter. It is not an R-CLBF-QP certificate,
does not establish forward invariance, and does not constitute a real-flight
safety proof. Tube radii are empirical constraint tightening, not a probability
coverage guarantee.

## Next experimental decision

Because the pre-registered development gate failed, the correct next step is a
new calibration-only experiment that independently varies QDR and asynchronous
scheduling (for example B0/B1/B4/B5/B6 with identical update budgets), adds the
remaining delay/noise and communication cells, and freezes a fresh confirmation
manifest before any locked-diagnostic run. The current Phase 56 locked split
must remain untouched.
