# Phase 30: Escape-Gap-Aware Cooperative MPC Pilot

> Status: development-only diagnostic. The current EGC formulation is **No-Go
> for promotion**. The pilot is retained because it is a reproducible,
> geometry-only negative result that identifies a failure mode in the proposed
> topology objective.

## 1. Motivation and information boundary

This pilot evaluates a low-complexity innovation that can be reproduced
without changing the predictor backbone:

**Escape-Gap-Aware Cooperative MPC (EGC-MPC)** adds a finite soft cost for
large angular gaps in the defender ring and for a target velocity direction
that points into an uncovered gap. The term is computed from the predicted
target candidates and defender rollout geometry. It is not a hard safety
constraint, does not replace local CBF, and is not a reachability proof.

The distributed implementation only uses the latest peer action/position
paths that are actually available. If peer information is incomplete, it
does not fabricate a global state; the team-level score adds the full EGC
term exactly once when the required paths are available. This preserves the
delayed-information boundary and makes the mechanism auditable.

The implementation is in `src/encirclement3d/escape_gap.py`, with planner
integration in `minimax_mpc.py` and `distributed_dn_mpc.py`. The evaluator
logs per-step and per-episode EGC metrics and writes explicit TensorBoard
scalars.

## 2. Frozen development protocol

This is a paired development smoke, not a locked-test experiment:

- scene source: `results/phase27_rnic_fresh_validation_scenes/scenes.jsonl`
- episodes: 20, one checkpoint seed (`727201`), validation-selection mode
- checkpoint: `results/phase16_gru_both_seed727201/checkpoint.pt`
- prediction: one candidate, eight sampling steps, projection iterations 4,
  refresh every control step, CPU
- execution: two-step command delay, zero command noise
- safety: local CBF
- planner branches: centralized `worst_case` and `distributed_delayed`
- scene manifest SHA-256:
  `5aaaa79c6dbef2d346ff57d48d238a34fe911d3534ebb576dff0252ba0593022`

The paired baseline disables EGC and sets its weight to zero. The pilot uses
the same protocol with `weight_escape_gap=0.30`; a separate centralized
weight-1.0 run probes the direction of the trade-off. No parameter was tuned
on the locked test.

## 3. Results

All percentages below are descriptive proportions over 20 episodes. No
confidence interval is claimed for this smoke.

### 3.1 Centralized worst-case branch

| Arm | Safe capture | Collision | Boundary | Timeout | Mean clearance (m) | Mean max gap (rad) | Coverage ratio | Total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EGC off | 85.0% | 15.0% | 0.0% | 0.0% | 0.3928 | 5.4929 | 0.1664 | 65.23 / 71.69 / 83.13 |
| EGC on, weight 0.30 | 90.0% | 10.0% | 0.0% | 0.0% | 0.4233 | 5.4972 | 0.1659 | 65.18 / 71.98 / 78.14 |
| EGC on, weight 1.00 | 75.0% | 25.0% | 0.0% | 0.0% | 0.3557 | 5.3971 | 0.1887 | 67.52 / 74.51 / 78.16 |

The weight-0.30 arm has a descriptive `+5` percentage-point safe-capture
difference and `-5` percentage-point collision difference, but the mechanism
metric does not improve: the mean maximum gap changes from `5.4929` to
`5.4972` rad and coverage changes from `0.1664` to `0.1659`. The weight-1.00
probe reduces the mean maximum gap by only about `1.7%` and increases coverage,
but safe capture falls by `10` percentage points and collision rises by `10`
percentage points. This is a negative safety--performance Pareto signal, not
evidence for promotion.

The component latency quantiles for the three centralized arms are:

| Arm | Predictor p50/p95/p99 | Planner p50/p95/p99 | Safety p50/p95/p99 | Total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| EGC off | 34.51 / 38.32 / 47.16 | 22.56 / 25.39 / 28.96 | 3.77 / 4.43 / 4.97 | 65.23 / 71.69 / 83.13 |
| EGC on, weight 0.30 | 33.83 / 37.13 / 40.66 | 23.41 / 25.87 / 29.46 | 3.71 / 4.20 / 4.76 | 65.18 / 71.98 / 78.14 |
| EGC on, weight 1.00 | 35.14 / 38.55 / 40.88 | 24.11 / 27.19 / 29.46 | 3.81 / 4.51 / 5.09 | 67.52 / 74.51 / 78.16 |

### 3.2 Distributed delayed branch

| Arm | Safe capture | Collision | Boundary | Timeout | Mean clearance (m) | Mean max gap (rad) | Coverage ratio | Total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EGC off | 95.0% | 5.0% | 0.0% | 0.0% | 0.2503 | 5.6531 | 0.1398 | 75.50 / 82.07 / 85.81 |
| EGC on, weight 0.30 | 95.0% | 5.0% | 0.0% | 0.0% | 0.2503 | 5.6531 | 0.1398 | 88.94 / 98.93 / 108.07 |

The distributed arm has no episode-level behavioral change and no measurable
geometry improvement, while planner p95 increases from `36.74` to `52.23 ms`
and total p95 increases from `82.07` to `98.93 ms`. Its total p99 reaches
`108.07 ms`. The user-defined 100 ms reference is not treated as a hard gate,
but the p50/p95/p99 profile shows that the current distributed implementation
does not justify the added computation.

The component latency quantiles are:

| Arm | Predictor p50/p95/p99 | Planner p50/p95/p99 | Safety p50/p95/p99 | Total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| EGC off | 34.22 / 37.04 / 39.10 | 33.30 / 36.74 / 38.30 | 3.74 / 4.14 / 4.35 | 75.50 / 82.07 / 85.81 |
| EGC on, weight 0.30 | 34.42 / 37.93 / 42.36 | 46.49 / 52.23 / 56.80 | 3.80 / 4.34 / 5.41 | 88.94 / 98.93 / 108.07 |

## 4. TensorBoard and artifact audit

Each of the five retained smoke directories contains the copied effective
configuration, source hashes, episode/step JSONL, summary, and TensorBoard
event files. The following scalar tags were present in every checked EGC run:

```text
Summary/EGC/enabled_rate
Summary/EGC/mean_gap_cost
Summary/EGC/mean_max_gap_rad
Summary/EGC/mean_escape_gap_rad
Summary/EGC/mean_coverage_ratio
Summary/EGC/mean_violation_rate
```

The source and configuration snapshots make the comparison reproducible:

```text
configs/phase30_egc_distance_baseline.yaml
configs/phase30_egc_gap_mpc_pilot.yaml
results/phase30_egc_baseline_smoke_seed727201_refresh2/
results/phase30_egc_gap_smoke_seed727201_refresh2/
results/phase30_egc_baseline_dist_smoke_seed727201_refresh2/
results/phase30_egc_gap_dist_smoke_seed727201_refresh2/
results/phase30_egc_weight1_smoke_seed727201/
```

The generated `results/` directories remain Git-ignored by policy; the source
implementation, tests, configs, and this report are versioned.

## 5. Decision and next experiment

**Decision: No-Go for promoting the current EGC formulation as a main
contribution.** The diagnostic is still useful: a large angular gap is
present, but adding its soft cost does not reliably reduce that gap in the
centralized branch and is behavior-neutral but expensive in the distributed
branch. The likely bottleneck is not the absence of a scalar gap penalty; it
is the mismatch between candidate generation, discrete role/topology changes,
and delayed cooperative information.

The next reproducible candidate should therefore change the decision
mechanism rather than scan more EGC weights:

1. implement a **Feasible-Consensus Distributed Barrier Formation (FC-DBF)**
   baseline using a small finite set of certified formation slots;
2. select a slot only when its local predicted arrival and inter-agent
   constraints are feasible, otherwise retain the previous slot;
3. compare off/on on the same validation scenes and seeds, with the same
   predeclared `-2 pp` non-inferiority gate, safety metrics, and latency
   quantiles;
4. only if FC-DBF passes paired confirmation should it be evaluated on one
   untouched OOD axis; no locked-test opening is justified by this EGC smoke.

EGC can be revisited later as a diagnostic feature or as a terminal tie-breaker
after candidate generation and discrete role assignment expose a controllable
topology action. It should not currently be described as a safety certificate,
reachability-normalized objective, or proof of improved interception.
