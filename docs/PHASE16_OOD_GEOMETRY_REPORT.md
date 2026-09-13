# Phase 16: OOD Geometry-Shift Closed-Loop Evaluation

> Status: completed diagnostic. This is one fresh geometry-shift axis, not a
> full robustness claim and not a new locked-test selection result.

## 1. Purpose

The evaluation tests whether the selected GRU `both` predictor and the current
local-CBF closed loop transfer to obstacle geometries outside the training
range. Only the wall geometry is shifted; the predictor checkpoint, sampling
protocol, planner, safety layer, and scene pairing are frozen. This provides a
reproducible robustness result without adding a new learned component.

## 2. Frozen protocol

- scene file: `results/phase16_ood_geometry_shift/scenes.jsonl`
- episodes: 100, consisting of 50 mirror groups; 50 upper and 50 lower scenes
- scene manifest SHA-256:
  `47c393d1d678f25aad14c978734aae5d89105d826ea3e7b32b16a46cc1654501`
- predictor: GRU `both`, checkpoint seeds `727201/727202/727203`
- prediction: one candidate, eight sampling steps, sampling seed `745102`
- planner: distributed delayed DN-MPC or centralized worst-case
- execution: local CBF, CPU, projection iterations 4, refresh every step
- decision label: `ood_diagnostic`
- QDR, RNIC and escape-gap additions disabled

The geometry ranges are deliberately disjoint from the training range:

| Parameter | Training range | OOD range |
| --- | ---: | ---: |
| wall half-x (m) | 0.45--0.65 | 0.70--0.85 |
| wall half-y (m) | 3.85--4.35 | 4.38--4.42 |
| wall height (m) | 9.25--9.55 | 9.60--9.80 |

The OOD generator validates each route before writing the manifest and keeps
mirror groups intact. No OOD result was used to tune a checkpoint or a planner
parameter.

## 3. Aggregate results

Values are means over three predictor seeds and 100 episodes per seed. The
intervals are hierarchical bootstrap 95% CIs over matched seed/episode units.

| Branch | Safe capture | Collision | Boundary | Timeout | Mean capture time (s) | Mean min clearance (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Centralized worst-case | 46.33% [39.67, 53.33] | 1.67% | 1.67% | 52.00% [45.00, 58.67] | 3.580 | 0.368 |
| Distributed delayed | 84.00% [79.33, 88.33] | 0.00% | 0.00% | 16.00% [11.67, 20.67] | 1.633 | 0.471 |

The main failure mode in this geometry block is timeout, especially for the
centralized worst-case branch. Distributed delayed control remains safer than
the centralized worst-case branch in this protocol, but its safe-capture rate
is substantially below the in-distribution locked-test reference. The result
therefore exposes a geometry-transfer boundary; it does not demonstrate that
the model is robust to arbitrary geometry shifts.

## 4. Latency

All values are predictor/planner/safety/total p50/p95/p99 in milliseconds.

| Branch | Predictor | Planner | Safety | Total |
| --- | ---: | ---: | ---: | ---: |
| Centralized worst-case | 33.61 / 42.74 / 55.47 | 20.06 / 25.75 / 33.18 | 3.79 / 5.41 / 6.94 | 61.77 / 77.13 / 97.40 |
| Distributed delayed | 33.06 / 43.07 / 52.01 | 31.61 / 43.25 / 53.38 | 3.63 / 5.48 / 7.04 | 72.93 / 93.60 / 111.26 |

The 100 ms reference remains descriptive rather than a hard gate. The
distributed p99 exceeds it, so future robustness work must report the full
tail and should not claim deployment readiness from p50 alone.

## 5. Reproducibility and TensorBoard

The three run directories contain effective configurations, source hashes,
episode/step JSONL, and summaries:

```text
results/phase16_ood_geometry_gru_seed727201_local_refresh1/
results/phase16_ood_geometry_gru_seed727202_local_refresh1/
results/phase16_ood_geometry_gru_seed727203_local_refresh1/
results/phase16_ood_geometry_gru_aggregate.json
```

The original OOD evaluation had complete JSON summaries, but its writer was
interrupted before event files were materialized. The retained results were
backfilled from the immutable method summaries with:

```powershell
python scripts/backfill_closed_loop_tensorboard.py `
  results/phase16_ood_geometry_gru_seed727201_local_refresh1 `
  results/phase16_ood_geometry_gru_seed727202_local_refresh1 `
  results/phase16_ood_geometry_gru_seed727203_local_refresh1
```

The generated event files contain outcome metrics and predictor/planner/safety/
total p50/p95/p99 latency tags. This is a reporting backfill, not a new
experiment or a tuning step; source/config/scene hashes remain those recorded
by the original runs. The aggregate was produced with 10,000 bootstrap samples
and seed `20260911`.

## 6. Decision and next steps

**Decision: diagnostic pass, no promotion gate.** The block is useful because
it is a clean, one-factor geometry transfer test, but it does not justify a
new robustness claim or opening another locked-test comparison.

The most reproducible next innovations remain:

1. Freshness-covariance weighted public-belief fusion, first as a deterministic
   planner-side input transformation;
2. failure-conditioned replay curriculum, using only development failures and
   mirror-preserving one-factor counterfactuals;
3. if a topology mechanism is revisited, a discrete communication-aware slot
   surrogate rather than another soft geometric penalty.

Each candidate should first pass a 20-scene development smoke, then a fresh
three-seed mirror-group confirmation with a pre-registered non-inferiority
gate. The current FC-DBF and EGC formulations should remain frozen negative
results.
