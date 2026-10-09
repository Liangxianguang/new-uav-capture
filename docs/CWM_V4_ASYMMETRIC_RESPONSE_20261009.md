# Asymmetric action-response diagnostic on original TRAIN scenes

Date: 2026-10-09. Decision: **data adequacy NO-GO; keep original DN-MPC + local CBF**.
No new model was trained or enabled in this stage. No S4, target-rule changes,
original validation/holdout tuning or planner-horizon change was introduced.

## Question and frozen protocol

Previous six-model anchored residual training did not beat zero response.
This stage tests whether public-geometry single-agent interventions and longer
observation windows expose response missing from the original 8-step horizon.

- Ten unused original TRAIN mirror groups: two per variant, one mirror member.
- Exclude the full original validation pool and all V1/V2 sampled groups.
- Fixed snapshot steps 3, 7, 11, 15, 23, 31; no response/mode-based selection.
- Original planned reference plus left/right tangent bias and brake for each
  of four defenders: 13 branches, 12 intervention/reference pairs per window.
- Tangent bias 3m/s, command speed at most 5m/s; original local CBF and
  execution remain active in every cloned branch.
- Horizons 8/16/24 steps at dt=0.1s. Steps 9-24 hold the final original planned
  command. This is an open-loop probe, NOT a longer DN-MPC plan or rollout of
  the original closed-loop policy.
- Step-keyed common random streams, cloned parent state, deterministic repeated
  reference, terminal censoring. Private mode/route/direction/sensor data are
  diagnostic labels only; commands use public geometry.

The predeclared screen requires at least 5% of valid paired points to exceed
5cm response, supported by at least two independent groups. It is a training
data heuristic, not a significance, promotion, physical necessity or safety
test. All candidates require independent confirmation even if this screen passes.

## Results

54 windows, 648 intervention/reference pairs; 452 pairs have full 24-step
common support. All five variants fail the frozen screen at every horizon.

| Variant | Level | Windows | Max 8-step response (m) | Max 24-step response (m) | 24-step valid points >5cm | Full 24-step pairs | Groups with observed >5cm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed_three_evasive | 5 | 10 | 0.009009 | 0.225532 | 0.9934% | 12 | 1 |
| dense_mixed | 6 | 11 | 0.006995 | 0.697280 | 2.0192% | 117 | 2 |
| fast_target | 7 | 12 | 0.001489 | 0.379570 | 1.5068% | 132 | 2 |
| agile_target | 7 | 11 | 0.011568 | 1.012038 | 2.9595% | 96 | 2 |
| sensor_degraded | 8 | 10 | 0.001191 | 0.271221 | 1.0691% | 95 | 2 |

These maxima describe common-prefix observations, not guaranteed 24-step
endpoints. In Level5 the complete 24-step endpoint maximum is zero: only 12
pairs survive the entire horizon. Larger prefix effects occur in other,
censored pairs. For the other four variants, the endpoint maxima equal the
table's 24-step maxima. Read the detailed endpoint support in the JSON report.

Post-hoc threshold-crossing analysis: 58 pairs ever exceed 5cm, all after the
original 8-step horizon. Earliest crossing is step 13 (1.3s), median step 21
(2.1s). This is observed displacement-threshold latency, NOT identified
physical reaction delay; censored pairs have unknown subsequent responses.
The 648 pairs are not independent trials; only ten scene groups were sampled.

Forty of 54 snapshot modes are boundary recovery. This is consistent with
geometry/feasibility limiting response, but does not identify its contribution
relative to mode holding, observation delay or motion limits.

## What this changes, and what it does not

The original scenes do contain occasional sizeable delayed response under
asymmetric interventions. The claim that target position never responds is
therefore too broad. However, this sample still gives no sufficient signal
for the current short-horizon residual model, no advantage over a matched
ordinary network, and no enhanced closed-loop capture/safety result.

The next research hypothesis should distinguish **whether a response occurs**
from **its delayed mode/trajectory outcome**. Before expanded training, obtain
independent original-TRAIN confirmation with a predeclared data/split protocol,
retain null examples, report the full natural distribution, and compare against
zero response plus matched ordinary action-conditioned models. Do not change
the failed gate after observing this result or treat strong windows as a test.

Any later control proposal must explain how delayed information can improve
candidate action ranking within the unchanged 8-step DN-MPC (for example a
separately validated terminal-value contribution). Simply lengthening the
diagnostic window or adding a fixed forecast is not that integration.
No such objective modification or controller activation is implemented here.

## Preservation and reproducibility

Each of the ten TRAIN scenes was rerun with and without the observer. All
target/defender trajectory shapes, dtypes and raw bytes match; all physical
outcomes match. These ten original-controller runs captured safely, but they
are TRAIN samples, not generalization evidence or an improvement claim.

Complete parent-environment fingerprints and repeated reference branches pass
for all 54 windows. The adapter uses the already-computed original K=1 GRU
prediction with the unchanged 252-feature encoder. The original nine provenance
hashes and old milestone archives were rechecked successfully. The earlier
256/256 historical replay remains separately documented in the baseline report;
this observer test does not broaden it into enhanced-controller safety.

34 focused V1-V4 unit tests pass, including negative release tests for falsified
statistics, resumed post-terminal masks, duplicate invariance coverage, altered
trajectory arrays and Windows source-path normalization.

Archive: `experiments/cwm_v4/artifacts/asymmetric_diagnostic_20261009.zip`.
Bytes: 4,840,579. SHA256:
`bd5f97f5a47ec73422e004cd53e729b8337a80597950ce34de5154fa02eccbef`.
It contains the original TRAIN pool, exclusions, selected scenes, raw paired
data, observed/unobserved trajectories, statistics, protocol and source closure.
Archive verification recomputes statistics/latency, verifies actual trajectory
byte equality, original TRAIN membership, exclusions and every member hash.

Baseline capsule SHA256 remains
`c83a27a4360ad055b69dbced4c28370cb31a32ae89ea9fbab13523fa6218a329`.
Exact numerical replay is scoped to the recorded local CPU/runtime; cross-host
bitwise equality and wall-clock latency are not guaranteed. Original local CBF
remains empirical, not a proved strict safety certificate. Enhancement stays off.

Reproduction commands and archive routing: `experiments/cwm_v4/README.md`.
