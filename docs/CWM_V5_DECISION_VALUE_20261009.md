# Offline decision value of response information

Date: 2026-10-09. Status: **diagnostic complete, keep baseline, enhancement off**.
This stage does not override V4's failed data screen, train a new model,
use original validation/holdout, add S4, change target rules, extend the
planner horizon, alter costs, or execute oracle/learned enhanced actions.

## Why this diagnostic

V4 exposed occasional sizeable response only after the original 8-step
planning horizon. Before investing in another short-horizon response model,
ask whether exact response information could change offline probe selection
under the original objective. Separate this from ordinary target-motion
prediction errors, which are not evidence of causal benefit.

The same 54 exploratory TRAIN windows from ten scene groups are reused.
Thirteen predeclared public-geometry command probes are scored using copies
of the original planner's post-plan state, including delayed peer messages.
The complete frozen `_team_scenario_costs` and original risk aggregation are
used, with no replacement distance-only objective. For the reference probe,
the score matches the original planner's diagnostic objective at every
rankable window. All costs use the original 8-step proposed-command rollout.

Four target-information conditions:

1. Original frozen GRU trajectory, shared across all probes.
2. That trajectory plus actual paired target response relative to the
   reference command. This isolates action-response information while
   retaining the common GRU motion-prediction error.
3. The reference command's actual target trajectory, shared across probes.
4. Each probe's own actual target trajectory.

Oracle truth/corrections are `raw` diagnostic containers, not certified
dynamics-projected forecasts. They are never passed into online `plan` or
execution. These conditions are upper-information references, **not formal
bounds on learned-controller performance**.

All thirteen probes must have complete valid 8-step target support. Seven
windows fail this joint requirement and are excluded from every condition;
47 remain. No scores are padded past termination.

## Results

| Information condition | Choices differing from original GRU | Tie-optimal choices under action-specific-truth diagnostic cost | Mean diagnostic-cost regret |
| --- | ---: | ---: | ---: |
| Original GRU | 0/47 | 41/47 | 6.271137 |
| GRU + exact paired response | 0/47 | 41/47 | 6.271137 |
| Fixed reference truth | 6/47 | 47/47 | 0 |
| Action-specific truth | 6/47 | 47/47 | 0 |

"Regret" is the chosen probe's weighted diagnostic cost minus the minimum
among these thirteen probes, evaluated with action-specific target truth.
It is not metres, physical capture time, realized closed-loop return or a
collision/safety metric. Optimality is only within this probe set and cost.

Motion-truth choice changes by variant: Level5 mixed 1/8, Level6 dense 2/11,
Level7 fast 1/11, Level7 agile 2/9, Level8 sensor 0/8. Adding exact response
changes no choice, either around the original GRU or around corrected motion
truth. The matched group-equal response regret reduction is exactly zero on
this sample. Descriptive scene-group bootstrap intervals are [0,0]; they do
not constitute fresh statistical validation or universal zero-value proof.

## Interpretation and next scope choice

In these archived windows/probes, even exact response information does not
improve short-horizon team diagnostic selection. Ordinary motion prediction
has visible selection sensitivity instead. Therefore do not claim another
short-horizon residual network is valuable merely because its trajectory ADE
changes, and do not label autonomous motion calibration a causal improvement.

This result does **not** cover the optimizer's full local candidate library,
all sequential best responses, unseen scene groups, longer horizons or actual
closed-loop policy improvement. The score rolls out proposed commands; future
CBF filtering/execution may differ from that rollout. A learned imperfect
model can behave differently from these information probes. No formal
impossibility or upper-bound claim is justified.

The next model-training phase needs a concrete scope decision:

- Add the existing S4 two-exit mechanism benchmark separately, with unchanged
  original baseline and explicit separation from Level1-8 performance. Its
  `adaptive_branching` adversary differs from the current target strategy;
  user authorization is still required before collecting/training there.
- Remain in original Level scenarios and first independently validate a
  delayed-response/terminal-value hypothesis. This changes the enhanced
  decision formulation and cannot be silently equated with the original
  8-step residual design or promoted using the already-inspected V4 data.

Neither option was launched here. Original DN-MPC + local CBF remains supported.

## Reproduction and preservation

Two independent process replays each cover all ten original TRAIN episodes.
Every target/defender trajectory shape, dtype and raw byte matches V4, and
physical outcomes match. The two runs' complete window score/metric JSON bytes,
statistics, model/source hashes and episode evidence also match. Complete
parent environment and planner fingerprints pass around every offline score.
This proves read-only isolation, not enhanced-controller safety.

41 focused V1-V5 unit tests pass. New negative tests reject falsified rankings,
rankable booleans inconsistent with actual terminal masks, and altered
trajectories despite matching report booleans/new hashes. Test module naming
and explicit import loading prevent V4/V5 `release.py` collisions. The first
unpublished draft archive was retained locally, then rebuilt after the test
entry fix; it is not the authoritative release.

Archive: `experiments/cwm_v5/artifacts/decision_value_20261009.zip`.
Bytes: 181,358. SHA256:
`b9ffa783f9bad9492d3cf9ae58e6754842d7c3b76fcffc83e9feae2ddaf538e8`.
It contains both replays, all probe scores/metrics, actual trajectory files,
protocol and source closure. It references the already-released immutable V4
archive and baseline capsule. The release verifier checks member hashes,
recomputes ranking/aggregate statistics, cross-checks actual V4 terminal masks,
compares trajectories and requires exact replay agreement.

Commands: `experiments/cwm_v5/README.md`. Exact replay remains scoped to the
recorded local CPU/runtime, not arbitrary hosts or deterministic latency.
Original local CBF remains empirical rather than a proved strict certificate.
