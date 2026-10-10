# S4 geometry compatibility qualified; model training preparation

Date:2026-10-10. Separate new-scene mechanism benchmark, not original Level
performance. Original baseline preserved, enhancement off. This report covers
completed geometry qualification, not a claim of newly trained-model utility.

## Root-cause evidence

V6 had nine target-invalid wall collisions across16 original episodes.
Tracing the unchanged target-policy return in two separate runs reproduces all
16 target/defender trajectories and every stored trajectory array byte in
each run. The trace records the attraction to exit waypoints, each defender
repulsion, obstacle repulsion, current velocity and returned target command.
All decomposed sums reproduce the frozen target command within1e-12.

Example961014, immediately before its wall collision: current x=-0.699m,
wall clearance0.109m; exit attraction's x component0.324, defender repulsion
x sum12.799, obstacle repulsion x=-7.319. Combined desired x=5.804 and returned
x velocity command2.653m/s, directed into the wall. These are direction-policy
components, not physical forces. The target also switches waypoint phase by
x position rather than full waypoint arrival. Acceleration limits and strong
defender repulsion make route-existence validation insufficient.

No target code, branch threshold, exit, acceleration, avoidance gain or safety
margin is changed to fix this compatibility issue.

## Geometry-only experiment

Predeclared translations x=0,1.5,3.0m move only the existing S4 wall and its
declared obstacle zone. The seed-matched initial agents and all frozen target
rules are identical. Eight original episodes per candidate and128 fixed
pressure/brake probes per candidate are tested. Stress snapshots0/4, eight
probe types,80-step horizon, original CBF; after eight commands the last planned
command is held. It is not a longer MPC optimizer.

| Candidate wall x | Original target-invalid | Stress target-invalid | Stress complete80-step branches | Stress defender safety terminations |
| --- | ---: | ---: | ---: | ---: |
| 0m | 4/8 | 0/128 | 78/128 | 45/128 |
| 1.5m | 0/8 | 0/128 | 82/128 | 39/128 |
| 3m | 0/8 | 0/128 | 84/128 | 38/128 |

Selection is the smallest translation with zero tested target-invalid cases,
so1.5m is selected. Capture and response signal are not selection criteria.
Development episode seeds963010-963017, layouts1963010-1963013; development
geometry data are never model validation/test.

Fresh qualification freezes1.5m, uses16 episodes, eight groups, episode
seeds964010-964025/layouts1964010-1964017. Zero original target-invalid episodes,
zero target-invalid among256 stress branches.185/256 stress branches observe
the complete80-step horizon;20 terminate at safe capture and51 at defender
safety failure. Thus censored later target behavior remains unobserved, and
the intervention set is not declared safe. Original unchanged controller has
16/16 safe captures here; this says nothing about an enhanced controller.

All original observed/unobserved trajectories are byte-equal and physical
outcomes agree. All tested scene routes pass the original conservative checks.
The release auditor recomputes target-wall and effective-boundary violations
from actual coordinates under the frozen world/target contract, verifies
actual masks and terminal coverage, checks original arrays against plain
replays and traces against V6, and recomputes the geometry selection.
It does not trust a `target_invalid=false` report alone.

## Frozen training task

The geometry is now qualified for finite testing within this distribution,
not universally safe. Training collection is separate, using64 new episodes,
layout seeds1965010-1965041/episode seeds965010-965073. Before seeing results,
first six four-variant mirror cycles (24 groups/48 episodes) are train;
last two (8 groups/16 episodes) are development validation. The reserved
test layouts1966010-1966017/episode seeds966010-966025 remain uncollected.
Mirrored scenes stay together and geometry development/qualification pools
are explicitly excluded.

Two matched predictors (`plain`, graph-pooled `structured`) use only public
8x252 history, belief-relative defender state, proposed/reference commands
and externally supplied frozen GRU trajectories. Branch labels are matched
auxiliary targets, never model inputs. Same three seeds, same batches and
80 fixed epochs; normalizers use train only. Relative graph messages are the
architectural difference and model parameter counts differ by less than5%.

The predeclared development gate requires each structured seed to reduce
group-equal paired response error by at least20% over zero response, and its
median error to beat the matched plain median by at least5%. All models must
be reported, not just the best run; later scoring chooses the median seed
per architecture. Regardless of this gate, enhanced control remains off.
Decision value under original DN-MPC, untouched holdout, optional-module
fallback, latency and closed-loop safety/capture remain later tasks.

Paired interventional learning plus graph pooling is not automatically a
causally identified SCM or innovative performance advantage. Ordinary motion
correction, causal-response prediction and actual control benefit must stay
separate. These results are not Level1-8 generalization evidence.

## Release and checks

Archive `experiments/cwm_v7/artifacts/geometry_qualification_20261010.zip`:
9,074,490 bytes, SHA256
`aacfddf2ee51b6bdc9547cdfb9a518c64f65e7c718e086fb6aa7835ab65ebe5e`.
The original capsule hash remains
`c83a27a4360ad055b69dbced4c28370cb31a32ae89ea9fbab13523fa6218a329`.
Archive contains geometry evidence and exact sources as run, not later-trained
weights. Current training source/protocols are versioned separately in the repo.

The first unpublished geometry implementation attempted mutation of a frozen
obstacle dataclass and exited after completing16 old-scene trace replays but
before new-scene testing. That failed directory is preserved; successful
development uses `_v2` and creates translated dataclasses with `replace`.
No live process was restarted for an observation timeout.

60 focused V1-V7 tests pass, including model input/anchor/graph/mask/split
checks and negative tests of concealed target-invalid episodes and selection.
Complete commands
are in `experiments/cwm_v7/README.md`. This milestone uses git-commit's scoped
Conventional Commit workflow; it does not stage the unrelated publishing helper
or user-owned dirty original checkout.
