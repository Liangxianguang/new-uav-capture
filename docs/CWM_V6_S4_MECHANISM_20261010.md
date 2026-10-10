# S4: replicated response signal, not a validated new model

Date: 2026-10-10. Status: mechanism signal replicated; new model not trained;
enhancement disabled; original DN-MPC + local empirical CBF preserved.

## Scope and frozen contracts

The user authorized adding new scenes. We use the repository's existing
S4 `adaptive_branching` target as a separate mechanism benchmark. It chooses
between upper/lower exits using defender geometry and projected velocity;
branch labels/scores remain absent from public observation. This strategy is
different from the original Level benchmark. No Level holdout was used.

Original GRU seed745101 weights, 252-dimensional encoding, distributed-delayed
DN-MPC (8 steps, four iterations, planner delay2) and local CBF margin0.35m
are unchanged. Environment route-validation safety margin is separately
0.75m, drone radius0.25m. These two margins must not be conflated. No claim
of a proved strict CBF certificate is justified by this experiment.

The legacy S4 random wall half-y range [3.85,4.35] sometimes failed the
preserved environment's exit clearance check. Before collecting any paired
data, its upper limit was reduced to4.20m: exit5.30m minus radius0.25m and
route margin0.75m leaves at least0.10m extra clearance. This changes only
new-scene geometry, not safety or target branch rules. All16 generated scenes
pass original conservative-route validation for both exits and participants.

## Experiments and results

Scout uses episode seeds961010-961017 and layout seeds1961010-1961013.
Independent confirmation uses962010-962017 and1962010-1962013. Each has
four groups with two mirrored members, crossed speeds0.55/0.75 and nominal/
delayed-noisy sensing. No scene response is used to select snapshot times
(0,1,2,4,8,16) or change the eight command probes/gates between stages.

| Eight-step response metric | Scout | Fresh-group confirmation |
| --- | ---: | ---: |
| Recorded state windows | 40 | 41 |
| Valid paired target time points | 2,183 | 2,186 |
| Points with displacement difference >5cm | 46.91% | 48.67% |
| Mean paired displacement difference | 0.2255m | 0.2327m |
| Exit-flip pairs / both-committed support | 104/280 | 98/287 |
| Exit-flip fraction | 37.14% | 34.15% |
| Independent groups supporting response and flips | 4/4 | 4/4 |
| Complete eight-step pairs | 251 | 251 |
| Original-controller safe captures | 4/8 | 3/8 |
| Original episodes invalid due to target-wall violation | 4/8 | 5/8 |

Both meet the predeclared mechanism screen: at least5% effective time
points,5% exit-flip pairs and signal in at least two independent groups.
Exit flips are counted only when both branches committed and both target
points are valid. Invalid/post-terminal points are not padded into signal.
Mirror members and repeated points are correlated; percentages are
descriptive, not independent-trial confidence claims.

## Interpretation and next training task

Unlike the prior Level response diagnostics, S4 contains substantial
short-horizon action-dependent signal under unchanged target rules. This
supports studying an action-conditioned model on S4; it does not prove a
learned causal model, identifiability from passive logs, information value
under the original MPC objective, cross-scene generalization or better capture.

Target validity is a separate unresolved problem. Route existence and wall
clearance do not guarantee execution by the legacy target policy under the
preserved dynamics. For example, scout episode961014 ends at approximately
(-0.434,0.819,6.047)m inside its wall's x/y footprint while below its top.
It is explicitly target-invalid, not a defender loss or ordinary timeout.
No synthetic success rate should silently exclude those episodes.

Next task: establish a valid new-scene suite through documented geometry/
operating-condition compatibility checks with the same safety/target rules,
then fresh-seed validation. Do not repair the target by private-label inputs,
post-hoc trajectory edits, lowered margins or forced branch decisions. Keep
current results as exploratory data, not a benchmark acceptance claim.

Only after that qualification: freeze group-separated train/validation/test
protocols (mirrors stay together); train matched ordinary action-conditioned
and structured response models plus a zero-response control using public
history/proposed actions; retain the original GRU as frozen anchor. Test
whether better paired response prediction changes frozen DN-MPC scoring
before enabling closed-loop enhancement, then verify fallback and separately
test transfer to original Levels. No enhanced model has been enabled here.

## Evidence, debugging history and reproducibility

Each branch repeat is byte-exact and each snapshot leaves the parent unchanged.
All16 original episodes reproduce target/defender array bytes and seven
physical outcomes with versus without the observer. This demonstrates
measurement isolation, not enhanced-policy safety or full paired-dataset
independent rerun equality. Confirmation uses fresh groups, not identical seeds.

The first scout setup failed before data collection (exit clearance). The
first confirmation setup failed before data collection because a relative
protocol path was compared to an absolute source root. Both failed directories
are preserved locally; the successful `_v2` runs are the released evidence.
The collector now resolves explicit protocol paths and snapshots its sources.
The pre-fix scout source was copied and checked against its recorded digest
before modifying the collector. Both versions are archived as actually run.

Archive: `experiments/cwm_v6/artifacts/s4_mechanism_20261010.zip`.
Size:2,485,859 bytes. SHA256:
`2d4b0cd57b7e7b016ce7557dab39822e621b831db96d01b3b05e1c7bd247ac3a`.
It includes actual data, scenes, masks, outcomes, all observed/unobserved
trajectories and per-run exact source hashes. The verifier recomputes actual
statistics/masks, group separation, snapshot population, and trajectory array
equality instead of trusting booleans. The original capsule hash remains
`c83a27a4360ad055b69dbced4c28370cb31a32ae89ea9fbab13523fa6218a329`.

Commands are in `experiments/cwm_v6/README.md`. Baseline full historical
256/256 exact replay remains the previously published, local CPU/runtime
evidence; this stage does not extend that guarantee to arbitrary hosts.
49 focused V1-V6 tests pass, including generated-scene validation and negative
release tests for forged statistics, changed trajectories with updated hashes,
and changed terminal masks with an updated dataset digest. Original baseline,
V1 pilot and interaction archives were independently rechecked unchanged.
