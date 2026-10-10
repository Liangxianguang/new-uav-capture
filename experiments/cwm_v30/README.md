# V30: complete original Levels and fixed later closed-loop evaluation

This protocol does NOT demonstrate causal-model efficacy or activate a model.
`baseline_full_levels.py` evaluates ONLY the original controller on ALL1280 frozen
development records: Level1–8 including BOTH Level7 variants, plus separate L0
warmup. No easier sample, scene filtering, singleton dropping, training, CWM
construction or reserved holdout access. Physical obstacles, original GRU745101/
252 encoding, DN-MPC/message/timeout, margin.35, speed5, horizon8/dt.1 and original
250step limit are unchanged. All256 historically archived L5/6 records must also
reproduce every numeric trajectory field and physical outcome. Commands/plans,
all native step diagnostics, absolute/group-level rates and capped capture times
are retained. Infinite/absent legacy diagnostics retain explicit tagged JSON,
never fake zero; numerical trajectories are unchanged. Native timing is observed,
not a passed real-time contract. Failures/invalids/timeouts stay in the population.

Protocol/source must be committed AND exactly pushed BEFORE a new baseline run.
Use short absolute new output paths on Windows. Partial progress never qualifies
full population, models, safety or latency. A terminal full summary still requires
independent ALL-population replay and complete reproducible archive publication.

```powershell
python -m pytest experiments/cwm_v30 -q
python experiments/cwm_v30/baseline_full_levels.py --output D:\uav-capture\cwm30-baseline-20261011
```

`audit_full_baseline.py` requires the COMPLETE baseline before any new replay
output. In a fresh process it reexecutes ALL1280 episodes in EACH off/refusal
mode using the exact original parent planner/runtime, with no optional model or
context reads. Every numeric trajectory/plan/CBF command and every non-timing
native episode/step diagnostic must match; all source/protocol/population/hash
and group-aware summary support is rechecked before and after execution. Only
six explicitly named native timing fields per episode/step are nondeterministic;
they remain saved, not silently dropped or latency-qualified. Fixture tests are
NOT actual full replays. Complete result packaging/archive replay and active
model faults remain separate requirements.

The independent audit scans EVERY step in streaming order, stores only per-episode
byte offsets/counts, then decodes all steps of the episode being replayed. File
digests use bounded1MiB chunks. No step/episode subsampling or support reduction;
reordered/repeated/interleaved blocks are rejected. Native non-timing JSON scalar
types/signs are exact too (True cannot impersonate1, nor +0.0 become-0.0).

```powershell
python experiments/cwm_v30/audit_full_baseline.py --reference D:\uav-capture\cwm30-baseline-20261011 --output D:\uav-capture\cwm30-audit-20261011
```

Later causal-policy evaluation is separately frozen by `closed_loop_protocol.json`:
seven scoring/library controls, fixed final80epoch ADE-median models, freshly
generated geometry shifts on new identities, reserved16episode holdout ONLY after
qualified artifact and passing development closed-loop gates. S4 new geometry
specification comes from prior V16 public axes, not reused scene outcomes; values
are fixed BEFORE evaluation. No new/holdout scene generation is called by this
baseline-only entry. `paired_closed_loop.py` and `closed_loop_gates.py` now
implement the fixed complete development execution/statistical entry, but active
execution remains unperformed/unqualified. The sealed holdout entry and all real
active/fault/archive evidence still need implementation/verification; listing
code or a protocol is not execution.

The primary outcome is restricted SAFE capture time: failures/unsafe/invalids
cost the original25s horizon, so success-only timing cannot hide regressions.
Require strict positive paired lower95 time gain vs original, own motion and L2;
per-level/variant safe-capture noninferiority and zero additional paired safety/
local-failure outcomes, and fixed end-to-end p95 latency<=100ms. Original8Level
aggregate gives each Level equal weight, Level7 variants equal weight and existing
mirror groups equal weight; L0 excluded. New geometry axes similarly use shared
group-aware aggregation, never windows as independent samples. All absolute rates
still reported. These gates do NOT alter V28's failed/passing qualification or
baseline safety parameters and are not a formal guarantee. If any required gate
fails, enhanced deployment remains off and original DN-MPC+CBF remains usable.

Closed-loop policy states/candidate libraries naturally diverge after different
actions; identical generation algorithms/initial seeds do not make future noise
or later libraries identical. Exact same-library response contribution remains
the V28 all-call offline comparison. Model-fault equivalence, full result archive
and independently replayed active/holdout safety/capture/latency evidence remain
required, not fulfilled by protocol/population tests or baseline capture rates.

The paired DEVELOPMENT entry requires the full independently replayed V28 fixed
primary release, exact published source HEAD, all7 policies on ALL1280 original
and ALL64 newly generated scenes. A failed primary refuses before new generation
or active entry; route/call errors retain evidence and STOP, never filter/replace
scenes, choose another model or silently claim CWM while using original control.
There is deliberately NO holdout CLI. Saved-outcome statistics always report
holdout authorization/native replay/deployment as false; an arithmetic pass is
not a native result. Fresh-axis bootstrap resamples the same layout groups across
axes; original Levels/variants/groups are equal weighted, with L0 diagnostic.

`measured_original_cycle.py` only observes the actual original evaluator's
`control_started` timestamp at its environment-step boundary. No clock/deadline
replacement, subtraction or changed physical configuration. End-to-end samples
include inference/proposals/original costs/CBF, V29 call-output IO, numeric command
observer and native environment step/history; first and warmup samples retained.
This is a conservative full simulation-cycle measurement, not inference-only
timing; preload and post-episode artifact writes are separately outside the
control cycle. A fixed two-record ORIGINAL-only smoke is provided; even a real
successful smoke does not qualify full-Level or active latency/model faults.

The V30 evidence adapter adds the already-public ordinal to V29's copied call
before its unchanged serializer runs. V29 released source bytes are preserved.

```powershell
python experiments/cwm_v30/measurement_smoke.py --output D:\uav-capture\cwm30-cycle-smoke-20261011
# Only after V28 artifact-qualified fixed primary; NOT authorized/running now:
python experiments/cwm_v30/paired_closed_loop.py --artifact VERIFIED_V28_ARTIFACT --certificate VERIFIED_V28_CERTIFICATE --output NEW_SHORT_OUTPUT
```

`package_full_baseline.py` requires the complete original baseline AND complete
off/refusal audit, inventories ALL trajectories/plans/CBF and native diagnostics,
includes the pinned original capsule/source snapshots, and checks every ZIP CRC/
length/hash/path. It then runs BOTH ALL1280 modes again in a fresh subprocess
against extracted archive evidence; only real success issues a certificate. No
arbitrary extracted Python execution. Optional45MiB lossless parts are created
only afterward. Use a short new replay path; matching Git history and original
Python runtime are required (not bare-ZIP or cross-platform proof). Packaging
has NOT actually run while the prerequisite full audit is incomplete.

```powershell
python experiments/cwm_v30/package_full_baseline.py --baseline D:\uav-capture\cwm30-baseline-20261011 --audit D:\uav-capture\cwm30-audit-20261011 --artifact NEW_FULL_ARCHIVE --replay-output D:\uav-capture\cwm30-archive-replay-20261011 --parts-output NEW_PARTS_MANIFEST
```
