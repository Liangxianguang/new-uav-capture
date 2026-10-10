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

Later causal-policy evaluation is separately frozen by `closed_loop_protocol.json`:
seven scoring/library controls, fixed final80epoch ADE-median models, freshly
generated geometry shifts on new identities, reserved16episode holdout ONLY after
qualified artifact and passing development closed-loop gates. S4 new geometry
specification comes from prior V16 public axes, not reused scene outcomes; values
are fixed BEFORE evaluation. No new/holdout scene generation is called by this
baseline-only entry. The active paired runner/gates and sealed holdout entry still
need implementation and real verification; listing a protocol is not execution.

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
