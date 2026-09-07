# Phase 5 Execution-Perturbation and Multi-Step Audit

> Date: 2026-09-07
> Run: `results/phase5_execution_perturbation_full_v4/`
> Decision: **execution-robustness No-Go for the current robust CBF-QP**

## 1. Scope

This audit tests the velocity-level safety filter when the commanded velocity
is delayed, noisy, tracked by a first-order execution model, and subject to
bounded acceleration and drag. It is an extension of the conditional P5
validation in `docs/PHASE5_STRATIFIED_VALIDATION_REPORT.md`.

The fixed reset seeds are the eight robust-safe seeds selected by the previous
P5 protocol. Both execution variants use the same scene RNG stream. Execution
parameter randomization and command noise use a separate seed stream with
`random_seed_offset = 104729`; this prevents an execution variant from
silently changing the target, obstacle layout, or initial defender state.

The independent checker is run in three ways:

1. on the commanded action before `env.step()`;
2. on the action actually executed by the environment;
3. on the actual post-step defender state after `env.step()`.

The third check is the relevant multi-step audit. It checks the current
post-step state against the contracted robust safe set and is not inferred
from solver status.

## 2. Protocol

| Item | Value |
| --- | --- |
| Episodes | 8 fixed robust-safe resets |
| Seeds | `649118, 649121, 649122, 649129, 649101, 649104, 649107, 649111` |
| Obstacles | 3, mixed environment geometry |
| Target | `flee_persistence`, speed scale `0.45` |
| Horizon | 250 steps, `dt = 0.1 s` |
| Methods | nominal, local CBF, robust CBF-QP |
| Safety margin | `0.35 m + 0.46 m` contracted robust margin |
| Execution RNG offset | `104729` |

Variants:

| Variant | Delay | Command noise | Tracking time constant | Randomized parameters |
| --- | ---: | ---: | ---: | --- |
| `mild_delay_noise` | 1 step | `0.03` | `0.20 s` | no |
| `hard_randomized_execution` | 2 steps | `0.08` | `0.40 s` | speed, acceleration, mass, drag |

## 3. Results

The environment's `collision` field is a safety-termination field and includes
boundary violations. Boundary violations are therefore reported separately;
the table does not reinterpret that field as physical contact alone.

| Variant | Method | Safe capture | Safety termination | Boundary violation | Actual post-step safe rate | Fallbacks | Filter p95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mild | nominal | 87.5% | 12.5% | 0.0% | 84.17% | 0 | 0.002 |
| mild | local CBF | 87.5% | 12.5% | 0.0% | 84.17% | 0 | 8.595 |
| mild | robust CBF-QP | 50.0% | 50.0% | 37.5% | 74.84% | 117 | 6.670 |
| hard | nominal | 100.0% | 0.0% | 0.0% | 87.30% | 0 | 0.002 |
| hard | local CBF | 100.0% | 0.0% | 0.0% | 87.54% | 0 | 8.596 |
| hard | robust CBF-QP | 0.0% | 100.0% | 87.5% | 58.18% | 200 | 7.749 |

The robust CBF-QP command certificate rate was 75.48% in the mild variant and
57.73% in the hard variant. The executed-action certificate rates were 74.09%
and 56.89%, respectively. The mismatch is caused by execution delay,
tracking, and the resulting loss of the contracted margin; it is not a solver
success guarantee. Nominal and local CBF have better capture in this audit,
but their commands are not valid under the robust contracted certificate and
therefore cannot be used as evidence of robust safety.

## 4. Decision

The current robust CBF-QP passes the original frozen velocity-level one-step
gate, but fails the execution-perturbation extension. The current evidence
does **not** support any of the following claims:

- execution-invariant robust safety;
- multi-step forward invariance under the configured delay/noise/tracking
  model;
- R-CLBF-QP safety certification;
- closed-loop or real-flight safety.

The next safety milestone requires a controller/filter contract that models
the delayed command queue and tracking state directly, or a margin derived
from a validated reachable-set bound. Simply increasing the static CBF margin
is not sufficient without rechecking QP feasibility and capture performance.
P6 learned CLBF work remains paused until this boundary is addressed.

## 5. Reproduction artifacts

```powershell
conda run --no-capture-output -n uav-encirclement-gpu python `
  scripts/evaluate_safety_execution.py `
  --config configs/innovation_safety_execution.yaml `
  --output-dir results/phase5_execution_perturbation_full_v4
```

The result directory contains configuration snapshots, source hashes, episode
and step JSONL, summaries, and TensorBoard event files for every variant and
method. The execution RNG isolation is implemented in
`src/encirclement3d/pursuit_env.py`.
