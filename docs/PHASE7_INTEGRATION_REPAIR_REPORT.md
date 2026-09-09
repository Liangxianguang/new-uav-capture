# Phase 7 Integration Repair Report

> Version: v1.1 (2026-09-09)
> Scope: frozen 100-episode unseen adaptive-target locked test

## 1. Outcome

The complete velocity-level composition is now reproducible on the frozen
Phase 4 scene file:

```text
projected portable-SSM diffusion candidates
        -> worst-case DN-MPC
        -> robust CBF-QP safety filter
```

The repaired run achieved:

| Metric | Original P7 | Matched-margin diagnostic | Contract-repaired P7 |
| --- | ---: | ---: | ---: |
| Safe capture | 50% | 30% | **99%** |
| Collision | 49% | 70% | **0%** |
| Boundary violation | 16% | 3% | **0%** |
| Timeout | 1% | 0% | 1% |
| Planner valid/effective | 100% / 100% | 100% / 100% | **100% / 100%** |
| Independent certificate valid | not reported as valid | 93.61% | **100%** |
| Safety-filter certificate valid | 64.87% | 86.73% | **99.98%** |
| Total-control p95 | 114.80 ms | 46.95 ms | **41.57 ms** |

The final run is a velocity-level matched-contract result, not a proof of
execution-invariant safety. It uses the P4 reset/scenario contract with the
base 0.35 m safety margin and disables unmodelled execution, delay,
observation-error, and disturbance margins. The 100 ms value is retained as a
reference metric only, not a hard success gate.

## 2. What was fixed

### 2.1 QP speed contract

The old QP used an inscribed component box of
`[-v_max/sqrt(3), v_max/sqrt(3)]` while the simulator and independent checker
used the Euclidean constraint `||v|| <= v_max`. This created artificial
infeasibility for legal axis-aligned commands. The filter now uses broad
component bounds plus an explicit differentiable Euclidean speed-ball
constraint. Because that exact speed ball is quadratic, the repaired online
projection is technically a small convex QCQP rather than a pure QP; the
existing `RobustCBFQPFilter` class name is retained for API compatibility.

### 2.2 Explicit optional command-increment contract

`MinimaxMPCConfig.action_change_limit_mps` is now optional. When enabled, the
centralized finite-shooting candidates are projected onto the same speed and
command-increment contract as the safety layer. This is useful for an
execution-dynamics experiment, but is disabled in the ideal velocity-level
P7 run because the environment does not execute acceleration-limited commands.

`RobustCBFQPConfig.enforce_action_change` is also explicit. The final P7
configuration sets it to `false`; execution-perturbation studies must use a
separately generated robust-safe reset protocol and the Phase 5 execution
checker.

### 2.3 Fallback interpretation

The final run had one SLSQP fallback at episode 79, step 8. The fallback action
passed the independent safety certificate. That episode timed out and did not
collide or violate the boundary. This is an engineering robustness event, not
a silent safety violation.

## 3. Reproduction

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate_minimax_mpc_s3.py `
  --protocol configs/phase4_unseen_adaptive_target.yaml `
  --split locked_test `
  --mpc-config configs/phase4_dn_mpc.yaml `
  --scene-records results/phase4_unseen_adaptive_locked_seed745101_gpu_v2/scenes.jsonl `
  --checkpoint results/phase2_formal_diffusion_seed745101/checkpoint.pt `
  --candidate-source checkpoint `
  --methods worst_case `
  --device cpu `
  --safety-layer robust_cbf_qp `
  --safety-config configs/phase7_matched_margin_safety.yaml `
  --prediction-refresh-interval-steps 20 `
  --output-dir results/phase7_joint_velocity_level_locked_seed745101_v4
```

The machine-readable result is in
`results/phase7_joint_velocity_level_locked_seed745101_v4/worst_case/summary.json`.

## 4. Remaining boundary

This result supports the claim that the three modules can be composed under a
consistent velocity-level state/action contract. It does not close the
following gaps:

- Phase 5 multi-step execution-perturbation audits still fail their current
  recoverability/capture gates.
- The robust margins in `configs/innovation_safety.yaml` are not compatible
  with the original P4 reset distribution; a new robust-safe reset protocol is
  required before enabling them.
- The diffusion checkpoint backend is `portable_diagonal_ssm`, not the
  official `mamba_ssm` CUDA implementation.
- One seed and one locked scene file are evidence for this integration audit,
  not a general deployment guarantee.
