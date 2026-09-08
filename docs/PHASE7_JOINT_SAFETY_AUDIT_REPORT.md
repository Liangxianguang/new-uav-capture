# Phase 7 Joint Prediction--DN-MPC--Safety Audit

> Version: v1.0 (2026-09-08)
> Decision: `No-Go for the integrated robust-CBF-QP path; retain modular P4 result`
> Scope: velocity-level robust CBF-QP after frozen P4 worst-case DN-MPC

## 1. Executive conclusion

The joint evaluator now supports a selectable `local_cbf` or
`robust_cbf_qp` safety layer. It records safety solver status, certificate
validity, fallback, barrier values, constraint violation and safety latency in
the episode/step JSONL and TensorBoard artifacts.

The formal audit reused the P4 locked protocol: one shared 100-episode
`adaptive_adversarial` scene file, checkpoint `745101`, `phase4_dn_mpc.yaml`,
and a 20-step prediction refresh interval. The existing local-CBF run on the
same scene file is the comparison baseline.

The integrated robust-CBF-QP path is a clear No-Go under this reset and
geometry contract. It reduces safe capture from `97%` to `50%`, increases
collision from `1%` to `49%`, and introduces `16%` boundary violations. The
planner itself remains valid at `100%`; the failure is introduced by the
robust filter and its fallback behavior, not by the DN-MPC solver.

This is a velocity-level integration audit only. The environment execution
model is disabled in this P4 run, so the result neither passes nor replaces
the separate Phase 5 execution-perturbation No-Go audit.

## 2. Frozen protocol

| Item | Value |
| --- | --- |
| Protocol | `configs/phase4_unseen_adaptive_target.yaml` |
| MPC config | `configs/phase4_dn_mpc.yaml` |
| Split | `locked_test` |
| Target policy | `adaptive_adversarial` |
| Episodes | 100 |
| Checkpoint | `results/phase2_formal_diffusion_seed745101/checkpoint.pt` |
| Prediction refresh | 20 control steps |
| Safety layer | velocity-level `robust_cbf_qp` |
| Scene file | `results/phase7_joint_safety_locked_seed745101_robust_p4protocol/scenes.jsonl` |
| Scene SHA-256 | `22c7a825cb11f0642b8c4f47671856e754982cea818619e110740b50bb21b13f` |

The scene hash is identical to the published P4 scene file. The run retains
root and method configuration snapshots, protocol, episode/step JSONL,
summary JSON, TensorBoard events and source hashes.

## 3. Results

| Metric | P4 local CBF baseline | Joint robust CBF-QP | Change |
| --- | ---: | ---: | ---: |
| Safe capture | 97.00% | 50.00% | -47.00 pp |
| Collision | 1.00% | 49.00% | +48.00 pp |
| Boundary violation | 0.00% | 16.00% | +16.00 pp |
| Timeout | 2.00% | 1.00% | -1.00 pp |
| Planner solver success | 100.00% | 100.00% | unchanged |
| Safety certificate valid | n/a | 64.87% | diagnostic |
| Safety fallback | n/a | 35.13% | diagnostic |
| Mean minimum clearance | 0.309 m | 0.228 m | -0.081 m |
| Worst minimum clearance | -0.039 m | -0.231 m | -0.192 m |
| Safety p95 | 9.55 ms | 10.57 ms | +1.02 ms |
| Total-control p95 | 236.47 ms | 114.80 ms | -121.67 ms |

The robust filter reported a minimum barrier of `-1.052 m` and a maximum
constraint violation of `0.519 m`. These values explain why the one-step
certificate rate is only `64.87%`: the P4 locked-test reset distribution was
not selected to satisfy the larger robust margin used by the Phase 5 safety
configuration. A fallback action without a valid certificate cannot be
counted as safe.

## 4. Interpretation and boundary

The result rejects the current direct composition of P4 planner outputs with
the Phase 5 robust filter. It does not reject the prediction or DN-MPC
modules independently: the same locked-test baseline already demonstrated
their modular planning value. It also does not establish that a different
reset protocol, margin contract, or safety architecture could not work.

The correct research claim is therefore:

```text
projected diffusion + DN-MPC: conditional modular planning evidence
robust CBF-QP: conditional one-step safety filter
direct end-to-end composition on the frozen P4 reset: No-Go
execution-invariant safety and R-CLBF-QP: unproven / No-Go
```

The 100 ms value remains a deployment reference, not a method-success hard
gate. The integrated run's total-control p95 is below 100 ms only because the
prediction cache is refreshed every 20 control steps; this engineering fact
does not repair the safety failure.

## 5. Reproducibility

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
  --prediction-refresh-interval-steps 20 `
  --output-dir results/phase7_joint_safety_locked_seed745101_robust_p4protocol
```

The formal result directory is intentionally under ignored `results/`; the
report and source changes are the versioned evidence, while large JSONL and
TensorBoard artifacts remain locally reproducible.
