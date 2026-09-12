# Phase 28 P7.2 Independent-Certificate Re-audit

> Status: audit artifacts complete; exact behavior-parity gate failed because
> the current source differs from the historical P7 run.

## Scope and protocol

This run closes the missing-artifact part of P7.2 by re-evaluating the original
P7 locked-test contract with the current repository source. It uses the same
100-episode scene file, checkpoint, P4 planner configuration, robust-CBF-QP
safety configuration and 20-step prediction refresh as the historical P7
diagnostic. It is a diagnostic audit only; no parameter was selected from the
locked-test result.

| Item | Value |
| --- | --- |
| Run directory | `results/phase7_p7_2_independent_certificate_reaudit_v5/` |
| Protocol | `configs/phase4_unseen_adaptive_target.yaml` |
| MPC config | `configs/phase4_dn_mpc.yaml` |
| Safety config | `configs/phase7_matched_margin_safety.yaml` |
| Episodes | 100 locked-test episodes |
| Checkpoint | `results/phase2_formal_diffusion_seed745101/checkpoint.pt` |
| Refresh interval | 20 control steps |
| Scene SHA-256 | `22c7a825cb11f0642b8c4f47671856e754982cea818619e110740b50bb21b13f` |
| Result decision | `diagnostic_only` |

The run contains root and method `config.yaml`, `protocol.yaml`,
`scenes.jsonl`, `summary.json`, `episodes.jsonl`, `steps.jsonl`, source hashes
and two TensorBoard event files. Every step also records the independent
current-state and next-state certificate fields.

## Current-source result

| Metric | Result |
| --- | ---: |
| Safe capture | 99.00% |
| Ordinary capture | 99.00% |
| Collision | 0.00% |
| Boundary violation | 0.00% |
| Timeout | 1.00% |
| Robust-filter certificate valid | 99.9804% |
| Independent certificate valid | 100.00% |
| Independent current-state safe | 100.00% |
| Independent next-state safe | 100.00% |
| Robust precondition valid | 100.00% |
| Safety fallback | 0.0196% (1 step) |
| Planner success / valid / effective | 100% / 100% / 100% |

Latency is reported by component; the predictor p50 is zero because the
20-step cache intentionally performs no predictor call on most control steps.

| Component | p50 / p95 / p99 (ms) |
| --- | ---: |
| Predictor | 0.00 / 65.71 / 72.96 |
| Planner | 70.41 / 82.00 / 88.73 |
| Robust safety filter | 6.27 / 8.58 / 10.35 |
| Total control | 86.12 / 146.66 / 159.58 |

## Behavior-parity audit

The historical authoritative P7 diagnostic is
`results/phase7_joint_safety_locked_seed745101_robust_p4protocol_v2_diagnostics/`.
Its episode-level result was safe capture `50.00%`, collision `49.00%`,
boundary violation `16.00%` and timeout `1.00%`. The current-source re-audit
is `99.00%`, `0.00%`, `0.00%` and `1.00%`, respectively. Only 50 of 100
episode outcome rows match exactly; 50 rows changed.

This difference means the new result cannot be interpreted as evidence that
the independent checker alone repaired the historical P7 failure. The source
hashes in the retained configs are different, and the current safety/filter
contract has evolved after the historical run. The completed artifact proves
that the checker and logging path are available now; it does not prove exact
reproduction of the old behavior.

## Decision

- **Artifact-completeness gate: pass.** The run is fully traceable and
  independently checkable.
- **Exact behavior-parity gate: No-Go.** The 50/100 changed episode outcomes
  prevent this run from replacing the historical P7 result.
- **Safety claim: not promoted.** The 100% independent certificate rate is
  conditional on the current velocity-level assumptions and the locked-test
  audit implementation. It is not a multi-step execution-invariant proof,
  forward-invariance proof, or R-CLBF-QP result.
- **Main performance claim: unchanged.** Retain the historical P7 direct
  robust-CBF-QP composition as No-Go until a source-frozen paired experiment
  is designed on an allowed development/confirmation split.

## Reproduction

```powershell
python scripts/evaluate_minimax_mpc_s3.py `
  --protocol configs/phase4_unseen_adaptive_target.yaml `
  --split locked_test `
  --mpc-config configs/phase4_dn_mpc.yaml `
  --scene-records results/phase4_unseen_adaptive_locked_seed745101_gpu_v2/scenes.jsonl `
  --checkpoint results/phase2_formal_diffusion_seed745101/checkpoint.pt `
  --candidate-source checkpoint `
  --methods worst_case `
  --episodes 100 `
  --device cpu `
  --safety-layer robust_cbf_qp `
  --safety-config configs/phase7_matched_margin_safety.yaml `
  --prediction-refresh-interval-steps 20 `
  --output-dir results/phase7_p7_2_independent_certificate_reaudit_v5
```

