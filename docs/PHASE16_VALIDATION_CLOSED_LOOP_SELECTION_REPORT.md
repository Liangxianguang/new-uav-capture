# Phase 16 Validation Closed-Loop Selection

> Status: configuration selection on the frozen 90-scene validation split.
> This report is not a locked-test result and must not be used to tune after
> the locked-test closed-loop matrix begins.

## Protocol

The prediction dataset uses only the public-team-belief reference at the
forecast origin. All evaluated learned checkpoints use the frozen `both`
action condition: executed action history plus the causal previous DN-MPC plan
shifted into the forecast horizon. The validation scene manifest SHA-256 is
`5ba54afc62d2cf123014f1c5e598a0e3e914e9067870bc7a199ba3393d8f8d99`.

The primary validation comparison uses 90 frozen scenes, matched checkpoint
seeds `727201`, `727202`, and `727203`, and hierarchical bootstrap intervals
that resample both training seed and paired scene episode. The safety-layer
comparison is deliberately separate: `none`, `local_cbf`, and
`robust_cbf_qp` answer different questions and are not interchangeable.

## Refresh Selection

GRU with no safety layer was evaluated at prediction refresh intervals of 1,
5, and 10 control steps. Caching predictions reduced compute latency, but it
also increased candidate age and degraded collision outcomes. In particular,
distributed delayed DN-MPC safe capture fell from 88.9% at every-step refresh
to 73.3% at 5 steps and 28.9% at 10 steps in the seed-727201 validation run.

**Decision:** freeze every-step prediction refresh for Phase 16 closed-loop
testing. The 100 ms reference is recorded as an engineering reference, not a
selection gate; p50/p95/p99 component and total latency remain mandatory.

## Safety-Layer Selection

With GRU seed `727201` and every-step refresh, no safety layer yielded 86.7%
safe capture / 13.3% collision for worst-case MPC and 88.9% / 11.1% for
distributed delayed DN-MPC. Local CBF projected these nominal controls to zero
collision, at the cost of timeouts:

| Planner | Safe capture | Collision | Timeout | Total p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| GRU + worst-case MPC + local CBF | 84.4% | 0.0% | 15.6% | 37.4 |
| GRU + distributed delayed DN-MPC + local CBF | 93.3% | 0.0% | 6.7% | 44.1 |

The robust CBF-QP diagnostic remains No-Go. With the same GRU checkpoint,
worst-case MPC obtained 50.0% safe capture and 50.0% collision; its filter
reported 79.1% solver/certificate success, 20.9% fallback, 91
precondition-invalid events, and 279 QP-infeasible events. Distributed
delayed DN-MPC was less poor but still had 86.7% safe capture, 13.3% collision,
and 16.5% fallback. It is retained only as a diagnostic branch.

**Decision:** use local CBF as the Phase 16 primary closed-loop safety layer;
run robust CBF-QP only as a separately labelled diagnostic.

## Three-Seed GRU Result

| Planner | Safe capture, 95% CI | Collision, 95% CI | Timeout, 95% CI | Total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| GRU + worst-case MPC + local CBF | 84.81% [80.37%, 88.89%] | 0.00% [0.00%, 0.00%] | 15.19% [11.11%, 19.63%] | 20.46 / 37.62 / 39.82 |
| GRU + distributed delayed DN-MPC + local CBF | **93.70% [90.74%, 96.30%]** | 0.00% [0.00%, 0.00%] | 6.30% [3.70%, 9.26%] | 25.92 / 44.78 / 49.12 |

The causal future-action condition was available for 92.18% of distributed
steps, and the mean candidate age was 0 steps because predictions refreshed
every step. These values must continue to be reported on the locked test.

## SSM and Official-S4 Check

The matched seed-727201 local-CBF check used K=8 candidates for the two
diffusion models. Both diagonal SSM and official upstream S4 produced 94.4%
safe capture and 0% collision with distributed delayed DN-MPC. Their total
p95 was 62.2 ms and 61.8 ms respectively, versus 44.1 ms for the single-mode
GRU seed-727201 run. The official model ran through the upstream CPU naive
fallback because structured CUDA kernels are unavailable.

This single-seed closed-loop similarity is insufficient to claim that either
SSM/S4 architecture is superior to GRU. The locked test must retain GRU as the
accuracy reference and diagonal SSM / official S4 as pre-registered K=8
multimodal comparisons.

## Frozen Closed-Loop Configuration

- scene split: Phase 16 locked-test 90-scene manifest;
- checkpoints: seeds `727201`, `727202`, `727203` for GRU, diagonal SSM,
  dense S4, and official S4;
- action condition: `both`, with causal online previous-plan action sequence;
- GRU: K=1, 8 sampling steps; diffusion/S4: K=8, 4 sampling steps;
- planner comparison: Dynamic Encirclement, Pure Pursuit, worst-case MPC,
  and distributed delayed DN-MPC;
- primary safety branches: `none` and `local_cbf`; robust CBF-QP diagnostic
  is reported separately and cannot support a safety proof;
- prediction refresh: every control step.

Generated aggregate: `results/phase16_validation_closed_loop_gru_aggregate.json`.
