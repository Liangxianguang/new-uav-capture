# Phase 18b Balanced-Context Adaptive Conformal Tube

Status: **Promising diagnostic; not promoted to a formal main method.**

Phase 18b repairs two weaknesses found in the first conformal-tube pilot:
the validation split was ordered by episode seed and one mirror group crossed
the split, while the runtime tube already had an uncertainty-scaling hook but
the offline report did not evaluate it. Phase 18b keeps complete mirror groups
together, stratifies a public-observation context score, and reports both the
fixed-radius and context-adaptive empirical coverage.

The result is a useful methodological signal, not a safety claim. The context
score is a proxy built from normalized public observations (message age,
confidence, covariance, estimated target speed and visibility). It never uses
future target truth at runtime. The conformal artifact is still empirical and
does not prove conditional coverage or forward invariance.

## Protocol

- Predictor: frozen Phase 16 Diagonal SSM conditional diffusion checkpoint,
  action conditioning `both`.
- Calibration/confirmation: complete mirror groups from the Phase 16
  validation archive, deterministically stratified with split seed `20260912`.
- Groups: 44 total, 20 calibration and 24 confirmation; 987 and 935 windows.
- Coverage target: `0.90`.
- Candidate set: `K=8`, four diffusion sampling steps, four projection steps.
- Context inflation: `radius × (1 + 0.25 × public_context_score)`.
- Locked-test: not read.

The reproducible configuration is
`configs/phase18b_balanced_context_conformal_tube.yaml`; the calibration
entry point is `scripts/calibrate_delay_aware_conformal_tube.py` with
`--split-strategy balanced_mirror_context`.

## Offline coverage

| Evaluation | Fixed-radius full trajectory | Context-adaptive full trajectory | Mean effective radius | Maximum effective radius |
| --- | ---: | ---: | ---: | ---: |
| Calibration, 987 windows | 90.17% | 100.00% | 8.587 m | 10.404 m |
| Confirmation, 935 windows | 83.32% | **99.79%** | 8.416 m | 10.404 m |

Confirmation mean per-horizon coverage is 91.55% for the fixed schedule and
99.96% after context inflation. The fixed schedule therefore remains a
failure for trajectory-level coverage; the adaptive result shows that the
public context score identifies the high-error long-route cases in this
development archive.

The base radius schedule is approximately 8.58 m at horizon 1 and 6.04 m at
horizon 12. The context-adaptive maximum reaches 10.40 m. This is large for
the current scene scale, so coverage alone is insufficient: a future version
must pass a pre-registered coverage--compactness gate.

## Online paired smoke

Both runs use the same 8 validation scenes, checkpoint, sampling seed,
distributed-delayed DN-MPC, UAKR, RNIC, local CBF and CPU execution. The only
difference is whether the frozen tube artifact is supplied.

| Metric | No tube | Context-adaptive tube |
| --- | ---: | ---: |
| Safe capture | 87.50% | 87.50% |
| Collision | 12.50% | 12.50% |
| Boundary violation | 0.00% | 0.00% |
| Episode outcomes identical | — | 8/8 |
| RNIC p50/p95/p99 | 0.77/1.51/2.19 ms | 0.75/1.70/1.90 ms |
| Total-control p50/p95/p99 | 36.16/55.99/63.40 ms | 36.20/56.27/64.49 ms |

The tube was enabled for 100% of tube-run steps. The small sample does not
support a performance improvement claim; it confirms runtime compatibility
and negligible observed latency change in this smoke.

## Decision and next gate

Phase 18b is retained as a **coverage-repair diagnostic**, not promoted as a
new capture controller. The 99.79% confirmation coverage is encouraging, but
the effective radius is broad and the compactness threshold was not frozen
before this exploratory confirmation. Therefore it cannot be used as a
formal main-result gate.

Before opening OOD or the full QDR × UAKR × RNIC matrix:

1. Freeze a radius/volume upper bound and a trajectory-coverage target on a
   fresh development split.
2. Repeat the complete-mirror, public-context split with a new confirmation
   archive and no post-hoc tuning.
3. Compare fixed, context-adaptive and trajectory-level scalar conformal
   scores under the same coverage--compactness gate.
4. Run a controlled single-process 3-way closed-loop comparison and report
   predictor, planner, RNIC and total p50/p95/p99.
5. Only a passing confirmation plus non-inferior ID capture and no added
   collision/boundary may reopen an OOD block.

## Local artifacts

Generated artifacts are ignored by Git and retained locally:

```text
results/phase18b_conformal_tube_balanced_seed727201_retry2/
results/phase18b_conformal_tube_online_smoke_seed727201/
results/phase18b_conformal_tube_online_smoke_seed727201_no_tube/
```

Each run retains JSONL logs, summary, source/config hashes and TensorBoard
events. The calibration run additionally retains the split metadata and
per-window public context scores.
