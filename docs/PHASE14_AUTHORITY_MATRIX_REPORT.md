# Phase 14 Queue-Authority Matrix Report

> Status: latest eight-seed execution audit, not a completed closed-loop proof
>
> Configuration: `configs/phase9_queue_authority.yaml`
>
> Evidence: `results/phase14_authority_matrix_v1/`

## 1. Conclusion

The latest code completed the full hard-execution authority matrix on the
eight retained robust-safe seeds. The result confirms that queue authority is
a first-order execution assumption, but it does not recover the hard-scene
capture objective under the current reachable-tube and multi-step certificate.

`flush_pending` is the best of the three simulated authority modes for safety:
it reaches `99.75%` actual post-state safety and `0%` collision/boundary
failure. Its safe capture is only `25%`, however, and its continuous-segment
certificate is valid on only `38.60%` of control steps. It must therefore be
reported as a simulated executor capability, not as evidence that a physical
flight controller can clear its command queue.

The current decision remains **Conditional Go for modular planning and No-Go
for the complete execution-aware robust controller**.

## 2. Protocol

| Item | Value |
| --- | --- |
| Episodes | 8 retained seeds |
| Scene | 3 obstacles, flee-persistence target |
| Target speed scale | 0.45 |
| Maximum episode length | 250 steps |
| Execution | hard randomized delay/noise/tracking |
| Delay | 2 steps |
| Reachable-tube multiplier | 2.1 |
| Prediction/authority method | robust CBF-QP only |
| Continuous-segment QP constraints | disabled in this authority matrix; independently audited |
| QP backend | analytic rollout Jacobian |

The matrix uses the same eight seeds as the prior locked authority audit. The
new result directory contains configuration snapshots, episode/step JSONL,
summary JSON, TensorBoard events, and source hashes.

## 3. Results

| Authority | Safe capture | Collision | Boundary | Timeout | Actual post-state safety | Rollout cert. | Continuous cert. | Prefix admissible | Abort required | Fallback cert. | p50 / p95 / p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `immutable` | 12.5% | 0% | 0% | 87.5% | 29.55% | 18.47% | 18.77% | 20.42% | 79.58% | 0.06% | 21.0 / 117.8 / 150.4 |
| `replace_nonexecuting` | 0% | 0% | 0% | 100% | 90.05% | 40.45% | 40.55% | 89.95% | 10.05% | 83.25% | 119.7 / 460.7 / 507.2 |
| `flush_pending` | 25.0% | 0% | 0% | 75.0% | 99.75% | 38.45% | 38.60% | 93.26% | 6.74% | 87.34% | 126.4 / 348.4 / 405.6 |

Additional observations:

- `immutable` has `1,426` abort-required steps and a mean queue depth of `2`.
  The filter cannot repair an unsafe committed prefix after it has entered the
  queue; the low capture rate is therefore mainly a recoverability failure,
  not evidence that the nominal pursuit planner cannot make progress.
- `replace_nonexecuting` and `flush_pending` substantially improve prefix
  admissibility, but they still reject many full-horizon rollouts. Their
  fallback counts are `1,200` and `1,177`, respectively, so the controller
  spends most of the hard episode in recovery rather than pursuit.
- The `flush_pending` minimum actual post-state robust barrier is
  `-0.00399 m`. This is close to the contracted boundary but is still a failed
  robust certificate; `0%` observed collision is not a replacement for a
  valid barrier certificate.
- The dominant failed barriers are horizon-5 boundary and obstacle barriers,
  together with the independent continuous-segment minimum barrier. This
  identifies the next bottleneck as the conservatism and feasibility of the
  execution certificate/recovery contract, not the DN-MPC planner.

## 4. Interpretation

The authority comparison separates three quantities that were previously easy
to conflate:

1. **Prefix feasibility:** whether commands already in the queue can still be
   executed under the declared authority.
2. **Post-state safety:** whether the simulated execution happened to end in a
   safe state.
3. **Task progress:** whether the defenders capture the target before timeout.

`flush_pending` improves the first two quantities in this simulation, but its
safe capture remains far below the working hard-scene target of `80%`. The
latest matrix therefore does not justify selecting an authority mode as the
final real-system controller.

The gap between actual post-state safety and the full rollout/continuous
certificate is also material. It may indicate a deliberately conservative
certificate, a mismatch between the sampled continuous model and the
execution model, or a recovery policy that is safe only for one step. These
possibilities must be separated experimentally before changing the tube
multiplier or relaxing a barrier.

## 5. Next TodoList

### P14.1 Diagnostic decomposition

- [x] Re-run the complete eight-seed hard authority matrix after the prefix
  screening and geometry-cache changes.
- [x] Preserve per-step queue depth, first failed barrier, fallback type,
  certificate scope, target-distance progress, and actual post-state checks.
- [ ] Aggregate failures by prefix, horizon index, barrier family, execution
  error magnitude, and fallback candidate.
- [ ] Compare horizon `1/2/3/5` certificates on exactly the same step states,
  so certificate conservatism is measured without changing the trajectory.

### P14.2 Certificate and recovery repair

- [ ] Keep the empirical tube multiplier at `2.1` until an independent
  calibration/holdout split justifies a change.
- [ ] Check whether the negative post-state robust barriers are caused by the
  robust margin contract, by execution noise outside its declared bound, or by
  a genuine state leaving the robust set.
- [ ] Make recovery lexicographic: first satisfy queue-prefix and full declared
  safety constraints, then maximize belief-only capture progress inside the
  certified feasible set.
- [ ] Keep one-step receding fallback explicitly separate from a full-horizon
  certificate; never report it as a multi-step invariance proof.
- [ ] Add a no-clear diagnostic variant that records infeasibility without
  silently changing the executor authority.

### P14.3 Authority and holdout validation

- [ ] Confirm with the target flight/executor interface whether pending
  commands can be replaced or flushed atomically. Until confirmed, treat
  `immutable` as the conservative physical assumption.
- [ ] After a candidate passes the diagnostic gate, evaluate at least 30--50
  unseen seeds with Wilson intervals.
- [ ] Require hard safe capture at least `80%`, collision and boundary failure
  `0%` in the evaluation set, actual post-state safety at least `99%`, and
  certificate coverage at the declared confidence level.
- [ ] Report latency p50/p95/p99 as an engineering metric; the former `100 ms`
  value remains a target, not a hard method gate.

### P14.4 End-to-end reopening

- [ ] Reconnect Mamba/SSM diffusion and DN-MPC only after the safety candidate
  passes P14.2/P14.3.
- [ ] Re-run the same prediction checkpoint and scene protocol with local CBF,
  current robust CBF-QP, and certified recovery as separate baselines.
- [ ] Keep the old P7 `50% safe capture / 49% collision / 16% boundary` result
  as a negative integration result; do not overwrite it with the authority
  ablation.

## 6. Evidence boundary

This report supports an execution-audit conclusion only. It does not establish
a learned CLBF, an R-CLBF-QP theorem, continuous-time invariance for a real
airframe, or physical queue authority. The generated result directory is
ignored by Git; the versioned report records the protocol and the authoritative
metrics needed to reproduce and interpret it.
