# Phase 5 Safety-QP Diagnostic Report

> Date: 2026-09-07
> Protocol: `configs/innovation_safety.yaml`
> Run: `results/phase5_diagnostic_v5_seed649101/`
> Status: diagnostic complete; P5 hard-barrier gate remains **not passed**.

## 1. Scope and claim boundary

This report audits the velocity-level robust CBF-QP filter and its independent
one-step checker. It does not claim a learned CLBF, a closed-loop forward
invariance proof, or real-flight safety. The checker recomputes current and
next-step geometric barriers from the public observation and does not reuse the
QP residual as a certificate.

The experiment uses eight fixed episode seeds (`649101`--`649108`), three
obstacles, `dt = 0.1 s`, hard barriers (`slack_enabled: false`), a base safety
margin of `0.35 m`, and a summed robustness margin of `0.46 m`. The action
change limit is derived from `6.0 m/s^2 * 0.1 s = 0.6 m/s`.

## 2. Implementation changes

The safety filter now persists, per control step:

- solver status and message;
- failure category (`none`, `precondition_invalid`, `qp_infeasible`,
  `solver_failure`, or `inconsistent_action_bounds`);
- fallback reason and whether the recovery action was used;
- precondition validity and phase (`initial` or `rollout`);
- constraint count, active count, minimum residual, maximum violation, and
  maximum slack;
- barrier values and named constraint residuals.

`barrier_recovery` is an explicit fallback policy. It moves with the violated
barrier normals subject to the current action-change and speed limits. Every
fallback action is still checked by `check_one_step_safety`; a fallback is
never counted as a valid certificate merely because it is finite.

## 3. Results

| Method | Safe capture | Collision | Timeout | Initial robust-set valid | Solver success | Fallback | Next-state safe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal | 100% | 0% | 0% | 100% | N/A | 0 | 98.6% |
| local CBF | 100% | 0% | 0% | 100% | N/A | 0 | 100% |
| robust CBF-QP | 100% | 0% | 0% | 37.5% | 84.7% | 19 | 88.7% |

The aggregate robust result is dominated by the reset contract, not by an
observed QP infeasibility:

- 5/8 episodes start outside the robust shrink-safe set.
- 19 steps are classified as `precondition_invalid`.
- 0 steps are classified as `qp_infeasible`.
- 0 steps are classified as numerical `solver_failure`.
- Among the three initially valid episodes, conditional solver success is
  100% and conditional independent next-state safety is 100%.
- The recovery fallback is safe on 26.3% of fallback steps. The remaining
  fallback steps cannot return to the contracted set in one step under the
  `0.6 m/s` action-change limit; this is an expected limitation of starting
  outside the assumed invariant set, not evidence of a valid certificate.
- Robust-filter latency p95 is about `4.86 ms` in this run, below the
  `100 ms` control-cycle budget.

The complete machine-readable evidence is in:

```text
results/phase5_diagnostic_v5_seed649101/summary.json
results/phase5_diagnostic_v5_seed649101/robust_cbf_qp/episodes.jsonl
results/phase5_diagnostic_v5_seed649101/robust_cbf_qp/steps.jsonl
results/phase5_diagnostic_v5_seed649101/robust_cbf_qp/tensorboard/
```

The TensorBoard event files contain the configuration, source hashes,
per-episode metrics, summary metrics, and failure-category text. The result
directory is intentionally ignored by Git and can be regenerated from the
configuration.

## 4. Root cause

The initial environment reset clips defender altitude to `lower + 0.6 m`,
which is `1.1 m` for this benchmark. The robust checker additionally requires
the drone radius, base margin, and robustness margin. Its lower-altitude
barrier is therefore negative for some reset states; the worst observed
initial barrier is `-0.46 m`. The filter correctly refuses to present this
state as a feasible robust-QP certificate.

This separates two gates that were previously conflated:

```text
reset state satisfies robust contracted set?
    no -> precondition failure and recovery diagnostic
    yes -> solve hard-barrier QP and independently check next state
```

## 5. Decision and next actions

P5 is still **pending**. The current evidence supports the narrower statement
“conditional velocity-level robust CBF-QP filter with an auditable independent
one-step checker.” It does not support “R-CLBF-QP” or a formal closed-loop
safety guarantee.

Before P6 or end-to-end claims, the following must be completed:

1. Define and freeze a reset protocol that either samples only robust-safe
   initial states or reports initial-condition failures as a separate stratum.
2. Run hard-case scans for narrow channels, box/cylinder corners, boundaries,
   and close inter-agent states using the same checker.
3. Compare hard-barrier, soft-slack diagnostic, local CBF, and fallback paths;
   preserve certificate-invalid counts in every comparison.
4. Re-evaluate the safety gate on initially valid episodes and report both
   unconditional and conditional statistics.
5. Keep P6 learned CLBF and the end-to-end locked test blocked until the
   conditional P5 gate has no unexplained solver or certificate failures.

The deterministic scan is now archived in
`docs/PHASE5_HARD_CASE_SCAN_REPORT.md`; it confirms hard-barrier feasibility on
the four robust-safe cases but does not remove the reset-contract gate.

## 6. Verification

```text
python -m py_compile src/encirclement3d/safety_qp.py scripts/evaluate_safety_qp.py
pytest -q tests/test_safety_qp.py          # 9 passed
git diff --check                           # passed
```
