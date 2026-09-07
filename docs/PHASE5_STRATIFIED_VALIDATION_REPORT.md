# Phase 5 Stratified Robust CBF-QP Validation

> Date: 2026-09-07
> Run: `results/phase5_stratified_v8_seed649101/`
> Hard-case run: `results/phase5_hard_case_scan_v2/`
> Decision: **conditional P5 pass** for the velocity-level robust CBF-QP contract.

## 1. Frozen reset protocol

The previous unstratified run mixed states inside and outside the contracted
robust safe set. The formal P5 validation now uses a deterministic,
geometry-only reset audit:

- candidate seeds: `649101`--`649164` (64 total);
- accepted episodes: 8 total;
- tight stratum: four seeds with initial minimum robust barrier in
  `[0, 0.25) m`;
- nominal stratum: four seeds with initial minimum robust barrier in
  `[0.25, +inf) m`;
- every candidate seed is audited and every rejected seed is retained in the
  protocol snapshot;
- selection uses only public defender state, obstacle geometry, bounds and a
  zero-action one-step check. Target ground truth is not used.

Accepted seeds, in evaluation order, are:

```text
tight:   649118, 649121, 649122, 649129
nominal: 649101, 649104, 649107, 649111
```

The snapshot contains 56 rejected seed audits, including their initial
minimum barrier and rejection reason.

## 2. Formal validation results

| Method | Safe capture | Collision | Timeout | Initial robust-set valid | Independent certificate valid | Next-state safe | p95 filter latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal | 100% | 0% | 0% | 100% | 100% | 100% | 0.002 ms |
| local CBF | 100% | 0% | 0% | 100% | 100% | 100% | 10.90 ms |
| robust CBF-QP | 100% | 0% | 0% | 100% | 100% | 100% | 6.27 ms |

The robust filter executed 130 control steps across the eight accepted
episodes:

- solver success: `130/130`;
- QP infeasible: `0`;
- numerical solver failure: `0`;
- fallback: `0`;
- nonzero slack: `0`;
- maximum QP constraint violation: `2.1e-15 m`;
- minimum independently checked next-state barrier: `0.1514 m`;
- robust safe capture: `8/8`;
- boundary violation and collision: `0/8`.

The robust method takes longer to capture on average than the local CBF on
this easy benchmark (`1.625 s` versus `0.925 s`), while preserving the same
safe-capture rate. This is a safety-versus-efficiency trade-off, not evidence
of a universal performance improvement.

## 3. Hard-case cross-check

The deterministic scan has 20 case-method rows covering a narrow channel, box
corner, upper boundary, close inter-agent state, and an intentionally unsafe
reset altitude. Hard-barrier passes the four robust-safe cases and correctly
classifies the unsafe reset as `precondition_invalid`. No QP infeasible case is
observed. Soft-slack produces three nonzero-slack diagnostic rows; those are
not counted as certificates. Full details are in
`docs/PHASE5_HARD_CASE_SCAN_REPORT.md`.

## 4. Decision boundary

This evidence is sufficient to promote the contribution from “pending
feasibility” to:

> a conditional velocity-level robust CBF-QP safety filter with an auditable
> independent one-step certificate, under the frozen reset, margin,
> action-change and kinematic assumptions.

It is not sufficient to claim R-CLBF-QP, learned barrier certification, or a
closed-loop formal safety guarantee. The following remain outside this gate:

- continuous-time swept-volume safety between discrete checks;
- execution delay/noise and acceleration perturbation Monte Carlo;
- adaptive unseen target policies and P4 locked-test generalization;
- multi-step forward-invariance proof with all fallback and solver tolerance
  cases included.

P6 remains blocked until these assumptions are made explicit and tested.

## 5. Reproduction and artifacts

```powershell
python scripts/evaluate_safety_qp.py `
  --config configs/innovation_safety.yaml `
  --output-dir results/phase5_stratified_v8_seed649101

python scripts/scan_safety_hard_cases.py `
  --config configs/innovation_safety.yaml `
  --output-dir results/phase5_hard_case_scan_v2
```

TensorBoard event files, configuration snapshots, source hashes, episode
JSONL and step JSONL are preserved under both result directories. Generated
results remain ignored by Git; the reports and commands are versioned.
