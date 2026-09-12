# Phase 18d Conformal-Tube Compactness Gate

Status: **Gate implementation verified; current artifact rejected.**

This phase turns the Phase 18b coverage--compactness requirement into an
executable, logged gate. It replays the frozen balanced-context calibration
with no parameter changes and checks the untouched development confirmation.
It does not access the locked test.

## Predeclared gate

- Full-trajectory confirmation coverage must be at least `90%`.
- Mean context-adaptive effective radius must be at most `8.0 m`.
- Maximum context-adaptive effective radius must be at most `10.0 m`.
- Any compactness failure returns `development_confirmation_no_go`.

The exact configuration is
`configs/phase18d_compactness_gate.yaml`. The gate is implemented in
`scripts/calibrate_delay_aware_conformal_tube.py` and records
`Gate/coverage_pass`, `Gate/compactness_pass`, and `Gate/overall_pass` in
TensorBoard.

## Result

| Gate | Observed | Limit | Pass |
| --- | ---: | ---: | --- |
| Confirmation full-trajectory coverage | 99.79% | ≥ 90.00% | Yes |
| Mean effective radius | 8.416 m | ≤ 8.000 m | No |
| Maximum effective radius | 10.404 m | ≤ 10.000 m | No |
| Overall | — | all conditions | **No-Go** |

The result is useful because it prevents the high coverage number from being
reported without its cost: the current tube covers the development archive
only by becoming too wide for the intended interception geometry.

## Decision

The Phase 18b artifact remains a coverage-repair diagnostic and is not
promoted to the main method. The next version must reduce tube volume while
retaining trajectory-level coverage. Its compactness limits must be frozen
before confirmation, and any new OOD or full factorial experiment remains
blocked until the gate passes.

## Local artifact

```text
results/phase18d_compactness_gate_seed727201/
```

The ignored result contains effective configuration, split metadata, source
hashes, calibration/confirmation scores, summary JSON and TensorBoard events.
