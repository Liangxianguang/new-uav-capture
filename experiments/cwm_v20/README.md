# V20: descriptive decomposition of the failed V19 response stage

This is **not another trained model or a new qualification**. It diagnoses the
already inspected V19 train/development population to decide what to change
next. V19 failures stay failed; the enhanced controller stays off; the original
GRU + distributed-delayed DN-MPC + local CBF and reserved holdout are untouched.

The diagnostic loads the immutable published V19 archive, verifies its full
manifest and independent two-run training/public-feature/original-cost audit,
and re-infers every final checkpoint on both train and development. Development
predictions must exactly match published arrays, and the decomposed response
metric must match the original gate-compatible metric. No optimizer exists in
this diagnostic. Labels only determine descriptive bins, never model inputs.

Every configuration and all three seeds are reported, including motion-only as
zero-response control. Fixed bins and horizon offsets are specified before the
diagnostic run, but because V19 development was already inspected this remains
a **posthoc descriptive analysis**, not preregistered fresh validation.

Report vector error, coordinate MSE, predicted magnitude, and false responses
over 5cm within each true-response bin. Bins are point-wise, not classifications
that a controller can know at runtime. Conditional averages disclose supported
groups; population contributions retain all original groups so disjoint bins
sum to the global error change. The V19 response training objective used
call-balanced coordinate MSE; the response gate used group-balanced points and
Euclidean vector error. Both aggregation conventions are reported explicitly.

Mediation diagnostics include absolute command errors and paired command-delta
errors. Good absolute command estimation alone need not imply good causal
contrasts. Estimated commands are frozen public estimates, not oracle commands.

Run with the original research Python environment, in a new process and an
exclusive output directory:

```powershell
python -m pytest -q experiments/cwm_v20
python experiments/cwm_v20/response_diagnostics.py --output results/cwm_v20/diagnostic
python experiments/cwm_v20/response_diagnostics.py --output results/cwm_v20/diagnostic_repeat
python experiments/cwm_v20/release_diagnostics.py --primary results/cwm_v20/diagnostic/summary.json --repeated results/cwm_v20/diagnostic_repeat/summary.json --output results/cwm_v20/reproduced_diagnosis.zip
```

The two `summary.json` files must be identical. Results have no timestamps or
machine-dependent paths. The output's `restored` directory is the frozen capsule
used for audit, not a changed original checkout. No new training or validation
may be claimed from this experiment. A subsequent architecture/loss change
needs a separate frozen protocol and independent fresh groups before training.

Released two-run summary SHA256:
`349ae06a2d27fb9cdb4fe1f0abc1a053e37591a14bd90ba6bf168e132448a817`.
Archive `artifacts/response_failure_diagnosis_20261010.zip`: 111,399 bytes,
SHA256 `067b2570f2c868480f7e187718fbbe4a733d82e27632247f4156f340d1f01fa8`.
The archive snapshots the pre-release README; this release information is added
after bundling. V19+V20 combined tests: 29 passed, one expected duplicate-member
rejection-test warning. Weak-response false predictions offset strong-response
gains; all three seeds remain reported and V19 failed gates are unchanged.
