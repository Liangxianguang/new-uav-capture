# Phase 21 UAKR Reliability Audit

Status: **Ranking signal confirmed; calibrated closed-loop policy not yet
confirmed.**

## Protocol

This audit reprocesses the retained Phase 17 UAKR validation logs for seeds
`727201`, `727202`, and `727203` (270 episodes total). For each episode it
aggregates the step-level public uncertainty into mean and maximum uncertainty
and evaluates whether those scores rank realized closed-loop failure,
collision, boundary violation, or timeout.

The script is `scripts/analyze_uakr_reliability.py`. It reads only
episode/step logs, does not feed realized outcomes into the controller, and
does not access locked-test data. It writes JSON/Markdown and TensorBoard
artifacts. The local output is:

```text
results/phase20_uakr_reliability_audit.json
results/phase20_uakr_reliability_audit.md
results/phase20_uakr_reliability_audit_tensorboard/
```

## Result

| Score | Failure AUROC | Collision AUROC | Boundary AUROC | Timeout AUROC |
| --- | ---: | ---: | ---: | ---: |
| Mean uncertainty | 0.740 | 0.740 | NaN | NaN |
| Maximum uncertainty | 0.703 | 0.703 | NaN | NaN |

The mean-uncertainty quartile failure rates were `0.00%`, `4.41%`, `2.99%`,
and `16.42%` from low to high. The highest quartile is therefore riskier,
but the middle bins are not strictly monotonic. This is useful ranking
evidence, not a calibrated probability or a safety certificate.

## Interpretation

The UAKR score contains exploitable information, but the earlier threshold
pilot demonstrates that simply lowering the bucket thresholds increases
compute without guaranteeing better control outcomes. The correct next step
is an independent development calibration map from uncertainty to a
predeclared failure label, followed by a fresh confirmation block. The
calibration must be frozen before any locked-test run.

Recommended labels are one-step public-prediction miss, next-step safety
margin violation, and episode timeout; each must be audited separately rather
than collapsed into an unexplained success score.

## Decision

UAKR is retained as a promising, reproducible scheduling direction but is not
promoted as a closed-loop performance contribution yet. No additional
threshold sweep is justified on the current validation archive.
