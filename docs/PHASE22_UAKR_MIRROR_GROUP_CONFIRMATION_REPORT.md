# Phase 22 UAKR Mirror-Group Confirmation Audit

Status: **Independent confirmation retains a weak ranking signal; online
risk-calibrated UAKR is not yet validated.**

## Protocol

This audit uses the three retained Phase 17 UAKR validation runs (seeds
`727201`, `727202`, and `727203`) and splits complete mirror groups rather
than relying on episode order. The scene manifest is
`results/phase16_validation_closed_loop_gru_seed727201_local_refresh1/scenes.jsonl`.

The deterministic split uses seed `20260912`, with 135 per-run mirror-group
units assigned to 67 calibration groups and 68 confirmation groups. This
produces 134 calibration episodes and 136 confirmation episodes. No mirror
group is shared across the two subsets. The controller is not changed by the
audit and no locked-test data is accessed.

The reproducible command is:

```powershell
python scripts/analyze_uakr_reliability.py `
  --run results/phase17_validation_uakr_adaptive_seed727201 `
        results/phase17_validation_uakr_adaptive_seed727202 `
        results/phase17_validation_uakr_adaptive_seed727203 `
  --scene-manifest results/phase16_validation_closed_loop_gru_seed727201_local_refresh1/scenes.jsonl `
  --split-strategy mirror_group_half `
  --split-seed 20260912 `
  --output results/phase22_uakr_mirror_group_reliability.json
```

## Result

| Split | Episodes | Mean-U AUROC | Max-U AUROC | Mean-U Q1 failure | Mean-U Q4 failure |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibration | 134 | 0.771 | 0.720 | 0.00% | 24.24% |
| Confirmation | 136 | 0.687 | 0.667 | 0.00% | 8.82% |

The first-step uncertainty AUROC is `0.728` on confirmation. The high-risk
quartile is riskier than the low-risk quartile, but the absolute rate drops
from calibration to confirmation. Thus the signal has some transferability,
but its magnitude is not calibrated across a held-out group split.

## Decision

Do not promote the current UAKR thresholds or interpret uncertainty as a
failure probability. The next implementation should fit a calibration map on
the calibration groups using a predeclared label, then evaluate that frozen
map on the confirmation groups. Candidate labels must be logged separately:
one-step prediction miss, next-step safety-margin violation, and episode
timeout. If the map is not stable, UAKR remains a compute-efficiency
ablation, and Phase 16 remains the primary model.

The JSON/Markdown/TensorBoard output is retained locally at:

```text
results/phase22_uakr_mirror_group_reliability.json
results/phase22_uakr_mirror_group_reliability.md
results/phase22_uakr_mirror_group_reliability_tensorboard/
```
