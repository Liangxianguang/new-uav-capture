# Phase 71: Multi-seed validation and defender adaptation to Maneuvering Adversary v2

## Scope and guardrail

This phase evaluates the new `adaptive_maneuvering` target and attempts to
adapt the released V5 recurrent defender. It is development validation only.
No locked-test scene, checkpoint, or metric was read, modified, tuned, or
replayed in this phase. The validation seed override is implemented in
`scripts/evaluate_random_central_mixed_obstacles.py` and rejects any use on
`locked_test`.

The validation contract contains three independent seed blocks (`711201`,
`711202`, `711203`), 20 episodes per block (60 total), randomized 3--5 mixed
obstacles, both defender sides, target speed scales `0.55/0.65/0.75`, and
nominal/delayed-noisy observation conditions. Scene generation is cached before
policy rollout so the V5 and fine-tuned policy see identical episodes within
each seed block.

The cache files are retained locally under `results/` and are ignored by Git:

| Seed | Episodes | SHA-256 of `scenes.jsonl` |
| ---: | ---: | --- |
| 711201 | 20 | `36F03494DC16FBD33F8575D71DC3BFB6D1BF6E2B5ACBE3D254C77396F2748BCB` |
| 711202 | 20 | `A1E8A7AFCC716F1B55A65D13FC67C88CCED588B0967AA157BCB6A4CD286A011E` |
| 711203 | 20 | `19EDA2FA794C34660EC58B1C61A610538B8A5AD41AA0B3EEEE8875639C9C4297` |

## Defender adaptation attempt

The first adaptation warm-started
`models/v5_development_exact_reactive_seed661606.pt` and collected 48 mixed
demonstration episodes with `flee_persistence`, `s_curve`, and
`adaptive_maneuvering`. It deliberately retained all demonstrations to test
whether a small domain-adaptation archive was sufficient. The archive had
1,656 frames; its rule-expert terminal safe-capture rate was only 31.25% and
collision rate was 68.75%. It was therefore retained as a negative transfer
control, not selected as a deployable model.

The quality-gated variants were also attempted with
`expert_require_safe_capture=true` and cooperative-entry gating. The broad
random 3--5 obstacle collector was stopped after route-feasibility search
became impractical on CPU; the fixed `cylinder_box` pilot was also stopped
before producing an accepted checkpoint. Neither partial run is used as a
result or model.

## Validation results

All results below use local CBF as an empirical action filter. They are not an
R-CLBF-QP certificate, forward-invariance proof, or real-flight guarantee.

| Method | Seed 711201 | Seed 711202 | Seed 711203 | Pooled safe capture (60 ep) | Pooled collision | Mean clearance (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Released V5 + local CBF | 0.0% | 5.0% | 15.0% | **6.67%** `[1.67%, 13.33%]` | 93.33% | 0.411 |
| 48-episode all-demo fine-tune + local CBF | 0.0% | 5.0% | 5.0% | **3.33%** `[0.00%, 8.33%]` | 96.67% | 0.419 |

Intervals are 10,000 episode-bootstrap samples. The fine-tuned candidate is
3.33 percentage points below V5 on the pooled development validation and is
not selected. Its extra demonstrations increased the mean maneuver-switch
diagnostic from 1.42 to 1.72, but did not improve interception; this is
consistent with negative transfer from collision/failed labels rather than a
successful adaptation.

For context, the `dynamic_encirclement + local CBF` rule expert on seed 711201
also obtained 0/20 safe captures and 20/20 safety failures. This single-seed
upper-bound diagnostic indicates that the current adversary/obstacle/CBF
combination is substantially harder than the V5 training support; it does not
prove that the task is mathematically infeasible.

The valid local artifacts are:

- `results/phase71_validation_v5_aggregate.json`
- `results/phase71_validation_bad_finetune_aggregate.json`
- `results/phase71_validation_v5_seed711201_cached/` (and the two other cached seeds)
- `results/phase71_validation_bad_finetune_seed711101/` for the rejected candidate
- TensorBoard event files under each training/evaluation output directory

Generated artifacts remain ignored; the versioned configs and scripts are the
reproduction contract.

## Interpretation

The requested multi-seed validation and defender fine-tuning were performed,
but the current evidence does not support claiming that the defender can
effectively intercept this new adversary. The key failure is not a small
confidence interval around an otherwise strong result: both learned V5 and
the rule baseline fail almost all randomized episodes, with collision/safety
failure occurring before cooperative capture. The new target implementation
does show the intended behavior—mode switches, obstacle-bypass selection, and
delayed/noisy private target observations—but it shifts the task beyond the
released defender's validated operating region.

## Next experiments, still validation-only

1. Repair the expert-data bottleneck before any more policy fine-tuning. Add a
   route-cache API and flush one rejected attempt to disk, then collect only
   safe/cooperative demonstrations. Do not lower the quality gate.
2. Use a staged curriculum: easy fixed central scenes at target speeds
   `0.45/0.55`, then one maneuver family at a time, then randomized 3--5
   obstacle scenes. Keep `adaptive_maneuvering` in every final-stage archive.
3. Add a defender-side diagnostic ablation with `use_cbf=false` versus
   `use_cbf=true` on the cached validation scenes. If raw policy avoids the
   early safety abort but fails to capture, the bottleneck is CBF feasibility;
   if both fail, the bottleneck is perception/coordination/adaptation.
4. After a quality-gated checkpoint exists, repeat the same three cached seed
   blocks and require a predeclared improvement over the 6.67% V5 reference.
   Report safe capture, capture, collision, boundary, timeout, clearance,
   maneuver switches, and target obstacle-bypass selection by seed.
5. Add a controlled process-isolated runtime benchmark. This Phase 71 actor
   validation does not instantiate predictor, DN-MPC, or QDR, so those latency
   columns are **not applicable** here; no p50/p95/p99 claim should be inferred
   from this report. When the full stack is re-evaluated, report predictor,
   planner, QDR/tube, safety, and total p50/p95/p99 separately.
6. Keep the locked-test gate closed until the new checkpoint passes the fresh
   validation gate across all three seeds. The current V5 locked-test evidence
   remains unchanged and is not updated by this phase.
