# Experiment Results Index

This repository keeps generated experiment artifacts under `results/`. The
directory is intentionally ignored by Git because it contains large JSONL,
checkpoint, and TensorBoard files. Formal reports and source code remain
versioned; each retained run keeps its configuration, source hashes, metrics,
and TensorBoard event files locally for reproduction.

## Current Evidence Levels

The current consolidated status and executable follow-up plan are in
`EXPERIMENT_STATUS_AND_NEXT_TODOLIST.md`. It is the latest decision snapshot;
the phase reports below remain the authoritative source for each individual
metric.

| Phase | Evidence | Status | Authoritative report |
| --- | --- | --- | --- |
| Phase 2 | Portable SSM + conditional diffusion prediction | Conditional Go | `PHASE2_FORMAL_ANALYSIS_REPORT.md`, `PHASE2_ADAPTIVE_GENERALIZATION_AUDIT_REPORT.md` |
| Phase 3 | Centralized scenario MPC | Pass for the frozen validation gate | `PHASE3_S3_VALIDATION_REPORT.md` |
| Phase 4 | Distributed DN-MPC and unseen adaptive-target locked test | Pass for modular planning | `PHASE4_DN_MPC_VALIDATION_REPORT.md`, `PHASE4_UNSEEN_ADAPTIVE_VALIDATION_REPORT.md` |
| Phase 5 | Velocity-level robust CBF-QP | Conditional pass only | `PHASE5_STRATIFIED_VALIDATION_REPORT.md` and execution audit reports |
| Phase 7 | DN-MPC + robust CBF-QP joint integration | No-Go on the frozen P4 reset | `PHASE7_JOINT_SAFETY_AUDIT_REPORT.md` |
| Phase 7.1 | Safety-contract failure diagnosis and rejected L2-SLSQP ablation | Contract repair required; direct integration remains No-Go | `PHASE7_SAFETY_CONTRACT_DIAGNOSTIC_REPORT.md` |
| Phase 10 | Continuous-segment, reachable-tube, and queue-recovery profiling | Unit tests pass; profiling remains below the formal capture/certificate gate | `PHASE10_QUEUE_AUTHORITY_RECOVERY_PROFILE_REPORT.md`, `PHASE10_REACHABLE_TUBE_HORIZON_AUDIT_REPORT.md` |
| Phase 14 | Latest eight-seed hard queue-authority matrix | Execution audit complete; hard capture gate remains No-Go | `PHASE14_AUTHORITY_MATRIX_REPORT.md` |
| Phase 15 | S4-v3 action-conditioned predictor and causal online DN-MPC | Data/audit and three-seed formal comparison complete; GRU `both` condition selected on validation and confirmed on locked test; S4 does not beat GRU offline; feasibility and multimodality ablations remain | `PHASE15_S4_V3_FORMAL_MULTISEED_REPORT.md`, `PHASE15_S4_V3_ACTION_CONDITION_ABLATION_REPORT.md`, `PHASE15_S4_V3_LOCKED_TEST_REPORT.md`, `OFFICIAL_S4_SETUP.md` |
| Phase 16 | Policy-safe predictor-v4 conversion and expanded S4 dataset | Dataset audit, three-seed prediction selection, validation closed-loop selection, GRU/diagonal-SSM/official-S4 locked-test matrix, K=1/K=8, safety-layer, and action-condition closed-loop ablations complete. Geometry, target-speed, delay/execution, and target-behavior OOD blocks are complete; controlled runtime and safety-contract repair remain. | `PHASE16_POLICY_SAFE_PREDICTOR_V4_PLAN.md`, `PHASE16_VALIDATION_CLOSED_LOOP_SELECTION_REPORT.md`, `PHASE16_LOCKED_TEST_CLOSED_LOOP_REPORT.md`, `PHASE16_OOD_GEOMETRY_REPORT.md`, `PHASE16_OOD_TARGET_SPEED_REPORT.md`, `PHASE16_OOD_DELAY_EXECUTION_REPORT.md`, `PHASE16_OOD_TARGET_BEHAVIOR_REPORT.md` |
| Phase 17 | Queue-aware delayed-state rollout, uncertainty-triggered adaptive K/replanning, and reachability-normalized interception | QDR is No-Go on validation; UAKR gives reproducible compute savings but misses the -2 pp ID non-inferiority gate; RNIC is behavior-neutral or worse on ID, target-speed OOD and delay/execution OOD while increasing latency; its delay/execution-OOD slack is not predictive of collision (pooled AUROC 0.504). | `PHASE17_QUEUE_ADAPTIVE_REACHABILITY_TODOLIST.md`, `PHASE17_M0_QDR_SMOKE_REPORT.md`, `PHASE17_QDR_VALIDATION_REPORT.md`, `PHASE17_UAKR_VALIDATION_REPORT.md`, `PHASE17_RNIC_VALIDATION_REPORT.md`, `PHASE17_RNIC_TARGET_SPEED_OOD_REPORT.md`, `PHASE17_RNIC_DELAY_EXECUTION_OOD_REPORT.md`, `PHASE17_RNIC_DIAGNOSTIC_ANALYSIS.md`, `PHASE17_UAKR_RNIC_PILOT_REPORT.md`, `PHASE17_OOD_GEOMETRY_PILOT_REPORT.md` |
| Phase 18 | Delay-aware split-conformal reachable-tube repair candidate | Implementation, calibration and online smoke are complete; confirmation full-trajectory coverage is 85.92% versus the 90% target and the tube is too wide, so the candidate is No-Go for promotion. | `PHASE18_DELAY_AWARE_CONFORMAL_TUBE_PILOT_REPORT.md` |
| Phase 18b | Balanced-mirror public-context adaptive conformal tube | Fixed-radius confirmation coverage remains 83.32%, but public-context inflation reaches 99.79% on the untouched development confirmation; the tube remains broad and the compactness gate was not pre-frozen, so this is a promising diagnostic rather than a promoted main result. | `PHASE18B_BALANCED_CONTEXT_TUBE_REPORT.md` |
| Phase 18c | Tube width as an explicit UAKR feature | In an 8-scene paired smoke, mean K and refresh ratio increase without changing safe capture or collision; total p95 increases to 57.87 ms. Retain as a negative ablation, not a promoted UAKR gain. | `PHASE18C_TUBE_UAKR_SMOKE_REPORT.md` |
| Phase 18d | Executable conformal-tube coverage--compactness gate | Coverage passes at 99.79%, but mean/max effective radius 8.416/10.404 m exceed the 8/10 m limits; the artifact is automatically classified No-Go. | `PHASE18D_COMPACTNESS_GATE_REPORT.md` |
| Phase 19 | Severe-unreachability-gated RNIC pilot | On a paired 20-scene validation pilot, thresholds `-0.75/-0.50/-0.25 s` leave safe capture/collision unchanged at 95.0%/5.0% while increasing total p95 to 66.15--67.97 ms; retain as a negative ablation, not a promoted method. | `PHASE19_GATED_RNIC_PILOT_REPORT.md`, `phase19_gated_rnic_validation.yaml` |

Phase 15's raw data collection and mirror-disjoint split audit remain valid.
Its predictor results used the former simulator-truth forecast origin and are
therefore historical evidence rather than the fair comparison for Phase 16.
The new Phase 16 plan freezes all future predictor selection on public-belief
references before its locked test is opened.

## Formal Phase 7 Run

The current locked joint audit is retained in:

```text
results/phase7_joint_safety_locked_seed745101_robust_p4protocol/
```

It contains the root and method `config.yaml`, protocol and scene records,
episode/step JSONL, summary JSON, source hashes, and TensorBoard events. The
run uses the P4 `adaptive_adversarial` 100-episode scene file and must be
treated as the authoritative result for the direct robust-CBF-QP composition.

The richer diagnostic rerun is retained separately at:

```text
results/phase7_joint_safety_locked_seed745101_robust_p4protocol_v2_diagnostics/
```

The `v3_l2speed` directory is a retained rejected ablation: it documents a
nonlinear SLSQP speed-constraint replacement that regressed to 1% safe capture
and must not be used as a formal result.

## TensorBoard Retention

Training and evaluation scripts write TensorBoard event files alongside their
run artifacts. Training runs also record the effective configuration,
hyperparameters, source hashes, and checkpoint metadata. To inspect all local
runs:

```powershell
.\scripts\start_tensorboard.ps1 -LogDir results -Port 6006
```

Do not delete a retained training or evaluation run merely because its output
is ignored by Git. Smoke runs and failed intermediate runs are useful for
debugging unless a later report explicitly supersedes them and the run has
been classified as disposable.

## Cleanup Policy

Only generated caches (`__pycache__`, `.pytest_cache`) and empty temporary
directories are disposable by default. Formal result directories, training
logs, checkpoints, scene files, and TensorBoard event files are retained.
