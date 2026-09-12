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

The complete forward-looking research protocol for the three proposed
contributions—Queue-Aware Delayed-State Rollout, Uncertainty-Triggered
Adaptive K and Replanning, and Reachability-Normalized Interception Cost—is
in `THREE_INNOVATIONS_MASTER_TODOLIST.md`.

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
| Phase 20 | UAKR threshold-calibration pilot | Lowering thresholds to `0.30/0.55` raises mean K to 3.19 but reduces the paired 20-scene safe capture from 95.0% to 90.0% and raises total p95 to 138.04 ms; retain as a negative ablation. | `PHASE20_UAKR_THRESHOLD_CALIBRATION_PILOT_REPORT.md` |
| Phase 21 | UAKR reliability audit | On 270 retained validation episodes, mean uncertainty ranks failure/collision with AUROC 0.740 and maximum uncertainty with AUROC 0.703; the signal is useful but quartile risk is not strictly monotonic, so calibration and confirmation remain open. | `PHASE21_UAKR_RELIABILITY_AUDIT_REPORT.md`, `scripts/analyze_uakr_reliability.py` |
| Phase 22 | UAKR mirror-group confirmation audit | With complete mirror groups split into 134 calibration and 136 confirmation episodes, mean-U AUROC is 0.771/0.687 and max-U AUROC 0.720/0.667; ranking transfers weakly but is not calibrated enough for online risk gating. | `PHASE22_UAKR_MIRROR_GROUP_CONFIRMATION_REPORT.md` |
| Phase 23 | Risk-calibrated UAKR confirmation | Frozen monotone calibration improves safe capture over original UAKR by +1.45 pp on 138 confirmation episodes, but remains 2.17 pp below fixed K=8 and raises total p95 to 141.98 ms; retain as an ablation. | `PHASE23_UAKR_RISK_CALIBRATED_CONFIRMATION_REPORT.md`, `configs/phase23_uakr_risk_calibrated.yaml`, `src/encirclement3d/uncertainty_calibration.py`, `scripts/calibrate_uakr_risk_map.py` |
| Phase 24 | UAKR residual-triggered budget pilot | A public residual trigger at 0.40 m raises mean K to 3.02 but lowers the paired 20-scene safe capture from 90.0% to 85.0%, raises collision from 10.0% to 15.0%, and increases total p95 to 141.61 ms; retain as a negative ablation. | `PHASE24_UAKR_RESIDUAL_TRIGGER_REPORT.md`, `configs/phase24_uakr_residual_trigger.yaml` |
| Phase 25 | UAKR residual-only refresh pilot | Keeping K unchanged but forcing refresh at residual 0.40 m still lowers the paired 20-scene safe capture to 85.0%, raises collision to 15.0%, and increases total p95 to 142.34 ms; retain as a negative ablation. | `PHASE25_UAKR_RESIDUAL_REFRESH_REPORT.md`, `configs/phase25_uakr_residual_refresh.yaml` |
| Phase 26 | QDR delayed-state safety projection, prefix-risk audit, and authority diagnostic | Evaluating local CBF at the first controllable delayed state improves the logged nominal prefix minimum clearance but leaves the paired 20-scene safe capture at 80.0% with 20.0% collision; the released prefix audit finds 92.47% admissible QDR control steps. Explicit `replace_nonexecuting` and `flush_pending` authority trade collision against timeout and do not pass safe-capture non-inferiority. Retain as a contract diagnostic, not a promoted gain. | `PHASE26_QDR_SAFETY_PROJECTION_PILOT_REPORT.md`, `scripts/analyze_qdr_prefix_failures.py`, `tests/test_phase17_qdr_safety_projection.py` |
| Phase 27 | RNIC formation-slot reachability cost | A bounded exact-permutation formation-slot objective is implemented and integrated into centralized and distributed team scoring. On a fresh 60-episode/30-mirror-group, three-seed centralized ID confirmation it reaches 89.44% safe capture versus 87.22% RNIC-off, paired delta +2.22 pp [-0.56,+5.00], supporting the predeclared -2 pp non-inferiority direction but not superiority. In the same distributed-delayed confirmation all arms are 96.67% safe capture / 3.33% collision; formation RNIC adds about 13.08/15.19/17.10 ms RNIC p50/p95/p99 and pooled total is 205.76/220.87/235.10 ms versus 75.10/82.40/89.86 ms off, so current distributed promotion is No-Go. | `PHASE27_RNIC_FORMATION_SLOT_PILOT_REPORT.md`, `configs/phase27_rnic_formation_slot_pilot.yaml`, `scripts/generate_phase27_rnic_confirmation_scenes.py` |

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
