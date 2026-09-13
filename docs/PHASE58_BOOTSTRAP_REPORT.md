# Phase 58：协议、场景与 QDR 审计阶段性结果

## 1. 阶段状态

Phase 58-0/1 已完成，闭环 calibration 与 process-isolated runtime 也已完成。
该报告前半部分只证明协议、场景和 QDR 时间索引审计；闭环数值与 gate 判定见
`docs/PHASE58_CALIBRATION_REPORT.md`。任何结果都不构成安全证明。

执行日期：2026-09-14
源代码提交：见仓库 Phase 58 最新提交（协议、runner、block aggregator 与报告）
主计划：`docs/PHASE58_REDESIGNED_DELAYED_PLANNING_TODOLIST.md`

## 2. 冻结的场景矩阵

生成目录（results 不提交 Git）：

```text
results/phase58_delayed_planning_matrix/
results/phase58_scene_audit/
```

| 项目 | 结果 |
| --- | ---: |
| Episodes | 480 |
| Mirror groups | 240 |
| Blocks | 4 |
| Episodes/block | 120 |
| Development-calibration | 160 episodes / 80 groups |
| Development-confirmation | 160 episodes / 80 groups |
| Locked-diagnostic | 160 episodes / 80 groups |
| Unique layout seeds | 240 |
| Manifest SHA-256 | `f4787252c2ca18e99316f0630f15c31c610504b60bba818b301b1cbf6af4207f` |

四个 block 为：`id_reference`、`delay_noise_factorial`、
`communication_execution_factorial` 和 `joint_stress_transfer`。执行压力覆盖
action delay `0/2/4/6/8/10` steps、command noise `0/0.04/0.08/0.12/0.16`
m/s、message delay `0/2/4/6`、message dropout `0/0.1/0.2/0.3`、tracking
time constant `0/0.1/0.2 s` 和 drag `0/0.05`。joint block 使用经过 route
validity 验证的 transfer geometry；所有 block 仍使用相同 predictor 条件。

## 3. 审计结果

`scripts/audit_phase58_scene_matrix.py` 已对冻结 manifest 完成以下检查：

- 480 episode、240 mirror group，四个 block 和三个 split 数量平衡；
- 每个 mirror group 完整包含 upper/lower，且 layout seed、障碍物和 split 一致；
- 240 个 layout seed 不重复；
- target speed 保持在训练范围，geometry transfer 只出现在 joint block；
- nominal sensing、延迟、噪声、tracking 和 drag 均属于预注册 level；
- scene bytes SHA-256 与 manifest 一致；
- audit result：`audit_pass: true`。

数据审计 TensorBoard：

```text
results/phase58_scene_audit/tensorboard/
```

## 4. QDR 时间索引结果

`scripts/check_phase58_qdr_time_index.py` 在 queue length
`q={0,2,4,8,10}`、4 defenders、8-step suffix、固定 deterministic replay
下完成检查：

- full rollout 与 shifted suffix 的最大位置误差：`0.0 m`；
- 最大速度误差：`0.0 m/s`；
- candidate action application index 均为 `t+q+j`；
- 正常合同的 `overall_pass: true`；
- 对每个 q 注入的 `t+q+2H` double-delay probe 均被识别并拒绝。

QDR TensorBoard：

```text
results/phase58_qdr_time_index/tensorboard/
```

这只是 queue composition 和 time-index equivalence 检查，不是 collision-free、
forward invariance、概率覆盖率或真实飞行安全证明。`local_cbf` 在后续闭环中
仍只标记为 empirical filter；不宣称 R-CLBF-QP 或安全证书。

## 4.1 Runner smoke（非正式性能结果）

在 calibration selection 的前 6 个 episode、GRU `both` seed `727201`、CPU
Torch `1/1` 线程、`K=1`、sampling steps `8`、local CBF 下，M0/M1/M6 均完成
闭环并写入 method-specific TensorBoard。该 smoke 只验证接口连通和计时字段，
不能作为 480 episode 性能结论。

| Method | Safe capture | Collision | Timeout | Total p50/p95/p99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| M0 current-state delayed-MPC | 100.0% | 0.0% | 0.0% | 8.08/8.92/10.94 |
| M1 known-delay delayed-MPC | 100.0% | 0.0% | 0.0% | 8.02/9.00/9.69 |
| M6 QDR+synchronous distributed | 100.0% | 0.0% | 0.0% | 15.91/18.72/20.21 |

Smoke TensorBoard：

```text
results/phase58_smoke_seed727201_m0_m1_m6/*/tensorboard/
```

## 5. 闭环 calibration 与运行时

三种 predictor seed 已在 160-episode calibration selection 上完成 M0--M8，
并按 mirror group 聚合。M6 QDR+synchronous 的 safe capture 为 `88.75%`
（95% CI `85.00--92.08%`），M7 QDR+asynchronous 为 `86.25%`
（`82.29--90.00%`）；但 ID non-inferiority 与 joint-stress liveness 仍需按
预注册 gate 判定，不能直接打开 confirmation。完整行为、分 block paired contrast
和五段 latency 见 `docs/PHASE58_CALIBRATION_REPORT.md`。

process-isolated matched-prefix 已完成 9 methods × 3 seeds，统一 Torch 1/1
线程；结果保留于 `results/phase58_isolated_runtime/aggregate.json` 与其
TensorBoard。该视图用于公平比较尾延迟，不是部署吞吐保证。

## 6. 下一步

1. M1 known-delay delayed-MPC 与 M6 QDR+synchronous evaluator contract、独立
   QDR checker、三 seed M0--M8 calibration 和 isolated runtime 已完成；
2. 自动执行 P58-5a gate 审计，并核验每个 block 的 failure taxonomy、queue age、
   fallback、QDR prefix/suffix coverage 与 tube 来源；
3. 只有 calibration gate 通过，才使用 80 个 confirmation mirror groups；
4. confirmation 只运行一次且配置冻结；locked-diagnostic 在 confirmation 之前
   保持关闭；
5. 如 gate 失败，冻结为 stress-conditional/negative result，不通过调参改写结论。
