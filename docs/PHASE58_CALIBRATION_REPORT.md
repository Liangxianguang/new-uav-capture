# Phase 58：延迟规划校准矩阵报告

## 1. 结论摘要

Phase 58 development-calibration 已完成：同一冻结场景 manifest 上，3 个
predictor seeds、M0--M8 九种方法全部完成闭环；另行完成了 process-isolated、
fixed-thread 的运行时基准。当前结果支持一个清晰但有边界的结论：

> Queue-aware delayed rollout（M6/M7）在 delay-noise 和 joint-stress 条件下
> 显著降低碰撞并提高 safe capture，但在 ID block 上 non-inferiority 仍未被
> 充分证明，且异步组合增加 timeout/尾延迟。因此目前只能称为
> **stress-conditional empirical improvement**，不能称为全场景优势或安全保证。

本阶段尚未打开 development-confirmation 或 locked-diagnostic，也没有使用
locked-test 调参。`local_cbf` 仅作为经验动作过滤器；本报告不宣称
R-CLBF-QP、CLBF certificate、forward invariance 或安全证明。

## 2. 实验范围与可追溯性

| 项目 | 值 |
| --- | ---: |
| Calibration episodes | 160（80 个 mirror groups） |
| Predictor seeds | 727201 / 727202 / 727203 |
| Methods | M0--M8 |
| Blocks | ID、delay-noise、communication-execution、joint-stress-transfer |
| Bootstrap unit | predictor seed + mirror group（upper/lower 先配对） |
| Scene manifest SHA-256 | `b65bf733d9cca3f9ae2f8e601ea726b8b6b4d9a7ec9dbea3642b7031a8e12e45` |
| QDR audit | `q=0/2/4/8/10`，正常合同通过，double-delay probes 全部拒绝 |
| Safety wording | local CBF = empirical filter only |

源数据和 TensorBoard 均保留在本地 `results/`（不提交 Git）：

```text
results/phase58_calibration_aggregate.json
results/phase58_calibration_by_block.json
results/phase58_calibration_by_block/tensorboard/
results/phase58_isolated_runtime/aggregate.json
results/phase58_isolated_runtime/tensorboard/
results/phase58_calibration_gate_audit.json
```

可复现入口：

```text
scripts/aggregate_phase58_calibration.py
scripts/benchmark_phase57_isolated_runtime.py
scripts/check_phase58_qdr_time_index.py
scripts/audit_phase58_calibration_gates.py
```

## 3. 三 seed 总体闭环结果

数值为三 seed 的 mirror-group 聚合均值；方括号为 hierarchical bootstrap 95%
区间。百分比中的 collision、boundary 和 timeout 是 episode-level failure
比例，不能用 safe capture 替代它们。

| Method | Safe capture | Collision | Boundary | Timeout | Capture time (s) | Min clearance (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M0 current-state delayed-MPC | 55.62% [49.38, 61.67] | 44.38% [38.33, 50.42] | 7.71% [4.79, 10.83] | 0.00% [0.00, 0.00] | 2.33 | 0.189 |
| M1 known-delay delayed-MPC | 62.29% [56.67, 68.12] | 37.71% [32.08, 43.54] | 11.67% [7.71, 15.62] | 0.00% [0.00, 0.00] | 2.36 | 0.243 |
| M2 fixed-tube MPC | 55.83% [49.79, 61.88] | 43.96% [37.92, 50.21] | 7.50% [4.58, 10.62] | 0.21% [0.00, 0.83] | 2.28 | 0.193 |
| M3 queue-aware tube MPC | 74.79% [68.75, 80.62] | 24.38% [19.17, 29.79] | 12.71% [9.38, 16.46] | 0.83% [0.00, 2.29] | 4.96 | 0.262 |
| M4 synchronous distributed MPC | 54.58% [48.54, 60.63] | 44.79% [38.75, 50.83] | 7.50% [4.58, 10.62] | 0.63% [0.00, 1.46] | 2.13 | 0.171 |
| M5 asynchronous distributed MPC | 50.83% [45.00, 56.67] | 48.54% [42.71, 54.37] | 6.25% [3.54, 9.38] | 0.63% [0.00, 1.46] | 2.13 | 0.115 |
| M6 QDR + synchronous distributed | 88.75% [85.00, 92.08] | 9.38% [5.83, 12.92] | 6.87% [3.96, 10.00] | 1.88% [0.63, 3.33] | 6.23 | 0.438 |
| M7 QDR + asynchronous distributed | 86.25% [82.29, 90.00] | 10.21% [7.08, 13.96] | 6.46% [3.33, 10.00] | 3.54% [1.46, 6.25] | 6.33 | 0.480 |
| M8 fixed-K=8 QDR | 75.00% [68.12, 81.04] | 24.17% [17.92, 31.25] | 13.75% [9.17, 18.54] | 0.83% [0.21, 1.88] | 5.16 | 0.279 |

M6 相对 M0 的总体 safe-capture 点差为 +33.13 pp，但这里是跨 block 的描述性
汇总；正式 promotion 仍必须按预注册 block gate 和 paired CI 判定。

## 4. 校准闭环的五段 latency

下表是所有 calibration 闭环 step 的 pooled p50/p95/p99，单位 ms。total 是
独立 wall-clock 测量，不由各组件相加得到。

| Method | Predictor | Planner | QDR/tube | Safety | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 | 4.33/6.47/9.88 | 3.29/4.73/7.04 | 0.00/0.00/0.00 | 0.51/0.76/1.53 | 8.87/12.73/17.91 |
| M1 | 4.66/7.31/11.49 | 3.52/5.34/8.15 | 0.00/0.00/0.00 | 0.56/0.77/1.77 | 9.57/14.53/20.16 |
| M2 | 4.68/8.28/12.50 | 3.58/6.23/8.89 | 0.00/0.00/0.00 | 0.56/1.08/1.92 | 9.73/16.29/21.95 |
| M3 | 4.79/7.64/11.42 | 9.56/14.38/19.85 | 0.74/1.37/2.57 | 0.56/0.84/1.79 | 17.75/26.14/34.51 |
| M4 | 4.71/8.16/12.30 | 5.18/8.93/12.57 | 0.00/0.00/0.00 | 0.56/1.04/1.93 | 11.38/18.41/24.83 |
| M5 | 4.70/7.86/12.13 | 5.05/8.41/11.96 | 0.00/0.00/0.00 | 0.56/0.93/1.79 | 11.18/17.90/23.71 |
| M6 | 4.79/7.68/11.56 | 11.11/17.46/23.55 | 0.80/1.37/2.47 | 0.57/0.88/1.80 | 19.38/28.91/37.62 |
| M7 | 4.82/8.10/11.81 | 10.50/16.78/22.45 | 0.76/1.41/2.67 | 0.57/0.99/1.88 | 18.76/28.55/37.62 |
| M8 | 4.75/7.74/11.69 | 9.43/14.47/19.78 | 0.73/1.26/2.36 | 0.56/0.86/1.74 | 17.34/26.35/34.62 |

顺序为 `p50/p95/p99`。100 ms 不是硬门槛；本阶段所有 total p95 均低于该
参考线，但这不能替代安全、活性和过程隔离的公平比较。

## 5. 四个 block 的行为响应

以下为点估计，用于显示因素响应面方向；完整 bootstrap 和每个组件的分位数
见 `results/phase58_calibration_by_block.json`。

| Block | Method | Safe capture | Collision | Timeout | Total p95 (ms) |
| --- | --- | ---: | ---: | ---: | ---: |
| ID | M0 | 96.7% | 3.3% | 0.0% | 10.6 |
| ID | M3 | 90.8% | 9.2% | 0.0% | 23.9 |
| ID | M6 | 98.3% | 1.7% | 0.0% | 27.6 |
| ID | M7 | 99.2% | 0.8% | 0.0% | 25.5 |
| delay-noise | M0 | 36.7% | 63.3% | 0.0% | 11.3 |
| delay-noise | M3 | 56.7% | 41.7% | 1.7% | 26.9 |
| delay-noise | M6 | 72.5% | 24.2% | 3.3% | 28.8 |
| delay-noise | M7 | 71.7% | 22.5% | 5.8% | 28.2 |
| communication-execution | M0 | 89.2% | 10.8% | 0.0% | 13.0 |
| communication-execution | M3 | 80.0% | 19.2% | 0.8% | 24.9 |
| communication-execution | M6 | 97.5% | 2.5% | 0.0% | 29.1 |
| communication-execution | M7 | 94.2% | 5.8% | 0.0% | 28.4 |
| joint-stress-transfer | M0 | 0.0% | 100.0% | 0.0% | 14.9 |
| joint-stress-transfer | M3 | 71.7% | 27.5% | 0.8% | 27.8 |
| joint-stress-transfer | M6 | 86.7% | 9.2% | 4.2% | 29.4 |
| joint-stress-transfer | M7 | 80.0% | 11.7% | 8.3% | 30.5 |

关键 paired contrasts（candidate minus reference）为：

- ID：M6−M0 safe capture `+1.7 pp [-4.2, +6.7]`，未达到 `-2 pp`
  non-inferiority 下界；M7−M5 为 `+14.2 pp [8.3, 20.0]`，但这是 QDR 与
  asynchronous 的组合对照，不能归因于单一模块。
- delay-noise：M6−M0 safe capture `+35.8 pp [25.8, 46.7]`，collision
  `-39.2 pp [-50.8, -28.3]`；M7−M5 safe capture `+40.8 pp [29.2, 52.5]`。
- communication-execution：M6−M0 safe capture `+8.3 pp [1.7, 15.8]`，
  collision `-8.3 pp [-15.8, -1.7]`。
- joint-stress-transfer：M6−M0 safe capture `+86.7 pp [80.0, 92.5]`，
  collision `-90.8 pp [-96.7, -85.0]`，但 M6 timeout 为 4.2%；M7 timeout
  为 8.3%，超过预注册 5% liveness 上限。

因此本阶段不能打开 confirmation。自动 gate audit 的结果为：M6 的 ID
non-inferiority 为 No-Go（CI 下界 `-4.2 pp`），joint timeout-delta 的 CI
上界为 `+8.3 pp`，最大 exhaustion streak 为 `126` steps（阈值 24）；M7
相对 M6 的 isolated total-p95 reduction 为 `14.06%`（阈值 15%），其 ID
non-inferiority CI 下界为 `-3.3 pp`。M6 的 joint collision gate 和 timeout
point gate 虽然通过，但不足以抵消上述失败。完整机械化结果见
`results/phase58_calibration_gate_audit.json`，总体状态为 `NO-GO`。

最合理的当前表述是：**QDR 具有强烈的压力条件性收益，但本阶段不能把它升级
为经确认的论文主方法；应冻结为 conditional/negative calibration result，
除非另行设计并预注册新的修复实验。**

## 6. Process-isolated matched-prefix runtime

9 methods × 3 seeds 顺序运行，每个 seed 取 24 episodes、首 18 个 control steps，
Torch intra/inter-op threads 均为 1。下表为隔离视图的
`predictor/planner/QDR/safety/total` p50/p95/p99，单位 ms：

| Method | Predictor | Planner | QDR | Safety | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 | 4.54/7.24/11.72 | 3.45/5.26/8.35 | 0.00/0.00/0.00 | 0.54/0.76/1.46 | 9.26/13.94/19.97 |
| M1 | 4.42/6.21/9.13 | 3.32/4.77/7.47 | 0.00/0.00/0.00 | 0.52/0.75/1.37 | 9.01/12.57/18.19 |
| M2 | 4.36/6.63/11.24 | 3.32/4.76/8.47 | 0.00/0.00/0.00 | 0.51/0.75/1.49 | 8.99/13.26/20.99 |
| M3 | 4.43/5.90/8.57 | 8.85/10.66/13.78 | 0.52/0.69/0.99 | 0.52/0.66/1.15 | 16.13/19.51/25.34 |
| M4 | 4.43/5.87/7.58 | 4.78/5.93/7.23 | 0.00/0.00/0.00 | 0.52/0.64/0.92 | 10.49/12.77/15.15 |
| M5 | 4.51/7.39/10.49 | 4.76/7.61/10.41 | 0.00/0.00/0.00 | 0.53/0.84/1.60 | 10.59/16.42/22.85 |
| M6 | 4.60/7.62/12.02 | 10.64/16.44/22.80 | 0.53/0.96/1.52 | 0.53/0.99/1.72 | 18.16/27.87/37.76 |
| M7 | 4.64/6.30/9.96 | 10.06/14.11/22.10 | 0.53/0.72/1.34 | 0.54/0.77/1.66 | 17.54/23.95/34.68 |
| M8 | 4.50/5.97/7.73 | 8.91/10.80/15.10 | 0.51/0.65/1.13 | 0.53/0.69/1.36 | 16.20/19.63/26.01 |

异步的效率收益仍需谨慎：M7 的行为不差，但它是 QDR+async 组合；本阶段不
把组合结果解释成 asynchronous 单模块带来的 p95 加速。

## 7. QDR 形式化审计与安全声明

独立 checker 已验证：queue prefix 只推进到 `t+q`，candidate suffix 的首次
物理作用时刻为 `t+q+j`；固定 disturbance replay 下 full rollout 与 shifted
rollout 的位置/速度误差为 0；显式注入的 `t+q+2H` double-delay probes 均被
识别并拒绝。这证明的是 queue composition 和 time-index equivalence，不是
collision-free 或概率覆盖率。

`local_cbf` 的实际含义是对上层动作做局部经验过滤/回退。由于当前没有完成
连续时间动力学假设、QP 全局可行性、forward-invariance 证书和长时闭环证明，
本阶段不得使用以下措辞：`R-CLBF-QP safety proof`、`guaranteed safe`、
`CLBF certificate`、`forward invariant`。

## 8. 下一步实验计划

### P58-5a：calibration gate 审计（已完成，No-Go）

- [x] 用 `phase58_calibration_by_block.json` 自动判定 M6/M7 的 ID
  non-inferiority、joint collision、timeout、exhaustion streak 和多重比较；
- [ ] 补齐每个 block/method/seed 的 failure taxonomy、queue age、stale-plan、
  fallback 和 QDR prefix/suffix coverage 表；
- [ ] 对 tube 半径 `0.35 m` 完成 calibration-only 的来源、分项误差和 objective
  生效审计，禁止在 confirmation 上重新估计。

### P58-5b：confirmation 决策（当前关闭）

- [ ] 若 M6 通过 ID non-inferiority、stress collision 和 liveness，冻结配置后
  在全新的 80 mirror groups 上只运行一次 confirmation；
- [ ] 若 M6 未通过而 M7 通过，必须将论文主张改为 QDR+async 的组合条件性结果，
  不宣称 async 单模块加速；
- [x] 当前 gate 为 No-Go，冻结为 stress-conditional/negative result，不访问
  confirmation 或 locked-diagnostic。

### P58-6：只有 confirmation 通过才开放 locked-diagnostic

- [ ] 运行新 locked split；locked 只验证冻结配置，不允许调参；
- [ ] 生成 delay/noise response surface、失败案例视频和 QDR appendix；
- [ ] 复核所有 TensorBoard、JSONL、summary、manifest hash、source hash 和
  命令行，最后再准备论文主表。

## 9. 运行命令与统计真值

统计真值是 `summary.json`、`aggregate.json` 和 JSONL 原始记录；TensorBoard 是
可视化/审计入口。建议查看：

```powershell
python scripts\aggregate_phase58_calibration.py `
  --group "phase58=results\phase58_calibration_seed*_m0_m8" `
  --scenes results\phase58_calibration_selection\scenes.jsonl `
  --output results\phase58_calibration_by_block.json `
  --tensorboard-dir results\phase58_calibration_by_block\tensorboard
```
