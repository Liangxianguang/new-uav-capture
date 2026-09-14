# Phase 61：QDR prefix-recovery authority 单变量消融报告

## 1. 目的与结论

Phase 60 的 failure taxonomy 指出，当前 QDR 的主要 liveness 瓶颈是
immutable prefix 之后的长期 exhaustion，而不是未解释的 planner solver failure。
Phase 61 在冻结的 development-calibration 场景上，只改变 QDR prefix-recovery
authority，比较：

1. `immutable`：已经进入执行队列的 prefix 不允许重写；
2. `replace_nonexecuting`：只替换尚未执行的队列后缀；
3. `flush_pending`：清空 pending 队列并从当前状态重新建立候选。

三种策略均未改善预注册的 liveness gate，因此本阶段为**可复现的负消融**，不
进入 confirmation 或 locked-test。最重要的结果是：恢复动作虽然确实被触发，
但没有降低 exhaustion；相反，`replace_nonexecuting` 和 `flush_pending` 的
safe capture 均显著低于 immutable。当前推荐保留 `immutable` 作为 nominal
QDR 合同，并把 recovery 仅作为显式诊断信号，不把“允许重写队列”当作默认修复。

本报告不把 local CBF 称为 R-CLBF-QP，也不宣称 forward invariance、闭环安全
证明或真实飞控安全保证。

## 2. 冻结的实验合同

| 项目 | 设置 |
|---|---|
| 场景 | Phase 60 calibration selection，`120 episodes / 60 mirror groups` |
| 场景 hash | `1b178f151f5dbaee3af94542caab62584de64ccf45213ca48efb54aece0bff24` |
| predictor | Diagonal-SSM checkpoint，seeds `727201/727202/727203` |
| planner | M6 queue-aware synchronous delayed DN-MPC |
| candidate budget | fixed `K=1`，`adaptive_k=false` |
| sensing/execution | action delay 4，message delay 2，dropout 0.05，command noise 0.08 m/s |
| QDR | queue-aware rollout + queue-aware safety projection，sampling steps 8 |
| safety | `local_cbf`，仅经验过滤器 |
| 线程/设备 | CPU，Torch intra/inter-op `1/1` |
| 分析 split | `validation_selection` / development calibration |
| 统计 | 3 seeds；episode 结果按 mirror group 做 hierarchical bootstrap，10,000 samples |

每种 authority 均运行 `3 seeds × 120 episodes = 360` 次闭环 episode；总计
`1,080` 次 episode。每个运行均保存 root/method config、source hashes、episode
JSONL、step JSONL、summary 和 TensorBoard event files。

## 3. 闭环结果

### 3.1 主要 episode 指标

数值为三种 predictor seed 的均值；括号为按 mirror group 的 95% bootstrap CI。

| authority | safe capture | collision | boundary | timeout | mean capture time (s) | mean min clearance (m) |
|---|---:|---:|---:|---:|---:|---:|
| immutable | 86.67% [81.39, 91.11] | 1.11% [0.28, 2.50] | 1.11% [0.28, 2.50] | 12.22% [8.06, 17.22] | 7.989 [7.150, 8.851] | 0.486 [0.469, 0.504] |
| replace_nonexecuting | 78.33% [72.78, 83.33] | 1.94% [0.28, 4.44] | 0.56% [0, 1.94] | 19.72% [15.28, 24.17] | 7.858 [6.947, 8.778] | 0.476 [0.451, 0.497] |
| flush_pending | 81.39% [75.56, 86.94] | 1.67% [0.28, 3.89] | 0.28% [0, 1.11] | 16.94% [11.67, 21.94] | 8.405 [7.019, 9.858] | 0.473 [0.449, 0.496] |

相对 immutable 的配对结果为：

- `replace_nonexecuting` safe capture `-8.33 pp [-12.22, -4.72]`，timeout
  `+7.50 pp [+3.61, +11.94]`；
- `flush_pending` safe capture `-5.28 pp [-10.83, -0.56]`，timeout
  `+4.72 pp [-0.28, +10.83]`。

因此两种恢复策略都没有通过“保持捕获性能并降低超时”的目标。collision 的
配对区间均跨零，不能声称恢复策略改善碰撞；clearance 反而分别下降约
`0.011 m` 和 `0.013 m`，对应的 95% CI 均低于零。

### 3.2 QDR recovery 与 exhaustion 诊断

| authority | recovery request/apply step rate | mean prefix admissible | mean suffix admissible | mean suffix exhaustion step rate | max exhaustion streak (跨 seed 最大值) |
|---|---:|---:|---:|---:|---:|
| immutable | 0 / 0 | 94.50% | 85.69% | 22.02% | 93 |
| replace_nonexecuting | 11.30% / 11.30% | 88.70% | 80.18% | 23.89% | 285 |
| flush_pending | 10.04% / 10.04% | 89.96% | 80.92% | 24.59% | 290 |

恢复调用在实现层面确实生效，但 prefix/suffix 可行性没有改善，且最大
exhaustion streak 从 immutable 的 `93` 步上升到 `285/290` 步。三组的
episode-level exhaustion 均接近 `100%`，说明“曾经发生过 exhaustion”本身
不是有辨识力的失败标签；论文和后续实验应继续使用 timeout、最大连续
exhaustion streak、prefix-recoverable/unrecoverable 等联合归因。

## 4. 延迟分位数

所有延迟均为 step-level pooled measurements，单位为 ms；按要求同时报告
`p50/p95/p99`。100 ms 不是硬门槛。

| authority | predictor p50/p95/p99 | planner p50/p95/p99 | QDR/tube p50/p95/p99 | safety p50/p95/p99 | total p50/p95/p99 |
|---|---:|---:|---:|---:|---:|
| immutable | 9.82 / 13.74 / 16.40 | 15.67 / 21.71 / 25.37 | 1.23 / 1.86 / 2.60 | 0.79 / 1.24 / 1.63 | 30.52 / 39.34 / 45.01 |
| replace_nonexecuting | 9.88 / 13.86 / 16.49 | 15.72 / 21.73 / 25.41 | 1.20 / 1.82 / 2.50 | 0.79 / 1.24 / 1.63 | 30.58 / 39.48 / 45.28 |
| flush_pending | 9.73 / 14.22 / 17.80 | 15.56 / 21.97 / 26.88 | 1.19 / 1.91 / 2.68 | 0.78 / 1.26 / 1.67 | 30.23 / 40.26 / 50.10 |

恢复 authority 没有造成主要的计算延迟爆炸；失败主要表现为闭环 liveness 和
队列一致性，而不是 p95 超过某个实时阈值。flush_pending 的 total p99 略高，
但该差异不是本阶段的主要判据。

## 5. Gate 判定与可解释性

预注册 gate：timeout rate 上限 `5%`、相对 timeout 增量上限 `5 pp`、最大
exhaustion streak `24` 步、unexplained planner failure `1%`。本阶段在前两个
以及 exhaustion streak 上均失败：

- immutable timeout `12.22%`，最大 streak `93`；
- replace_nonexecuting timeout `19.72%`，最大 streak `285`；
- flush_pending timeout `16.94%`，最大 streak `290`。

所以本阶段不能支持“recovery authority 修复了 QDR liveness”的论文结论。
它支持的较弱而诚实的结论是：在当前 fixed-K、delay/noise 和 local-CBF
合同下，允许修改未执行队列或清空 pending 队列会引入队列一致性代价；恢复
策略应先解决 prefix/suffix 的可行性与切换语义，再考虑性能晋级。

## 6. 可复现工件与下一步

配置和代码已提交到仓库；本阶段结果目录被 `.gitignore` 忽略，但保留在本地
用于复核：

- 配置：`configs/phase61_qdr_recovery_authority_ablation.yaml`；
- 聚合：`results/phase61_qdr_recovery_authority_aggregate.json`；
- 原始运行：`results/phase61_qdr_recovery_seed{727201,727202,727203}_{immutable,replace_nonexecuting,flush_pending}/`；
- 每个 method 目录内含 `episodes.jsonl`、`summary.json` 和 TensorBoard；
- QDR 形式化/时间索引 checker 继续沿用 Phase 60，并不因本次 recovery
  消融而变成安全证明。

后续只建议做以下工作：

1. 冻结 immutable 为 nominal reference，不在本 calibration 上继续扫描
   recovery 阈值；
2. 增加逐步的 prefix-recoverable → suffix-admissible 转移审计，区分
   “恢复请求被拒绝”“恢复后仍不可行”和“恢复导致队列不一致”；
3. 在全新的 calibration manifest 上单独验证 bounded queue replacement，
   明确版本号、队列 token 和执行器 ACK 后再考虑 confirmation；
4. 继续把 local CBF 写成 empirical filter，把 robust CBF-QP/R-CLBF-QP
   写成 diagnostic No-Go，不把本阶段结果当作安全证书。

