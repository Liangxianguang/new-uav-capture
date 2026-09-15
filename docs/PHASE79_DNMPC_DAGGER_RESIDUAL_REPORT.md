# Phase 79：DN-MPC Teacher + DAgger Recovery Residual Actor 结果

## 结论

Phase79 的正式 Nominal 复核没有通过 promotion gate，当前 checkpoint 不能替代
Phase78 reference。完整 250 步验证得到：

| 方法 | 场景数 | safe capture | collision | boundary | timeout |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase78 三 seed recurrent BC（加权合计） | 252 | 59.92% | 18.25% | 6.35% | 21.83% |
| Phase79 residual actor（seed 791501） | 84 | 34.52% | 65.48% | 9.52% | 0% |

Phase79 的结果是单 seed，不能据此声称三 seed 的最终统计结论；但在同一批 84 个
有效 Nominal 场景、相同 Phase78 目标合同下，碰撞已显著增加，方向上已经足以判定
当前训练配置为 No-Go。旧 checkpoint 和结果目录均保留。

## 协议与数据

- 目标合同仍是 Phase78 obstacle-avoidance-first：不要求主动变道或穿过障碍物，
  关闭 intentional reverse lane change，目标初始优先远离最近障碍物；
- 训练和评估只使用 development calibration/validation，`locked_test_used=false`；
- 初始示范复用已保留的 `64+64+64=192` 条 Phase78 接受示范；
- 完成两轮闭环 DAgger，每轮 12 个 episode，共 24 个 actor-visited episode；
- DAgger recovery 标注从第一轮的 537 个步骤降至第二轮的 414 个步骤；聚合数据共
  16,247 帧，其中 recovery 标记 951 帧（5.85%）；
- 当前 teacher 只用于 DAgger 标签，部署评估实际运行的是 public-belief route base
  + recurrent residual actor + local CBF，而不是 DN-MPC teacher 本身。

正式训练产物：

```text
results/phase79_dnmcp_dagger_residual_seed791501_final/
```

完整 250 步复核产物：

```text
results/phase79_dnmcp_dagger_residual_nominal_full_seed791501/
```

## Conservative candidate：三 seed Nominal 复核

为修复 Phase79 的 safety-filtered base 合同问题，并在 delayed/noisy execution 下保留
安全裕度，新增独立配置
`configs/phase79_dnmcp_dagger_residual_conservative.yaml`，只改变 validation 配置
中的经验 `safety_margin=1.0` 和 `residual_scale_mps=0.5`；Phase78 目标合同、192 条
bootstrap 示范和两轮 DAgger 流程保持不变。三个 seed 均使用 84 个固定有效 Nominal
场景和完整 250 步复核：

| seed | safe capture | collision | boundary | timeout | mean min clearance (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 791601 | 94.05% | 3.57% | 3.57% | 2.38% | 0.785 |
| 791602 | 94.05% | 3.57% | 3.57% | 2.38% | 0.768 |
| 791603 | 94.05% | 4.76% | 4.76% | 1.19% | 0.786 |
| **合计（252 episodes）** | **94.05%** | **3.97%** | **3.97%** | **1.98%** | **0.780** |

三个 seed 均通过 Nominal gate（safe capture ≥70%、collision ≤5%、boundary ≤5%、
timeout ≤10%）。这证明了当前 conservative Phase79 candidate 在 Nominal
development validation 上具有稳定拦截能力，但不等于 Hard、Stress 或 locked-test
已经通过。

三 seed 的完整聚合结果保存在：

```text
results/phase79_dnmcp_dagger_residual_conservative_nominal_aggregate.json
```

统一延迟报告（各 seed 分位数的均值，CPU 诊断）为：

| 模块 | p50 (ms) | p95 (ms) | p99 (ms) |
| --- | ---: | ---: | ---: |
| route intent | 2.40 | 147.69 | 216.04 |
| actor | 0.53 | 1.17 | 1.75 |
| local CBF | 1.75 | 3.85 | 4.24 |
| total | 4.78 | 150.65 | 218.95 |

这组结果仍只使用 local CBF 经验过滤器，不构成 R-CLBF-QP 或 robust CBF-QP
安全证明。

## 为什么看起来比之前差很多

### 1. 初次结果被 `--max-steps 60` 截断

训练脚本末尾的快速验证使用了 `--max-steps 60`，即最多 6 秒。该输出为
14/84 safe capture（16.67%），不能与 Phase78 的 250 步结果直接比较。Phase78
成功轨迹的平均捕获时间为 16.76 秒，很多成功本来就不可能在 6 秒内发生。

使用完整 250 步复核后，Phase79 仍只有 29/84 safe capture（34.52%），所以截断
解释了第一轮数字过低，但不是退化的全部原因。

### 2. residual 的 base-action 合同没有完全对齐

Phase78 接受示范中的 `actions` 来自 safety-filtered route controller。Phase79
bootstrap loader 将这些动作同时作为 `base=target`，因此初始 residual 标签为零。
但是 Phase79 闭环 rollout 和部署评估把未经 safety filter 的 route action 作为
`base_action`，只在 residual 相加之后再调用一次 local CBF。也就是说，训练时 actor
学习的 base 分布与部署时输入的 base 分布不同；初始 192 条数据并没有真正告诉 actor
如何修正当前部署 base。这是本次退化最重要的实现级原因。

### 3. recovery 信号太稀疏且动作差距过大

24 个 DAgger episode 中，第一轮只有 2/12 safe capture，第二轮 5/12 safe capture；
其余主要在早期 safety failure 终止。有效 recovery 只有 951/16,247 帧，但
teacher-base 的 recovery gap 平均约 3.11 m/s、最大约 8.03 m/s，而 actor 的
residual scale 只有 2.5 m/s。当前数据更像少量大幅纠偏标签，不足以学习连续、可执行
的恢复策略，且容易使 residual 在早期就把队形推入碰撞状态。

### 4. 当前组合不是 Phase16 的完整部署组合

Phase16 的 94.81% 结果是在闭环中直接运行 distributed delayed DN-MPC；Phase79
目前是“DN-MPC 只提供训练标签，评估时不运行 DN-MPC”。因此不能把 Phase79 residual
actor 的结果解释为完整 DN-MPC 组合的结果。与 Phase78 比较时，Phase79 仍然退化，
但与 Phase16 比较则属于不同部署路径。

### 5. 当前只完成单 seed，不能用来宣称稳定性

Phase79 只有 seed `791501` 的正式 checkpoint。三 seed 复核尚未开始，且在 base 合同
修复前没有继续打开 Hard、Stress 或 locked-test 的必要。

## 当前性能与延迟

完整 250 步 Phase79 复核的平均最小净空为 `0.0916 m`，平均捕获时间为 `4.739 s`；
捕获时间偏小是因为 55 个 episode 很早以 safety failure 结束，不代表拦截速度更快。

各模块延迟（CPU、开发验证诊断）为：

| 模块 | p50 (ms) | p95 (ms) | p99 (ms) |
| --- | ---: | ---: | ---: |
| route intent | 0.72 | 173.34 | 224.26 |
| actor | 0.52 | 1.16 | 1.79 |
| local CBF | 1.66 | 3.66 | 4.02 |
| total | 2.98 | 176.83 | 226.99 |

100 ms 不是硬门槛；p95/p99 受 CPU 调度影响，不能直接作为部署吞吐声明。local CBF
仍然只是经验过滤器，不构成 R-CLBF-QP 或 robust CBF-QP 安全证明。

## 后续修复顺序

1. 先统一 base-action 合同：训练、DAgger 和部署均使用同一版本的
   safety-filtered route base，或重新收集同时保存 raw-base/safe-base/teacher-action
   三元组的 192 条示范；
2. 将 recovery 样本按 episode 和失败前缀分层重采样，增加中后期队形恢复样本，避免
   只学习早期碰撞前的巨大修正；
3. 在 validation 上先做 residual scale `0.5/1.0/1.5` 的预注册小矩阵，并增加
   residual smoothness/动作变化惩罚；不得查看或调参 locked-test；
4. 分别报告 route-base residual、DN-MPC teacher、完整 DN-MPC deployment 三条路径，
   不再把 teacher-label 训练结果与完整 planner 闭环混称；
5. 修复后至少完成三个 seed 的 Nominal gate：safe capture ≥70%、collision ≤5%、
   boundary ≤5%、timeout ≤10%，并完整报告 route/predictor/planner/QDR/safety/total
   的 p50/p95/p99；Nominal 未通过前不运行 Hard、Stress 或 locked-test。
