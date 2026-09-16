# Phase 81：Nominal-plus 过渡验证与三 seed DAgger 复核

## 结论

Phase 81 保持 Phase 78 的目标行为合同不变：目标初始优先远离最近障碍物，
不要求主动变道，不要求穿越中心障碍物；目标机动仍使用有限时域候选、边界/障碍物
可行性过滤以及速度、加速度、转向角速度和 jerk 限制。该阶段只把命令噪声从
Nominal 的 `0.020` 轻微提高到 `0.025 m/s`，并使用独立生成的 holdout，作为
Nominal 到更高难度之间的过渡验证。

所有结果均来自 development validation，`locked_test_used=false`。local CBF 仍是
经验过滤器，不构成 R-CLBF-QP 或 robust CBF-QP 安全证明。

## 数据与训练合同

- 训练池：900 场景（Easy/Nominal/过渡 Hard 各 300），450 个 mirror groups；
  训练场景哈希为 `ddd9bdfb83e5a52b820de94332804281b4a743d12d6ac37456b5deaf5a5c4e74`。
- 独立 holdout：900 场景、450 个 mirror groups；哈希为
  `293f4147e1b8fe76bfe8b87b6040b80325c4368dcbd668b8a93faeab1e35c7e5`。
- holdout 的 public-belief route calibration 接受 `267/300` 个过渡 Hard 场景，
  接受率为 `89%`；模型评估分母固定为这 267 个已标定场景。
- 每个 seed 收集 `192` 条质量门控 DN-MPC teacher 示范，并执行两轮闭环 DAgger，
  每轮 12 个 episode；训练器仅保留 safe capture、无碰撞、无越界、非 timeout 的
  初始示范。
- 三个训练 seed 为 `813601/813602/813603`，每个 seed 均产生独立 checkpoint、
  manifest 和 recovery 数据；未使用 locked-test。

三组初始示范收集统计如下：

| seed | 接受示范 | 尝试次数 | 拒绝数 | DAgger round 1 | DAgger round 2 |
|---:|---:|---:|---:|---:|---:|
| 813601 | 192 | 220 | 28 | 12/12 safe | 9/12 safe |
| 813602 | 192 | 219 | 27 | 12/12 safe | 11/12 safe |
| 813603 | 192 | 216 | 24 | 11/12 safe | 10/12 safe |

## 独立 holdout 闭环结果

每个 checkpoint 在同一 267 个 holdout 场景上运行完整 250 步；不是训练器内部的
40-episode smoke。

| seed | episodes | safe capture | collision | boundary | timeout | mean min clearance (m) |
|---:|---:|---:|---:|---:|---:|---:|
| 813601 | 267 | 92.51% | 6.37% | 6.37% | 1.12% | 0.781 |
| 813602 | 267 | 92.13% | 7.12% | 7.12% | 0.75% | 0.785 |
| 813603 | 267 | 92.51% | 6.74% | 6.74% | 0.75% | 0.780 |
| **pooled** | **801** | **92.38%** | **6.74%** | **6.74%** | **0.87%** | **0.782** |

以 episode 为单位的 bootstrap 95% 区间为：

- safe capture：`[90.51%, 94.26%]`；
- collision：`[4.99%, 8.49%]`；
- boundary：`[5.12%, 8.49%]`；
- timeout：`[0.25%, 1.62%]`。

预先设定的 promotion gate 为 safe capture ≥ `70%`、collision ≤ `5%`、boundary
≤ `5%`、timeout ≤ `10%`。三个 seed 的 safe capture 和 timeout 达标，但三个 seed
的 collision/boundary 点估计均略高于 5%，因此 Phase 81 过渡候选判定为
**Nominal-plus safety gate No-Go**，不开放 Hard、Stress 或 locked-test。

## 延迟

以下是三 seed 分位数的均值，运行于 CPU 单线程诊断环境；100 ms 不是硬门槛，但完整
报告 p50/p95/p99：

| 模块 | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---:|---:|---:|
| route intent | 1.326 | 61.435 | 91.787 |
| recurrent actor | 0.322 | 0.484 | 0.638 |
| local CBF | 0.938 | 1.205 | 1.759 |
| total | 2.623 | 62.844 | 93.305 |

平均捕获时间为 `8.135 s`。当前主要问题不是计算延迟，而是少量危险前缀下的
闭环安全恢复。

## 与 Phase 79 的关系及“效果变差”的解释

Phase 79 conservative reference 在固定 84 个有效 Nominal 场景、3 seed、252 个
episode 上为 `94.05%/3.97%/3.97%/1.98%`（safe capture/collision/boundary/timeout）。
Phase 81 pooled holdout 为 `92.38%/6.74%/6.74%/0.87%`。safe capture 只下降
`1.67` 个百分点，主要变化是安全失败率上升，而不是捕获能力大幅崩溃。

两组结果不能视为严格 paired regression，原因是 Phase 81 使用了独立 holdout、
更大的场景分母和 `0.025 m/s` 命令噪声；Phase 79 使用固定有效 Nominal 集。此前
出现的 `73.03%` 是 12 条初始示范和 120-step smoke，不能与完整 250-step 结果
比较。Phase 79 曾出现的 `34.52%` 还包含 raw route base 与 safety-filtered
bootstrap base 不一致的实现问题，修复后已恢复到 94.05%。

为区分场景分布与策略变化，进一步将 Phase79 conservative checkpoint 放到完全
相同的 267 个 holdout、相同 Phase81 执行配置和完整 250 步上运行。该对照不是
Phase79 原始正式 reference 的替代，只是同 holdout diagnostic：

| checkpoint | episodes | safe capture | collision | boundary | timeout | total p50/p95/p99 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Phase79 conservative reference | 267 | 94.01% | 4.87% | 4.87% | 1.12% | 2.645/64.063/95.811 |
| Phase81 Nominal-plus actor | 267 | 92.51% | 6.37% | 6.37% | 1.12% | 2.604/61.970/91.639 |

因此，在同一 holdout 上 Phase81 相对该 reference 的 safe capture 下降约
`1.50` 个百分点，collision/boundary 各上升约 `1.50` 个百分点。结合三 seed
Phase81 pooled 结果，当前退化应归因于两部分：独立 holdout/轻微噪声变化带来的
分布变化，以及 Nominal-plus 训练中 actor 对危险前缀恢复的负迁移；不能再把全部
差异归因于 smoke 截断或旧的 base-action 合同错误。

本阶段的可核查诊断是：

1. 三 seed 的 holdout 结果高度接近，说明当前 6%--7% 的碰撞/越界不是单 seed
   偶然波动；
2. timeout 只有 `0.87%`，因此当前主要瓶颈不是仿真时域不足；
3. 初始 teacher 质量门控和 DAgger 已经提供了足量数据，但第二轮仍出现安全失败，
   说明恢复前缀覆盖仍不足；
4. local CBF 能限制危险动作，但它是经验投影，可能在接近障碍物或队形冲突时
   无法同时保证安全和捕获，不能被解释为形式安全证书；
5. DN-MPC 在本实验中主要提供 teacher/recovery 标签，部署时执行的是 route base
   + recurrent residual actor + local CBF，因此仍存在 teacher 到 actor 的近似误差。

## 结果文件

- 配置：`configs/phase81_nominal_plus_transition.yaml`、
  `configs/phase81_dnmcp_dagger_residual_nominal_plus.yaml`；
- 聚合结果：
  `results/phase81_dnmcp_dagger_residual_nominal_plus_holdout_aggregate.json`；
- 三个训练 checkpoint：
  `results/phase81_dnmcp_dagger_residual_nominal_plus_formal_seed813602/`、
  `results/phase81_dnmcp_dagger_residual_nominal_plus_formal_seed813603/`，以及
  `results/phase81_dnmcp_dagger_residual_nominal_plus_seed813601/`；
- 三个 full holdout 结果目录：
  `results/phase81_dnmcp_dagger_residual_nominal_plus_full_seed813601/`、
  `...full_seed813602/`、`...full_seed813603/`。

## 后续决策

当前不继续增加难度。下一步应在不改变目标合同的前提下，固定 Phase 81 holdout，
做失败前缀归因和同场景 Phase 79 reference 对照；优先补充 obstacle-near、
inter-agent-separation 和 local-CBF projection 后的 recovery 数据。只有在新的
Nominal/过渡验证中 collision 和 boundary 同时回到 5% 以下，才考虑 Hard；Stress
和 locked-test 继续关闭。
