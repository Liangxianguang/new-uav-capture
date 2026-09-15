# Phase 76：路线感知防守方重训练与闭环 pilot

更新时间：2026-09-15

## 结论先行

Phase 76 完成了第一套真正可执行的重训练方案，但 pilot 未通过闭环性能门槛。
路线教师能够生成无碰撞的高质量示范，然而只用“路线特征 + 低层循环行为克隆”训练出的
策略在 200 个 development-validation 场景上只有 `0.5%` safe capture，碰撞率为
`95.5%`。因此该 checkpoint 不能替代现有模型，也不能支持“有效拦截”或论文正向结论。

这次失败是有用的边界证据：离线动作拟合已经变好，但策略仍然受到闭环 covariate shift
影响；下一步应该做真正的闭环 DAgger/恢复训练，而不是继续单纯增加 epoch 或扫描学习率。

## 新训练方案

本阶段实现了 `PublicBeliefRouteIntentController`，其输入边界为：

- 延迟、带噪声的 team target belief；
- 公开的形状感知障碍几何和队友状态；
- 对目标尚未进入探测范围时使用公共的中央 rally prior，不读目标真实位置；
- 每 8 步重选一次左绕、右绕、上绕候选路线，最短保持 6 步；
- 对每一段 3-D 线段做精确障碍/边界清除检查；
- 对速度命令施加最大加速度、jerk 和转向角速度限制，并经过 local CBF 经验过滤器。

为避免左右绕行动作在 MSE 中互相平均，训练观测在原有 63 维 local observation 后追加
7 维路线条件：下一航点相对向量 3 维、路线 one-hot 3 维、belief-blind 标志 1 维。
低层 GRU 只学习在该路线条件下跟踪航点和保持捕获编队。

训练分布包含双侧镜像、3--5 个混合障碍、目标 crossing、`flee_persistence`/
`s_curve`/`adaptive_maneuvering` 模式，以及 2-step execution delay 和 `0.04 m/s`
command noise。示范质量门槛为 safe capture 且无碰撞；被拒绝的专家轨迹不进入正样本。

## 已完成实验

| 实验 | 数据/训练 | 结果 |
| --- | --- | --- |
| Route-aware BC v1 | 48 个合格示范；180 次尝试；12 epoch；`3e-5` | generic online validation safe capture `0%` |
| 延长训练对照 | 同一示范集；60 epoch；`3e-5` | generic online validation safe capture `0%`，collision `50%` |
| 从零训练学习率对照 | 同一示范集；40 epoch；`5e-4` | 离线 action MSE `0.1978`；generic validation safe capture `0%`，collision `25%` |
| 正式 development validation | 200 场景 / 100 mirror groups；checkpoint seed `761201`；2-step delay + `0.04 m/s` noise + local CBF | 见下表 |

正式 200 场景结果：

| 指标 | 结果 |
| --- | ---: |
| safe capture | `0.5%` |
| capture event | `3.0%` |
| collision | `95.5%` |
| boundary violation | `90.0%` |
| timeout | `1.5%` |
| target crossing | `1.0%` |
| mean capture time（未捕获按 25 s 计） | `24.5465 s` |
| mean minimum clearance | `0.5281 m` |

闭环单步时延分位数如下。100 ms 不作为硬门槛，但 p95/p99 尾部必须保留并解释：

| 阶段 | p50 | p95 | p99 |
| --- | ---: | ---: | ---: |
| route-intent feature | `0.103 ms` | `87.496 ms` | `131.945 ms` |
| recurrent actor | `0.580 ms` | `0.935 ms` | `1.450 ms` |
| local CBF filter | `1.711 ms` | `3.862 ms` | `4.437 ms` |
| total | `2.509 ms` | `90.344 ms` | `135.241 ms` |

路线规划 p95/p99 的尾部来自首次/重新生成路线的几何检查，说明下一版需要缓存障碍
union、路线候选和增量更新；不能只报告平均时延。

## 与既有结果的关系

旧实验结果和 checkpoint 均保留，没有被覆盖。Phase 74 在同一 200 场景 development
validation 上的 nominal GRU + local CBF safe capture 为 `10.83%`，但该配置与本阶段的
执行条件、目标控制和策略输入不同，不能写成严格配对比较。Phase 76 使用了更强的
2-step delay + `0.04 m/s` noise 诊断，结果仍然明显失败；这足以判定当前重训练方案
不具备 promotion 资格，但不能把差异归因于某一个因素。

本阶段只使用 development-validation 场景；`external_holdout` 和 locked-test 均未使用。
local CBF 是经验过滤层，本文档不宣称 R-CLBF-QP 或 robust CBF-QP 形式化安全证明。

## 下一步 TodoList

1. 在 `development_calibration` 上实现闭环 DAgger：让当前策略真实滚动，教师在每个策略
   状态查询 route/waypoint/action；按 75%/50%/25% expert mix 做 3 轮，失败回合只作为
   recovery 诊断，不作为正向安全标签。
2. 将训练目标改为多任务 loss：route-intent cross-entropy、role/waypoint loss、
   action Huber 和 action-smoothness；保留 public-belief-only 输入边界。
3. 先用 20--40 个 calibration 场景做快速 gate：safe capture ≥50%、collision ≤15%、
   timeout ≤20%；未通过则不再增加模型容量。
4. 修复路线特征时延：缓存固定障碍 union，只有目标 belief/队友状态变化时更新下一航点；
   重新报告 route-intent/actor/safety/total 的 p50/p95/p99。
5. 通过快速 gate 后，用 3 个训练 seed 在至少 300 个新的镜像互斥场景上确认；确认门槛为
   safe capture ≥70%、collision/boundary ≤5%、timeout ≤10%、target crossing ≥95%。
6. 若 DAgger 三种子仍低于 50% safe capture 或 collision 高于 15%，停止继续堆叠学习器，
   将路线规划 + DN-MPC 作为主控制器，把学习器降级为候选路线评分/预测模块。
