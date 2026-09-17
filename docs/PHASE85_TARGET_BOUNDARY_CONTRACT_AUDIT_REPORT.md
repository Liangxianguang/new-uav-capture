# Phase 85：目标边界合同审计报告

> 日期：2026-09-17
> 范围：目标边界语义、目标候选 rollout、episode 分类和训练数据门控
> locked-test：未读取、未修改

## 1. 审计结论

旧实现存在两处会污染实验结论的语义缺口：

1. 目标候选的 `min_boundary` 使用物理世界边界距离，但可行性阈值来自另一套 `target_maneuver_feasibility_margin`，没有扣除配置的 `target_boundary_margin`。
2. 所有候选不可行时仍从不安全候选中选择 fallback，导致目标轨迹可能继续执行，却没有被明确标记为目标合同失败。

因此，目标触及有效边界应先归类为目标合同问题，不能直接作为 defender collision 或策略失败。

## 2. 修复后的合同

目标有效区域定义为：

```text
target_safe_lower = world_lower + target_radius + target_boundary_margin
target_safe_upper = world_upper - target_radius - target_boundary_margin
```

当前配置没有显式 `agents.target_radius` 时，目标按点模型处理，半径为 `0`，但 `target_boundary_margin` 始终保留。环境运行时、候选 rollout、边界恢复和指标记录均使用同一安全集函数。

- 候选 rollout 的每个 endpoint 都必须保持正的有效边界净空。
- 目标预测会离开安全集时，先尝试 inward boundary recovery。
- 没有安全的一步 fallback 时，episode 标记 `target_candidate_invalid` 和 `target_invalid_episode`。
- 运行时进入有效边界后，立即终止并标记 `target_boundary_violation`。
- 目标非法不会设置 defender `collision`；防守方碰撞和防守方边界越界分别由显式字段记录。
- 原始 `world_violation_steps` 和 `target_world_violation_steps` 保留，用于兼容历史诊断。

## 3. 新增结果字段

环境 `info` 现在同时提供：

- `target_invalid_episode`
- `target_boundary_violation`
- `target_obstacle_violation`
- `target_candidate_invalid`
- `task_valid_for_policy_evaluation`
- `defender_physical_collision`
- `defender_boundary_violation`
- `defender_safety_failure`
- `minimum_target_boundary_clearance_m`
- `minimum_defender_boundary_clearance_m`
- `target_command_clipped`
- `target_candidate_rejection_reason`

Phase 77 专家评估、Phase 79 DAgger 数据筛选和 execution evaluator 已改用这些字段。目标非法 episode 不再进入合法 teacher demonstration，也不再被计入 defender collision/boundary 主指标。

## 4. 测试证据

新增回归覆盖：

- x/y/z 六个世界边界方向的目标裁剪和计数。
- 向外目标命令的边界恢复。
- 使用有效目标安全集拒绝候选 rollout。
- 所有候选不安全时拒绝不安全 fallback。
- 目标非法与 defender collision 的指标分离。

验证结果：

- Phase 85 边界测试：`12 passed`
- 环境、专家、训练筛选和 execution 相关聚焦测试：`35 passed`
- 完整测试套件：`396 passed`

## 5. 尚未证明的内容

本阶段只修复和验证仿真环境中的目标合同与评价分离，不构成真实飞行安全证明，也不构成 robust CBF-QP 形式化安全证明。下一步必须在新 manifest 上运行 `nominal_repaired` smoke，确认目标非法率为零后，才允许生成合法训练池。
