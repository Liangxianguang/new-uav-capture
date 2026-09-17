# 下一阶段实验 TodoList：目标边界合同修复与有效闭环验证

> 更新时间：2026-09-17  
> 当前阶段：Phase 85 规划  
> 前提：任务定义要求逃逸目标始终处于世界有效区域内。目标触及世界边界属于目标生成器或环境合同违规，而不是防守策略失败。

## 0. 当前问题判断

Phase 72 的显式归因表明，CBF 开启时：

- [ ] 56/60 个 episode 出现目标边界接触。
- [ ] 防守方边界接触为 0/60。
- [ ] 目标障碍物碰撞为 0/60。
- [ ] 关闭 CBF 后才额外出现防守方队形分裂和负净空。

因此下一阶段的首要目标是：先保证目标轨迹在整个评估时域内合法、可执行、不会自行触碰世界边界，再重新评价防守策略。

不要继续把目标非法越界 episode 作为防守策略失败样本，也不要在包含这类失败标签的数据上继续增加 BC/DAgger 训练。

## 1. 总体实验门控

### 1.1 所有阶段都必须遵守

- [ ] 不读取、修改或调参 locked-test。
- [ ] 保留历史失败结果，不覆盖原始结果目录。
- [ ] 所有新场景使用新的 manifest、scene hash、配置快照和 source hash。
- [ ] 镜像场景必须作为完整 mirror group 放在同一 split。
- [ ] 训练、验证、external holdout 不能按窗口随机划分。
- [ ] 不在目标非法越界的旧 episode 上重新训练策略。
- [ ] 不把 target_boundary_violation 计入防守方 collision 的主指标。
- [ ] local CBF 继续标记为经验过滤器，不写成形式化安全证明。

### 1.2 新的 episode 分类

每个 episode 必须区分：

- [ ] safe_capture
- [ ] defender_physical_collision
- [ ] defender_boundary_violation
- [ ] target_obstacle_violation
- [ ] target_boundary_violation
- [ ] timeout

同时记录：

- [ ] target_invalid_episode
- [ ] defender_safety_failure
- [ ] task_valid_for_policy_evaluation
- [ ] first_target_boundary_violation_step
- [ ] first_defender_boundary_violation_step
- [ ] minimum_target_boundary_clearance_m
- [ ] minimum_defender_boundary_clearance_m
- [ ] target_command_clipped
- [ ] target_candidate_fallback_count
- [ ] target_candidate_rejection_reason

主表必须同时报告三种分母：全部原始 episode、目标合法 episode、专家接受的目标合法 episode。不能只报告专家接受子集。

## 2. Phase 85：目标边界合同审计与实现修复

### 2.1 审计当前执行语义

重点检查：

- [ ] 世界边界是主体中心边界，还是已经考虑机体半径后的有效边界。
- [ ] 位置更新、速度裁剪、动作裁剪和边界裁剪的先后顺序。
- [ ] _enforce_world_bounds 是否在目标已经越界后才进行裁剪。
- [ ] 目标越界后是否仍被允许继续产生捕获统计。
- [ ] target_boundary_margin、route boundary buffer 和场景生成器 buffer 是否使用同一语义。
- [ ] 目标预测 horizon 是否覆盖 command delay、maneuver hold 和执行惯性。
- [ ] x、y、z 六个边界方向是否都有独立测试。

重点文件：

- src/encirclement3d/pursuit_env.py
- src/encirclement3d/showcase.py
- scripts/generate_phase82_nominal_plus1_scenes.py
- 当前 Phase 84 场景生成和专家评估脚本

### 2.2 定义统一目标安全集

目标有效区域使用世界边界减去 target radius 和 target margin。即使目标被建模为点，也必须保留明确的 target_boundary_margin，不能直接使用世界边界。

- [ ] 将目标安全集写入统一配置字段。
- [ ] 环境、场景生成器、目标候选 rollout、专家 evaluator 共用同一安全集函数。
- [ ] 目标候选的每个中间点和终点均检查安全集。
- [ ] 对速度、加速度、jerk 和 command delay 做 forward reachable tube 检查。
- [ ] 候选若无法在有限 horizon 内保持安全集内，必须被拒绝。
- [ ] 所有候选不可行时执行 inward boundary-recovery fallback，并记录原因。
- [ ] fallback 仍不可行时，场景标记为 invalid，不进入训练池。
- [ ] 不再只依赖当前时刻的 boundary repulsion。

### 2.3 固定环境处理语义

推荐语义：

- [ ] 场景生成阶段进行有限时域目标可行性证书。
- [ ] 合法 episode 中目标不应触及边界。
- [ ] 若运行时仍触及边界，记录合同失败并立即终止。
- [ ] 该 episode 不作为策略能力的有效分母。

如果保留硬裁剪用于诊断：

- [ ] 首次越界时记录 target_boundary_violation。
- [ ] 裁剪后的后续捕获不能算作 safe_capture。
- [ ] 裁剪后的 episode 不能作为正常训练标签。
- [ ] 只能作为目标生成器压力测试单独报告。

### 2.4 Phase 85 测试

- [ ] 靠近每个边界且速度朝外的目标。
- [ ] 高速目标经过一个 command delay 后仍处于安全集内。
- [ ] maneuver horizon 内无候选可行时能正确拒绝场景。
- [ ] boundary-recovery 会降低速度并转向内部。
- [ ] 目标越界后不会被记为 safe_capture。
- [ ] 目标边界违规不会污染 defender boundary 指标。
- [ ] 镜像场景的目标可行性标签保持对称。
- [ ] 延迟、噪声和不同 target speed 下的证书字段一致。

Phase 85 完成条件：

- [ ] 新增单元和回归测试全部通过。
- [ ] 旧 Phase 72--84 结果目录未被覆盖。
- [ ] 安全集、margin、fallback 语义可由配置快照复现。

## 3. Phase 86：重新生成目标合法场景池

### 3.1 Split 和规模

建立独立的新场景池，不修改 Phase 78--84 数据：

- [ ] development_calibration：至少 300 episodes / 150 mirror groups。
- [ ] development_validation：至少 300 episodes / 150 mirror groups。
- [ ] external_holdout：至少 300 episodes / 150 mirror groups。
- [ ] 资源不足时可先用每个 block 100 episodes / 50 mirror groups 做 smoke。
- [ ] 所有 split 的完整目标轨迹通过全时域合法性证书。
- [ ] 各 split 使用不同 layout seed。
- [ ] 所有镜像成员放在同一 split。

### 3.2 单因素 block

先拆分因素，不要一开始叠加所有困难条件：

- [ ] nominal_repaired
- [ ] target_speed
- [ ] obstacle_near
- [ ] formation_tight
- [ ] route_exercise
- [ ] execution_delay
- [ ] command_noise

每个场景记录目标初始和全时域最小边界净空、最大速度/加速度/jerk、目标障碍物最小净空、防守方初始最小机间距、route intent、bypass route 数量、target boundary certificate hash 和 mirror group id。

### 3.3 场景质量门槛

- [ ] 目标边界违规率为 0%。
- [ ] 目标障碍物碰撞率为 0%。
- [ ] full-horizon target certificate 通过率为 100%。
- [ ] public-belief expert acceptance 至少 80%。
- [ ] oracle expert acceptance 至少 80%。
- [ ] public-belief 与 oracle 差异单独报告。
- [ ] route-exercise block 不能只报告 direct route。

如果 block 不达标，只修场景合同或目标候选生成器，不降低 acceptance gate，不直接训练 defender，不访问 locked-test。

## 4. Phase 87：专家和基线重新验证

在新场景池上先运行基线，不训练新模型：

- [ ] oracle route expert。
- [ ] public-belief route expert。
- [ ] Phase83 Hard-1 checkpoint。
- [ ] Phase79 conservative checkpoint 作为历史参考。
- [ ] local CBF on/off 诊断。

分别报告 target_invalid_rate、target_boundary_rate、target_obstacle_rate、defender_collision_rate、defender_boundary_rate、safe_capture_rate、timeout_rate、route intent、route-exercise rate、首个失败时间步和 route/actor/safety/total 延迟。

Phase 87 通过条件：

- [ ] oracle 和 public-belief 的目标非法率均为 0%。
- [ ] 目标非法率与防守方失败率分开存储。
- [ ] 至少一个 repaired nominal block 的 public-belief acceptance 达到 80%。
- [ ] 不再出现目标边界失败被聚合为 defender collision 的错误。

## 5. Phase 88：只使用合法轨迹重新训练策略

### 5.1 数据清理

- [ ] 删除 target_invalid_episode=true 的完整 episode。
- [ ] 删除目标边界违规后的 recovery 标签。
- [ ] 保留目标非法样本作为诊断数据，不作为正向 expert action。
- [ ] recovery 标签只来自 defender safety failure、formation separation、obstacle proximity 和 timeout prefix。
- [ ] 保存 base_action_gap、base_cbf_correction、teacher_cbf_correction 连续值。

### 5.2 训练协议

- [ ] 至少 192 条合法、质量门控的初始示范。
- [ ] 2--3 轮闭环 DAgger/recovery。
- [ ] 训练使用与部署完全一致的 safety-filtered base-action 合同。
- [ ] calibration sweep residual scale 0.25/0.35/0.50。
- [ ] 不在 external holdout 或 locked-test 上收集 DAgger。
- [ ] 三个独立训练 seed。
- [ ] 保存 checkpoint、manifest、source hash、配置、recovery dataset 和 TensorBoard。

### 5.3 Nominal promotion gate

每个 seed 必须满足 safe capture >=70%、defender collision <=5%、defender boundary <=5%、timeout <=10%、target invalid rate =0%。

如果 safe capture 低但目标非法率为 0，进入策略/规划失败归因。如果目标非法率大于 0，返回 Phase 85/86，不调 actor。

## 6. Phase 89：Hard、Stress 和 route-exercise 验证

只有 Phase 88 Nominal 通过后才打开：

- [ ] Hard-1 重新验证。
- [ ] Stress-1 target-speed block。
- [ ] Stress-1 obstacle-near block。
- [ ] Stress-1 formation-tight block。
- [ ] route-exercise block。

每个 block 使用相同的三 seed checkpoint、相同的 manifest、相同的 target validity certificate，并逐 block 统计。

promotion gate：

- [ ] 每个 seed safe capture >=70%。
- [ ] 每个 seed defender collision <=5%。
- [ ] 每个 seed defender boundary <=5%。
- [ ] 每个 seed timeout <=10%。
- [ ] target invalid rate =0%。
- [ ] 报告 per-seed、pooled 和 95% CI。

如果目标合法但策略失败，才进入防守策略改进。如果目标再次越界，停止该 block 的策略结论并返回目标合同修复。

## 7. Phase 90：执行扰动与安全层实验

目标合同修复并通过 Hard 后，才重新评估执行扰动。

### 7.1 单因素矩阵

- [ ] nominal execution。
- [ ] command delay 1/2 steps。
- [ ] command noise 0.025/0.04/0.08 m/s。
- [ ] velocity tracking error。
- [ ] drag/mass randomization。
- [ ] pending-command queue。

第一轮不要同时叠加 delay、noise、tracking、drag 和 mass。

### 7.2 安全层对照

- [ ] no safety layer，作为失败诊断。
- [ ] local CBF，作为当前经验基线。
- [ ] queue-aware execution projection。
- [ ] certified fallback / emergency braking。
- [ ] robust CBF-QP，继续作为诊断分支。

记录 current-state certificate、next-state certificate、swept-volume certificate、post-execution robust-state rate、QP infeasible 次数、fallback 次数、queue authority、filter p50/p95/p99，以及 target invalid 与 defender failure 的分离结果。

只有目标合法率为 100% 且 defender safety failure 同时达标，才考虑升级为主要安全结果。

## 8. Phase 91：方法归因消融

在修复后的同一 validation block 上比较：

- [ ] route base + local CBF。
- [ ] route base without CBF。
- [ ] residual actor + local CBF。
- [ ] residual actor without CBF。
- [ ] DN-MPC teacher + local CBF。
- [ ] current GRU/DN-MPC reference，若执行合同兼容。

回答四个问题：

1. 性能提升来自 route intent 还是 residual actor？
2. CBF 是降低碰撞，还是把碰撞转成 timeout？
3. DAgger recovery 是否改善危险前缀恢复？
4. 目标合同修复后，策略是否仍然稳定？

Phase 15/16 已没有显示稳定的 S4 主分支优势，因此该阶段不优先继续扫描 S4/SSM 骨干。

## 9. Phase 92：external holdout 和一次性 locked-test diagnostic

只有 Phase 89、90、91 完成后：

- [ ] 冻结模型、配置、manifest 和 source hash。
- [ ] 在 untouched external holdout 上运行三 seed。
- [ ] 同时报告全部合法场景和专家接受子集。
- [ ] external holdout 之后不再调参。
- [ ] external holdout 达标后，才考虑一次性 locked-test diagnostic。
- [ ] locked-test 结果不再用于修改目标边界合同。

最终主结果必须包含目标合法率、safe capture、defender collision、defender boundary、timeout、最小净空、target/defender first-failure step、route/predictor/planner/safety/total 延迟、三 seed 结果和 paired bootstrap CI。

## 10. 失败分支决策树

### A. 目标仍然越界

- [ ] 停止策略训练和 promotion。
- [ ] 检查安全集、候选 rollout、command delay 和裁剪顺序。
- [ ] 新场景不进入训练池。
- [ ] 只运行目标合同单元测试和 expert feasibility audit。

### B. 目标合法，但防守方碰撞/越界

- [ ] 固定目标轨迹和场景。
- [ ] 运行 CBF on/off、residual scale 和 execution projection 对照。
- [ ] 按 formation、obstacle、queue、boundary 分层采样 recovery。
- [ ] 不把更换 S4/SSM 骨干作为第一反应。

### C. 目标合法且安全，但大量 timeout

- [ ] 按目标速度、route intent、观测延迟和 candidate age 分层。
- [ ] 比较 prediction refresh、DN-MPC horizon 和 route replanning interval。
- [ ] 检查预测误差是否传递为 interception delay。
- [ ] 记录 planner/route p95/p99，不只看平均捕获率。

### D. Nominal 通过但 Stress 失败

- [ ] 只对失败因素做单因素 calibration。
- [ ] 不把所有压力因素同时加入训练。
- [ ] 重新估计 public-belief expert acceptance。
- [ ] 只有目标合法且专家可行时，才收集新的 recovery 数据。

## 11. 当前明确禁止的实验

- [ ] 不在旧的目标边界违规 episode 上继续训练。
- [ ] 不把 target boundary violation 重新命名为 defender collision。
- [ ] 不只报告专家接受场景的 safe capture 而隐藏 raw pool。
- [ ] 不在目标合同未修复前打开更高 Stress 或 locked-test。
- [ ] 不继续无目的扫描 S4、SSM、QDR 参数来掩盖目标合同问题。
- [ ] 不在执行扰动 gate 失败时训练 R-CLBF。
- [ ] 不把速度级理想执行结果写成真实飞行安全结果。

## 12. 最终交付物

- [ ] Phase 85 目标边界合同审计报告。
- [ ] Phase 85 单元测试和回归测试报告。
- [ ] Phase 86 三 split 合法场景 manifest、README 和 hash。
- [ ] Phase 87 oracle/public-belief 专家聚合 JSON 和报告。
- [ ] Phase 88 三 seed 合法数据训练产物和 promotion report。
- [ ] Phase 89 Hard/Stress 分 block 闭环结果。
- [ ] Phase 90 execution-aware safety audit。
- [ ] Phase 91 residual/CBF/teacher 消融结果。
- [ ] Phase 92 external holdout 和一次性 locked-test diagnostic。
- [ ] 更新 docs/EXPERIMENT_STATUS_AND_NEXT_TODOLIST.md，明确旧目标边界失败是合同诊断，不是当前策略主结论。

## 13. 近期实际执行顺序

1. 完成 Phase 85 目标边界语义审计和测试。
2. 生成 Phase 86 nominal_repaired 小规模 smoke，确认目标非法率为零。
3. 生成完整 calibration、validation 和 external holdout。
4. 运行 Phase 87 专家和当前 checkpoint 基线。
5. 若专家通过，使用合法轨迹重新收集 192 条示范并完成三 seed DAgger。
6. 先通过 repaired Nominal，再逐步打开 Hard、Stress 和 route-exercise。
7. 最后进入执行扰动、安全层和 external holdout 确认。

当前最重要的成功标准不是再得到一个更高的 capture 数字，而是证明：

> 在目标全程合法、场景物理可行且执行合同明确的条件下，防守策略仍能跨三个 seed 保持低碰撞、低越界和可复现的 safe capture。
