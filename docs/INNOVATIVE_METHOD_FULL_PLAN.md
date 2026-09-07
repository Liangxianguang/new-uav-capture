# Mamba-SSM + Conditional Diffusion + DN-MPC + R-CLBF-QP 完整可行性计划书

> 版本：v1.0（2026-09-07）  
> 目标仓库：[Liangxianguang/new-uav-capture](https://github.com/Liangxianguang/new-uav-capture)  
> 适用基准：四架追捕无人机、一个高机动目标、三维障碍物、部分观测与通信延迟  
> 当前判断：整体方向为 **Conditional Go**。预测模块已有正式结果，但尚未证明完整的 DN-MPC、R-CLBF-QP 和端到端方法成立。

## 1. 先给结论

这套方法在工程上可以分阶段实现，但不能把三个创新点一次性堆叠后再看结果。当前最合理的研究路线是：

```text
统一动力学与实验协议
    -> 可校准的多模态预测
    -> 集中式 scenario min-max MPC 诊断
    -> 分布式 DN-MPC
    -> robust CBF-QP
    -> R-CLBF-QP 与独立证书检查
    -> 端到端锁定测试
```

当前三项创新的可行性判断：

| 方向 | 当前判断 | 主要原因 | 最小可发表版本 |
| --- | --- | --- | --- |
| SSM + 条件扩散预测 | 中高，Conditional Go | 已有稳定多模态候选和较低 minFDE，但 raw 候选不可执行，且平均提升低于 10% 门槛 | portable SSM diffusion + dynamics projection + coverage/energy/latency 审计 |
| 极小极大 DN-MPC | 中 | 场景候选可以直接进入滚动优化，但规划收益尚未验证，通信和求解失败是主要风险 | 预测候选驱动的集中式 risk-sensitive MPC；分布式作为后续实验 |
| R-CLBF-QP | 中低 | QP 过滤器可实现，完整闭环证明需要统一动力学、扰动界、离散时间不变性和可行性证明 | 可审计 robust CBF-QP；只有证书检查通过才升级为 R-CLBF-QP |
| 三者端到端组合 | 低到中 | 预测误差、求解延迟和安全保守性会累积 | 在困难场景安全捕获不劣于基线，并公开所有失败模式 |

### 1.1 当前正式证据

Phase 2 使用 `v3_multimodal` 数据集、三个训练种子和 locked-test。正式平均结果如下，数值来自 `docs/PHASE2_FORMAL_ANALYSIS_REPORT.md`：

| 模型 | minFDE | Energy score | 全轨迹 coverage | 可行候选比例 |
| --- | ---: | ---: | ---: | ---: |
| GRU raw | 0.6825 | 1.6402 | 0.9006 | 0.0782 |
| portable SSM diffusion raw | **0.6379** | 1.7242 | 0.8926 | **0.0000** |
| GRU projected | 0.7075 | 1.6456 | 0.8962 | 0.9884 |
| portable SSM diffusion projected | **0.6802** | 1.6918 | 0.8874 | **0.9971** |

结论：raw diffusion 相对 raw GRU 的 minFDE 改善为 6.54%，projected diffusion 相对 projected GRU 的改善为 3.87%，均未达到预注册的 10% 强门槛；但投影后的候选可行率达到 99.71%，说明它可以作为规划器的候选生成器继续诊断。预测单模块 p95 延迟约 21 ms，尚未包含 MPC 和安全层延迟。

### 1.2 必须避免的表述错误

- 当前实现是 `portable_diagonal_ssm`，不是官方 `mamba_ssm` CUDA kernel。论文中必须按实际实现命名；除非加入并验证真正的 Mamba block，否则不能直接宣称“官方 Mamba”。
- raw diffusion 候选不可直接执行。规划器只能消费 `dynamics-projected` 候选，并在接口中保留候选状态。
- 当前环境主要是速度级运动学，不是真实飞控动力学。仿真结果不能写成真实无人机安全保证。
- `uniform_uncalibrated` 只能表示候选均匀权重，不能解释为已经校准的 mode probability。
- 当前已有的是局部 CBF 投影，不是 R-CLBF-QP，也不是完整闭环形式化安全证明。
- 脚本化目标模式不能直接等同于完整零和博弈对手。必须加入自适应目标和未见策略测试后，才能讨论博弈泛化。

## 2. 研究问题、假设与成功等级

### 2.1 总研究问题

在目标不可完全观测、通信存在延迟和丢包、目标进行高机动逃逸的三维围捕任务中，能否通过：

1. 用 SSM 编码长时序观测，并用条件扩散生成多条带覆盖信息的目标轨迹；
2. 将这些候选轨迹放入极小极大滚动时域优化，显式考虑追逃风险；
3. 用鲁棒屏障 QP 将规划动作投影到可验证安全集；

同时提高捕获性能、困难场景鲁棒性和安全可审计性？

### 2.2 可检验假设

| 编号 | 假设 | 最小验证 |
| --- | --- | --- |
| H1 | 多模态候选比单轨迹点预测更能覆盖高机动和转弯目标 | minFDE、energy score、coverage、候选 spread、按模式结果 |
| H2 | 预测候选只有在被纳入风险敏感规划后，才会转化为围捕收益 | 同场景、同候选下比较 expected/worst-case/CVaR MPC 与 DynamicEncirclement |
| H3 | 极小极大目标在观测退化和策略切换条件下比平均代价目标更稳健 | delayed-noisy、丢包、未见目标策略的最坏候选指标 |
| H4 | 安全过滤能降低碰撞和边界违规，且不破坏捕获率 | 同一 nominal planner 输出下比较当前 CBF、robust CBF-QP、R-CLBF-QP |
| H5 | 统一动力学、显式 fallback 和低频规划/高频过滤可以满足控制周期 | p50/p95/p99 总延迟、求解成功率、fallback 后的安全指标 |

### 2.3 成功等级

不要把“网络训练成功”当作“方法成功”。本项目按以下等级判断：

| 等级 | 名称 | 必须证明的内容 | 可以形成的结论 |
| --- | --- | --- | --- |
| L0 | 工程可运行 | 数据、模型、规划器和过滤器可复现运行 | 工程实现完成 |
| L1 | 预测模块成功 | 多模态覆盖有价值、置信度可校准、候选物理可行 | 多模态轨迹预测贡献 |
| L2 | 规划模块成功 | 预测候选在不访问目标真值时提升困难场景围捕 | scenario MPC/DN-MPC 贡献 |
| L3 | 安全模块成功 | QP 约束可审计，明确假设下通过独立证书检查 | robust CBF-QP 或 R-CLBF-QP 贡献 |
| L4 | 完整方法成功 | L1-L3 且三种子端到端不劣于基线，满足实时与 fallback 要求 | 完整组合方法 |

如果某一等级失败，立即停在上一等级，保留已经证实的贡献，不为保留方法名称继续堆叠模块。

## 3. 现有代码结构与接口边界

当前仓库的主要实现边界如下：

| 组件 | 现有位置 | 现有职责 | 后续约束 |
| --- | --- | --- | --- |
| 三维围捕环境 | `src/encirclement3d/pursuit_env.py` | 状态、障碍物、观测、目标模式、step/reset、捕获与碰撞 | 必须成为 rollout 的唯一动力学真值源 |
| 现有控制器 | `src/encirclement3d/pursuit_controllers.py` | PurePursuit、PredictionPursuit、DynamicEncirclement、局部 CBF | 作为所有新方法的公平基线和 fallback |
| 预测模型 | `src/encirclement3d/prediction.py` | GRU、portable SSM diffusion、校准、投影和候选可行性 | planner 只依赖稳定的数据契约，不依赖模型内部张量 |
| 数据集 | `src/encirclement3d/trajectory_dataset.py` | 轨迹窗口、split、标签和几何上下文 | 禁止 episode 内窗口泄漏和 target truth leakage |
| 预测训练 | `scripts/train_prediction_models.py` | GRU/SSM diffusion 训练和 TensorBoard | 训练配置、归一化器和 source hash 必须落盘 |
| 预测评估 | `scripts/evaluate_prediction_models.py` | locked-test、coverage、可行性、延迟 | raw/projected 结果必须同时保留 |
| 规划评估 | 待新增 `scripts/evaluate_minimax_mpc.py` | planner 场景实验和统计 | 不得读取目标真值 |
| 安全层 | 待新增 `src/encirclement3d/safety_qp.py` | robust CBF-QP/R-CLBF-QP | 先实现可审计 QP，再考虑 learned CLBF |

### 3.1 统一数据契约

规划器和安全层之间必须使用明确的结构，建议至少包含：

```text
PredictionCandidateSet:
  trajectories: [K, H_p + 1, 3]
  candidate_weights: [K]
  score_kind: uniform_uncalibrated | calibrated_region
  dynamics_status: raw | projected
  conformal_radius_m: scalar
  source_model_hash: string
  timestamp_step: integer

PlannerOutput:
  nominal_actions: [N_defenders, 3]
  objective_value: scalar
  worst_case_cost: scalar
  solver_status: success | timeout | infeasible | numerical_error | fallback
  latency_ms: scalar
  fallback_reason: optional string

SafetyOutput:
  safe_actions: [N_defenders, 3]
  barrier_values: mapping
  residuals: mapping
  slack: mapping
  certificate_status: valid | invalid | not_checked
  solver_status: string
  latency_ms: scalar
```

### 3.2 统一主动力学

在进入安全证明前，训练 rollout、MPC rollout、安全 QP 和环境 `step()` 必须使用同一主模型。第一版可以继续使用速度级离散模型：

```text
p_i[t+1] = p_i[t] + dt * v_i[t]
v_i[t]   = clip(u_i[t], max_speed)
```

如果加入执行动态，则必须同时固定：动作延迟、速度跟踪时间常数、命令噪声、加速度上界和扰动集合。不能让预测器、MPC 和 QP 各自使用不同的 dynamics。

## 4. 总体执行顺序

以下是主线任务，所有任务按依赖顺序完成。编号前的复选框用于后续持续更新。

```text
P0 冻结协议
  -> P1 统一动力学与目标策略
  -> P2 关闭预测模块缺口
  -> P3 集中式 scenario min-max MPC
  -> P4 分布式 DN-MPC
  -> P5 robust CBF-QP
  -> P6 R-CLBF-QP 与证书
  -> P7 端到端集成
  -> P8 locked-test、统计和论文证据
```

不得跳过 P0/P1 直接训练联合模型；不得在 P5/P6 完成前写“闭环安全证明”。

## 5. P0：冻结基线、协议和仓库状态

### 5.1 任务清单

- [ ] 确认远程仓库为 `https://github.com/Liangxianguang/new-uav-capture.git`。
- [ ] 记录 Python、PyTorch、NumPy、SciPy、OSQP/CasADi 等版本。
- [ ] 运行完整测试并保存日志：`python -m pytest -q`。
- [ ] 重新运行当前 `DynamicEncirclementController + local CBF` 基线。
- [ ] 在相同场景种子下运行 RAW controller 和当前安全过滤 controller。
- [ ] 保存每个 episode 的 seed、目标模式、障碍物 profile、观测退化参数和 checkpoint hash。
- [ ] 固定 train/validation/locked-test 三个不重叠 episode seed block。
- [ ] 固定 locked-test 只允许在最终评估阶段读取标签。
- [ ] 预先登记 primary metric、secondary metric、停止条件和统计方法。
- [ ] 为每个大阶段建立独立 commit，不把用户已有的 `README.md` 或实验总结混入提交。

### 5.2 基线指标

- [ ] Cooperative Safe Capture rate。
- [ ] 普通 capture rate。
- [ ] collision rate。
- [ ] boundary violation rate。
- [ ] timeout rate。
- [ ] mean/median capture time。
- [ ] mean/worst minimum clearance。
- [ ] 总路径长度和控制变化量。
- [ ] CBF action correction norm。
- [ ] 控制周期 p50/p95/p99 延迟。
- [ ] 目标观测可见率、observation age、message age。

### 5.3 P0 验收条件

- [ ] 测试全通过。
- [ ] 同一 seed 重复运行的关键指标在预设误差内一致。
- [ ] 基线结果与仓库已有 V4/V5 证据没有未解释的大幅漂移。
- [ ] 所有输出均保存配置、seed、版本和 hash。

若基线不能稳定重现，P0 不通过，暂停创新模块。

## 6. P1：统一动力学、目标策略和数据协议

### 6.1 动力学与执行器

- [ ] 在 `src/encirclement3d/dynamics.py` 或环境内部形成唯一的离散动力学函数。
- [ ] 用随机动作逐步比较环境 rollout 与独立 dynamics rollout。
- [ ] 明确追捕机和目标的最大速度、最大加速度、动作延迟和执行噪声。
- [ ] 明确障碍物、边界、机间距离的几何定义和安全 margin。
- [ ] 明确捕获半径 `0.80 m` 仍是任务指标，不把它误写成碰撞半径。
- [ ] 对执行动态做 nominal、delayed-noisy、随机参数三组测试。
- [ ] 为任何仿真安全结论写出适用范围：速度级运动学、已知障碍几何和指定扰动范围。

### 6.2 目标策略

- [x] 保留 `flee_persistence`、`random_turn`、`s_curve`、`burst`、`boundary_escape`。
- [ ] 增加基于追捕机位置、速度和障碍物的自适应逃逸策略。
- [ ] 增加策略参数随机化，避免预测器记忆固定脚本。
- [ ] 增加至少一个训练未出现的 target policy 用于 locked generalization test。
- [ ] 增加用于规划诊断的 worst-case target policy，但只能在环境内部生成标签，不能进入 planner 输入。
- [ ] 记录 policy ID、随机种子和参数；不把真实 policy ID 泄露给观测。
- [ ] 验证所有 target trajectory 满足物理速度、加速度和边界限制。

### 6.3 数据集

- [x] 使用 episode 级切分，禁止同一 episode 的窗口跨 split。
- [x] 保存配置 hash、source hash、标签协议和几何上下文。
- [x] 检查输入、标签、参考位置、边界和障碍物信息均 finite。
- [ ] 生成 nominal、delayed-noisy、严重遮挡、消息丢失四种观测条件。
- [ ] 对每个 target mode 和 observation condition 统计样本量，避免某一模式主导总体指标。
- [ ] 完成人工抽检，确认未来 target label 没有通过观测字段泄漏到输入。
- [ ] 固定预测 horizon、MPC horizon 和时间步长的换算关系。

### 6.4 P1 验收条件

- [ ] 环境与独立 dynamics 在随机动作下逐步一致。
- [ ] 至少一个非固定规则 adversarial target 通过基础物理检查。
- [ ] nominal 与 delayed-noisy 数据可以由独立命令重新生成。
- [ ] 任意样本不存在 target truth leakage。

## 7. P2：SSM + 条件扩散预测模块

P2 的目标不是追求模型名字，而是验证“在可见信息约束下，多模态候选能否为规划提供额外有效信息”。

### 7.1 模型实现

- [x] 保留 GRU Gaussian baseline。
- [x] 保留 portable diagonal SSM diffusion 实现。
- [ ] 如果环境允许，再加入官方 `mamba_ssm` 或清晰可复现的 Mamba 对照；否则论文中使用 portable SSM 命名。
- [ ] 固定 token：belief position/velocity、confidence、age、队友摘要、障碍物摘要。
- [ ] 固定 team pooling 和 defender permutation 处理方式。
- [ ] 固定 padding/mask、reset 和 deterministic inference 行为。
- [ ] 增加 SSM deterministic head、GRU diffusion head、single-mode diffusion 三个消融。
- [ ] 测量 history length、horizon、候选数 `K` 和采样步数的扩展曲线。

### 7.2 预测输出与物理约束

- [x] 预测空间固定为相对 team belief reference 的未来位置位移。
- [x] 输出候选集合、候选状态和 `uniform_uncalibrated` 标记。
- [x] 对 raw 和 projected 候选分别计算速度、加速度、边界和障碍物可行性。
- [x] 使用 dynamics projection 修复速度/加速度不连续，不把 projection 当作安全证明。
- [ ] 将 projected candidate contract 接入 planner，并禁止 planner 读取 raw 候选。
- [ ] 在目标策略切换和观测退化时测量 coverage 退化是否可解释。

### 7.3 预测指标

- [x] ADE、FDE、minADE、minFDE。
- [x] energy score。
- [x] full-trajectory conformal coverage 和逐 horizon coverage。
- [x] 候选 spread 与 mode collapse 检查。
- [x] p50/p95/p99 单样本推理延迟。
- [ ] CRPS 或等价分布质量指标。
- [ ] nominal、delayed-noisy、丢包、未见策略的分层结果。

### 7.4 P2 验收条件

主门槛在新实验开始前冻结，建议为：

- [ ] projected diffusion 在全部主测试条件下 minFDE 相对 projected GRU 改善至少 10%，或在误差相当时 coverage 提高至少 5 个百分点。
- [x] 90% conformal 目标的经验 coverage 在 `0.85--0.95`，但未见策略仍需补测。
- [x] projected 候选可行比例不低于 95%；raw 候选不可执行时必须显式报告。
- [ ] 目标策略切换后，置信度随失配而退化，不能异常维持高置信度。
- [x] 单模型 p95 延迟低于 100 ms；总 planner/safety 延迟尚未验收。

当前状态为 Conditional Go。因此可以进入 P3 的集中式诊断，但不能宣称 P2 已完整通过，也不能直接宣称最终 DN-MPC 有效。

## 8. P3：集中式 scenario min-max MPC 诊断版

P3 用来回答最关键的转化问题：预测候选的改善是否真的变成围捕收益。第一版必须集中式、短 horizon、可解释，不能同时引入通信博弈和 learned safety certificate。

### 8.1 规划问题

给定 projected candidate set `Y = {y^k}`，求：

```text
U* = argmin_U max_k J(U, y^k)
```

建议使用速度级 rollout、短 horizon `H_m = 8`、控制保持 `3` 步。代价至少包含：

- 目标距离与终端捕获距离；
- 拦截点相对速度；
- 围捕角度、覆盖度和 blocker 位置；
- 障碍物净空、边界净空和机间距离；
- 控制量与控制变化量；
- solver slack 和 fallback 惩罚。

每个代价项必须记录单位、归一化和权重来源。先固定权重，再做小范围敏感性实验。

### 8.2 必须实现的三种风险模式

- [ ] expected-cost：候选代价均值。
- [ ] worst-case：候选代价最大值。
- [ ] CVaR：高风险候选尾部平均代价。

第一版只允许使用预测候选、观测 belief 和环境公开几何，禁止读取目标真值、未来真实位置和目标 policy ID。

### 8.3 求解器与 fallback

- [ ] 选择 Windows 环境稳定的 SciPy/OSQP/CasADi 方案，记录版本。
- [ ] 支持线性化动力学、短 horizon 和 warm start。
- [ ] 记录 solver status、objective、worst-case cost、最大约束残差和 latency。
- [ ] 对 timeout、infeasible、NaN、异常数值建立统一 fallback。
- [ ] fallback 首选当前 `DynamicEncirclementController + local CBF`，并记录触发原因。
- [ ] 所有 planner output 做 finite、shape、速度上限和动作范围检查。

### 8.4 P3 对照矩阵

| 编号 | 预测输入 | 规划器 | 安全层 | 用途 |
| --- | --- | --- | --- | --- |
| P3-A | 常速度 | DynamicEncirclement | local CBF | 当前基线 |
| P3-B | GRU projected | expected MPC | local CBF | 点/概率规划对照 |
| P3-C | SSM projected | expected MPC | local CBF | 预测输入效果 |
| P3-D | SSM projected | worst-case MPC | local CBF | 极小极大效果 |
| P3-E | SSM projected | CVaR MPC | local CBF | 尾部风险效果 |
| P3-F | oracle target | 同上 | local CBF | 仅作上限诊断 |

P3-F 只能作为上限，不能作为主方法结果。

### 8.5 P3 验收条件

- [ ] 在相同场景种子和相同 projected candidates 下，worst-case 或 CVaR 至少改善一个困难层指标。
- [ ] S3 随机混合障碍物或 S5 退化观测条件下，安全捕获率提升至少 5 个百分点，或者最坏候选 capture distance 降低至少 10%。
- [ ] collision rate 不高于基线超过 1 个百分点。
- [ ] solver success rate 不低于 99%，失败均进入记录过的 fallback。
- [ ] p95 planner latency 能够放入控制预算，或明确采用低频 planner + 高频 safety 架构。
- [ ] 规划器不读取 target truth。

若 P3 只有 oracle 输入有效，则判定为诊断工具，停止向 DN-MPC 扩展。

## 9. P4：分布式 DN-MPC

只有 P3 有明确收益后才执行 P4。P4 的创新点不是把集中式代码复制四份，而是证明有限通信下的分布式博弈规划仍保持可接受性能。

### 9.1 信息和通信契约

- [ ] 每架无人机只能使用自身状态、局部 belief、邻机摘要、障碍物局部信息和允许接收的消息。
- [ ] 明确共享内容：候选目标轨迹摘要、角色、局部控制序列或预测协方差。
- [ ] 固定通信频率、最大消息年龄、延迟分布和丢包率。
- [ ] 消息中不能包含真实未来目标状态。
- [ ] 通信异常时使用上一控制序列、局部 MPC 或 DynamicEncirclement fallback。

### 9.2 算法选择和实现

- [ ] 用集中式 oracle 建立性能上限。
- [ ] 用 sequential best-response 作为第一种分布式机制，先验证闭环可行性。
- [ ] 再与 consensus/ADMM 做小规模对照，不同时实现多个主算法。
- [ ] 固定每周期最大迭代次数和超时上限。
- [ ] 记录局部子问题 infeasible、未收敛次数、通信量和角色切换频率。
- [ ] 分析分布式输出和集中式 oracle 的代价差、捕获差和安全差。

### 9.3 P4 验收条件

- [ ] 分布式安全捕获率在主困难场景中高于当前 DynamicEncirclement，或至少不低于集中式结果 10 个百分点以上。
- [ ] 与集中式 oracle 的主要指标差距不超过 10 个百分点，超出时必须解释为通信限制并给出曲线。
- [ ] delayed-noisy 与丢包条件下性能下降不超过预注册阈值。
- [ ] 99% 以上控制周期有有效 planner 或记录过的安全 fallback。
- [ ] p95 总 planner latency 满足预算，或低频规划架构的端到端频率明确。

若 P4 不通过，保留 P3 的集中式结果；不能把“集中式场景 MPC”写成“分布式 DN-MPC”。

## 10. P5：robust CBF-QP 安全过滤层

P5 是 R-CLBF-QP 前的必要基线。先解决 QP 可行性、离散时间约束和执行误差，再讨论 learned CLBF。

### 10.1 安全集

至少定义：

- `h_obstacle`：追捕机到圆柱、箱体、墙体的净空；
- `h_boundary`：到三维世界边界的净空；
- `h_inter_agent`：追捕机之间的最小距离；
- optional `h_execution`：速度、加速度和动作变化约束。

目标本身不能被错误地加入排斥障碍集合，因为接近目标是捕获任务的一部分。

### 10.2 QP 形式

建议第一版使用：

```text
min_{u, slack} ||u - u_nom||_R^2 + rho * ||slack||_1/2
s.t. discrete barrier constraints
     speed/acceleration/action-change constraints
     robust margins for observation, delay and execution disturbance
```

- [ ] 名义动作来自 P3/P4 planner。
- [ ] 安全约束优先级高于性能约束。
- [ ] 对安全 slack 使用分层惩罚；任何 slack 都必须记录，不能静默掩盖违反。
- [ ] infeasible 时进入应急动作，并记录 certificate invalid。
- [ ] 输出每个 barrier 值、残差、slack、active 状态、修正量和 solver status。

### 10.3 鲁棒 margin

必须明确 margin 的来源：

- 理论动力学扰动上界；
- 观测误差和消息延迟上界；
- 预测误差的校准分位数；
- 执行器跟踪误差和 solver tolerance。

统计 conformal margin 可以作为经验鲁棒项，但不能自动等同于全状态安全证明。

### 10.4 P5 验收条件

- [ ] 相同 planner 输出下，碰撞率不高于当前 local CBF。
- [ ] 所有可行 QP 的最大 barrier residual 小于预设数值容差。
- [ ] infeasible 比例低于 1%，并且 fallback 的安全结果完整记录。
- [ ] p95 QP latency 满足高频执行预算。
- [ ] 在窄通道、障碍拐角、边界附近和机间近距离做专门压力测试。

如果 P5 只能证明“多数时候求解器找到较安全动作”，只能命名为 `robust CBF-QP safety filter`，不能升级为闭环形式化证明。

## 11. P6：R-CLBF-QP 和独立安全证书

P6 是风险最高的理论阶段。它的成功标准不是训练出一个 barrier 网络，而是证明网络、QP、动力学和实际执行之间的契约一致。

### 11.1 CLBF 任务

- [ ] 明确 learned CLBF 的输入状态、训练域和部署域。
- [ ] 生成安全状态、危险状态、边界附近和障碍拐角 hard cases。
- [ ] 训练时禁止访问 locked-test 场景标签。
- [ ] 检查 `V_theta`、gradient、Lie derivative/discrete residual 的范围。
- [ ] 给出网络梯度/Lipschitz 或可替代的数值误差上界。
- [ ] 对窄通道、延迟、命令噪声和执行动态做 Monte Carlo 验证。

### 11.2 证书检查器

新增独立检查脚本，不能复用训练时未经审计的内部判断：

- [ ] 重新计算每个 barrier 和 robust margin。
- [ ] 重新验证 safe action 的约束残差。
- [ ] 检查动作饱和、离散步长和 solver tolerance 的最坏影响。
- [ ] 检查理论扰动范围是否覆盖实际采样扰动。
- [ ] 对证书失效原因分类：QP infeasible、扰动越界、模型状态越界、数值残差、fallback。
- [ ] 每个 episode 输出 certificate valid/invalid 及原因。

### 11.3 形式化结论门槛

只有同时满足以下条件，才可以使用“在给定假设下的闭环安全保证”：

- [ ] 动力学方程与环境 `step()` 一致。
- [ ] 所有观测、延迟、执行扰动有明确有界集合。
- [ ] barrier 定义和 QP 约束可由独立代码检查。
- [ ] 可行 QP 的离散时间正向不变性推导完成。
- [ ] slack、饱和、solver tolerance 和 fallback 的作用已纳入证明或明确标记为证书失效。
- [ ] 独立数值检查覆盖所有预设 hard cases。

否则最终表述只能是“条件性 robust CBF-QP safety filter”。

## 12. P7：端到端集成

### 12.1 集成顺序

- [ ] 保持现有 `DynamicEncirclement + local CBF` 可运行。
- [ ] 接入预测器但只记录，不改变控制，确认 latency 和数据契约。
- [ ] 接入 projected candidate 到集中式 MPC。
- [ ] 加入 P4 分布式通信模型。
- [ ] 接入 P5 robust CBF-QP，并保留 local CBF fallback。
- [ ] 只有 P6 通过后才启用 R-CLBF-QP 主路径。
- [ ] 所有组件异常都进入可解释 fallback。
- [ ] 最后才考虑联合训练；主结果必须先来自冻结模块组合。

### 12.2 每一步运行时检查

- [ ] observation、candidate、planner output、safe action 均 finite。
- [ ] candidate shape、horizon、坐标系与 `dt` 一致。
- [ ] 记录 predictor/planner/QP 三段 latency。
- [ ] 记录 solver status、fallback reason、nominal-safe action difference。
- [ ] 记录 target truth 只存在于环境评估侧，不进入 planner。
- [ ] episode 结束时保存配置、seed、model hash、scene hash、代码 hash。

### 12.3 端到端消融矩阵

| 编号 | 预测 | 规划 | 安全层 | 目的 |
| --- | --- | --- | --- | --- |
| E0 | 常速度 | DynamicEncirclement | local CBF | 当前基线 |
| E1 | GRU | DynamicEncirclement | local CBF | 预测输入对照 |
| E2 | SSM diffusion projected | DynamicEncirclement | local CBF | 单独预测贡献 |
| E3 | SSM diffusion projected | expected MPC | local CBF | 平均风险规划 |
| E4 | SSM diffusion projected | worst-case MPC | local CBF | 极小极大规划 |
| E5 | SSM diffusion projected | CVaR MPC | local CBF | 尾部风险规划 |
| E6 | SSM diffusion projected | DN-MPC | local CBF | 分布式贡献 |
| E7 | SSM diffusion projected | DN-MPC | robust CBF-QP | 安全过滤贡献 |
| E8 | SSM diffusion projected | DN-MPC | R-CLBF-QP + fallback | 完整方法 |
| E9 | oracle target | DN-MPC | 同上 | 上限诊断，不作为主结论 |

所有配置必须使用相同场景 seed、相同目标策略、相同观测退化和相同 episode 数量。

## 13. P8：正式实验、统计和 locked-test

### 13.1 场景分层

- [ ] S1：单类障碍物、nominal observation。
- [ ] S2：固定混合障碍物。
- [ ] S3：随机混合障碍物。
- [ ] S4：窄通道、墙体、拐角等 hard geometry。
- [ ] S5：观测延迟、消息延迟、丢包和噪声。
- [ ] S6：目标策略切换和未见自适应策略。
- [ ] S7：执行延迟、命令噪声、速度和加速度扰动。

### 13.2 规划和安全指标

必须同时报告均值、标准差/置信区间和最坏分层结果：

- safe capture rate、ordinary capture rate；
- worst-case candidate capture rate；
- expected candidate capture rate；
- CVaR capture distance；
- capture time；
- collision rate、boundary violation rate、timeout rate；
- minimum clearance；
- role switching frequency；
- solver success、fallback、certificate invalid rate；
- predictor/planner/QP/total p50/p95/p99 latency；
- communication bytes、message age、未收敛次数。

### 13.3 三种子与统计规则

- [ ] 至少三个独立训练种子。
- [ ] 场景 seed 与训练 seed 分离。
- [ ] 每个主要配置使用完全相同的 episode seed。
- [ ] 报告绝对百分点、相对变化和置信区间。
- [ ] 对多配置比较说明多重比较或只预先指定 primary comparison。
- [ ] 不隐藏 timeout、solver failure、fallback 和 certificate invalid。
- [ ] locked-test 只能做一次正式主评估；调参只能用 train/validation。

### 13.4 最终主门槛

建议在正式锁定前冻结下列门槛：

| 模块 | 通过条件 |
| --- | --- |
| 预测 | projected minFDE 相对 projected GRU 至少改善 10%，或误差相当且 coverage 至少提高 5 个百分点 |
| 规划 | S3/S5/S6 安全捕获率至少提升 5 个百分点，或最坏候选 capture distance 降低 10% |
| 安全 | 碰撞率不增加超过 1 个百分点，QP 残差和可行性可审计 |
| 实时 | 总控制 p95 在预算内；否则必须有低频 planner + 高频 safety 的明确频率 |
| 稳定性 | 三种子方向一致，不依赖单一 seed 或单一场景 |
| 理论 | 结论中的每个假设、barrier、扰动界和离散时间条件都能对应到代码 |
| 复现 | 新机器按文档可重建配置、checkpoint、日志和主表格 |

## 14. Go / Conditional Go / No-Go 决策表

### 14.1 完整 Go

只有同时满足以下条件，才可把三个模块作为完整方法主线：

- [ ] 多模态预测通过误差、coverage、可行性和延迟审计；
- [ ] DN-MPC 在未见策略和观测退化下仍有收益；
- [ ] R-CLBF-QP 通过独立证书检查，且安全假设与代码一致；
- [ ] 三种子端到端安全捕获率不低于基线，主要困难场景有统计稳定增益；
- [ ] solver/fallback/总延迟满足部署预算；
- [ ] 没有未解释的 target truth leakage 或结果选择性报告。

### 14.2 Conditional Go

- 预测通过、集中式规划通过、分布式不稳定：论文主线写集中式 scenario risk-sensitive MPC。
- 规划通过、安全证明不完整：写 robust CBF-QP 和条件安全保证。
- 预测有效但整体提升不足：写“可校准多模态目标预测及其规划诊断”，不宣称完整闭环收益。
- 安全提高但捕获下降：报告安全-效率 Pareto，不只报告安全率。
- 官方 Mamba 依赖不可复现：使用 portable SSM，并在限制中说明。

### 14.3 No-Go

出现以下情况时，停止继续堆叠：

- 预测候选不优于常速度/GRU，且 coverage 也没有改善；
- projected 候选可行率低于 95%；
- MPC 只在 target truth 或 oracle mode 下有效；
- DN-MPC 在通信限制下频繁不收敛，fallback 也不能保持安全；
- QP 经常不可行，或安全违规无法解释；
- 端到端收益依赖单一 seed、单一目标模式或单一场景；
- 总延迟无法满足控制需求，也无法设计低频规划/高频过滤架构。

No-Go 后的回退方向：

```text
预测失败 -> 改为观测鲁棒/校准预测研究
规划失败 -> 保留预测器，研究 risk-sensitive centralized planning
分布式失败 -> 保留集中式 MPC，报告通信限制
CLBF 失败 -> 保留 robust CBF-QP，不声称形式化证明
端到端失败 -> 分模块报告，不宣称组合方法有效
```

## 15. 推荐新增文件与提交边界

### 15.1 推荐文件

```text
src/encirclement3d/dynamics.py
src/encirclement3d/minimax_mpc.py
src/encirclement3d/safety_qp.py
src/encirclement3d/safety_certificate.py
scripts/evaluate_minimax_mpc.py
scripts/evaluate_safety_qp.py
scripts/run_innovation_ablation.py
configs/innovation_mpc.yaml
configs/innovation_safety.yaml
configs/innovation_end_to_end.yaml
tests/test_dynamics_contract.py
tests/test_minimax_mpc.py
tests/test_safety_qp.py
tests/test_safety_certificate.py
docs/PHASE3_MPC_REPORT.md
docs/PHASE4_SAFETY_REPORT.md
docs/FINAL_INNOVATION_REPORT.md
```

### 15.2 每阶段提交建议

- `chore(protocol): freeze baseline and dynamics contract`
- `feat(mpc): add centralized scenario minimax planner`
- `feat(mpc): add distributed best-response DN-MPC`
- `feat(safety): add robust cbf qp filter`
- `feat(safety): add certificate checker and robustness audit`
- `feat(experiment): add end-to-end locked evaluation`
- `docs(research): publish final innovation feasibility report`

每个 commit 必须附带测试命令、配置文件、结果目录说明和已知限制。训练结果放在被忽略的 `results/` 中，报告中写出路径和 hash，不把大模型工件直接提交到 Git。

## 16. 时间和资源估计

以一名研究者、一张中等 GPU、当前代码基础为前提，建议使用 10-14 周，但门槛优先于日历：

| 周期 | 阶段 | 主要产物 |
| --- | --- | --- |
| 第 1 周 | P0 | 基线报告、环境版本、seed protocol |
| 第 2-3 周 | P1 | dynamics contract、目标策略、退化数据 |
| 第 4 周 | P2 补强 | prediction ablation、未见策略和退化评估 |
| 第 5-6 周 | P3 | centralized scenario MPC、solver diagnostics |
| 第 7-8 周 | P4 | DN-MPC、通信和 fallback 对照 |
| 第 9-10 周 | P5 | robust CBF-QP、独立残差检查 |
| 第 11 周 | P6 | CLBF、证书检查和 hard-case 扫描 |
| 第 12 周 | P7 | 端到端集成和消融 |
| 第 13-14 周 | P8 | 三种子 locked-test、统计表格和最终报告 |

最可能的延期点是 P1 的动力学统一、P4 的分布式收敛和 P6 的安全证明。若其中任何一个阶段超出一周仍没有可审计进展，应执行 Conditional Go 降级，而不是继续增加模型复杂度。

## 17. 最终交付物检查表

- [ ] 可重建环境、动力学和目标策略配置。
- [ ] 三个不重叠 split 的数据集和 metadata/source hash。
- [ ] GRU、portable SSM diffusion 及所有主消融 checkpoint。
- [ ] raw/projected 候选评估、coverage、energy、可行性和延迟报告。
- [ ] centralized scenario min-max MPC 代码、配置和报告。
- [ ] DN-MPC 通信模型、求解日志和 fallback 统计。
- [ ] robust CBF-QP 过滤器、QP solver wrapper 和压力测试。
- [ ] R-CLBF 定义、训练域、扰动界和独立 certificate checker。
- [ ] 端到端三种子 locked-test 结果和完整消融矩阵。
- [ ] TensorBoard、JSON/CSV 指标、配置、seed、模型 hash 和版本信息。
- [ ] 失败模式、负结果、限制和安全边界说明。
- [ ] 最终论文中的每个结论都能追溯到一条实验或一个明确证明条件。

## 18. 当前下一步

在当前工作树上，下一步按以下顺序执行：

1. 完成当前 Phase 2 修改的完整测试、编译检查和阶段性提交。
2. 实现只消费 projected candidates 的集中式 scenario min-max MPC 诊断版。
3. 在与 `DynamicEncirclementController + local CBF` 相同的场景 seed 上进行 smoke 和小规模正式评估。
4. 根据 P3 结果决定是否实现 DN-MPC；P3 无收益时停止分布式扩展。
5. 在动力学契约冻结后实现 robust CBF-QP；没有独立证书前不称 R-CLBF-QP 已证明。

当前最重要的判断不是“能否把代码全部写出来”，而是：

```text
多模态候选
  是否覆盖真实逃逸可能性？
风险敏感规划
  是否把覆盖转化为安全捕获收益？
安全过滤
  是否在明确假设下可审计且不破坏性能？
```

只要这三个问题按阶段得到正面证据，这个方向就能形成可靠研究成果；不必强行要求三个模块一次性全部通过。
