# Mamba-Diffusion + DN-MPC + R-CLBF-QP 创新方法 TodoList

> 版本：v2.1（2026-09-07）
> 目标仓库：`https://github.com/Liangxianguang/new-uav-capture`
> 当前结论：预测模块处于正式实验前的 `Conditional Go` 阶段；DN-MPC、R-CLBF-QP 和端到端结论尚未成立。

## 0. 文档目的

本计划用于验证以下完整方法是否能够在当前四机三维协同围捕基准上成立：

```text
部分观测与延迟通信
        -> Mamba-SSM + 条件扩散多模态目标轨迹预测
        -> 极小极大分布式博弈 DN-MPC
        -> R-CLBF-QP 鲁棒安全过滤
        -> 速度/动力学执行与 Cooperative Safe Capture 评估
```

计划遵循“先验证可行性，再扩大创新范围”的原则。每一阶段都有可运行产物、对照实验、验收门槛和停止条件。任何阶段未通过门槛时，不进入下一阶段，也不把未验证的模块写成论文结论。

### 当前进度

- Phase 0：历史回归和基线代码已存在；本轮代码修改后必须重新运行完整测试，并补齐正式 V5 基线重跑和锁定评估。
- Phase 1：预测窗口、可见信息约束、五类目标模式、障碍物上下文、未来目标速度、数据元数据和数据集切分已经实现；统一执行动力学和自适应博弈目标仍未完成。
- Phase 2：portable SSM-conditioned diffusion、GRU 对照、归一化、候选可行性检查、conformal coverage、energy score、延迟审计和 TensorBoard 工件已经实现；正式三种训练种子尚未完成，当前 smoke 结果不能作为通过结论。
- 正式数据集：当前已有 train/validation/locked-test 三个不重叠 seed block，但环境硬速度上限修正晚于数据生成，必须重生成一次以保证 source hash 完全一致。
- Phase 3 及以后：尚未开始实现。

### 当前必须遵守的决策顺序

```text
完整回归
  -> 重生成正式数据集
  -> 三种子预测训练与冻结测试
  -> Phase 2 Go/Conditional Go/No-Go
  -> 集中式 scenario min-max MPC
  -> 分布式 DN-MPC
  -> 基础安全 QP
  -> R-CLBF-QP 与证书检查
  -> 端到端三种子锁定测试
```

在 Phase 2 没有通过前，不实现 DN-MPC 的最终版本；在动力学、扰动边界和安全 QP 都固定前，不宣称 R-CLBF-QP 有闭环安全证明。

### 当前立即执行清单

- [ ] 在最新工作树上重新运行完整 `pytest` 和 Python 编译检查。
- [ ] 重生成 `phase2_prediction_train_v2_multimodal`、`phase2_prediction_validation_v2_multimodal` 和 `phase2_prediction_locked_test_v2_multimodal`，确认 metadata 中的 source hash 对应当前源码。
- [ ] 使用训练种子 `745101`、`745201`、`745301`，分别训练正式 GRU 和 portable SSM diffusion；每个 seed 使用独立输出目录。
- [ ] 对每个 checkpoint 运行冻结 validation/locked-test 评估，汇总 ADE/FDE/minADE/minFDE、energy score、conformal coverage、候选可行率和 p50/p95/p99 延迟。
- [ ] 按五种目标模式和观测退化条件分层汇总，禁止只报告总体平均值。
- [ ] 检查每个训练目录是否同时包含 checkpoint、TensorBoard event、config、metadata、history 和 source hash。
- [ ] 写入新的 Phase 2 分析报告；保留旧的 `docs/PHASE2_PREDICTION_REPORT.md` 作为历史 NO-GO 记录，不覆盖它。
- [ ] 只有 Phase 2 达标后，才开始集中式 scenario min-max MPC；只有集中式版本有收益后，才实现分布式 DN-MPC。
- [ ] 每完成一个重大阶段，单独提交到 `origin/main`；提交前不 stage 用户已有的 `README.md` 或实验总结。

### 正式预测实验命令模板

以下命令中的路径应替换为实际生成目录；命令本身是协议的一部分，训练参数不得在不同 seed 间临时修改：

```powershell
conda run --no-capture-output -n uav-encirclement-gpu python -m pytest -q
conda run --no-capture-output -n uav-encirclement-gpu python -m py_compile `
  scripts/train_prediction_models.py `
  scripts/collect_prediction_dataset.py `
  src/encirclement3d/prediction.py `
  src/encirclement3d/trajectory_dataset.py `
  src/encirclement3d/pursuit_env.py
```

正式训练要求：

- [ ] `--target-normalization train_split_standardize`，归一化器只能拟合 train split。
- [ ] 使用训练配置中的正式 history/horizon/diffusion/sampling 参数。
- [ ] 三个训练 seed 只改变随机初始化和训练采样，不改变数据切分、网络规模和评估协议。
- [ ] evaluation sampling seed 独立于 training seed，并在所有模型间保持可复现。
- [ ] `candidate-max-speed` 使用物理硬上限，不使用行为强度 `target_speed_scale` 代替。
- [ ] 任何候选可行率不足的结果必须同时报告原始候选和筛选后候选，不能静默删除失败样本。

### 初步可行性判断

基于当前代码和已完成的 smoke 实验，这个方向不是“直接跑一次就能证明”的方案，而是一个有条件可行的分层研究计划：

| 子方法 | 当前可行性 | 最大风险 | 最小可发表版本 |
| --- | --- | --- | --- |
| portable SSM + 条件扩散预测 | 中高 | 候选轨迹可行率、置信度校准和实时采样 | SSM/GRU 对照 + 多模态 coverage/energy/延迟分析 |
| 集中式 scenario min-max MPC | 中 | 预测误差传递到规划后是否真的改善围捕 | 预测候选驱动的集中式 risk-sensitive MPC |
| 分布式 DN-MPC | 中低 | 通信延迟、子问题不收敛和实时预算 | 集中式 oracle 与有限通信的分布式对照 |
| R-CLBF-QP | 中低 | 统一动力学、扰动上界和离散时间证书 | 可审计 robust CBF-QP；CLBF 作为增强结果 |
| 三者端到端组合 | 低到中 | 误差和延迟在模块间累积 | 在困难场景安全捕获不劣于基线，并报告代价 |

因此推荐的研究成功定义是分层的：

1. **最低成功**：Phase 2 预测器在冻结测试集上证明多模态候选有覆盖价值，且输出经过物理可行性检查。
2. **方法成功**：集中式 scenario MPC 使用预测候选后，在未见目标模式或退化观测条件下改善安全捕获；分布式版本只在确有增益时纳入主方法。
3. **完整成功**：R-CLBF-QP 在明确假设下通过独立证书检查，并且端到端三种子结果不劣于当前 `Policy + local CBF`。

如果第 1 层成立而第 2 层不成立，应停止堆叠模块，转写为“可校准的多模态目标轨迹预测”；如果第 2 层成立而第 3 层不成立，应将安全贡献表述为“条件鲁棒安全过滤”，不能声称完整闭环形式化证明。

### 正式数据集锁定表

正式数据集必须在最新动力学修正后重新生成，并在报告中固定以下协议：

| Split | Episodes | Samples（历史版本） | Seed block | 用途 |
| --- | ---: | ---: | --- | --- |
| `phase2_prediction_train_v2_multimodal` | 64 | 15,232 | 645101--645164 | 训练和 train-only normalization |
| `phase2_prediction_validation_v2_multimodal` | 24 | 5,712 | 646101--646124 | 模型选择、conformal/calibration |
| `phase2_prediction_locked_test_v2_multimodal` | 32 | 7,616 | 647201--647232 | 最终一次性评估 |

重新生成后，如果样本数量因环境修正发生变化，以新 metadata 为准；不能为了保持旧数量而修改环境逻辑。五种目标模式必须在三个 split 中有明确的 episode schedule，且 episode seed 不重叠。

---

## 1. 当前基线和问题边界

### 1.1 当前仓库基线

- 四架同质追捕机，一架逃逸目标。
- 三维障碍物环境，包含圆柱、箱体和墙体。
- `dt = 0.1 s`，最长 `250` 步。
- 当前主要是速度级运动学：动作直接作为速度并积分位置。
- 追捕机最大速度为 `5.0 m/s`，目标最大速度为 `3.6 m/s`。
- 局部观测包含自身状态、目标 belief、队友信息和近邻障碍物。
- 当前策略是循环行为克隆 actor。
- 当前目标预测器是 GRU 单峰均值/方差预测器。
- 当前安全层是局部迭代投影 CBF，不是 QP，也不是已证明的 CLBF。
- 当前正式基线是 `Policy + local CBF`，不能把裸策略结果作为部署结果。

### 1.2 必须保持的科学边界

- “捕获”仍然是进入 `0.80 m` 捕获半径，不等同于物理捕获。
- 在没有加入执行器、动力学和扰动模型之前，不宣称真实飞行安全。
- 在没有自适应目标策略之前，不把脚本化逃逸模式直接称为完整零和对手。
- 预测训练可以使用模拟器目标真值作为标签，但推理和策略输入不得泄露目标真值。
- 所有新结果必须与当前 V4/V5 结果分开报告，不能覆盖已有正式基线。

### 1.3 研究成功的最低定义

方法只有同时满足以下条件，才可以称为“方法在当前仿真基准上成立”：

1. 多模态预测在冻结测试集上比常速度和 GRU 基线改善至少一项核心预测指标，且不牺牲校准质量。
2. DN-MPC 在相同预测输入和相同场景种子下，比当前动态围捕控制器改善安全捕获或最坏逃逸模式下的捕获表现。
3. R-CLBF-QP 在明确的动力学与扰动假设下，满足离散时间安全约束；数值实验中不存在未解释的求解失败。
4. 端到端组合方法在三种子测试中不低于当前正式基线的安全捕获率，并且至少在一个预注册难例指标上有实质改善。
5. 所有结果可由独立命令、固定种子、配置快照和模型哈希复现。

---

## 2. 总体架构和模块接口

### 2.1 目标执行链

```text
env.reset/step
  -> BeliefStateEncoder
  -> MultiModalTrajectoryPredictor
  -> DistributedMinimaxMPC
  -> RobustCLBFQPFilter
  -> env.step
```

### 2.2 统一状态定义

需要先固定以下变量，所有训练、规划、证明和评估使用同一份定义：

- `x_i = [p_i, v_i]`：第 `i` 架追捕机状态。
- `y = [p_t, v_t]`：目标状态，仅用于环境内部和训练标签。
- `b_i`：第 `i` 架追捕机可获得的目标 belief。
- `z_i`：第 `i` 架追捕机的历史局部观测编码。
- `u_i`：规划器输出的名义控制量。
- `u_i_safe`：安全过滤后的最终控制量。
- `w`：观测、预测、延迟和执行扰动的有界集合。
- `H_p`：预测 horizon。
- `H_m`：MPC horizon。
- `K`：扩散候选轨迹数量。

初始推荐设置：

```yaml
prediction:
  history_length: 16
  horizon_steps: 12
  num_modes: 8
  diffusion_steps_train: 100
  diffusion_steps_sample: 8
planner:
  horizon_steps: 8
  control_horizon_steps: 3
  replanning_period_steps: 1
safety:
  solver: osqp_or_scipy
  slack_enabled: true
  disturbance_margin_mode: calibrated_quantile
```

这些数值只是起始配置，必须在配置文件中显式记录，不能写死在模块内部。

### 2.3 推荐新增文件边界

```text
src/encirclement3d/
  trajectory_dataset.py       # 轨迹窗口、标签、数据切分和校验
  mamba_encoder.py            # SSM 编码器，独立于扩散采样器
  diffusion_predictor.py      # 条件扩散训练、采样和 checkpoint
  prediction_contracts.py     # 候选轨迹和置信度的数据结构
  minimax_mpc.py              # 名义/鲁棒/分布式 MPC
  safety_qp.py                # CLBF/CBF 约束与 QP 求解
  dynamics.py                 # 统一离散动力学和扰动模型
  safety_certificate.py       # 证书检查、残差和假设记录

scripts/
  collect_prediction_dataset.py
  train_mamba_diffusion.py
  evaluate_prediction.py
  evaluate_minimax_mpc.py
  evaluate_safety_qp.py
  run_innovation_ablation.py

configs/
  innovation_prediction.yaml
  innovation_mpc.yaml
  innovation_safety.yaml
  innovation_end_to_end.yaml
```

模块必须通过明确的数据结构连接，避免让 MPC 直接依赖扩散模型内部张量格式，也避免安全层依赖神经网络内部状态。

---

## 3. Phase 0：冻结基线和实验协议

### 3.1 任务清单

- [x] 记录当前工作区状态、已有提交、当前未提交修改和远程仓库地址。
- [x] 确认基线运行环境、Python、PyTorch、NumPy、SciPy 版本。
- [x] 运行现有单元测试并保存完整日志。
- [ ] 重新运行当前 V5 development validation，保存场景、逐 episode 结果和 summary。
- [ ] 在相同场景上分别运行 RAW policy 和现有 Policy + CBF。
- [ ] 记录每个 episode 的 seed、障碍物布局、目标模式、观测条件和 checkpoint hash。
- [ ] 固定 train、validation、locked-test 三个不重叠 seed block。
- [ ] 固定新方法不得访问 locked-test 标签和结果。
- [ ] 预先登记所有主要指标和停止规则。
- [ ] 固定最终报告中的名称：`baseline`、`prediction-only`、`MPC-only`、`safety-only`、`end-to-end`。

### 3.2 基线必须输出的指标

- [ ] Cooperative Safe Capture rate。
- [ ] 普通 capture rate。
- [ ] collision rate。
- [ ] boundary violation rate。
- [ ] timeout rate。
- [ ] mean/median capture time。
- [ ] worst and mean minimum clearance。
- [ ] mean path length。
- [ ] CBF action correction norm。
- [ ] 平均控制周期、p95 控制周期和最坏控制周期。
- [ ] 目标可见率、观测 age、消息 age。

### 3.3 Phase 0 通过条件

- [ ] 测试全部通过。
- [ ] 基线结果与仓库现有证据在允许的随机误差范围内一致。
- [ ] 所有输出均能从固定 seed 重建。
- [ ] 如果基线无法复现，先修复实验协议，不开始创新模块。

---

## 4. Phase 1：统一动力学、扰动和数据记录

这是三个创新点共同依赖的基础阶段，优先级高于模型结构。

### 4.1 动力学契约

- [x] 明确追捕机控制变量是速度、加速度还是期望速度；当前主协议为速度指令加可选执行跟踪。
- [x] 若使用速度级模型，写出离散状态方程并明确最大速度约束。
- [ ] 若使用加速度级模型，实现速度、加速度和动作延迟的一致执行。
- [ ] 让训练、MPC rollout、安全 QP 和环境 `step()` 使用同一个 dynamics 实现。
- [ ] 为目标增加可配置的动作策略接口，而不是只在环境内部写死目标动作。
- [ ] 支持 `flee_persistence`、`s_curve`、`random_turn` 和自适应 adversarial policy。
- [ ] 明确障碍物、边界和机间碰撞的几何定义。
- [ ] 明确执行噪声、延迟、速度跟踪误差和加速度扰动的上界。

### 4.2 目标策略扩展

- [ ] 保留现有脚本化目标作为基础测试集。
- [ ] 增加基于追捕机状态的自适应逃逸策略。
- [ ] 增加随机化目标策略，避免预测器只记住固定模式。
- [ ] 增加 worst-case evaluation policy，用于测试极小极大规划。
- [ ] 记录目标策略 ID、随机种子和参数，不把策略信息泄露给追捕机观测。
- [ ] 验证目标策略不会产生不可行或超出物理约束的轨迹。

### 4.3 预测数据集

每条样本至少包含：

```text
history_observation: [history, defenders, feature_dim]
belief_state:        [history, defenders, belief_features]
future_target:       [horizon, 3]
future_target_vel:   [horizon, 3]
target_mode_id
obstacle_context
observation_condition
episode_id / timestep / seed
```

任务清单：

- [x] 增加只使用策略可见信息的 history encoder 输入。
- [x] 使用环境内部目标真值生成 future target 标签。
- [ ] 生成 nominal、delayed-noisy、遮挡严重和消息丢失数据。
- [ ] 按 episode 和 scene 切分 train/validation/test，禁止时间窗口泄漏。
- [ ] 按目标运动模式分层统计样本量。
- [x] 保存数据集版本、配置 hash、源代码 hash 和标签生成协议。
- [x] 检查所有输入和标签 finite、范围合理、坐标系一致。
- [x] 检查未来轨迹没有错误地使用当前之后的观测信息。
- [x] 生成小型 CPU smoke dataset，保证开发测试不依赖 GPU。

### 4.4 Phase 1 通过条件

- [ ] 速度级或加速度级动力学只能选择一个作为主协议。
- [ ] 环境 rollout 与独立 dynamics rollout 在随机动作下逐步一致。
- [ ] 目标策略至少包含一个非固定规则的 adversarial policy。
- [x] 预测数据集可由 clean run 重新生成。
- [ ] 任意样本人工检查时不存在 target truth leakage。

---

## 5. Phase 2：Mamba-SSM + 条件扩散预测器

## 5.1 预测任务定义

给定每架追捕机的局部历史和可用 belief，输出目标未来的多条候选轨迹：

```text
z_i[0:T] -> {trajectory_i[k, 0:H, 3], score_i[k], uncertainty_i[k, 0:H]}
```

建议同时保留一个团队级输出，供 DN-MPC 使用：

```text
team_history -> {trajectory[k, 0:H, 3], score[k], covariance[k, 0:H, 3, 3]}
```

第一版先采用团队级目标轨迹，避免四架无人机分别采样后出现候选集合不一致；之后再做分布式局部预测消融。

## 5.2 Mamba/SSM 编码器任务

- [ ] 评估当前环境是否允许引入 `mamba-ssm` 或等价依赖。
- [x] 如果 CUDA/平台兼容性不稳定，先实现纯 PyTorch SSM 兼容版本；当前实现为 `portable_diagonal_ssm`，不是官方 `mamba_ssm` CUDA kernel。
- [ ] 固定输入 token 定义：belief、速度、confidence、age、队友摘要、障碍物摘要。
- [ ] 明确是否使用 defender ID embedding；默认使用参数共享和 permutation-aware team pooling。
- [ ] 实现可变历史长度或固定 padding mask。
- [ ] 对比 GRU、TCN、Transformer-lite 和 Mamba 编码器。
- [ ] 测量显存、吞吐、单步推理延迟和序列长度扩展曲线。
- [ ] 编写 shape、mask、reset 和 deterministic inference 测试。

## 5.3 条件扩散解码器任务

- [ ] 选择轨迹空间：绝对坐标、相对目标坐标或速度增量；优先相对坐标。
- [x] 定义扩散目标为未来位置序列或未来速度序列，并固定一种主方案；当前使用相对团队 belief reference 的未来位置位移。
- [x] 实现 noise schedule、训练 loss 和条件注入。
- [x] 支持候选轨迹 batch sampling。
- [ ] 输出候选轨迹分数，并使用 softmax 或 energy normalization 得到相对置信度。
- [ ] 对候选轨迹进行速度、加速度、边界和障碍物可行性检查。
- [x] 设计快速采样路径：当前使用 few-step DDIM 风格采样。
- [x] 提供 deterministic seed，保证训练和采样可由固定配置重放。
- [x] 对所有预测输出做坐标反归一化和 finite 检查。

## 5.4 置信度与校准任务

- [ ] 定义置信度是 mode probability、轨迹级 energy 还是 coverage confidence。
- [ ] 在 validation set 上做 temperature scaling 或 conformal calibration。
- [ ] 统计不同置信度分位数对应的真实轨迹覆盖率。
- [ ] 测量 misspecified target mode 下的置信度退化。
- [ ] 记录 top-1、top-`K`、均匀候选和 oracle mode 的差异。
- [ ] 不能用未校准的 softmax 分数直接当作概率写入论文。

## 5.5 预测评估指标

- [ ] ADE：平均位移误差。
- [ ] FDE：终点位移误差。
- [ ] minADE/minFDE：候选集合覆盖能力。
- [ ] negative log-likelihood 或 energy score。
- [ ] CRPS 或等价分布质量指标。
- [ ] coverage@50/80/90/95%。
- [ ] calibration error。
- [ ] 长时域误差随 horizon 的曲线。
- [ ] 不同观测延迟、丢包和目标模式的分层结果。
- [ ] 单步推理 p50/p95/p99 延迟。

## 5.6 预测消融

- [x] Constant-velocity baseline（CPU smoke）。
- [x] 当前 GRU Gaussian predictor（CPU smoke）。
- [ ] Mamba + deterministic head。
- [ ] GRU + diffusion head。
- [x] Mamba + diffusion head 的 portable SSM smoke 实现；正式 Mamba kernel 对照尚未完成。
- [ ] 单模态 diffusion 与多模态 diffusion。
- [ ] 无 confidence calibration 与 calibrated confidence。
- [ ] 不同候选数量 `K`。
- [ ] 不同 history length 和 prediction horizon。

## 5.7 Phase 2 通过条件

建议采用以下 go/no-go 门槛：

- [ ] 在所有测试模式上，Mamba-Diffusion 的 minFDE 至少比 GRU 基线降低 10%，或在相同误差下达到更高的 coverage。
- [ ] 90% 置信区域的经验覆盖率位于 `0.85--0.95`，不能严重过度自信。
- [ ] 预测输出经过动力学和边界检查后，可行候选比例不少于 95%。
- [ ] 在目标策略切换后仍有可解释的置信度退化，而不是输出异常高置信度。
- [ ] p95 推理时间满足控制周期预算；若不能满足，必须先蒸馏或降低采样步数。
- [ ] 与 planner 的接口固定，候选轨迹不再因内部模型实现改变。

如果只提升 ADE/FDE，却没有提升候选覆盖率或校准质量，则不能声称“多模态预测成功”，只能称为更强的点预测器。

---

## 6. Phase 3：极小极大分布式 DN-MPC

## 6.1 规划问题定义

第一版建议先实现集中式 scenario min-max MPC，确认目标函数和候选轨迹有价值，再实现分布式版本。

```text
minimize_u  max_k J(u, target_trajectory[k])
subject to  dynamics
            speed/acceleration limits
            obstacle and boundary constraints
            inter-agent separation
```

其中 `k` 是扩散模型输出的候选逃逸轨迹。需要区分三种模式：

- expected-cost MPC：按候选置信度加权。
- worst-case MPC：直接取候选中最大代价。
- risk-sensitive MPC：使用 CVaR 或 top-q 平均代价。

## 6.2 代价函数任务

- [ ] 定义捕获距离代价。
- [ ] 定义拦截点和未来相对速度代价。
- [ ] 定义围捕角度、覆盖度和 blocker 分工代价。
- [ ] 定义障碍物净空代价。
- [ ] 定义机间距离和队形稳定代价。
- [ ] 定义控制量、控制变化和通信不一致惩罚。
- [ ] 定义终端捕获/拦截代价。
- [ ] 对每个代价项写出单位、归一化方式和权重来源。
- [ ] 先固定权重，再做预注册的小范围敏感性实验。

## 6.3 分布式机制任务

- [ ] 规定每架无人机可获得的信息范围。
- [ ] 规定共享消息内容：邻机状态、预测轨迹、候选控制序列或角色信息。
- [ ] 规定消息延迟、丢包和通信频率。
- [ ] 实现本地 MPC 子问题。
- [ ] 实现 consensus、ADMM 或 sequential best-response 中的一种。
- [ ] 规定每个控制周期的最大迭代次数。
- [ ] 记录未收敛次数和局部求解失败次数。
- [ ] 设计通信中断时的本地 fallback。
- [ ] 验证分布式输出与集中式 oracle 的差距。

## 6.4 求解器和实时性任务

- [ ] 评估 SciPy、OSQP、CVXPY/Clarabel 或 CasADi 的依赖和 Windows 安装稳定性。
- [ ] 第一版优先使用线性化动力学和 QP/二次代价，降低实时风险。
- [ ] 为非凸障碍物约束设计局部线性化或安全层后置处理。
- [ ] 设置 solver time limit。
- [ ] 设置 infeasible、timeout、NaN 和异常数值的 fallback。
- [ ] 记录每步 solver status、迭代次数、目标值、最大约束残差和运行时间。
- [ ] 对所有 planner output 做动作范围检查。

## 6.5 规划评估指标

- [ ] Cooperative Safe Capture rate。
- [ ] worst-case candidate capture rate。
- [ ] expected candidate capture rate。
- [ ] CVaR capture distance。
- [ ] capture time。
- [ ] collision and boundary violation。
- [ ] minimum clearance。
- [ ] role switching frequency。
- [ ] solver success rate。
- [ ] p95/p99 planning latency。
- [ ] 通信量和通信失败后的恢复时间。

## 6.6 DN-MPC 消融

- [ ] 当前 DynamicEncirclementController。
- [ ] 单一 GRU/常速度预测 + MPC。
- [ ] 多模态预测 + expected-cost MPC。
- [ ] 多模态预测 + worst-case MPC。
- [ ] 多模态预测 + CVaR MPC。
- [ ] 集中式 MPC。
- [ ] 分布式 MPC。
- [ ] 无通信、理想通信、延迟通信和丢包通信。
- [ ] planner 后置当前 CBF。
- [ ] planner 后置 R-CLBF-QP。

## 6.7 Phase 3 通过条件

- [ ] 在相同候选轨迹和相同场景种子下，worst-case 或 CVaR DN-MPC 的安全捕获率高于当前 DynamicEncirclementController。
- [ ] 在 delayed-noisy 条件下，性能下降不超过预注册阈值。
- [ ] 分布式结果与集中式 oracle 的主要指标差距不超过 10 个百分点，或能解释差距来自通信限制。
- [ ] solver success rate 不低于 99%，剩余失败均有安全 fallback。
- [ ] p95 planning latency 在单个控制周期内，或有明确的低频规划/高频安全执行架构。
- [ ] 仅使用预测候选而不访问目标真值。

如果 DN-MPC 只有在访问目标真值时有效，则该模块判定为实验诊断工具，不能作为方法主体。

---

## 7. Phase 4：R-CLBF-QP 鲁棒安全过滤

## 7.1 先明确证明对象

第一版不应直接声称“对真实无人机闭环安全”。应先证明以下较窄命题：

> 在给定离散运动学/动力学模型、已知障碍物几何、有限观测误差、有限执行扰动和 QP 可行的条件下，安全过滤后的状态保持在定义安全集内。

之后再扩展到执行动力学和更强的扰动模型。

## 7.2 安全集和 barrier 任务

- [ ] 定义障碍物安全函数 `h_obstacle`。
- [ ] 定义世界边界安全函数 `h_boundary`。
- [ ] 定义机间安全函数 `h_inter_agent`。
- [ ] 明确 barrier 使用净空、平方距离还是 signed distance。
- [ ] 处理圆柱、箱体和墙体的不可微边界/拐角。
- [ ] 明确目标本身是否属于安全约束；捕获目标不能被错误地排除或纳入排斥项。
- [ ] 对观测误差、延迟、预测误差和执行误差加入收缩项。
- [ ] 固定鲁棒 margin 的来源：理论上界、统计分位数或 conformal bound。

## 7.3 CLBF/QP 任务

建议先实现安全 QP，再加入 learned robust CLBF，防止理论和求解器问题同时出现。

- [ ] 定义 QP 决策变量：当前控制量和可选 slack。
- [ ] 定义名义动作跟踪目标。
- [ ] 加入速度/加速度/动作变化约束。
- [ ] 加入离散 CBF 或高阶 CBF 约束。
- [ ] 对机间约束处理相对控制耦合。
- [ ] 对边界和障碍物处理动作饱和。
- [ ] 实现 slack 分层惩罚，安全约束优先级高于性能约束。
- [ ] 实现 solver infeasible 的应急动作。
- [ ] 记录每个约束的 barrier 值、残差、slack 和 active 状态。
- [ ] 记录 solver status、数值条件和修正量。
- [ ] 为 learned CLBF 记录网络版本、训练域、验证域和 Lipschitz/gradient bound。

## 7.4 安全证明任务

- [ ] 写出系统动力学假设。
- [ ] 写出扰动集合和每一项上界。
- [ ] 证明 nominal action 到 safe action 的约束满足性。
- [ ] 证明在可行 QP 下安全集的离散时间正向不变性，或明确只能得到一步安全保证。
- [ ] 证明 slack 不会掩盖安全约束违反；若允许 slack，给出可接受条件。
- [ ] 证明动作饱和、延迟和求解误差如何进入 margin。
- [ ] 给出 solver numerical tolerance 对 barrier 的最坏影响。
- [ ] 用独立数值脚本检查理论 bound 是否覆盖实际扰动样本。
- [ ] 明确哪些条件无法满足时，证书自动失效并标记 episode。

## 7.5 R-CLBF 训练和验证任务

- [ ] 生成安全/危险状态样本和边界附近 hard cases。
- [ ] 训练 CLBF 或 robust value function 时禁止使用测试场景标签。
- [ ] 检查 barrier value、gradient、Lie derivative 和 residual 的范围。
- [ ] 在障碍拐角、窄通道、机间近距离和边界附近做网格扫描。
- [ ] 做随机扰动 Monte Carlo 验证。
- [ ] 做 action delay 和 command noise stress test。
- [ ] 验证 learned CLBF 不会在训练域外产生过度自信的安全证书。

## 7.6 安全层通过条件

- [ ] 所有可行 QP 的最大安全约束残差不超过预设数值容差。
- [ ] 证书假设覆盖评估中实际使用的速度、延迟、噪声和扰动范围。
- [ ] infeasible 比例低于预注册阈值，且 fallback 不产生未记录安全违规。
- [ ] 在相同 planner 输出下，R-CLBF-QP 不增加碰撞率。
- [ ] 与当前局部 CBF 比较时，至少改善一个难例安全指标或明显降低动作修正量/超时率。
- [ ] 证明文档和代码检查器使用同一套 barrier 定义。

如果只能证明“QP 求解器通常找到了较安全的动作”，则应称为 robust safety filter，不能称为闭环形式化安全证明。

---

## 8. Phase 5：端到端集成

## 8.1 集成顺序

- [ ] 保持现有 actor + 当前 CBF 可运行。
- [ ] 接入预测器，但先只把预测结果用于日志，不改变动作。
- [ ] 接入预测器摘要特征，运行 prediction-only 消融。
- [ ] 用预测候选驱动集中式 min-max MPC。
- [ ] 将 MPC 输出经过当前 CBF，验证接口和性能。
- [ ] 接入 R-CLBF-QP，保留当前 CBF 作为可选 fallback。
- [ ] 接入分布式通信和延迟模型。
- [ ] 最后才考虑 actor、预测器和 planner 的联合训练。

## 8.2 端到端运行时检查

- [ ] 每步检查观测、预测、规划、过滤动作均 finite。
- [ ] 每步检查候选轨迹 shape 和 horizon 一致。
- [ ] 每步记录 predictor latency、planner latency、QP latency。
- [ ] 每步记录 solver status 和 fallback reason。
- [ ] 每步记录 nominal/safe action 差异。
- [ ] 任何组件异常时进入可解释的安全 fallback。
- [ ] 终止时写入完整 provenance：配置、模型 hash、seed、场景、版本和日志。

## 8.3 端到端消融矩阵

| 编号 | 预测 | 规划 | 安全层 | 目的 |
| --- | --- | --- | --- | --- |
| A0 | 常速度 | DynamicEncirclement | 当前 CBF | 当前基线 |
| A1 | GRU Gaussian | DynamicEncirclement | 当前 CBF | 现有预测器贡献 |
| A2 | Mamba-Diffusion | DynamicEncirclement | 当前 CBF | 预测贡献 |
| A3 | Mamba-Diffusion | expected MPC | 当前 CBF | 规划基本效果 |
| A4 | Mamba-Diffusion | worst-case MPC | 当前 CBF | 极小极大效果 |
| A5 | Mamba-Diffusion | DN-MPC | 当前 CBF | 分布式效果 |
| A6 | Mamba-Diffusion | DN-MPC | R-CLBF-QP | 安全层效果 |
| A7 | Mamba-Diffusion | DN-MPC | R-CLBF-QP + fallback | 完整部署栈 |
| A8 | oracle target | DN-MPC | R-CLBF-QP | 上限诊断，不作为主结果 |

每个配置必须使用同一组 scene seeds，并同时报告 RAW、当前 CBF 和新安全层结果。

---

## 9. Phase 6：正式实验和统计协议

### 9.1 场景分层

- [ ] S1：单类障碍物基础安全。
- [ ] S2：固定混合障碍物。
- [ ] S3：随机混合障碍物。
- [ ] S4：窄通道和墙体难例。
- [ ] S5：延迟、丢包、噪声退化。
- [ ] S6：目标策略切换和自适应 adversarial target。
- [ ] S7：动力学扰动和执行延迟。

### 9.2 统计要求

- [ ] 至少三个独立训练种子。
- [ ] 训练、验证、锁定测试场景完全分离。
- [ ] 每个主要配置使用相同 episode seeds。
- [ ] 报告均值、样本标准差和置信区间。
- [ ] 对主要改进报告绝对百分点和相对变化。
- [ ] 对多配置比较进行多重比较说明。
- [ ] 不只报告成功率，同时报告安全失败、超时和求解失败。
- [ ] 报告最坏场景和分层难例，不隐藏失败模式。

### 9.3 主要结论门槛

建议预注册以下门槛，具体数值可在 Phase 0 依据算力和基线结果冻结：

| 层级 | 建议最低门槛 |
| --- | --- |
| 预测 | minFDE 或 energy score 相对 GRU 改善 >= 10%，coverage 不下降 |
| 规划 | S3/退化条件安全捕获率至少提升 5 个百分点，或最坏候选捕获距离下降 >= 10% |
| 安全 | 碰撞率不高于当前 CBF，且 QP 残差/可行性可审计 |
| 实时 | p95 总控制周期不超过预算，或有明确低频 planner + 高频 safety 架构 |
| 复现 | 三种子方向一致，关键指标不依赖单一 seed |
| 理论 | 假设、barrier、扰动界和离散时间条件全部可对应到代码 |

若预测提升但端到端不提升，应保留预测贡献结论，但不能声称完整方法有效。若安全层提升但捕获率大幅下降，应报告安全--效率权衡，不得只报告安全率。

---

## 10. 风险、失败模式和应对方案

### 10.1 Mamba 依赖和 Windows/CUDA 兼容性

风险：`mamba-ssm` 安装或 CUDA kernel 不稳定。

应对：保留纯 PyTorch SSM 后端；将 Mamba 作为可选依赖；先以等价 SSM 结构完成科学验证。

### 10.2 扩散采样过慢

风险：无法满足 `0.1 s` 控制周期。

应对：减少采样步数、蒸馏、候选缓存、低频预测、高频 MPC/安全层执行。

### 10.3 多模态只是伪多模态

风险：所有候选轨迹塌缩到同一条均值轨迹。

应对：报告候选多样性、mode coverage、energy score；加入 mode collapse 检测；不能只看 ADE。

### 10.4 模拟目标过于简单

风险：预测器只学会脚本模式，无法支持博弈结论。

应对：增加自适应目标策略、目标策略随机化和未见策略测试。

### 10.5 MPC 求解不可行或不实时

风险：窄通道、候选轨迹冲突和通信延迟导致 planner 失败。

应对：线性化、warm start、短 horizon、软性能约束、严格安全 QP、备用 controller。

### 10.6 QP 证书与实际执行不一致

风险：理论使用连续动力学，环境却直接积分速度或发生动作饱和。

应对：先统一 dynamics；所有饱和、延迟、噪声都进入安全 margin；独立 rollout 验证。

### 10.7 安全层过于保守

风险：碰撞减少但大量 timeout，安全捕获率下降。

应对：报告安全--效率 Pareto 曲线；使用 adaptive margin、风险敏感代价和 slack 分层，而不是盲目增大 margin。

### 10.8 复杂度过高导致无法完成

应对：采用以下降级路线：

```text
Mamba + deterministic prediction
  -> Mamba + Gaussian mixture
  -> Mamba + few-step diffusion
  -> scenario MPC
  -> distributed MPC
  -> robust CBF-QP
  -> learned CLBF-QP
```

只要前一级已经形成可验证贡献，就不要为了保留名词而强行进入下一级。

---

## 11. 推荐时间安排

以下按一个人、已有当前仓库和一张中等 GPU 估计，实际以阶段门槛为准：

| 周期 | 工作 | 产物 |
| --- | --- | --- |
| 第 1 周 | Phase 0 基线冻结 | baseline report、seed block、运行日志 |
| 第 2 周 | Phase 1 动力学与数据记录 | dynamics contract、dataset v1 |
| 第 3--4 周 | Phase 2 预测器 | GRU/Mamba/Diffusion checkpoints、prediction report |
| 第 5--6 周 | Phase 3 集中式和分布式 MPC | planner、solver diagnostics、MPC report |
| 第 7--8 周 | Phase 4 safety QP/CLBF | QP filter、certificate report |
| 第 9 周 | Phase 5 集成和消融 | end-to-end artifacts |
| 第 10 周 | Phase 6 locked test 和论文表格 | final report、复现脚本、模型清单 |

如果 Phase 2 不满足预测门槛，应暂停后续扩展，先检查数据和目标策略；如果 Phase 3 不满足实时性，应采用低频 planner + 高频安全层；如果 Phase 4 无法形成可审计证明，应降低论文表述为经验型鲁棒安全过滤。

---

## 12. 最终交付物清单

- [ ] 版本化环境和动力学配置。
- [ ] 可重建的预测数据集生成脚本。
- [ ] Mamba/SSM 预测器代码和 checkpoint。
- [ ] 条件扩散采样器和置信度校准器。
- [ ] 预测评估脚本和报告。
- [ ] DN-MPC 控制器、分布式通信模拟和 fallback。
- [ ] MPC 求解诊断和实时性报告。
- [ ] R-CLBF-QP 过滤器和 solver wrapper。
- [ ] 安全集定义、扰动边界和证明假设文档。
- [ ] QP 证书检查器和独立数值验证脚本。
- [ ] 完整消融矩阵结果。
- [ ] 三种子锁定测试结果。
- [ ] 每个结果的配置、seed、场景和模型 hash。
- [ ] README 使用说明和限制说明。
- [ ] 失败模式及负结果记录。

---

## 13. 最终 Go / No-Go 决策

### Go：可以作为完整创新方法继续

满足以下条件：

- [ ] 预测器的多模态性和置信度经过独立指标验证。
- [ ] DN-MPC 在未见目标策略和退化观测条件下仍有收益。
- [ ] R-CLBF-QP 的安全约束和假设可被代码检查器复核。
- [ ] 端到端方法在三种子测试中稳定优于或至少不劣于基线。
- [ ] 运行时满足部署预算，失败时有明确 fallback。

### Conditional Go：缩小贡献范围

出现以下情况时，只保留已经通过的部分：

- 预测有效，但扩散采样不实时：保留 Mamba 多模态预测，使用蒸馏或混合密度头。
- MPC 有效，但分布式不稳定：报告集中式 scenario MPC，分布式作为后续工作。
- QP 有效，但 CLBF 证明不完整：报告 robust CBF-QP 和条件性安全保证。
- 安全提升明显但捕获率下降：报告安全--效率 Pareto，而不是宣称全面提升。

### No-Go：暂停该方向

出现以下情况时，不应继续堆叠模块：

- 预测器不能超过常速度/GRU 基线。
- 候选轨迹覆盖率低且置信度严重失准。
- DN-MPC 只在访问目标真值时有效。
- QP 经常不可行且 fallback 不能保证安全。
- 端到端结果依赖单一 seed 或单一场景。
- 运行时无法满足控制周期，也无法设计合理的低频/高频分层架构。

此时应回退到“改进观测预测”或“鲁棒安全规划”中的单一贡献，不再同时追求三个创新点。

---

## 14. 当前建议

第一步只做 Phase 0 和 Phase 1，不要立即编写完整扩散模型、分布式 MPC 和 CLBF 网络。最先要回答的不是“网络能否训练”，而是：

1. 目标数据是否足够多样，能否支撑多模态预测；
2. 统一动力学后，规划器和安全层是否使用同一个系统模型；
3. 新模块的运行时间是否能满足控制周期；
4. 现有基线的失败是否确实来自预测和规划，而不是环境、数据或安全层接口问题。

只有这四个问题通过，才进入 Mamba-Diffusion；只有预测器通过，才进入 DN-MPC；只有动力学和扰动边界固定，才进入 R-CLBF-QP。
