# Mamba-Diffusion + DN-MPC + R-CLBF-QP 创新方法 TodoList

> 版本：v3.1（2026-09-08）
> 目标仓库：`https://github.com/Liangxianguang/new-uav-capture`
> 当前结论：整体为 `Conditional Go`；预测模块未达到 10% minFDE 强门槛，P3 集中式 Scenario MPC、P4 固定 S3 validation 和 P4 未见自适应目标 locked-test 的围捕效果门槛已通过；P5 分层 robust-safe reset 已通过 velocity-level 条件性 gate，但执行扰动扩展未通过。低频预测缓存的严格 10 Hz 部署参考（100 ms 总控制 p95）尚未满足，但这不是 P4 方法验证的硬门槛；R-CLBF-QP 形式化结论和端到端完整结论尚未成立。

完整的研究问题、代码接口、阶段门槛、实验矩阵、Go/No-Go 规则和时间安排见：
[`docs/INNOVATIVE_METHOD_FULL_PLAN.md`](INNOVATIVE_METHOD_FULL_PLAN.md)。本文档保留为日常执行清单。

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
- Phase 1：预测窗口、可见信息约束、六类目标模式、障碍物上下文、未来目标速度、数据元数据和数据集切分已经实现；自适应目标已通过物理约束单元测试和 smoke，统一高阶执行动力学仍未完成。
- Phase 2：正式三种训练种子、冻结 locked-test、按模式统计、conformal coverage、energy score、原始/投影候选可行性和 TensorBoard 工件均已完成；结论为 `Conditional Go`，未达到 10% minFDE 强门槛。
- 正式数据集：已生成 `v3_multimodal` train/validation/locked-test，三个 split 的 episode seed 不重叠，metadata source hash 与当前代码一致。
- Phase 3：集中式 Scenario MPC 已完成 formal-small 和 S3 validation；三个 prediction checkpoint seed 在同一组 12 个 S3 场景上均达到 100% safe capture、0% collision，已通过集中式规划门槛。
- Phase 4：有限通信 sequential best-response DN-MPC、ideal/delayed/dropout/none 通信模式、fallback 和三种 checkpoint seed 的 S3 对照均已完成；四轮 best-response + shifted warm start 后 planner p95 为 8.72--12.47 ms、total-control p95 为 50.38--51.51 ms，四种模式和三个 seed 的 effective/converged plan rate 均为 100%，固定 S3 validation gate 已通过。
- Phase 5：已实现 solver/fallback/precondition 分类、命名约束残差、独立 checker 结果和 TensorBoard 记录；安全单元测试 9/9 通过。分层协议从 64 个候选 seed 中固定选择 tight/nominal 两层共 8 个 robust-safe episode；130/130 步 solver、独立 certificate 和 next-state safety 均通过，QP infeasible、solver failure、fallback、slack 和碰撞均为 0。P5 通过的是冻结协议下的 velocity-level 条件性 gate，详见 `docs/PHASE5_STRATIFIED_VALIDATION_REPORT.md`；R-CLBF-QP 和闭环证明仍未开始。
- P4 未见自适应目标 locked-test：已完成同一份 100-episode 场景文件上的 3 checkpoint × 6 method 正式矩阵；共享 DynamicEncirclement baseline 为 95.00% safe capture、2.00% collision，预测驱动方法为 97.33--99.00% safe capture、0--1.00% collision，solver/valid/effective rate 均达到 99% 以上。正式结果见 `docs/PHASE4_UNSEEN_ADAPTIVE_VALIDATION_REPORT.md`；低频缓存的总控制 p95 为 167.73--229.62 ms，严格 10 Hz 部署参考未满足，但作为工程权衡记录，不阻塞 P4 方法验证。

### 成功等级总览

本项目按四个等级判断“方法是否成功”，不能把代码跑通或单个 seed 的高成功率当作完整方法成立：

| 等级 | 通过条件 | 当前状态 | 可以声称的结论 |
| --- | --- | --- | --- |
| L0 工程可运行 | 数据、模型、planner、过滤器可复现，测试通过 | 已通过 | 工程实现完成 |
| L1 预测模块 | 多模态 coverage/energy 有价值，候选可行且延迟可接受 | Conditional Go | 多模态预测贡献 |
| L2 规划模块 | 不访问目标真值，在未见策略和退化观测下改善围捕 | P4 固定 S3 与 adaptive locked-test 效果门槛已通过 | Scenario MPC/DN-MPC 贡献 |
| L3 安全模块 | 执行契约、扰动界和独立证书检查一致 | 一步 velocity-level 条件通过；执行扩展 No-Go | 条件性 robust CBF-QP |
| L4 完整方法 | L1-L3、三种子端到端不劣、部署架构与 fallback 可审计 | 尚未完成：P5 执行扰动 gate 未通过，严格 10 Hz 部署参考尚未满足 | 完整组合方法仍不可宣称 |

后续任务必须按 `P4 locked-test -> P5 执行契约修复 -> P6 CLBF（可选） -> 端到端` 的顺序推进。任一级失败时，立即停在上一等级并保留其可验证结果。

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

Phase 2 的 Conditional Go 只允许先做集中式诊断；集中式 P3 validation 门槛已通过后，才进入有限通信 DN-MPC。动力学、扰动边界和安全 QP 都固定前，不宣称 R-CLBF-QP 有闭环安全证明。

### 当前立即执行清单

- [x] 在最新工作树上重新运行完整 `pytest` 和 Python 编译检查。
- [x] 重生成 `phase2_prediction_train_v3_multimodal`、`phase2_prediction_validation_v3_multimodal` 和 `phase2_prediction_locked_test_v3_multimodal`，确认 metadata 中的 source hash 对应当前源码。
- [x] 使用训练种子 `745101`、`745201`、`745301`，分别训练正式 GRU 和 portable SSM diffusion；每个 seed 使用独立输出目录。
- [x] 对每个 checkpoint 运行冻结 validation/locked-test 评估，汇总 ADE/FDE/minADE/minFDE、energy score、conformal coverage、候选可行率和 p50/p95/p99 延迟。
- [x] 按五种目标模式分层汇总；观测退化条件仍需作为 Phase 1/正式规划实验补充。
- [x] 检查每个训练目录是否同时包含 checkpoint、TensorBoard event、config、metadata、history 和 source hash。
- [x] 写入新的 `docs/PHASE2_FORMAL_ANALYSIS_REPORT.md`；保留旧的 `docs/PHASE2_PREDICTION_REPORT.md` 作为历史 NO-GO 记录，不覆盖它。
- [x] 集中式 scenario min-max MPC 已在 S3 validation 上相对 DynamicEncirclement 提升 8.3 个百分点 safe capture，满足进入分布式 DN-MPC 的前置门槛。
- [x] 在相同 12 个 S3 场景、三个 checkpoint seed 上完成 ideal、delayed、dropout、none 和 centralized oracle 对照。
- [x] 记录消息发送、接收、丢包、字节数、消息年龄、planner 有效率、收敛率、局部失败和 fallback。
- [x] 已完成 locked-test 和未见自适应目标策略复验；固定 S3 validation gate 与 adaptive locked-test 分开报告，不将任一结果外推为真实部署保证。
- [x] 通过向量化局部 scenario-cost 评估解决分布式 planner p95 超过 100 ms 的问题；三 seed 的 planner p95 均在 100 ms 内。
- [x] 解决 ideal 模式 effective/converged plan rate 边界问题：增加一轮 best-response，并将上一周期序列按已执行动作左移后 warm start；三 seed 协议重跑均为 100%。
- [x] 加入初版 `src/encirclement3d/safety_qp.py`；当前只代表 velocity-level robust CBF-QP，不代表 R-CLBF-QP 或闭环证明。
- [x] 完成安全 QP 语法检查、9 个安全单元测试和四类最小数值场景：远离障碍物、边界投影、机间分离、solver fallback。
- [x] 新增 `configs/innovation_safety.yaml`、`scripts/evaluate_safety_qp.py`、`src/encirclement3d/safety_certificate.py` 和对应测试。
- [x] 在相同 nominal action 下完成 nominal、local CBF 和 robust CBF-QP 对照，并记录证书、残差、修正量、延迟、碰撞和超时。
- [x] 补齐每步 `solver_status`、`solver_message`、`fallback_reason`、约束数量、残差和 slack 的完整 JSONL/TensorBoard 记录，并完成不可行原因分类。
- [x] 分离“初始状态不在 robust 收缩安全集”和“QP 求解不可行”两个 gate；当前 8-seed 复验为 5 个初始前置条件失败、0 个 QP infeasible。
- [x] 在 margin、动作变化限制和 `barrier_recovery` fallback 冻结后重跑 hard-barrier validation；结果已归档，分层 velocity-level 条件性 P5 gate 已通过。
- [x] 完成窄通道、箱体拐角、边界和近距离机间的一步 hard-case scan；soft-slack、local CBF、zero fallback 和 hard-barrier 均保留独立 checker 结果。
- [x] 重新设计并冻结 robust-safe reset/stratified hard-case protocol；在初始有效层上完成 velocity-level formal gate。
- [x] 加入 `adaptive_adversarial` 未见目标策略，完成物理约束单元测试、validation smoke 和 100-episode locked-test。
- [x] 完成 checkpoint `745101` 的 `worst_case` locked-test：interval-20 结果为 100 episodes、safe capture 97%、collision 1%、timeout 2%；predictor p95 约 167 ms、total p95 约 236 ms，仍不能作为实时部署结论。更早的逐周期刷新诊断保留为历史延迟证据。
- [x] 使用同一份 `scenes.jsonl` 完成 3 个 checkpoint × 5 个 checkpoint-dependent 方法，并复用 checkpoint-independent `dynamic_encirclement` baseline 展开为 3 × 6 对照表：`dynamic_encirclement`、`worst_case`、`distributed_ideal`、`distributed_delayed`、`distributed_dropout`、`distributed_none`。
- [x] 写入 `docs/PHASE4_UNSEEN_ADAPTIVE_VALIDATION_REPORT.md`，固定记录场景 hash、source hash、目标真值隔离、通信审计和三段延迟。
- [x] 完成 predictor p95 超过 100 ms 的低频预测 + 候选缓存 + 高频 planner/safety 诊断；总控制 p95 超过严格 10 Hz 部署参考，因此保留部署限制说明，但不阻塞 P4 方法验证，后续转向 batch/异步/蒸馏优化。
- [x] 完成固定 40-episode 场景上的 `8x8`、`4x8`、`2x8`、`8x4` 采样消融；四组 safe capture 均为 92.5%、collision 均为 7.5%，`2x8` 的 total p95 最低，但暂不据此替换主配置。
- [x] 完成 P5 execution delay/noise/tracking/randomization multi-step audit，加入真实 post-step 独立 checker；robust CBF-QP execution extension 为 No-Go，详见 `docs/PHASE5_EXECUTION_PERTURBATION_AUDIT_REPORT.md`。
- [ ] 在 P5 通过执行扰动扩展前，禁止训练或接入 learned CLBF，也禁止写“R-CLBF-QP 闭环安全证明”。
- [x] P5 验证通过后单独提交；不得将未验证的 `README.md` 或 `docs/EXPERIMENTAL_STUDY_REPORT.md` 加入提交。
- [ ] 每完成一个重大阶段，单独提交到 `origin/main`；提交前不 stage 用户已有的 `README.md` 或实验总结。P4 locked-test 报告和缓存延迟消融待本轮回归后提交。

### P5 当前证据和剩余工作

当前 hard-barrier 配置为：

- `slack_enabled: false`；
- `action_change_limit_mps: null`，由 `max_acceleration * dt` 推导；
- 鲁棒 margin 合计为 `0.46 m`，另加基础 safety margin `0.35 m`；
- fallback 为 `barrier_recovery`，并对 fallback 动作执行独立 checker；
- 独立 checker 重新计算 `p_next = p + dt * u_safe`，不复用 QP 内部 residual。

历史 8-seed 未分层 diagnostic 结果：

| 方法 | Safe Capture | Collision | Timeout | Solver success | Fallback | Next-state safety | Filter p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal | 100% | 0% | 0% | 100% | 0 | 98.6% | 0.002 ms |
| local CBF | 100% | 0% | 0% | N/A | 0 | 100% | 6.59 ms |
| robust CBF-QP hard | 100% | 0% | 0% | 84.7% | 19 | 88.7% | 4.86 ms |

该历史结果中 5/8 episode 的初始状态不在 robust 收缩安全集，因此不作为最终 P5 估计。正式分层结果为：tight `649118, 649121, 649122, 649129`，nominal `649101, 649104, 649107, 649111`；safe capture、initial robust-set valid、independent certificate valid、next-state safety 和 solver success 均为 100%，QP infeasible、solver failure、fallback、slack、collision 和 boundary violation 均为 0，robust QP p95 latency 为 6.27 ms。完整证据见 `docs/PHASE5_STRATIFIED_VALIDATION_REPORT.md` 和 `results/phase5_stratified_v8_seed649101/`。

当前 P5 结论是 **conditional pass**，不是无条件安全保证。执行扰动扩展已经完成但未通过。仍需完成：

- [x] execution delay/noise/acceleration/tracking perturbation Monte Carlo 和真实 post-step 多步审计；当前 robust CBF-QP 未通过，不能宣称 execution-invariant safety；
- [ ] continuous-time swept-volume safety；
- [ ] multi-step forward-invariance proof/audit under a filter contract that includes the execution state;
- [ ] 理论/独立校准集支持的 robust margin coverage；
- [ ] 将 P4 locked-test、退化通信和未见自适应目标纳入联合安全评估。

### P4 locked-test 正式协议

- 固定 100 个 `adaptive_adversarial` 场景，所有 checkpoint 和方法复用同一份 `scenes.jsonl`，最终报告必须给出 SHA-256。
- 运行矩阵为 3 个 prediction checkpoint（`745101`、`745201`、`745301`）× 6 个方法，共 18 个完整运行；每次均保留 episode/step JSONL、summary、TensorBoard、配置和源码 hash。
- 非 oracle 方法只能消费 projected candidates 和可见观测；目标真实状态只允许存在于环境内部的标签和最终评估统计。
- 每个方法必须报告 safe capture、ordinary capture、collision、boundary violation、timeout、capture time、minimum clearance、solver success、valid/effective/converged plan rate、fallback、message age/dropout/bytes 以及 predictor/planner/QP/total p50/p95/p99 latency。
- P4 通过条件在正式 locked-test 打开前冻结：collision 不高于基线超过 1 个百分点；safe capture 的 aggregate 不低于基线，且至少一个 hard stratum 的 safe capture 或最坏候选距离改善 5% 以上；solver/valid/effective rate 不低于 99%；所有 fallback 和通信失败均可追溯。
- 若每步刷新 predictor 时总 p95 超过 100 ms，必须采用并审计低频刷新策略：记录候选年龄、刷新周期和 stale-candidate fallback；这份延迟契约用于明确部署限制和工程权衡，不把 100 ms 超限作为 P4 方法验证的硬门槛。

### P5 执行扰动恢复路线

当前审计已经证明静态 margin 不能覆盖延迟、tracking 和实际 post-step 状态。恢复路线如下：

- [ ] 将延迟命令队列、执行速度/跟踪状态和随机执行参数纳入安全过滤器状态，而不是只在命令速度上加 margin。
- [ ] 让环境 `step()`、独立 dynamics rollout、QP 约束和 certificate checker 使用同一执行动力学。
- [ ] 由理论 reachable-set bound 或独立 calibration set 冻结 observation/delay/tracking/noise 的 margin，并报告覆盖率。
- [ ] 增加 swept-volume 连续时间检查和多步 post-step forward-invariance audit；命令证书不能替代实际执行证书。
- [ ] 在 mild 和 hard 两类扰动、窄通道、障碍拐角、边界、近距离机间场景上重跑 8 个固定 seed，并报告 safe capture、termination、fallback、certificate invalid 和实际安全率。
- [ ] 只有在执行扰动扩展通过后，才开始 learned CLBF；否则将安全贡献固定命名为条件性 robust CBF-QP。

P5 恢复的停止条件：实际 post-step safety 低于 99%、QP/fallback 失败率高于 1%、或安全率提升以不可接受的 timeout/capture 损失为代价时，停止继续增大 margin，改为报告安全--效率 Pareto 或回退到可审计的一步过滤器。

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

- [x] `--target-normalization train_split_standardize`，归一化器只能拟合 train split。
- [x] 使用训练配置中的正式 history/horizon/diffusion/sampling 参数。
- [x] 三个训练 seed 只改变随机初始化和训练采样，不改变数据切分、网络规模和评估协议。
- [x] evaluation sampling seed 独立于 training seed，并在所有模型间保持可复现。
- [x] `candidate-max-speed` 使用物理硬上限，不使用行为强度 `target_speed_scale` 代替。
- [x] 任何候选可行率不足的结果同时报告原始候选和 dynamics-projected 候选，没有静默删除失败样本。

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
| `phase2_prediction_train_v3_multimodal` | 64 | 15,232 | 645101--645164 | 训练和 train-only normalization |
| `phase2_prediction_validation_v3_multimodal` | 24 | 5,712 | 646101--646124 | 模型选择、conformal/calibration |
| `phase2_prediction_locked_test_v3_multimodal` | 32 | 7,616 | 647201--647232 | 最终一次性评估 |

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
- [x] 增加基于追捕机状态、障碍物和边界的自适应逃逸策略，并完成物理可行性检查。
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

- [x] 选择轨迹空间：当前固定为相对团队 belief reference 的未来位置位移。
- [x] 定义扩散目标为未来位置序列或未来速度序列，并固定一种主方案；当前使用相对团队 belief reference 的未来位置位移。
- [x] 实现 noise schedule、训练 loss 和条件注入。
- [x] 支持候选轨迹 batch sampling。
- [x] 输出候选轨迹分数接口；当前明确记录为 `uniform_uncalibrated`，不能解释为概率。
- [x] 对原始和 dynamics-projected 候选分别进行速度、加速度、边界和障碍物可行性检查。
- [x] 设计快速采样路径：当前使用 few-step DDIM 风格采样。
- [x] 提供 deterministic seed，保证训练和采样可由固定配置重放。
- [x] 对所有预测输出做坐标反归一化和 finite 检查。

## 5.4 置信度与校准任务

- [x] 定义当前置信度输出为 split-conformal full-trajectory coverage；候选分数仍是未校准均匀权重。
- [x] 在 validation set 上做 split-conformal calibration；尚未把未校准分数包装成 mode probability。
- [x] 统计校准半径和全轨迹/逐 horizon 覆盖率。
- [ ] 测量 misspecified target mode 下的置信度退化。
- [x] 记录 top-1、top-`K`、均匀候选；oracle mode 尚未作为主结果使用。
- [x] 未使用未校准的候选分数作为概率。

## 5.5 预测评估指标

- [x] ADE：平均位移误差。
- [x] FDE：终点位移误差。
- [x] minADE/minFDE：候选集合覆盖能力。
- [x] energy score。
- [ ] CRPS 或等价分布质量指标。
- [x] 90% conformal coverage；50/80/95% coverage 尚未作为正式表格输出。
- [x] 校准半径和覆盖率审计。
- [x] 长时域 full-trajectory/per-horizon 覆盖结果。
- [x] 五种目标模式的分层结果；观测退化组合仍待 Phase 1/3。
- [x] 单步推理 p50/p95/p99 延迟。

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

- [ ] 在所有测试模式上，Mamba-Diffusion 的 minFDE 至少比 GRU 基线降低 10%，或在相同误差下达到更高的 coverage；当前 locked-test 仅达到 raw `6.54%`、projected `3.87%` 的平均 minFDE 改善，且 coverage 略低。
- [x] 90% conformal 目标的经验覆盖率位于 `0.85--0.95`，但按模式仍需继续做未见策略测试。
- [x] dynamics-projected 预测输出的可行候选比例不少于 95%；raw diffusion 候选仍为 0%，不能直接执行。
- [ ] 在目标策略切换后仍有可解释的置信度退化，而不是输出异常高置信度。
- [x] 已记录单模型 p95 推理时间；100 ms 仅作为严格 10 Hz 部署参考，总 planner/safety 延迟已通过低频缓存协议量化。
- [x] planner 接口固定为候选轨迹 + 明确的 raw/projected 状态 + `uniform_uncalibrated` score kind。

正式结果见 `docs/PHASE2_FORMAL_ANALYSIS_REPORT.md`。Phase 2 当前为 `Conditional Go`，因此下一步只实现集中式 scenario min-max MPC 诊断，不宣称最终 DN-MPC 或完整方法已经通过。

如果只提升 ADE/FDE，却没有提升候选覆盖率或校准质量，则不能声称“多模态预测成功”，只能称为更强的点预测器。

---

## 6. Phase 3：集中式 Scenario Min-Max MPC

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

## 6.3 分布式 DN-MPC 机制任务（Phase 4）

- [x] 规定每架无人机可获得的信息范围。
- [x] 规定共享消息内容：邻机状态、预测轨迹、候选控制序列或角色信息。
- [x] 规定消息延迟、丢包和通信频率。
- [x] 实现本地 MPC 子问题。
- [x] 实现 sequential best-response 作为第一种分布式机制。
- [x] 规定每个控制周期的最大迭代次数。
- [x] 记录未收敛次数和局部求解失败次数。
- [x] 设计通信中断时的本地 fallback。
- [x] 验证分布式输出与集中式 oracle 的安全指标差距。

## 6.4 求解器和实时性任务

- [ ] 评估 SciPy、OSQP、CVXPY/Clarabel 或 CasADi 的依赖和 Windows 安装稳定性。
- [ ] 第一版优先使用线性化动力学和 QP/二次代价，降低实时风险。
- [ ] 为非凸障碍物约束设计局部线性化或安全层后置处理。
- [x] 设置 solver time limit。
- [x] 设置 infeasible、timeout、NaN 和异常数值的 fallback。
- [x] 记录每步 solver status、迭代次数、目标值、最大约束残差和运行时间。
- [x] 对所有 planner output 做动作范围检查。

## 6.5 规划评估指标

- [x] Cooperative Safe Capture rate。
- [ ] worst-case candidate capture rate。
- [ ] expected candidate capture rate。
- [ ] CVaR capture distance。
- [x] capture time。
- [x] collision and boundary violation。
- [x] minimum clearance。
- [ ] role switching frequency。
- [x] solver success rate。
- [x] p95/p99 planning latency。
- [x] 通信量和通信失败后的恢复时间。

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

## 6.7 Phase 3 集中式规划通过条件

- [x] 在相同候选轨迹和相同场景种子下，worst-case 或 CVaR 至少改善一个困难层指标。
- [x] 在 S3 随机混合障碍物场景下，安全捕获率相对 DynamicEncirclement 提升至少 5 个百分点。
- [x] collision rate 不高于基线超过 1 个百分点。
- [x] solver success rate 不低于 99%，剩余失败均有记录过的 fallback。
- [x] 已明确采用低频 prediction cache + 高频 planner/safety 架构，并记录总控制 p95；100 ms 仅作为严格 10 Hz 部署参考，不作为 P4 方法验证硬门槛。
- [x] 仅使用预测候选而不访问目标真值。

这些条件只证明集中式 scenario planner 的 P3 validation gate，不证明分布式 DN-MPC。

## 6.8 Phase 4 分布式 DN-MPC 通过条件

- [x] 在 delayed-noisy 条件下，安全捕获和 collision 未出现下降；性能代价已记录。
- [x] 分布式结果与集中式 oracle 的主要安全指标差距不超过 10 个百分点，或能解释差距来自通信限制。
- [x] solver success rate 不低于 99%，剩余失败均有安全 fallback；三种 seed 均为 100%。
- [x] 每个 seed 的 effective/converged plan rate 不低于 99%；四种通信模式在三 seed 上均为 100%。
- [x] p95 planning latency 在单个控制周期内；修复后三种 seed 的分布式 planner p95 均值为 8.72--12.47 ms，total-control p95 均值为 50.38--51.51 ms。
- [x] 仅使用预测候选而不访问目标真值。

如果 DN-MPC 只有在访问目标真值时有效，则该模块判定为实验诊断工具，不能作为方法主体。

### 当前 Phase 3 诊断状态

- [x] 已实现集中式 finite-shooting scenario MPC，支持 expected、worst-case 和 CVaR。
- [x] 已实现 projected-candidate contract、solver/fallback diagnostics、逐步 JSONL 和 TensorBoard 记录。
- [x] 已完成 8 个固定 seed 的 formal-small 诊断；四种配置均无碰撞，total control p95 约为 45--52 ms。
- [x] 已在 S3 随机混合障碍物、delayed_noisy 和 s_curve 条件下完成 12 个共享场景的 validation。
- [x] 已保存候选级 minimum/terminal distance、scenario cost、worst/CVaR 聚合统计。
- [x] 已使用 prediction checkpoint seed `745101/745201/745301` 重复运行，三次 `scenes.jsonl` hash 一致且方向一致。
- [x] 已完成 ideal、2-step delayed、10% dropout 和 no-communication 四种 P4 通信条件，并保留 TensorBoard、episode/step JSONL 和 summary JSON。
- [x] 已确认四种分布式条件均为 100% safe capture、0% collision，DynamicEncirclement 为 91.7% safe capture、8.3% collision。
- [x] locked-test 和未见自适应目标策略已完成，详见 `docs/PHASE4_UNSEEN_ADAPTIVE_VALIDATION_REPORT.md`。
- [x] planner p95 和 per-seed effective/converged plan rate 在固定 S3 validation gate 中通过；adaptive locked-test 的规划效果门槛也通过，低频缓存延迟作为工程权衡记录，且仍无形式化博弈保证。

正式验证报告见 `docs/PHASE3_S3_VALIDATION_REPORT.md` 和 `docs/PHASE4_DN_MPC_VALIDATION_REPORT.md`；当前结论是集中式 planner 和固定 S3 分布式 validation gate 均已通过，但泛化、R-CLBF-QP 和端到端结果仍待验证。

---

## 7. Phase 5：robust CBF-QP 与 R-CLBF-QP 鲁棒安全过滤

## 7.1 先明确证明对象

第一版不应直接声称“对真实无人机闭环安全”。应先证明以下较窄命题：

> 在给定离散运动学/动力学模型、已知障碍物几何、有限观测误差、有限执行扰动和 QP 可行的条件下，安全过滤后的状态保持在定义安全集内。

之后再扩展到执行动力学和更强的扰动模型。

## 7.2 安全集和 barrier 任务

- [x] 定义障碍物安全函数 `h_obstacle`。
- [x] 定义世界边界安全函数 `h_boundary`。
- [x] 定义机间安全函数 `h_inter_agent`。
- [x] 明确当前实现使用 signed clearance/净空，而不是平方距离。
- [x] 为圆柱和轴对齐箱体提供几何处理；拐角和不可微区域仍需 hard-case 扫描。
- [x] 明确目标本身不加入追捕机排斥障碍集合。
- [x] 将观测误差、延迟和执行误差以显式 margin 写入配置；预测误差 margin 的校准来源仍待冻结。
- [ ] 用理论上界或独立校准集冻结每一项 robust margin，并证明实际扰动被覆盖。

## 7.3 CLBF/QP 任务

建议先实现安全 QP，再加入 learned robust CLBF，防止理论和求解器问题同时出现。

- [x] 定义 QP 决策变量：当前控制量和可选 slack。
- [x] 定义名义动作跟踪目标。
- [x] 加入速度/加速度/动作变化约束。
- [x] 加入当前 velocity-level 离散 CBF 约束；高阶 CBF 仍未实现。
- [x] 对机间约束处理相对控制耦合。
- [x] 对边界和障碍物处理动作饱和。
- [x] 实现 slack 惩罚；hard-barrier 主配置禁用 slack，soft-slack 只能诊断。
- [x] 实现 solver infeasible 的应急动作，并将 certificate 标记为 invalid。
- [x] 记录 barrier 值、残差、slack、active 状态、修正量和 solver status 的接口。
- [x] 补齐失败原因和数值条件的逐步持久化，并验证 fallback 本身满足一步安全契约。
- [ ] 为 learned CLBF 记录网络版本、训练域、验证域和 Lipschitz/gradient bound。

## 7.4 安全证明任务

- [x] 写出当前速度级系统动力学假设。
- [ ] 写出并冻结观测、延迟、执行和预测扰动集合及每一项上界。
- [x] 对 nominal action 到 QP action 的约束残差进行数值检查。
- [x] 明确当前独立 checker 只能给出一步安全检查；离散时间正向不变性仍未证明。
- [ ] 证明 slack 不会掩盖安全约束违反；若允许 slack，给出可接受条件。
- [ ] 证明动作饱和、延迟和求解误差如何进入 margin。
- [ ] 给出 solver numerical tolerance 对 barrier 的最坏影响。
- [ ] 用独立数值脚本检查理论 bound 是否覆盖实际扰动样本。
- [x] 明确 QP infeasible、fallback、初始状态越界、下一状态越界、非有限动作和扰动假设不覆盖时证书自动失效。

## 7.5 R-CLBF 训练和验证任务

- [ ] 生成安全/危险状态样本和边界附近 hard cases。
- [ ] 训练 CLBF 或 robust value function 时禁止使用测试场景标签。
- [ ] 检查 barrier value、gradient、Lie derivative 和 residual 的范围。
- [ ] 在障碍拐角、窄通道、机间近距离和边界附近做网格扫描。
- [ ] 做随机扰动 Monte Carlo 验证。
- [ ] 做 action delay 和 command noise stress test。
- [ ] 验证 learned CLBF 不会在训练域外产生过度自信的安全证书。

## 7.6 安全层通过条件

- [x] 在可行 QP 样例中检查最大安全约束残差；分层 formal gate 已通过。
- [ ] 证书假设覆盖评估中实际使用的速度、延迟、噪声和扰动范围。
- [x] 在冻结分层协议上 infeasible 比例为 0，且 fallback 不产生未记录安全违规；执行扰动扩展已测但未通过。
- [ ] 在相同 planner 输出下，R-CLBF-QP 不增加碰撞率。
- [ ] 与当前局部 CBF 比较时，至少改善一个难例安全指标或明显降低动作修正量/超时率。
- [x] 独立一步 checker 与当前 barrier 定义对齐；完整证明文档仍未完成。

### P5 当前决策

- 工程实现等级 `L0`：通过，9 个安全单元测试通过。
- 可审计一步安全检查：通过接口验证；失败原因、命名残差和 fallback checker 已归档。
- `robust CBF-QP safety filter` 主结果：在冻结分层 robust-safe reset、速度级动力学和一步 checker 假设下条件性通过。
- `R-CLBF-QP` / 闭环形式化保证：未开始验收，不得使用该表述。

如果只能证明“QP 求解器通常找到了较安全的动作”，则应称为 robust safety filter，不能称为闭环形式化安全证明。

---

## 8. Phase 6：端到端集成

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

## 9. Phase 7：正式实验和统计协议

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

以下按一个人、已有当前仓库和一张中等 GPU 估计，实际以阶段门槛为准；从当前状态重新计算：

| 周期 | 工作 | 产物 |
| --- | --- | --- |
| 第 1 周 | P4 locked-test 补齐与报告 | 18-run matrix、common-scene hash、P4 report |
| 第 2--3 周 | P4 延迟架构和退化观测复验 | cached prediction protocol、latency report |
| 第 4--5 周 | P5 执行状态建模与安全审计 | augmented filter、swept-volume、certificate report |
| 第 6 周 | P6 learned CLBF（仅当 P5 通过） | CLBF checkpoint、独立 checker |
| 第 7--8 周 | 端到端消融和三种子复验 | ablation matrix、E2E summaries |
| 第 9 周 | 统计、失败分析和论文材料 | final report、复现脚本、模型清单 |

如果 Phase 2 不满足预测门槛，应暂停后续扩展，先检查数据和目标策略；如果 Phase 3 不满足实时性，应采用低频 planner + 高频安全层；如果 Phase 5 无法形成可审计证明，应降低论文表述为经验型鲁棒安全过滤。

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

当前不应回到“先把所有模块写完”的路线，而应按以下顺序收敛：

1. 保留 P4 locked-test 三 checkpoint × 六方法结果作为泛化证据；固定 S3 的 100% 结果和 adaptive locked-test 的 98--99% 结果都不能外推为真实部署结论。
2. 针对执行扰动 No-Go，重构包含延迟队列与 tracking state 的安全过滤契约，再重做 continuous-time swept-volume 和 multi-step audit。
3. 只有 P4 泛化和 P5 扩展安全审计均通过，才做 P6 learned CLBF；如果只能稳定完成一步检查，就把贡献限定为条件性 robust CBF-QP。
4. 只有 P4 泛化、P5 扩展安全审计和 P6 证书边界均通过，才做 Phase 6 端到端三种子 locked-test；联合训练放到所有冻结模块之后。
5. 任何一级失败都保留上一级已证实贡献：预测失败回退到校准预测，分布式失败回退到集中式 scenario MPC，CLBF 失败回退到 robust CBF-QP，端到端失败则分模块报告。
