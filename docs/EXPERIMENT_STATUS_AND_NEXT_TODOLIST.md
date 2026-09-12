# 当前实验状态与后续 TodoList

> 更新时间：2026-09-11
> 仓库：[Liangxianguang/new-uav-capture](https://github.com/Liangxianguang/new-uav-capture)
> 当前总判定：**Conditional Go**。预测与 DN-MPC 已形成可复现的模块化证据；robust CBF-QP 只通过了冻结条件下的一步安全 gate；三者直接端到端组合失败，完整方法尚未完成。

> Phase 16 v4 更新：扩建的 600 场景 S4 原始档案已按 300 个镜像组冻结为 420/90/90 场景的 train/validation/locked-test；由公开 team belief 重新构建预测原点后得到 13,224/1,922/1,539 个窗口。预测的 validation-only 选型和 locked-test 预测确认均已完成：GRU 是误差主参考，Diagonal SSM、dense S4 与 official S4 保留为 K=8 多模态比较，不能声称 S4 优于 GRU。三种子 validation 闭环选择冻结为每步刷新、`both` 因果动作条件和 local CBF；GRU + distributed delayed DN-MPC 的 validation safe capture 为 `93.70% [90.74%, 96.30%]`，collision 为 `0%`、total p95 为 `44.78 ms`。locked-test 三种子矩阵现已完成：GRU K=1 的 distributed safe capture 为 `94.81% [91.85%, 97.41%]`、total p95 为 `43.58 ms`；Diagonal SSM K=8 与 official S4 K=8 均为 `95.56% [92.96%, 97.78%]`、0 collision，但 total p95 分别为 `90.81/90.02 ms`。相对 GRU 的 distributed gain 均为 `+0.74` 点且 CI 下界为 0，不能声称主分支提升。新增严格 K=1/K=8 配对消融显示：主 distributed 分支均为 95.56% safe capture，候选数无可见收益；worst-case 中 K=8 对 Diagonal SSM 增益 `+7.78` 点 `[+3.70,+12.22]`，对 official S4 增益 `+5.56` 点 `[+1.85,+9.26]`，且 K=1 出现少量 collision/boundary failures、K=8 为 0。因此多候选的当前证据限于保守 worst-case 规划，不可泛化为所有场景收益。新增 GRU none/local-CBF 三种子安全消融显示：distributed 中 local CBF 将 collision 从 `5.56%` 降到 `0%`（paired `-5.56` 点 `[-8.89,-2.96]`），safe capture 差异不显著；worst-case 中 local CBF 同样消除 collision，但 safe capture 下降 `8.15` 点 `[-14.44,-1.48]`，表现为安全--捕获 Pareto，非无代价优势。新增 action-conditioning 闭环消融显示 `none/history/future` 相对 `both` 的 safe-capture paired CI 均跨 0；当前只支持动作条件接口已因果实现和审计，不能支持“动作条件提升闭环捕获率”。并行运行的 221--223 ms p95 不可与 earlier `both` 批次作因果延迟比较。robust CBF-QP diagnostic 仍为 No-Go。完整记录见 `docs/PHASE16_LOCKED_TEST_CLOSED_LOOP_REPORT.md`。

> Phase 16 几何 OOD 更新：新增并冻结 100 个 episode / 50 个上下镜像组的几何外推诊断；wall half-x、half-y、height 均与训练范围不重叠。三种子 GRU `both` + distributed delayed DN-MPC + local CBF 的 safe capture 为 `84.00% [79.33%, 88.33%]`，collision / boundary 均为 `0%`，timeout 为 `16.00% [11.67%, 20.67%]`，total p50/p95/p99 为 `72.93/93.60/111.26 ms`。同一 OOD 块上 worst-case 仅为 `46.33% [39.67%, 53.33%]`，且 collision / boundary 各为 `1.67%`，不可作为 OOD 几何主方案。该块只改变几何，且与 locked-test 场景文件不同，因此相对 ID `94.81%` 的下降只能作为描述性 transfer gap，不能写成配对显著性结论，更不能写成全面鲁棒性或安全证明。详见 `docs/PHASE16_OOD_GEOMETRY_REPORT.md`。

> Phase 16 目标速度 OOD 更新：新增并冻结 `0.82/0.90` 两个训练外速度档，各 50 个 episode，共 50 个上下镜像组，几何仍在训练范围内。三种子 GRU `both` + distributed delayed DN-MPC + local CBF 的 safe capture 为 `42.67% [37.00%, 48.33%]`，collision / boundary 均为 `0%`，timeout 为 `57.33% [51.67%, 63.00%]`，total p50/p95/p99 为 `83.94/106.72/130.87 ms`。worst-case safe capture 为 `55.00% [47.33%, 63.00%]`，timeout 为 `45.00%`，collision / boundary 均为 `0%`。相对 ID `94.81%` 的差异因场景文件不同仅作描述性 transfer gap；该结果表明速度外推是当前主要性能瓶颈，不代表安全证明。详见 `docs/PHASE16_OOD_TARGET_SPEED_REPORT.md`。

> Phase 16 通信/执行 OOD 更新：新增并冻结 `delay6_dropout20` 与 `delay8_dropout25` 两个感知条件，各 50 个 episode，并真正激活 2-step command delay、执行噪声、速度/加速度随机缩放和一阶跟踪 dynamics。三种子 distributed delayed + local CBF 的 safe capture 为 `47.00% [41.00%, 53.00%]`，collision / boundary 为 `53.00% / 19.00%`，total p50/p95/p99 为 `86.09/122.82/178.46 ms`；worst-case safe capture 为 `53.67%`，collision / boundary 为 `46.33% / 24.00%`。这是当前执行感知安全契约的 No-Go 诊断，不是模型成功结果或安全证明。详见 `docs/PHASE16_OOD_DELAY_EXECUTION_REPORT.md`。

> Phase 16 目标行为 OOD 更新：新增并冻结 `short_lookahead`（0.20 s）与 `long_lookahead_margin`（1.20 s / 0.20 s）两种训练外的目标分支决策规则，各 50 个 episode，保持速度、几何和执行配置在 ID 支持内。三种子 distributed delayed + local CBF 的 safe capture 为 `94.33% [91.33%, 97.33%]`，collision / boundary 均为 `0%`，timeout 为 `5.67%`，说明该两种行为参数外推没有可见退化；这只是单轴描述性证据，不能外推到任意对抗策略。详见 `docs/PHASE16_OOD_TARGET_BEHAVIOR_REPORT.md`。

> Phase 15 v3 正式更新：600 场景的数据划分和动作/时间戳契约审计已通过；30 epoch、3 seed 的冻结离线测试中，GRU 的 projected minFDE 为 `0.7244 +/- 0.0352 m`，官方 S4 为 `1.3373 +/- 0.0275 m`。每步刷新且因果动作条件可用率约 `92%--95%` 时，GRU + distributed delayed DN-MPC 为 `95.56% [92.96%, 97.78%]` safe capture，官方 S4 为 `94.44% [91.48%, 97.04%]`。不能声称 S4 优于 GRU；下一步是 validation-only 的 `none/history/future/both`、单/多模态和风险/刷新消融。详见 `docs/PHASE15_S4_V3_FORMAL_MULTISEED_REPORT.md`。

> Phase 15 action-conditioning 消融已完成：GRU 在相同 30 epoch、3 个匹配 seed 下，`both` 的 validation minFDE 为 `0.8000 +/- 0.0064 m`，优于 `future` 的 `0.8388 +/- 0.0099 m`、`history` 的 `0.9994 +/- 0.0076 m` 和 `none` 的 `1.0288 +/- 0.0101 m`；相对 `both` 的配对 minFDE delta 区间均为正。后续冻结测试确认了这一排序，`both` 的 projected minFDE 为 `0.7000 +/- 0.0057 m`、coverage 为 `94.59%`，但 candidate feasibility 只有 `77.60%`。`both` 已锁定为后续主配置，不能继续用 locked test 选模型；下一步是 no/local/robust safety 三路闭环和单/多模态消融。详见 `docs/PHASE15_S4_V3_ACTION_CONDITION_ABLATION_REPORT.md`。

> 最新 Phase 14 authority matrix：hard `immutable` / `replace_nonexecuting` / `flush_pending` 的 safe capture 分别为 `12.5% / 0% / 25.0%`，actual post-state safety 为 `29.55% / 90.05% / 99.75%`，collision/boundary 均为 `0% / 0%`。`flush_pending` 仍只是仿真执行器能力，不能直接外推到真实飞控；完整执行感知安全控制仍为 No-Go。详见 `docs/PHASE14_AUTHORITY_MATRIX_REPORT.md`。

## 1. 先给结论

目前不能把项目描述为“实验已经完善”或“R-CLBF-QP 闭环安全证明已经成立”。更准确的表述是：

1. Mamba/portable SSM + 条件扩散预测器在未见 `adaptive_adversarial` 目标策略上有效，候选轨迹必须经过动力学投影。
2. 投影候选驱动的 worst-case / distributed DN-MPC 在 100 个 locked-test 场景上达到 97.33%--99.00% safe capture，优于 95.00% 的 DynamicEncirclement 基线。
3. velocity-level robust CBF-QP 在预先筛选的 robust-safe reset 上通过了一步独立证书检查，但这不是执行扰动下的闭环安全证明。
4. 将 robust CBF-QP 直接接入 P4 流程后，safe capture 只有 50.00%，collision 为 49.00%，boundary violation 为 16.00%；因此当前端到端组合路径为 **No-Go**。
5. 当前 v4 独立证书审计目录只有配置和场景文件，没有完整 `summary.json`、episode/step 日志，不能作为新实验结果引用。

6. 最新 Phase 14 八种子 authority matrix 已完成，但三种权限均未达到 hard `80%` safe-capture 工作门槛；下一步必须先做证书/队列失败分解，再扩大 unseen seed。
7. 最新 continuous-QP progress/recovery audit 仍未通过：2-seed/250-step hard `flush_pending` 的 safe capture 为 `0%`，actual post robust safety 为 `71.4%`，continuous certificate 为 `12.4%`；重复 emergency brake 消除了该小样本中的物理碰撞/越界，但没有恢复 robust tube 不变性。详见 `docs/PHASE14_PROGRESS_RECOVERY_REPORT.md`。

## 2. 当前捕获率

这里的主指标是 `safe capture`：目标进入捕获半径且没有被碰撞、越界或其他安全失败污染。普通 `capture` 不等于安全捕获，不能只报告普通捕获率。

### 2.1 P4：预测 + DN-MPC，100 个未见策略 locked-test

所有方法复用同一份 `adaptive_adversarial` 场景文件；下表为三种 prediction checkpoint 的聚合结果，DynamicEncirclement 是共享场景基线。

| 方法 | Safe capture | Collision | Boundary | Timeout | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| DynamicEncirclement baseline | 95.00% | 2.00% | 0.00% | 3.00% | 基线 |
| Worst-case DN-MPC | 97.33% | 1.00% | 0.00% | 1.67% | 通过 |
| Distributed ideal | 98.00% | 1.00% | 0.00% | 1.00% | 通过 |
| Distributed delayed | **99.00%** | **0.00%** | 0.00% | 1.00% | 通过 |
| Distributed dropout | **99.00%** | **0.00%** | 0.00% | 1.00% | 通过 |
| Distributed none | 98.00% | 0.00% | 0.00% | 2.00% | 通过 |

P4 的 planner solver/valid/effective rate 为 99.97%--100%，没有出现正式结果级别的 planner fallback。这个结果支持“预测候选能够转化为围捕收益”的模块化结论，但不等于真实飞行安全保证。

### 2.2 P5：robust CBF-QP 独立安全过滤

在冻结的 8 个 robust-safe episode、130 个控制步上：

| 指标 | 结果 |
| --- | ---: |
| Safe capture | 100%（8/8） |
| Collision / boundary violation | 0% / 0% |
| Solver success | 130/130 |
| Independent one-step certificate | 100% |
| Next-state safety | 100% |
| Fallback / QP infeasible | 0 / 0 |
| Filter p95 latency | 6.27 ms |

这是一个**有前置条件的 velocity-level 一步安全结果**。执行延迟、跟踪误差、动作队列、连续段 swept-volume 和多步 forward invariance 审计均未通过，因此不能命名为 R-CLBF-QP，也不能写成闭环安全证明。

### 2.3 P7：三模块直接组合

同一份 P4 100-episode locked-test 上，使用 worst-case DN-MPC + robust CBF-QP：

| 指标 | Local CBF baseline | Joint robust CBF-QP |
| --- | ---: | ---: |
| Safe capture | 97% | **50%** |
| Collision | 1% | **49%** |
| Boundary violation | 0% | **16%** |
| Safety certificate valid | n/a | 64.87%（原始诊断） |
| Safety fallback | n/a | 35.13% |
| Planner success / valid / effective | n/a | 100% / 100% / 100% |

5,289 个控制步中的 1,858 个 fallback 已分类为：`precondition_invalid=1,225`、`qp_infeasible=350`、`inconsistent_action_bounds=283`。因此主要问题在安全层的 reset/margin/action-bound/fallback 契约，不是 DN-MPC planner。

## 3. 预测模块目前能声称什么

在未见 `adaptive_adversarial` 审计中，三种冻结训练 seed 的均值为：

| 模型 | Projected minFDE | Full-trajectory coverage |
| --- | ---: | ---: |
| GRU | 0.9033 ± 0.0648 m | 82.22% ± 4.86% |
| Projected SSM diffusion | **0.5363 ± 0.0071 m** | **90.31% ± 0.03%** |

相对 projected GRU，diffusion 的 minFDE 改善 40.63%，coverage 提高 8.09 个百分点。但原五模式正式 locked-test 的 minFDE 仅改善 3.87%，没有达到预注册的 10% 门槛；候选权重仍是 `uniform_uncalibrated`，不能解释为每条轨迹的概率。

因此预测模块是 **Conditional Go**，不是无条件通过。

## 4. 延迟结论

P4 使用每 20 个控制步刷新预测并缓存候选。三 checkpoint 聚合的 total-control p95 约为 167.73--229.62 ms，严格 100 ms/10 Hz 参考尚未达到；但用户已明确 100 ms 不作为硬门槛，因此它属于部署架构和工程优化项，不推翻 P4 方法结果。

后续应分别报告 predictor、planner、safety 和 total-control 延迟，并验证 batch inference、异步预测、少步扩散或蒸馏方案。不能用低频缓存的平均收益掩盖候选年龄和执行安全问题。

## 5. 实验完成度判断

| 层级 | 当前状态 | 是否完成 |
| --- | --- | --- |
| 工程实现、日志、基础回归 | 代码和主要审计链已具备 | 基本完成 |
| SSM/扩散预测贡献 | 未见策略结果成立，但原始 10% 门槛未过 | 条件完成 |
| Scenario MPC / DN-MPC | S3 与 adaptive locked-test 已通过 | 完成当前模块 gate |
| 一步 velocity-level robust CBF-QP | 冻结 robust-safe reset 上通过 | 条件完成 |
| 执行扰动下安全 | 多轮 audit 为 No-Go | 未完成 |
| R-CLBF-QP 形式化证明 | 尚未成立 | 未完成 |
| 三者端到端完整方法 | 50% safe capture / 49% collision | 未完成，No-Go |
| P7.2 独立证书复核 | v4 运行缺少完整结果工件 | 未完成 |

结论：**P4 模块化实验已经完善到可以写阶段性结果；P5/P7 还没有完善到可以宣称完整方法成功。**

## 6. 接下来的 TodoList

### P7.2：完成独立证书审计

- [ ] 先确认 v4 目录是否因中断而只有 `config.yaml`、`protocol.yaml`、`scenes.jsonl`；当前已确认缺少完整 summary 和 episode/step 日志。
- [ ] 使用与 P7 原始 v2 完全相同的 100 个场景、checkpoint、MPC 配置和刷新周期，重新运行独立 `check_one_step_safety` 审计。
- [ ] 验证 `summary.json`、`episodes.jsonl`、`steps.jsonl`、TensorBoard event、配置快照和 source hash 全部落盘。
- [ ] 对比重跑前后的 safe capture、collision、boundary violation、planner success，确认新增 checker 不改变控制行为。
- [ ] 汇总 independent certificate valid、current-state safety、next-state safety、barrier 和 violation counts。
- [ ] 更新 `PHASE7_JOINT_SAFETY_AUDIT_REPORT.md` 与 `PHASE7_SAFETY_CONTRACT_DIAGNOSTIC_REPORT.md`；若重跑失败，明确记录失败原因，不把缺失结果当成通过。

**P7.2 gate：** 结果工件完整、控制结果与原始 v2 一致、独立证书指标可逐步追溯。该 gate 只完善审计，不会自动把 P7 的 No-Go 改成 Go。

### P8：修复安全契约，而不是继续堆叠模型

- [x] 完成最新 hard `immutable`、`replace_nonexecuting`、`flush_pending` 八种子 authority matrix；结果为执行审计证据，仍未通过 hard 捕获与连续证书 gate。
- [ ] 按队列前缀、horizon index、barrier family、execution error 和 fallback candidate 分解 Phase 14 失败。
- [ ] 在同一批 step states 上对比 horizon `1/2/3/5`，区分证书保守性与真实状态离开安全集。

- [ ] 冻结 P4 的同一组场景，分别复现 local CBF、当前 robust CBF-QP 和 nominal action。
- [ ] 将 `precondition_invalid`、`qp_infeasible`、solver numerical failure、fallback 后证书失败严格分开统计。
- [ ] 重新定义 reset protocol：初始状态必须明确属于 robust safe set，并分别测试 tight / nominal / out-of-contract 三类状态。
- [ ] 统一 safety margin、速度盒约束、加速度约束、动作变化约束与环境实际执行模型。
- [ ] 明确 pending command 的 authority；不可修改的 queue prefix 必须进入 certificate，而不是只检查当前动作。
- [ ] 设计经过独立 checker 的 certified fallback，并记录 fallback 前后 current/next/prefix/swept safety。
- [ ] 在 mild / hard delay-noise-tracking 扰动下执行多步 post-step 与 continuous-segment audit。
- [x] 修正 delayed fallback 的进展评分：在 `queue length + 1` 的首个可控时刻评价新动作，而不是只评价旧队列前缀。
- [x] 对 prefix 仍不可行的状态重复保持 emergency brake；只有 prefix 恢复可行后才允许 resume。
- [x] 修正 `actual_post_robust_state_safe` 的统计口径，使其真正包含 reachable-tube robust barrier。
- [ ] 增加 braking/viability-aware recovery guard，在当前状态进入不可恢复区之前触发保守恢复。
- [ ] 将 safe abort、robust-contract violation 和 physical collision 分开建模，并在 episode 终止协议中明确其优先级。

**P8 gate（建议预注册）：** 初始契约有效率 100%；post-step independent safety、fallback 后安全率和关键场景证书覆盖率至少 99%；QP infeasible、未解释 solver failure 和未认证 fallback 均低于 1%；collision/boundary 不得劣于 local CBF baseline；否则停止扩大模型，报告安全--捕获 Pareto。

### P9：运行时工程优化

- [ ] 先保留当前 8×8 配置作为主结果，不因一次 40-episode 消融直接替换。
- [ ] 在同一 locked-test 上比较 batch inference、异步 predictor、少步 diffusion、蒸馏和候选缓存年龄。
- [ ] 报告 predictor/planner/filter/total p50、p95、p99 及 stale-candidate 比例。
- [ ] 100 ms 只作为部署参考；若仍超过参考值，给出明确的低频预测 + 高频 planner/filter 架构和安全假设。

### P9.1：Phase 16 OOD 泛化矩阵

- [x] 完成 geometry-only OOD：100 episode / 50 镜像组，三种子 distributed 分支为 `84.00% [79.33%, 88.33%]` safe capture，0 collision/boundary；worst-case 为 `46.33% [39.67%, 53.33%]` 且各有 `1.67%` collision/boundary。该结果仅是单轴诊断，详见 `docs/PHASE16_OOD_GEOMETRY_REPORT.md`。
- [x] 完成 target-speed OOD：速度 `0.82/0.90` 各 50 episode，几何保持 ID；三种子 distributed 分支为 `42.67% [37.00%, 48.33%]` safe capture、`57.33%` timeout、0 collision/boundary，total p50/p95/p99 为 `83.94/106.72/130.87 ms`。该结果说明速度外推是主要性能瓶颈，详见 `docs/PHASE16_OOD_TARGET_SPEED_REPORT.md`。
- [x] 完成 delay/execution OOD：消息延迟 6/8 步、dropout 20/25%，并激活 2-step command delay 和 tracking/noise dynamics；三种子 distributed 分支为 `47.00% [41.00%, 53.00%]` safe capture、`53.00%` collision、`19.00%` boundary。该结果是当前 local-CBF execution-safe contract 的 No-Go，详见 `docs/PHASE16_OOD_DELAY_EXECUTION_REPORT.md`。
- [x] 完成 unseen target behavior OOD：两种未见 lookahead/commit 规则，各 50 episode，速度/几何/执行保持 ID；三种子 distributed 为 `94.33% [91.33%, 97.33%]` safe capture、0 collision/boundary。该结论限于两个冻结行为规则，详见 `docs/PHASE16_OOD_TARGET_BEHAVIOR_REPORT.md`。
- [ ] 冻结 stronger communication / execution delay / tracking-noise OOD 块，保持几何和目标策略在 ID 范围内。
- [ ] 每个 OOD 块固定三种子、镜像配对、source hash 和完整 episode/step 工件；报告 safe/ordinary capture、collision、boundary、timeout、clearance、动作条件可用率以及 p50/p95/p99。
- [ ] 在各单轴诊断完成前，不运行混合压力场景，也不基于 OOD 结果重新选 checkpoint、调整模型或改写 locked-test 主结论。

### P10：重新开放端到端三种子实验

- [x] 已完成 GRU 三种子 × 90 locked-test 场景的 validation-selected local-CBF 闭环确认；该结果是 locked-test diagnostic，不替代 P8 安全 gate。
- [ ] 只有 P8 通过后，才重开 robust CBF-QP/certified-fallback 端到端 locked-test。
- [ ] 固定同一 scenes、target-truth 隔离、seed、source hash 和日志格式。
- [ ] 至少保留 local CBF、robust CBF-QP、certified fallback 三条路径；不要只报告成功率。
- [ ] 同时报告 safe/ordinary capture、collision、boundary、timeout、capture time、minimum clearance、certificate、fallback、solver failure 和完整延迟。
- [ ] 端到端 gate：aggregate safe capture 不低于 local CBF baseline，collision/boundary 不增加，三种 seed 方向一致，且每个失败可追溯。

### P11：R-CLBF-QP 只在 P8/P10 通过后开展

- [ ] 明确定义 CLBF/robust CLBF、扰动集合、执行动力学和离散时间安全条件。
- [ ] 训练集、验证集、locked-test 完全分离，禁止用 locked-test 调 margin 或选模型。
- [ ] 独立 certificate checker 必须与训练损失和 QP 内部 residual 解耦。
- [ ] 证明或验证 multi-step forward invariance、fallback、solver tolerance、饱和、延迟和连续段安全。
- [ ] 若无法完成形式化闭环证明，贡献名称降级为“经验型/条件性 robust CBF-QP”，不使用 R-CLBF-QP 安全证明表述。

### P12：最终整理与论文证据

- [ ] 固化最终实验协议、配置、checkpoint SHA-256、场景 SHA-256 和源码 hash。
- [ ] 生成完整 ablation 表：GRU/SSM diffusion、raw/projected、expected/worst-case/CVaR、centralized/distributed、local/robust safety。
- [ ] 单独整理失败案例和 No-Go 结果，不删除失败运行目录。
- [ ] 更新 `RESULTS_INDEX.md`、阶段报告和 README 限制说明。
- [ ] 完整回归：`python -m pytest -q`、Python 编译检查、`git diff --check`。
- [ ] 仅提交代码、测试和正式报告；保留用户已有的 `README.md` 与 `docs/EXPERIMENTAL_STUDY_REPORT.md`，不得误暂存。

### P13：Phase 17 可复现三创新点路线

- [x] 按 `PHASE17_QUEUE_ADAPTIVE_REACHABILITY_TODOLIST.md` 完成第一版 Queue-Aware Delayed-State Rollout 适配器、单元测试、配置快照和 TensorBoard smoke 记录；它修正规划状态和候选时间索引，不重新打开 robust CBF-QP。
- [x] 完成 QDR 的 nominal immutable-prefix 几何诊断、post-hoc endpoint audit、TensorBoard 字段和 90 场景 validation paired comparison；QDR-on safe capture 为 `88.89%`，QDR-off 为 `94.44%`，collision 为 `11.11%/5.56%`，当前 gate No-Go。详见 `docs/PHASE17_QDR_VALIDATION_REPORT.md`。
- [x] Phase 17 geometry-OOD Diagonal-SSM 20-episode pilot 已完成；结果显示 no-QDR 100%/0% collision、QDR fixed-K 85%/15% collision、UAKR+QDR 90%/10% collision，仍不足以作为正式统计结论。
- [x] 完成 UAKR 的可解释策略、动态 K/refresh、缓存轨迹推进、残差触发与 TensorBoard 记录；当前 pilot 的 forced-refresh 偏高，尚未冻结阈值。
- [x] 完成 RNIC 的受限到达时间代价、central/local DN-MPC 接入、单元测试与 TensorBoard 记录；当前 pilot 未带来行为收益且明显增加 planner latency，尚未通过 speed-OOD gate。
- [x] 实现只使用 policy-safe uncertainty 的 K={1,4,8} 与 refresh={4,2,1} 自适应调度；主结果使用可移植 Diagonal SSM diffusion，GRU 保持 K=1 参考。当前仍需 validation 冻结阈值。
- [x] 实现基于加速度/限速到达时间裕量的 Reachability-Normalized Interception Cost；当前为 heuristic 版本，interceptor-only 与 formation-slot 消融仍未完成。
- [ ] 三个模块分别通过 pilot gate 后，运行 2×2×2 validation 全因子矩阵；按预注册规则保留最简单的通过组合。
- [ ] 方法冻结后才运行新建且未见的 ID/OOD confirmation；Phase 16 已公开 OOD 只作开发诊断。
- [ ] 不设 100 ms 硬门槛，但必须逐组件报告 p50/p95/p99，并在同硬件单进程条件下进行公平比较。

**P13 gate：** ID safe capture 满足 `-2 pp` 非劣；QDR/UAKR/RNIC 至少一个对应失败轴获得配对统计支持的改善；最终组合不增加 collision/boundary；所有阈值、seed、hash 和 step-level decision 可复现。详细 gate、指标、测试和六周日程见 `docs/PHASE17_QUEUE_ADAPTIVE_REACHABILITY_TODOLIST.md`。

## 7. 当前推荐执行顺序

```text
P7.2 完成独立证书工件
  -> P8 修复 reset/margin/action-authority/fallback 契约
  -> P8 多步执行扰动安全 gate
  -> P9 延迟架构优化
  -> P10 三种子端到端 locked-test
  -> P11 可选 R-CLBF-QP 与形式化证明
  -> P12 最终统计、复现和论文材料
```

在 P8 通过之前，不继续训练 learned CLBF，不继续扩大端到端矩阵，也不把 50% safe capture 的联合结果包装成成功结果。即使 P8 最终未通过，P2/P4 已完成的预测和 DN-MPC 模块化结果仍然可以独立形成阶段性贡献。

最新 continuous-QP 结论同样不改变该顺序：短 pilot 的 100% post-state safety 不能替代 250-step 闭环审计；在 recovery gate 通过前，不扩大到 30--50 unseen seeds，也不重新接入 Mamba/扩散/DN-MPC 端到端组合。
