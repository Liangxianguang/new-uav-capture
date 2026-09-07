# Phase 3 集中式 Scenario MPC 诊断报告

## 1. 决策

**实现状态：Go。科学结论：Conditional Go。**

集中式 scenario risk-sensitive MPC 已经可以消费 dynamics-projected 目标候选，并在不读取目标真值的条件下完成 expected、worst-case 和 CVaR 三种风险目标的闭环 rollout。当前正式小规模结果没有显示相对于 `DynamicEncirclementController + local CBF` 的捕获率增益，因此尚不能进入最终分布式 DN-MPC，也不能宣称 Phase 3 通过。

本阶段证明了：

- planner 的候选接口、风险聚合、fallback 和 local CBF 后置接口可运行；
- vectorized finite-shooting planner 的单步 planner 延迟约 9--12 ms；
- SSM diffusion 预测、planner 和 local CBF 组合的 total control p95 在本实验中低于 100 ms；
- 目标真值只用于环境内部的 capture/collision 评估，不进入预测候选或 planner。

本阶段没有证明：

- worst-case/CVaR 在困难场景上优于当前动态围捕控制器；
- 预测候选一定能改善真实围捕，而不只是改善轨迹预测指标；
- 分布式 DN-MPC 在通信延迟和丢包下有效；
- 任何 R-CLBF-QP 或闭环形式化安全保证。

## 2. 实现边界

新增实现：

```text
src/encirclement3d/minimax_mpc.py
scripts/evaluate_minimax_mpc.py
configs/innovation_mpc.yaml
tests/test_minimax_mpc.py
```

第一版采用集中式、速度级、短 horizon 的有限 shooting：

```text
U* = argmin_U risk({J(U, y_k)}_k)
```

其中 `risk` 可以是候选代价均值、最大值或上尾 CVaR。控制序列由参考候选轨迹、interceptor/blocker 角色、formation perimeter 和速度级离散 rollout 生成；同一个动作序列会在所有候选目标轨迹上评估，因此没有把“每个候选各自追踪”误写成 min-max。

障碍物、边界和机间距离在 planner 中作为 rollout penalty，名义动作随后进入当前 local CBF。这是诊断版设计，不是 R-CLBF-QP，也不把 planner penalty 当作硬安全约束。

## 3. 实验协议

- 预测 checkpoint：`results/phase2_formal_diffusion_seed745101/checkpoint.pt`。
- 预测 backend：`portable_diagonal_ssm`。
- 候选数：8；采样步数：8；固定 sampling seed：`745102` 加 episode 偏移。
- planner horizon：8 步；control horizon：3 步；`dt = 0.1 s`。
- 目标候选：全部经过 shared dynamics projection，planner 拒绝 `raw` 候选。
- 环境：`configs/capture_radius_pursuit_central_v4_flee.yaml`。
- 目标模式：`flee_persistence`。
- 障碍物数量：3；目标速度 scale：0.55。
- episode seeds：`648301--648308`，四种方法使用同一组 seeds。
- 安全层：当前 `PursuitCBFSafetyFilter`。
- locked-test 之外的诊断实验；不用于最终论文主结论。

复现实验命令：

```powershell
conda run --no-capture-output -n uav-encirclement-gpu python scripts/evaluate_minimax_mpc.py `
  --output-dir results/phase3_mpc_formal_small_seed648301 `
  --checkpoint results/phase2_formal_diffusion_seed745101/checkpoint.pt `
  --candidate-source checkpoint `
  --episodes 8 `
  --seed-start 648301 `
  --device cuda
```

每个方法目录保存：

```text
episodes.jsonl
steps.jsonl
summary.json
tensorboard/
```

根目录保存 `config.yaml`、source hash 和汇总 `summary.json`。

## 4. Formal-small 结果

以下结果来自 8 个 episode 的均值；这是诊断结果，不是完整 Phase 3 locked-test。

| 方法 | safe capture | collision | timeout | mean capture time (s) | mean min clearance (m) | worst min clearance (m) | solver success | fallback | total p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DynamicEncirclement + CBF | 1.000 | 0.000 | 0.000 | 0.9125 | 1.3849 | 1.1507 | N/A | 0.000 | 1.62 |
| Expected MPC + CBF | 1.000 | 0.000 | 0.000 | 0.9000 | 1.4336 | 0.9626 | 1.000 | 0.000 | 44.98 |
| Worst-case MPC + CBF | 1.000 | 0.000 | 0.000 | 0.9750 | 1.3380 | 0.7604 | 1.000 | 0.000 | 45.87 |
| CVaR MPC + CBF | 1.000 | 0.000 | 0.000 | 0.9000 | 1.4339 | 1.0080 | 1.000 | 0.000 | 52.24 |

planner p95 分别约为：expected `11.40 ms`、worst-case `11.37 ms`、CVaR `11.83 ms`。预测器 p95 分别约为：`33.50 ms`、`34.89 ms`、`39.94 ms`。local CBF p95 约为 `1.05--1.34 ms`。

## 5. 结果分析

### 5.1 正面结果

1. **接口成立。** planner 使用 `ScenarioTrajectorySet`，显式要求 `dynamics_status = projected`，raw diffusion 候选无法静默进入规划器。
2. **风险模式可区分。** expected、worst-case 和 CVaR 在同一候选集合上产生独立 objective 和 scenario cost 记录。
3. **实时性初步可行。** vectorized cost matrix 将此前约 600--700 ms 的 Python 循环降到约 9--12 ms planner p95；总控制 p95 在小规模实验中低于 100 ms。
4. **安全层边界清晰。** planner 只产生 nominal action，local CBF 仍是独立的后置安全层，动作修正和延迟均有日志。

### 5.2 不能过度解释的结果

- 所有方法在该简单 `flee_persistence` 场景上都达到 100% 捕获，因此没有足够难度区分规划器。
- worst-case 的 mean capture time 比基线更长，且 worst minimum clearance 低于 expected/CVaR，说明保守风险目标不自动等于更安全或更高效。
- 8 个 episode 不足以给出训练种子、场景分层和策略切换结论。
- 当前候选为单个冻结 checkpoint 的 uniform uncalibrated samples；没有把候选分数当作概率。

## 6. Phase 3 门槛检查

| 门槛 | 状态 | 说明 |
| --- | --- | --- |
| expected/worst-case/CVaR 可运行 | 通过 | 三种风险目标均有输出和 TensorBoard 记录 |
| planner 不访问 target truth | 通过 | 只读取 policy-safe observation 和候选轨迹 |
| solver/fallback 可审计 | 通过 | smoke 中 success=1、fallback=0，仍需困难场景压力测试 |
| planner p95 在预算内 | 通过（初步） | 11--12 ms；需要把困难场景和 CPU/低频部署纳入正式测试 |
| total control p95 在预算内 | 通过（初步） | 45.87--52.24 ms；仅针对本实验条件 |
| 困难场景安全捕获提升 >= 5 个百分点 | 未验证 | 当前 formal-small 太容易，所有方法均为 100% |
| 最坏候选 capture distance 降低 >= 10% | 未验证 | 当前评估还未形成完整 candidate-level capture rollout 指标 |
| 进入分布式 DN-MPC | 不通过 | 必须先补充 S3/S5/S6 困难条件并证明集中式收益 |

## 7. 下一步

在实现 DN-MPC 前必须完成：

1. 将评估扩展到随机混合障碍物、窄通道、delayed-noisy、丢包和未见目标策略。
2. 对每个真实 episode 保存候选级 rollout cost、worst-case candidate 和 CVaR candidate 统计。
3. 增加常速度候选、GRU projected、SSM projected 和 oracle target 的同场景消融。
4. 将正式场景 seed 固定为同一组，并至少运行三个 prediction checkpoint seed。
5. 若集中式 worst-case/CVaR 在困难场景仍没有收益，停止 DN-MPC 扩展，保留“多模态预测 + centralized diagnostic MPC”结论。

当前结论应写成：

> 集中式 scenario risk-sensitive MPC 已实现并满足初步实时性要求，但在简单 formal-small 场景上尚未证明相对 DynamicEncirclement 的性能增益；DN-MPC 暂不成立。
