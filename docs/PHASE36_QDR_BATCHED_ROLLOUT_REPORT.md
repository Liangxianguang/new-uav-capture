# Phase 36：QDR batched execution-rollout 优化验证报告

> 日期：2026-09-13
> 实验级别：validation-development，仅用于运行时优化验证
> 仓库：[Liangxianguang/new-uav-capture](https://github.com/Liangxianguang/new-uav-capture)

## 1. 目的与控制变量

Phase 35 的 suffix gate + recovery candidates 已恢复开发块的 episode-level 安全
outcome，但 planner p95 达到 `127.46 ms`。Phase 36 仅将本地候选 suffix 的执行动力学
rollout 从“逐候选 Python 循环”改为“候选维度批量、时间维度保留”的 NumPy rollout。
执行动力学、候选集合、gate 规则、MPC 权重、checkpoint、场景、采样 seed 和 safety
layer 均保持不变。

该优化不改变 QDR 的信息边界，也不引入新的 authority：immutable queue prefix 仍然
不可被事后取消；当所有 suffix 候选均不可行时仍只选择最小违反候选并记录 gate
exhaustion，因此本阶段不构成安全证明。

## 2. 可复现设置

| 项目 | 设置 |
| --- | --- |
| 场景 | `results/phase34_qdr_development_scenes_v2/scenes.jsonl` |
| 规模 | 40 episodes / 20 mirror groups |
| split | `validation-development`，不含 locked-test |
| 场景 SHA-256 | `4d99d829157145a6f747a4afde6dcbe7c0846e2bc44c5c0a5de5dbe65bd600d3` |
| predictor | 固定 GRU `both`，checkpoint seed `727201` |
| planner | distributed delayed DN-MPC，worst-case，horizon 8，control horizon 3 |
| execution | enabled，command delay 2，immutable，noise/tracking/drag 0 |
| safety | local CBF |
| sampling | `num_samples=1`，`sampling_steps=8`，`sampling_seed=745102` |
| run | `results/phase36_qdr_batched_rollout_dev_seed727201` |

## 3. 结果

### 3.1 episode-level outcome

| 指标 | Phase35 QDR gate/recovery | Phase36 batched rollout | 变化 |
| --- | ---: | ---: | ---: |
| safe capture | 100.0% | 100.0% | 0 pp |
| collision | 0.0% | 0.0% | 0 pp |
| boundary violation | 0.0% | 0.0% | 0 pp |
| timeout | 0.0% | 0.0% | 0 pp |
| mean capture time | 3.915 s | 3.915 s | 0 s |
| mean minimum clearance | 0.556 m | 0.556 m | 0 m |
| worst minimum clearance | 0.218 m | 0.218 m | 0 m |
| suffix admissible rate | 86.80% | 86.80% | 0 pp |
| gate exhaustion rate | 21.76% | 21.76% | 0 pp |

结果表明批量化只改变计算路径，没有改变控制决策和安全 outcome；这也是该优化
可以接受的回归条件。

### 3.2 latency

所有延迟均为 `p50 / p95 / p99`（ms）：

| 运行 | predictor | planner | QDR | safety | total |
| --- | --- | --- | --- | --- | --- |
| Phase35 | 10.94 / 21.90 / 23.77 | 87.56 / 127.46 / 146.08 | 0.77 / 1.42 / 1.68 | 1.18 / 2.53 / 2.89 | 107.44 / 150.21 / 168.13 |
| Phase36 | 10.98 / 21.83 / 23.53 | 57.10 / 82.39 / 93.80 | 0.79 / 1.39 / 1.72 | 1.21 / 2.55 / 2.90 | 78.06 / 105.27 / 120.09 |

相对 Phase35，planner p95 降低约 `35.1%`，total p95 降低约 `30.0%`；但相对
QDR-off 当前源码参考的 total p95 `46.53 ms`，Phase36 仍增加约 `126%`。100 ms
仍然只是参考值：Phase36 total p95 为 `105.27 ms`，planner p95 为 `82.39 ms`，
不能用“接近 100 ms”替代预注册的相对延迟门槛。

### 3.3 机制与求解诊断

- suffix gate active：`100%`；
- gate exhaustion：`21.76%`；
- rejected candidates：`28,856`；
- QDR prefix/suffix admissible：`98.64%/86.80%`；
- planner effective-plan rate：`97.89%`；
- 独立执行状态 auditor 的 current/next certificate valid rate：`98.28%/99.17%`，
  仍仅是诊断覆盖率。

## 4. TensorBoard 与测试审计

最新 run 保留 config、summary、episode/step JSONL 和 TensorBoard event file。事件
文件审计得到 `219` 个 scalar tags；要求的 QDR gate、predictor/planner/QDR/safety/
total latency p50/p95/p99 标签均存在。

批量 rollout 增加了与逐候选 rollout 的逐元素一致性测试；专项测试 `38 passed`，
此前全量回归基线为 `264 passed`。提交前还需在当前源码上再次运行全量回归。

## 5. 阶段判定

**计算实现：通过。** 在完全不改变 episode outcome、候选集合和 QDR 规则的条件下，
批量化显著降低 planner/total 延迟。

**QDR 性能晋级：仍为 No-Go。** total p95 相对 QDR-off 仍远超 `15%` 增幅门槛，
gate exhaustion 仍约五分之一，且没有改善 OOD。当前结果支持“执行 rollout 的可复现
运行时优化”，不支持“QDR 已通过延迟鲁棒性 gate”。

## 6. 下一步 TodoList

- [x] 对候选维度批量化执行 dynamics rollout；
- [x] 用逐候选参考实现做数值等价测试；
- [x] 在 40 集 development block 上复核安全 outcome 和五级 latency；
- [ ] 对 `qdr_candidate_violation` 做障碍/peer/boundary 的早停与缓存，避免对已
  明显不可行 suffix 重复计算；
- [ ] 评估固定候选预算、增量 peer rollout 和 warm-start，保持候选集合与安全规则
  可审计；
- [ ] 重新跑同一开发块的 QDR-off/QDR-on paired benchmark，确认 total p95 增幅
  是否达到 `15%` 以内；
- [ ] 只有效率 gate 通过后，才执行 fresh three-seed delay/authority/noise
  confirmation；之后才开放 UAKR/RNIC 组合。
