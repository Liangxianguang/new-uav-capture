# Phase 35：QDR suffix gate 与 recovery candidates 开发验证报告

> 日期：2026-09-13
> 实验级别：validation-development，仅用于实现验证与下一阶段决策
> 仓库：[Liangxianguang/new-uav-capture](https://github.com/Liangxianguang/new-uav-capture)

## 1. 阶段目的与边界

Phase 34 已确认 QDR 的执行感知 rollout 与 planner 接口可以工作，但大量新规划
suffix 不可行；在 `immutable` authority 下，已提交且不安全的 queue prefix 也不
可能被后续动作事后修复。Phase 35 在此基础上加入两项机制：

1. 在 centralized/distributed candidate scoring 中加入 suffix-feasibility gate；
2. 在 gate 的正常候选之外加入零动作 suffix 与当前速度保持 suffix，作为有限的
   recovery candidates。

当存在可行候选时，gate 对不可行候选施加有限的大惩罚；当所有候选都不可行时，
仍选择最小违反候选并记录 `qdr_suffix_gate_exhausted`。因此该实现改善了候选选择
和可观测性，但 gate exhaustion 本身不是 safety certificate，也不能修复
immutable prefix。

本阶段只使用 validation-development 场景，不读取 locked-test，不用本结果回调
 predictor、MPC 权重或安全阈值，也没有重新接入 robust CBF-QP。

## 2. 可复现设置

| 项目 | 设置 |
| --- | --- |
| 场景 | `results/phase34_qdr_development_scenes_v2/scenes.jsonl` |
| 场景规模 | 40 episodes / 20 mirror groups |
| split | `validation_development`，`locked_test_included=false` |
| 场景 SHA-256 | `4d99d829157145a6f747a4afde6dcbe7c0846e2bc44c5c0a5de5dbe65bd600d3` |
| predictor | 固定 GRU `both` checkpoint，seed `727201` |
| planner | distributed delayed DN-MPC，worst-case，horizon 8，control horizon 3 |
| safety | local CBF |
| execution | enabled，command delay 2，immutable authority，noise 0，tracking/drag 0 |
| sampling | `num_samples=1`，`sampling_steps=8`，`sampling_seed=745102` |
| run | `results/phase35_qdr_recovery_dev_seed727201` |
| decision | `validation_selection` |

QDR-off 对照沿用同一场景、checkpoint、采样和执行设置的当前源码运行
`phase34_qdr_dev_v6_off_delay2_current_seed727201`。QDR-on 运行开启
`queue_aware_rollout`、`queue_aware_safety_projection`、suffix gate 和 recovery
candidates。

## 3. 结果

### 3.1 episode-level outcome

| 指标 | Phase34 QDR-off | Phase35 QDR-on + gate/recovery | 相对 QDR-off |
| --- | ---: | ---: | ---: |
| safe capture | 100.0% | 100.0% | 0 pp |
| ordinary capture | 100.0% | 100.0% | 0 pp |
| collision | 0.0% | 0.0% | 0 pp |
| boundary violation | 0.0% | 0.0% | 0 pp |
| timeout | 0.0% | 0.0% | 0 pp |
| mean capture time | 1.963 s | 3.915 s | +1.953 s |
| mean minimum clearance | 0.446 m | 0.556 m | +0.110 m |
| worst minimum clearance | 0.222 m | 0.218 m | -0.004 m |
| planner effective-plan rate | 100.0% | 97.89% | -2.11 pp |

相对于 Phase34 未加 recovery candidates 的 QDR-on 版本，safe capture 从 `87.5%`
恢复到 `100.0%`，collision 从 `12.5%` 降到 `0%`；suffix admissible rate 从
`48.03%` 提升到 `86.80%`。这是开发块上的工程性改善，不是三种子统计确认。

### 3.2 QDR 诊断

- suffix gate active rate：`100.0%`；
- gate exhaustion rate：`21.76%`；
- rejected candidates：`28,856` 次局部候选拒绝计数；该计数不是候选总数占比；
- queue length / first controllable step：`2 / 2`；
- prefix admissible rate：`98.64%`；suffix admissible rate：`86.80%`；
- endpoint check coverage：`93.36%`；endpoint position/velocity error：`0`；
- precondition status：`1327` 个 `prefix_safe_suffix_safe`、`206` 个
  `prefix_safe_suffix_unsafe`、`33` 个 `prefix_unsafe_unrecoverable`；
- 独立当前/下一状态 certificate auditor 的 valid rate 为 `98.28%/99.17%`，
  仅作诊断覆盖率，不能写成闭环安全证明。

gate exhaustion 仍达到约五分之一的规划步，说明当前动作候选空间和执行动力学
之间还有明显不匹配；这也是本阶段不能直接晋级为主方法的关键原因。

### 3.3 latency

所有延迟均报告为 `p50 / p95 / p99`（ms）：

| 运行 | predictor | planner | QDR | safety | total |
| --- | --- | --- | --- | --- | --- |
| QDR-off | 11.03 / 22.04 / 23.80 | 12.26 / 22.51 / 25.27 | 0 / 0 / 0 | 1.16 / 2.52 / 2.85 | 27.84 / 46.53 / 51.59 |
| QDR-on + gate/recovery | 10.94 / 21.90 / 23.77 | 87.56 / 127.46 / 146.08 | 0.77 / 1.42 / 1.68 | 1.18 / 2.53 / 2.89 | 107.44 / 150.21 / 168.13 |

QDR-on 的 total p95 相对 QDR-off 增加约 `223%`，planner p95 增加约 `105 ms`。
100 ms 不是硬门槛，但当前 planner p95 和 total p95 都明显高于该参考值；同时
平均捕获时间约翻倍。因此 recovery candidates 解决了安全 outcome，却引入了
明显的 liveness/compute cost，不能称为完整成功。

## 4. TensorBoard 与 artifact 审计

最新运行目录保留：

```text
results/phase35_qdr_recovery_dev_seed727201/config.yaml
results/phase35_qdr_recovery_dev_seed727201/summary.json
results/phase35_qdr_recovery_dev_seed727201/distributed_delayed/episodes.jsonl
results/phase35_qdr_recovery_dev_seed727201/distributed_delayed/steps.jsonl
results/phase35_qdr_recovery_dev_seed727201/distributed_delayed/tensorboard/
```

TensorBoard event file 共审计到 `219` 个 scalar tags，包含：

- `Summary/QDR/suffix_gate_active_rate`、`suffix_gate_exhaustion_rate`、
  `suffix_gate_rejected_candidates`；
- `Summary/PredictorLatency/*`、`Summary/PlannerLatency/*`、
  `Summary/QDR/latency_*`、`Summary/SafetyLatency/*` 和
  `Summary/TotalControlLatency/*` 的 p50/p95/p99；
- episode/step 级 queue length、prefix/suffix admissibility、recovery 状态和
  planner status。

## 5. 阶段判定

**实现与安全 outcome：开发验证通过。** recovery candidates 使本开发块达到
`100%` safe capture、`0%` collision/boundary/timeout，并保留了可审计的 suffix gate
状态。

**性能晋级：No-Go。** gate exhaustion 仍为 `21.76%`，planner/total p95 分别为
`127.46/150.21 ms`，且 mean capture time 从 `1.963 s` 增至 `3.915 s`。因此不能
进入 QDR confirmation 或 locked-test，也不能把本阶段结果写成已证明的延迟鲁棒性
提升。当前最准确的表述是：

> QDR execution-aware candidate gate + finite recovery candidates can recover
> episode-level safety on the development block, but its computational and
> liveness cost remains unresolved.

## 6. 下一步 TodoList

- [x] 在 centralized/distributed candidate stage 接入 suffix-feasibility gate；
- [x] 增加零动作与当前速度保持 recovery candidates；
- [x] 增加 gate active/exhaustion/rejected-candidate 的 JSONL 与 TensorBoard 记录；
- [x] 完成 40-episode 开发验证和 `264 passed` 全量回归；
- [ ] 做候选批量化、增量 rollout、可行性预筛和重复 suffix 缓存，先把 gate
  exhaustion 降到预注册目标以下并将 total p95 增幅控制在 `15%` 内；
- [ ] 在新的 development-calibration 场景上做 recovery candidate 与普通 candidate
  的逐步失败回放，区分 immutable prefix failure 与 suffix candidate failure；
- [ ] 只有效率 gate 通过后，才按 `delay={0,1,2,4}`、authority、bounded
  execution-noise 做 fresh three-seed confirmation；
- [ ] QDR 通过前不开放 QDR×UAKR×RNIC full factorial，不重新接入 robust CBF-QP；
  robust CBF-QP 仍是独立诊断 No-Go。
