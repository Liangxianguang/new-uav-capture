# Phase 38--40：QDR rollout cache 与候选压缩验证报告

> 日期：2026-09-13
> 实验级别：validation-development，仅用于可复现的运行时优化验证
> 仓库：[Liangxianguang/new-uav-capture](https://github.com/Liangxianguang/new-uav-capture)

## 1. 目的与边界

Phase 35 的 suffix gate/recovery candidates 修复了 development block 的
episode-level safety outcome，但 planner 和 total latency 仍明显高于 QDR-off。
Phase 36 已经把本地 suffix execution rollout 改成候选维度批量计算；Phase 38--40
继续只优化重复计算，不改变预测器、MPC cost、候选排序、执行动力学、authority、
安全层或 QDR gate：

1. 对同一控制步内重复出现的本地候选执行 rollout 做缓存；
2. 对同一控制步内重复出现的 peer message action path 做缓存；
3. 移除与加权 reference 完全相同的重复候选，同时保留不同候选的原始顺序；
4. 用固定候选预算 `max_local_candidate_paths=2` 做一个小规模 smoke，作为预算
   诊断，不把它当作正式性能结论。

这些优化没有获得新的控制 authority。immutable queue prefix 仍然不可被事后取消；
当所有 suffix 候选不可行时，系统仍选择最小违反候选并记录 gate exhaustion。因此
本阶段不构成 safety certificate，也不代表 QDR 已通过 promotion gate。

## 2. 可复现设置

| 项目 | 设置 |
| --- | --- |
| 场景 | `results/phase34_qdr_development_scenes_v2/scenes.jsonl` |
| 规模 | 40 episodes / 20 mirror groups |
| split | `validation-development`，不含 locked-test |
| 场景 SHA-256 | `4d99d829157145a6f747a4afde6dcbe7c0846e2bc44c5c0a5de5dbe65bd600d3` |
| predictor | GRU `both`，checkpoint seed `727201` |
| planner | distributed delayed DN-MPC，worst-case，horizon 8，control horizon 3 |
| execution | enabled，command delay 2，immutable，noise/tracking/drag 0 |
| safety | local CBF |
| sampling | `num_samples=1`，`sampling_steps=8`，`sampling_seed=745102` |
| device | CPU |
| main config | `configs/phase34_qdr_precondition_validation.yaml` |
| Phase39 budget smoke | `configs/phase39_qdr_candidate_budget2.yaml`，5 episodes |
| Phase40 run | `results/phase40_qdr_candidate_dedup_dev_seed727201` |

Phase40 的运行配置、checkpoint/source hashes、episode/step JSONL、summary 和
TensorBoard event files均保留在本地结果目录；结果目录由 Git ignore，不提交大文件。

## 3. 结果

### 3.1 Episode-level outcome

Phase 35--40 使用同一 development block。下表的 35--38 是逐步优化过程，Phase40
是在当前源码上加入 exact duplicate candidate removal 后的 40 集复测。

| 运行 | safe capture | collision | boundary | timeout | mean capture time | mean min clearance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Phase35 suffix gate/recovery | 100.0% | 0.0% | 0.0% | 0.0% | 3.915 s | 0.556 m |
| Phase36 batched rollout | 100.0% | 0.0% | 0.0% | 0.0% | 3.915 s | 0.556 m |
| Phase37 local rollout cache | 100.0% | 0.0% | 0.0% | 0.0% | 3.915 s | 0.556 m |
| Phase38 peer rollout cache | 100.0% | 0.0% | 0.0% | 0.0% | 3.915 s | 0.556 m |
| Phase40 candidate dedup | 100.0% | 0.0% | 0.0% | 0.0% | 3.915 s | 0.556 m |

Phase40 的最小 clearance 为 `0.218 m`，solver/effective-plan rate 为 `97.89%`。
QDR 机制指标为：prefix admissible `98.64%`、suffix admissible `86.80%`、gate
exhaustion `21.76%`、endpoint check coverage `93.36%`、平均 QDR queue length
`2.0`。候选去重把本次记录的 rejected candidate evaluations 从 Phase38 的
`28,856` 降到 `18,387`，但没有改变 episode outcome 或 suffix admissibility。

Phase39 的 5 集 candidate-budget smoke 同样为 `100%/0%/0%/0%`（safe
capture/collision/boundary/timeout），总延迟为 `45.22/68.42/77.23 ms`，仅用于
预算可行性诊断；未用来替代 Phase40 的 40 集复测。

### 3.2 延迟

所有延迟均为 `p50 / p95 / p99`（ms）：

| 运行 | predictor | planner | QDR | safety | total |
| --- | --- | --- | --- | --- | --- |
| QDR-off current reference | 11.03 / 22.04 / 23.80 | 12.26 / 22.51 / 25.27 | 0 / 0 / 0 | 1.16 / 2.52 / 2.85 | 27.84 / 46.53 / 51.59 |
| Phase35 | 10.94 / 21.90 / 23.77 | 87.56 / 127.46 / 146.08 | 0.77 / 1.42 / 1.68 | 1.18 / 2.53 / 2.89 | 107.44 / 150.21 / 168.13 |
| Phase36 | 10.98 / 21.83 / 23.53 | 57.10 / 82.39 / 93.80 | 0.79 / 1.39 / 1.72 | 1.21 / 2.55 / 2.90 | 78.06 / 105.27 / 120.09 |
| Phase37 | 10.58 / 21.99 / 23.82 | 49.99 / 73.51 / 84.17 | 0.76 / 1.40 / 1.72 | 1.15 / 2.53 / 2.89 | 69.71 / 96.68 / 105.40 |
| Phase38 | 11.04 / 22.11 / 24.01 | 26.84 / 42.55 / 49.36 | 0.77 / 1.45 / 1.71 | 1.18 / 2.55 / 2.84 | 45.99 / 66.11 / 75.03 |
| Phase40 | 10.76 / 21.79 / 23.83 | 24.51 / 40.43 / 46.77 | 0.77 / 1.43 / 1.74 | 1.17 / 2.58 / 2.94 | 43.61 / 63.77 / 72.40 |

相对 Phase35，Phase40 将 planner p95 从 `127.46` 降至 `40.43 ms`，total p95
从 `150.21` 降至 `63.77 ms`；相对 Phase36，total p95 再下降约 `39.4%`。不过
相对 QDR-off 的 total p95 `46.53 ms`，Phase40 仍增加约 `37.1%`，超过预注册的
`15%` 相对延迟上限（对应约 `53.51 ms`）。因此 100 ms 参考值虽然已满足，但
QDR 的相对效率 gate 仍未通过。

### 3.3 TensorBoard 与测试审计

Phase40 event directory 为：

```text
results/phase40_qdr_candidate_dedup_dev_seed727201/distributed_delayed/tensorboard/
```

审计得到 2 个 event files 和 221 个 scalar tags。QDR suffix gate、prefix/suffix
admissibility、predictor/planner/QDR/safety/total latency 的 p50/p95/p99 共 17
个必需标签全部存在，包括：

```text
Summary/QDR/suffix_admissible_rate
Summary/QDR/suffix_gate_exhaustion_rate
Summary/PredictorLatency/{p50,p95,p99}_ms
Summary/PlannerLatency/{p50,p95,p99}_ms
Summary/QDR/latency_{p50,p95,p99}_ms
Summary/SafetyLatency/{p50,p95,p99}_ms
Summary/TotalControlLatency/{p50,p95,p99}_ms
```

本阶段新增了“逐步候选去重不误删不同轨迹”的单元测试；当前源码上的完整回归为
`266 passed in 84.24 s`。

## 4. 阶段判定

**运行时实现：通过。** 本地候选和 peer action-path 缓存、exact duplicate removal
均保持 episode outcome、候选顺序语义和 suffix 指标不变；40 集开发复测与前序结果
一致，且 planner/total p95 有确定性下降。

**QDR promotion：仍为 No-Go。** gate exhaustion 仍为 `21.76%`，suffix admissible
仍为 `86.80%`，且 total p95 相对 QDR-off 超出 `15%` 门槛。当前证据可以写成
“execution-aware QDR 的可审计实现与运行时压缩”，不能写成“QDR 已证明提升延迟
鲁棒性”，更不能写成闭环安全证明。

## 5. 下一步 TodoList

- [x] 本地候选 execution rollout 候选维度批量化；
- [x] 缓存同一控制步内重复的 local candidate rollout；
- [x] 缓存同一控制步内重复的 peer action path；
- [x] 去除与 weighted reference 完全相同的重复候选；
- [x] 在 40 集 development block 上复核 outcome、机制指标和五级 latency；
- [ ] 用当前源码重新跑 QDR-off paired reference，确认缓存/去重没有引入行为变化；
- [ ] 在 fresh validation 上固定 QDR，逐轴跑 delay、authority、bounded execution
  noise confirmation；
- [ ] 只有相对 latency 和安全 outcome gates 同时通过后，才开放三种子确认和
  QDR×UAKR×RNIC 交互矩阵；
- [ ] 将候选早停、增量 rollout、warm-start 作为独立工程 ablation，不能与主效应
  混写；
- [ ] OOD 继续一次只改变一个因素，并保持 locked-test 不可见。
